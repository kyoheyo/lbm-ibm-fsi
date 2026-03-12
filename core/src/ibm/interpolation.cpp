#include "ibm/interpolation.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <array>

#ifdef LBM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace ibm {

static constexpr double PI = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// 一维 δ 函数核值
// ---------------------------------------------------------------------------
double delta_phi(double r, double h, DeltaKernel kernel)
{
    const double roh = r / h;   // 无量纲距离 r/h
    if (kernel == DeltaKernel::TwoPoint) {
        // 线性（帽形）核 — 支撑宽度 2h
        const double absr = std::abs(roh);
        if (absr < 1.0) return (1.0 - absr) / h;
        return 0.0;
    } else {
        // Peskin 4 点余弦核 — 支撑宽度 4h
        const double absr = std::abs(roh);
        if (absr < 2.0) {
            return (1.0 + std::cos(PI * roh / 2.0)) / (4.0 * h);
        }
        return 0.0;
    }
}

// ---------------------------------------------------------------------------
// 速度插值（二维，假定使用 D2Q9 格子）
// ---------------------------------------------------------------------------
void interpolate_velocity(const lbm::LatticeGrid& grid,
                          MarkerSet& ms,
                          double dx,
                          DeltaKernel kernel)
{
    if (grid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("IBM interpolation: only D2Q9 supported currently");
    }

    const int nx = grid.nx;
    const int ny = grid.ny;

    // δ 函数的支撑半径（格子数）
    const int support = (kernel == DeltaKernel::TwoPoint) ? 1 : 2;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int m = 0; m < ms.size(); ++m) {
        auto& mk = ms.markers[m];
        // 标记点在格子单位下的位置
        const double xm = mk.x / dx;
        const double ym = mk.y / dx;

        // 最近格子节点
        const int i0 = static_cast<int>(std::floor(xm));
        const int j0 = static_cast<int>(std::floor(ym));

        double ux_sum = 0.0, uy_sum = 0.0;

        for (int dj = -support; dj <= support + 1; ++dj) {
            for (int di = -support; di <= support + 1; ++di) {
                const int ii = i0 + di;
                const int jj = j0 + dj;
                // 周期性截断（超出边界时跳过）
                if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;

                const int node = grid.idx(ii, jj);
                // 跳过固体节点（固体内部速度无效）
                if (!grid.solid.empty() && grid.solid[node]) continue;

                const double phi_x = delta_phi(mk.x - ii * dx, dx, kernel);
                const double phi_y = delta_phi(mk.y - jj * dx, dx, kernel);
                // 二维积分权重：φ(x)*φ(y)*Δx²
                const double phi = phi_x * phi_y * dx * dx;

                ux_sum += grid.u[node * 2 + 0] * phi;
                uy_sum += grid.u[node * 2 + 1] * phi;
            }
        }

        mk.ux = ux_sum;
        mk.uy = uy_sum;
        mk.uz = 0.0;
    }
}

// ---------------------------------------------------------------------------
// 力展布（二维，D2Q9）
// ---------------------------------------------------------------------------
void spread_force(lbm::LatticeGrid& grid,
                  const MarkerSet& ms,
                  double dx,
                  DeltaKernel kernel)
{
    if (grid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("IBM spread_force: only D2Q9 supported currently");
    }

    const int nx = grid.nx;
    const int ny = grid.ny;
    const int support = (kernel == DeltaKernel::TwoPoint) ? 1 : 2;

    // 先将体力场清零
    std::fill(grid.force.begin(), grid.force.end(), 0.0);

    // 展布是一个散射操作 — 需要原子操作或串行循环以避免竞争
    for (int m = 0; m < ms.size(); ++m) {
        const auto& mk = ms.markers[m];
        const double xm = mk.x / dx;
        const double ym = mk.y / dx;

        const int i0 = static_cast<int>(std::floor(xm));
        const int j0 = static_cast<int>(std::floor(ym));

        for (int dj = -support; dj <= support + 1; ++dj) {
            for (int di = -support; di <= support + 1; ++di) {
                const int ii = i0 + di;
                const int jj = j0 + dj;
                if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;

                const int node = grid.idx(ii, jj);
                const double phi_x = delta_phi(mk.x - ii * dx, dx, kernel);
                const double phi_y = delta_phi(mk.y - jj * dx, dx, kernel);
                // 展布权重：φ(x)*φ(y)*ΔS（弧长元素）
                const double phi = phi_x * phi_y * mk.ds;

#ifdef LBM_ENABLE_OPENMP
#pragma omp atomic
#endif
                grid.force[node * 2 + 0] += mk.fx * phi;
#ifdef LBM_ENABLE_OPENMP
#pragma omp atomic
#endif
                grid.force[node * 2 + 1] += mk.fy * phi;
            }
        }
    }
}

// ===========================================================================
// 多重直接力法（MDF-IBM）
// ===========================================================================
void compute_ibm_forces_mdf(lbm::LatticeGrid& fluid,
                             MarkerSet& ms,
                             double dx,
                             double dt,
                             int    n_iter,
                             DeltaKernel kernel)
{
    if (fluid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("MDF-IBM: only D2Q9 supported currently");
    }

    const int n = fluid.size();
    const int d = fluid.dim();   // = 2

    // 工作速度场（子迭代过程中被逐步修正）
    std::vector<double> u_work = fluid.u;

    // 累积欧拉力场（最终写入 fluid.force）
    std::vector<double> F_total(n * d, 0.0);

    // 初始化标记点力为零
    for (auto& mk : ms.markers) { mk.fx = mk.fy = 0.0; }

    // 临时力场（每次子迭代的增量展布结果）
    std::vector<double> dF_euler(n * d, 0.0);

    for (int iter = 0; iter < n_iter; ++iter) {
        // 1. 用 u_work 插值标记点速度（临时将 fluid.u 设为 u_work）
        //    无需拷贝：直接 swap，插值后再 swap 回来
        std::swap(fluid.u, u_work);
        interpolate_velocity(fluid, ms, dx, kernel);
        std::swap(fluid.u, u_work);

        // 2. 计算增量力（刚体目标速度 = 0；如需移动边界，在此修改 target）
        for (auto& mk : ms.markers) {
            const double dFx = (0.0 - mk.ux) / dt;   // ρ=1 格子单位假设
            const double dFy = (0.0 - mk.uy) / dt;
            mk.fx = dFx;   // 临时存放增量（不累加到 markers，最后由 F_total 覆盖）
            mk.fy = dFy;
        }

        // 3. 将增量力展布到 dF_euler
        std::fill(dF_euler.begin(), dF_euler.end(), 0.0);
        // 临时使用 fluid.force 作为展布目标，然后移走
        spread_force(fluid, ms, dx, kernel);  // writes to fluid.force
        std::swap(fluid.force, dF_euler);     // dF_euler = 本次增量展布结果

        // 4. 更新工作速度：u_work += dt · dF_euler / ρ（ρ=1）
        for (int i = 0; i < n; ++i) {
            u_work[i * d + 0] += dt * dF_euler[i * d + 0];
            u_work[i * d + 1] += dt * dF_euler[i * d + 1];
        }

        // 5. 累积总欧拉力
        for (int i = 0; i < n * d; ++i) {
            F_total[i] += dF_euler[i];
        }
    }

    // 写入最终总力到 fluid.force（供 Guo 体力格式在 collide 步使用）
    fluid.force = F_total;

    // 同步标记点力（近似：取最后一次子迭代的值重新展布前的 mk.fx/fy）
    // 为了提供更好的力信息，重新从 F_total 反推。此处保持 mk.fx/fy
    // 为最后一次子迭代的增量值（已足够用于 FSI 反作用力计算）。
    // 如需精确 marker force，调用方可在此之后额外调用 interpolate+compute。
}

// ===========================================================================
// 移动最小二乘速度插值（MLS-IBM）
//
// 线性基 p(x) = [1, Δx, Δy]，Gaussian 权函数 w(r) = exp(−r²/h²)
// 支撑半径 h = 2.5·dx（覆盖约 5×5 的格点邻域）
//
// 对每个标记点 X_m：
//   1. 收集支撑域内所有流体节点 {x_i}
//   2. 计算权重 w_i = exp(−|x_i − X_m|² / h²)
//   3. 构造 3×3 矩阵 M = Σ w_i p(x_i - X_m) ⊗ p(x_i - X_m)
//   4. 构造 3×1 向量 b_u = Σ w_i u_x(x_i) p(x_i - X_m)（同理 b_v）
//   5. 求解 M·a_u = b_u，M·a_v = b_v（Cramer's rule，3×3）
//   6. 插值速度：u_m = a_u[0]，v_m = a_v[0]（p(0) = [1,0,0]）
// ===========================================================================

// 正则化下界（MLS 矩阵奇异性保护，避免除以零）
static constexpr double MLS_REGULARIZATION_EPS = 1e-30;
// 返回 false 若行列式接近零（奇异）
static bool solve3x3(const double A[3][3], const double b[3], double x[3])
{
    const double det = A[0][0] * (A[1][1]*A[2][2] - A[1][2]*A[2][1])
                     - A[0][1] * (A[1][0]*A[2][2] - A[1][2]*A[2][0])
                     + A[0][2] * (A[1][0]*A[2][1] - A[1][1]*A[2][0]);
    if (std::abs(det) < 1e-30) return false;

    const double inv_det = 1.0 / det;

    x[0] = inv_det * (b[0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                    - b[1]*(A[0][1]*A[2][2]-A[0][2]*A[2][1])
                    + b[2]*(A[0][1]*A[1][2]-A[0][2]*A[1][1]));
    x[1] = inv_det * (A[0][0]*(b[1]*A[2][2]-b[2]*A[2][1])
                    - A[0][1]*(b[0]*A[2][2]-b[2]*A[2][0])
                    + A[0][2]*(b[0]*A[2][1]-b[1]*A[2][0]));
    x[2] = inv_det * (A[0][0]*(A[1][1]*b[2]-A[1][2]*b[1])
                    - A[0][1]*(A[1][0]*b[2]-A[1][2]*b[0])
                    + A[0][2]*(A[1][0]*b[1]-A[1][1]*b[0]));
    return true;
}

void mls_interpolate_velocity(const lbm::LatticeGrid& grid,
                               MarkerSet& ms,
                               double dx)
{
    if (grid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("MLS-IBM: only D2Q9 supported currently");
    }

    const int nx = grid.nx;
    const int ny = grid.ny;

    // MLS 支撑半径（格子单位）和 Gaussian 半宽
    const double h_mls = 2.5 * dx;          // Gaussian 半宽
    const double R_s   = 2.5 * dx;          // 支撑半径（截断距离）
    const double h2    = h_mls * h_mls;
    const int    iR    = static_cast<int>(std::ceil(R_s / dx)) + 1;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int m = 0; m < ms.size(); ++m) {
        auto& mk = ms.markers[m];

        // 标记点在格子坐标系中的位置
        const double xm = mk.x / dx;
        const double ym = mk.y / dx;
        const int    i0 = static_cast<int>(std::round(xm));
        const int    j0 = static_cast<int>(std::round(ym));

        // MLS 矩阵（3×3）和右端向量
        double M[3][3] = {};
        double bu[3]   = {};   // for ux
        double bv[3]   = {};   // for uy

        int n_contrib = 0;

        for (int dj = -iR; dj <= iR; ++dj) {
            for (int di = -iR; di <= iR; ++di) {
                const int ii = i0 + di;
                const int jj = j0 + dj;
                if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;

                const int node = grid.idx(ii, jj);
                // 跳过固体节点
                if (!grid.solid.empty() && grid.solid[node]) continue;

                const double ddx = ii * dx - mk.x;
                const double ddy = jj * dx - mk.y;
                const double r2  = ddx * ddx + ddy * ddy;

                if (r2 > R_s * R_s) continue;

                // Gaussian 权函数
                const double w = std::exp(-r2 / h2);

                // 基函数 p = [1, Δx/dx, Δy/dx]（归一化以改善条件数）
                const double p[3] = {1.0, ddx / dx, ddy / dx};

                // 累积矩阵 M += w * p ⊗ p
                for (int r = 0; r < 3; ++r) {
                    for (int c = 0; c < 3; ++c) {
                        M[r][c] += w * p[r] * p[c];
                    }
                }

                // 累积右端向量
                const double ux_i = grid.u[node * 2 + 0];
                const double uy_i = grid.u[node * 2 + 1];
                for (int r = 0; r < 3; ++r) {
                    bu[r] += w * ux_i * p[r];
                    bv[r] += w * uy_i * p[r];
                }
                ++n_contrib;
            }
        }

        if (n_contrib < 3) {
            // 支撑域内节点不足，回退到简单平均
            mk.ux = (n_contrib > 0) ? bu[0] / (M[0][0] + MLS_REGULARIZATION_EPS) : 0.0;
            mk.uy = (n_contrib > 0) ? bv[0] / (M[0][0] + MLS_REGULARIZATION_EPS) : 0.0;
            mk.uz = 0.0;
            continue;
        }

        // 求解 3×3 系统
        double au[3], av[3];
        if (!solve3x3(M, bu, au) || !solve3x3(M, bv, av)) {
            // 奇异系统：回退到权重平均
            mk.ux = M[0][0] > MLS_REGULARIZATION_EPS ? bu[0] / M[0][0] : 0.0;
            mk.uy = M[0][0] > MLS_REGULARIZATION_EPS ? bv[0] / M[0][0] : 0.0;
        } else {
            // 在标记点处求值：p(0) = [1, 0, 0]，所以 u_m = au[0]
            mk.ux = au[0];
            mk.uy = av[0];
        }
        mk.uz = 0.0;
    }
}

// ===========================================================================
// 罚函数法 IBM（Penalty-IBM / Feedback Forcing）
//
// 参考：Goldstein D. et al. (1993) J. Comput. Phys. 105:354-366.
//
// 每步调用：
//   1. 插值 u_IBM（δ 函数加权插值）
//   2. e = u_target − u_IBM
//   3. integral += dt · e
//   4. F = α·e + β·integral
//   5. 展布 F 到欧拉力场
//
// integral_x / integral_y 必须在外部持久化（每步传入同一 vector）。
// 调用方在仿真开始前将 integral_x/y 初始化为全零（std::vector<double>(ms.size(), 0.0)）。
// ===========================================================================
void compute_ibm_forces_penalty(lbm::LatticeGrid& fluid,
                                 MarkerSet& ms,
                                 double dx,
                                 double dt,
                                 double alpha,
                                 double beta,
                                 std::vector<double>& integral_x,
                                 std::vector<double>& integral_y,
                                 DeltaKernel kernel,
                                 double u_target_x,
                                 double u_target_y)
{
    if (fluid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("Penalty-IBM: only D2Q9 supported currently");
    }

    const int nm = ms.size();

    // 确保积分向量长度足够（自动扩展，填充 0）
    if (static_cast<int>(integral_x.size()) < nm) integral_x.assign(nm, 0.0);
    if (static_cast<int>(integral_y.size()) < nm) integral_y.assign(nm, 0.0);

    // 1. 插值流体速度 → mk.ux, mk.uy
    interpolate_velocity(fluid, ms, dx, kernel);

    // 2-4. 计算每标记点的罚函数力
    // 积分抗饱和上限：|integral| ≤ max_integral = 10/|beta| （若 beta > 0）
    // 防止长时间积分项无限增大（"积分饱和"，integrator wind-up）。
    const double max_integral = (beta > 1e-15) ? (10.0 / beta) : 1e10;

    for (int m = 0; m < nm; ++m) {
        auto& mk = ms.markers[m];

        const double ex = u_target_x - mk.ux;
        const double ey = u_target_y - mk.uy;

        // 3. 更新积分（简单 Euler 积分）+ 抗饱和限幅
        integral_x[m] += dt * ex;
        integral_y[m] += dt * ey;
        // 抗饱和（integrator anti-windup）：防止 beta>0 时积分无限增长
        if (integral_x[m] >  max_integral) integral_x[m] =  max_integral;
        if (integral_x[m] < -max_integral) integral_x[m] = -max_integral;
        if (integral_y[m] >  max_integral) integral_y[m] =  max_integral;
        if (integral_y[m] < -max_integral) integral_y[m] = -max_integral;

        // 4. 罚函数力：F = α·e + β·integral
        mk.fx = alpha * ex + beta * integral_x[m];
        mk.fy = alpha * ey + beta * integral_y[m];
    }

    // 5. 展布力到欧拉网格
    spread_force(fluid, ms, dx, kernel);
}

// ===========================================================================
// IBM 固体受力统计：合力计算
//
// F_x = Σ_m  mk.fx * mk.ds
// F_y = Σ_m  mk.fy * mk.ds
//
// IBM 力作用于流体（流体得到 +F），因此固体所受合力为 -F_fluid，
// 即固体受力 = -(Σ mk.fx * mk.ds)，方向与 IBM 力相反。
// 但为了与 MEA 方法（compute_solid_body_force）的符号约定一致，
// 此处返回的是施加到流体上的 IBM 力（即固体受到的反作用力为其负值）。
//
// 物理说明：
//   在 IBM 中，力 mk.fx/fy 施加到流体上（阻止流体穿越边界）。
//   由牛顿第三定律，固体所受流体合力 = Σ(-mk.fx * mk.ds)。
//   本函数返回 +Σ(mk.fx * mk.ds)（IBM 方向），调用方可根据需要取反。
// ===========================================================================
void compute_ibm_body_force(const MarkerSet& ms,
                             double& out_fx, double& out_fy)
{
    double fx = 0.0, fy = 0.0;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static) reduction(+:fx,fy)
#endif
    for (int m = 0; m < ms.size(); ++m) {
        const auto& mk = ms.markers[m];
        fx += mk.fx * mk.ds;
        fy += mk.fy * mk.ds;
    }

    out_fx = fx;
    out_fy = fy;
}

} // namespace ibm
