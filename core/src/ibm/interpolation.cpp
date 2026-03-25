#include "ibm/interpolation.hpp"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <array>

#ifdef LBM_ENABLE_OPENMP
#include <omp.h>
#endif

#ifdef LBM_ENABLE_MPI
#include <mpi.h>
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

        // 最近格子节点（本地坐标）
        const int i0 = static_cast<int>(std::floor(xm));
        const int j0 = static_cast<int>(std::floor(ym));

        // MPI 归属过滤：仅处理中心落在本进程物理域内的标记点，避免跨块重复计算
        if (i0 < ms.owner_i_lo || i0 >= ms.owner_i_hi) continue;
        if (j0 < ms.owner_j_lo || j0 >= ms.owner_j_hi) continue;

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

        // MPI 归属过滤：仅处理中心落在本进程物理域内的标记点，避免双重计数
        if (i0 < ms.owner_i_lo || i0 >= ms.owner_i_hi) continue;
        if (j0 < ms.owner_j_lo || j0 >= ms.owner_j_hi) continue;

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
//
// 实现参考：
//   Wang et al. (2008) Int. J. Multiphase Flow 34:283–302   — 原始 MDF 方法
//   Suzuki & Inamuro (2011) Comput. Fluids 49:173–187       — MDF 的 LBM 版本
//
// 算法（LBM 格子单位，Dx = Dt = 1）：
//   Step 0：g₀(Xₖ) = Uₖ − u*(Xₖ)
//   迭代 l = 0 … n_iter−1：
//     Step 1：将 gₗ(Xₖ) 展布到欧拉网格：gₗ(x) = Σₖ gₗ(Xₖ)·W(x−Xₖ)·ΔV
//     Step 2：修正工作速度：u_{l+1}(x) = u*(x) + Σᵢ≤ₗ gᵢ(x)
//     Step 3：插值：uₗ(Xₖ) = Σₓ uₗ(x)·W(x−Xₖ)·Δx²
//     Step 4：更新拉格朗日力：g_{l+1}(Xₖ) = gₗ(Xₖ) + (Uₖ − uₗ(Xₖ))
//   最终欧拉体力 = Σₗ spread(gₗ − g_{l-1}) = spread(g_{n_iter})
//   最终拉格朗日力 = g_{n_iter}(Xₖ) = Σₗ 各迭代增量之和 → 存入 mk.fx/fy
//
// 注意：mk.fx/fy 在本函数返回后保存的是各迭代增量的累加总和（即拉格朗日总体力
// g_L(Xₖ)），供 compute_ibm_body_force() 等 FSI 接口使用。
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
    const int nm = ms.size();

    // 工作速度场 u_work：从 u* 出发，每次子迭代后叠加展布力修正。
    // 对应论文 Step 2 的虚拟流速场，用于插值评估收敛程度。
    std::vector<double> u_work = fluid.u;

    // 累积欧拉力场 F_total：各迭代增量展布结果之和，最终写入 fluid.force。
    // 等价于论文中最终拉格朗日力 g_L 展布到欧拉网格的结果。
    std::vector<double> F_total(n * d, 0.0);

    // 拉格朗日力累加器：各迭代增量之和 = g_L(Xₖ)，供 FSI 反作用力计算。
    // 这是论文中 mk.fx/fy 应存储的正确值（总合力，而非末次迭代增量）。
    std::vector<double> total_lag_fx(nm, 0.0);
    std::vector<double> total_lag_fy(nm, 0.0);

    // 临时力场：每次子迭代的增量展布结果 spread(Δgₗ)
    std::vector<double> dF_euler(n * d, 0.0);

    for (int iter = 0; iter < n_iter; ++iter) {
        // Step 1 & 3：用 u_work 插值标记点速度
        //   对应论文：u*(x) → 插值 → u*(Xₖ)（第 0 次）或 uₗ(x) → 插值 → uₗ(Xₖ)
        //   无需拷贝：直接 swap，插值后再 swap 回来
        std::swap(fluid.u, u_work);
        interpolate_velocity(fluid, ms, dx, kernel);
        std::swap(fluid.u, u_work);

        // Step 0 / Step 4：计算本次迭代增量力 Δgₗ(Xₖ) = Uₖ − uₗ(Xₖ)
        //   目标速度 Uₖ 从 mk.ux_target/uy_target 读取（运动体由外部在每步前更新；
        //   静止体默认 0.0，与原先 u_target=0 行为完全一致）。
        for (int m = 0; m < nm; ++m) {
            auto& mk = ms.markers[m];
            const double dFx = (mk.ux_target - mk.ux) / dt;   // ρ=1 格子单位假设
            const double dFy = (mk.uy_target - mk.uy) / dt;
            // 暂存增量到 mk.fx/fy 供 spread_force() 使用（展布增量力，而非累积总力）
            mk.fx = dFx;
            mk.fy = dFy;
            // 同时累加到拉格朗日总力：g_L(Xₖ) = Σ Δgₗ(Xₖ)
            total_lag_fx[m] += dFx;
            total_lag_fy[m] += dFy;
        }

        // Step 1（展布）：将增量力 Δgₗ 展布到欧拉网格得到 dF_euler
        std::fill(dF_euler.begin(), dF_euler.end(), 0.0);
        spread_force(fluid, ms, dx, kernel);  // 写入 fluid.force（使用 mk.fx/fy 增量）
        std::swap(fluid.force, dF_euler);     // dF_euler = spread(Δgₗ)

        // Step 2：更新工作速度：u_work += Δgₗ(x)·dt（等价于 u_{l+1} = u* + Σᵢ≤ₗ gᵢ(x)）
        for (int i = 0; i < n; ++i) {
            u_work[i * d + 0] += dt * dF_euler[i * d + 0];
            u_work[i * d + 1] += dt * dF_euler[i * d + 1];
        }

        // 累积总欧拉力：F_total = Σₗ spread(Δgₗ) = spread(g_L)
        for (int i = 0; i < n * d; ++i) {
            F_total[i] += dF_euler[i];
        }
    }

    // 写入最终总欧拉力到 fluid.force（供 Guo 体力格式在 collide 步使用）
    fluid.force = F_total;

    // 将累积拉格朗日总力 g_L(Xₖ) 写入 mk.fx/fy。
    // 这是各迭代增量之和，等价于论文中的 g_L(Xₖ)，正确用于：
    //   compute_ibm_body_force() → Σ mk.fx·mk.ds（FSI 合力）
    for (int m = 0; m < nm; ++m) {
        ms.markers[m].fx = total_lag_fx[m];
        ms.markers[m].fy = total_lag_fy[m];
    }
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

// Cramér 法则求解 3×3 线性系统 A·x = b（仅适用于**对称矩阵**）
//
// 实现说明（对称矩阵 Cramér 等价形式）：
//   对 x[k]，Cramér 法则要求 det(A_{替换第k列为b}) / det(A)。
//   利用 det(M) = det(M^T)，对 k=1,2 将 "替换列" 操作等价转换为 "替换行" 操作，
//   使三个分量的计算结构完全统一（均为沿第一行展开）。
//   对称矩阵保证了等价性：A[i][j] = A[j][i] ↔ 转置 = 原矩阵。
//
//   故此函数**仅对对称矩阵（如 MLS 矩阵 M = Σ wᵢ pᵢ⊗pᵢ^T）正确**。
//
// 返回 false 若行列式绝对值 < 1e-30（奇异）
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

    // 2025 JCP 参数：H_k = 1.5·dx（每方向支撑域半宽），ε = 0.3（Gaussian 集中参数）
    // 权函数 w(r) = exp(−r²/(H_k·ε)²)，支撑域：|Δx|≤H_k 且 |Δy|≤H_k（矩形支撑域，Fig.2）
    // 参考：de Tullio & Pascazio (2016) J. Comput. Phys. 325:116-135
    //       2025 JCP Wu & Fu Eq.(14)：H_k = 1.5·dx（论文中 h_k 即格子间距 dx），ε = 0.3
    const double H_k   = 1.5 * dx;          // 每方向支撑域半宽
    const double eps   = 0.3;               // Gaussian 集中参数
    const double h_eff = H_k * eps;         // 有效 Gaussian 宽度 = 0.45·dx
    const double h2    = h_eff * h_eff;
    const int    iR    = static_cast<int>(std::ceil(H_k / dx));  // = 2（覆盖半径内所有节点）

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

        // MPI 归属过滤：仅处理中心落在本进程物理域内的标记点，避免跨块重复计算
        if (i0 < ms.owner_i_lo || i0 >= ms.owner_i_hi) continue;
        if (j0 < ms.owner_j_lo || j0 >= ms.owner_j_hi) continue;

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

                // 矩形支撑域过滤（2025 JCP Fig.2：2H_x × 2H_y 矩形支撑域）
                if (std::abs(ddx) > H_k || std::abs(ddy) > H_k) continue;

                const double r2  = ddx * ddx + ddy * ddy;

                // Gaussian 权函数 w = exp(−r²/(H_k·ε)²)（2025 JCP Eq.14）
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
// MLS 力展布（MLS-IBM 的伴随/转置展布算子）
//
// 参考：2025 JCP "An implicit moving-least-squares immersed boundary method
//       for high fidelity fluid-structure interaction simulations"
//
// 使用与 mls_interpolate_velocity 完全相同的 MLS 参数和矩阵，
// 但采用转置展布：对每个标记点 X_m，先构建 M 并求解 M·c=e_0，
// 再用 MLS 形状函数 φ_i=w_i·(c^T·p_i) 把力散布到支撑节点。
// ===========================================================================
void mls_spread_force(lbm::LatticeGrid& grid,
                      const MarkerSet& ms,
                      double dx)
{
    if (grid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("MLS-IBM spread: only D2Q9 supported currently");
    }

    const int nx = grid.nx;
    const int ny = grid.ny;

    // 与 mls_interpolate_velocity 完全相同的 MLS 参数（2025 JCP Eq.14）
    const double H_k   = 1.5 * dx;
    const double eps   = 0.3;
    const double h_eff = H_k * eps;
    const double h2    = h_eff * h_eff;
    const int    iR    = static_cast<int>(std::ceil(H_k / dx));

    // 先将体力场清零
    std::fill(grid.force.begin(), grid.force.end(), 0.0);

    // 展布是散射操作（一个标记点写多个节点），不使用并行以避免竞争
    for (int m = 0; m < ms.size(); ++m) {
        const auto& mk = ms.markers[m];

        const double xm = mk.x / dx;
        const double ym = mk.y / dx;
        const int    i0 = static_cast<int>(std::round(xm));
        const int    j0 = static_cast<int>(std::round(ym));

        // MPI 归属过滤：仅处理中心落在本进程物理域内的标记点
        if (i0 < ms.owner_i_lo || i0 >= ms.owner_i_hi) continue;
        if (j0 < ms.owner_j_lo || j0 >= ms.owner_j_hi) continue;

        // --- 第一遍：构造 MLS 矩阵（与插值完全相同）---
        double M[3][3] = {};
        int n_contrib = 0;

        for (int dj = -iR; dj <= iR; ++dj) {
            for (int di = -iR; di <= iR; ++di) {
                const int ii = i0 + di;
                const int jj = j0 + dj;
                if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;

                const int node = grid.idx(ii, jj);
                if (!grid.solid.empty() && grid.solid[node]) continue;

                const double ddx = ii * dx - mk.x;
                const double ddy = jj * dx - mk.y;
                if (std::abs(ddx) > H_k || std::abs(ddy) > H_k) continue;

                const double r2  = ddx * ddx + ddy * ddy;
                const double w   = std::exp(-r2 / h2);
                const double p[3] = {1.0, ddx / dx, ddy / dx};

                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        M[r][c] += w * p[r] * p[c];
                ++n_contrib;
            }
        }

        if (n_contrib < 3) continue;   // 支撑域节点不足，跳过此标记点

        // 求解 M·c = e_0（e_0=[1,0,0]^T），得到 MLS 形状函数系数 c
        // 满足：φ_i(X_m) = w_i · (c_0 + c_1·Δx_i/dx + c_2·Δy_i/dx)
        double e0[3] = {1.0, 0.0, 0.0};
        double c[3];
        if (!solve3x3(M, e0, c)) continue;   // 奇异矩阵：跳过此标记点

        // --- 第二遍：用 MLS 形状函数展布力 ---
        for (int dj = -iR; dj <= iR; ++dj) {
            for (int di = -iR; di <= iR; ++di) {
                const int ii = i0 + di;
                const int jj = j0 + dj;
                if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;

                const int node = grid.idx(ii, jj);
                if (!grid.solid.empty() && grid.solid[node]) continue;

                const double ddx = ii * dx - mk.x;
                const double ddy = jj * dx - mk.y;
                if (std::abs(ddx) > H_k || std::abs(ddy) > H_k) continue;

                const double r2  = ddx * ddx + ddy * ddy;
                const double w   = std::exp(-r2 / h2);
                const double p[3] = {1.0, ddx / dx, ddy / dx};

                // MLS 形状函数：φ_i = w_i · (c^T · p_i)
                const double phi = w * (c[0] * p[0] + c[1] * p[1] + c[2] * p[2]);

                // 展布权重包含弧长元素 ds（与 spread_force 保持量纲一致）
                grid.force[node * 2 + 0] += phi * mk.fx * mk.ds;
                grid.force[node * 2 + 1] += phi * mk.fy * mk.ds;
            }
        }
    }
}

// 内部辅助：计算每个 Lagrangian 点的 MLS 形状函数支撑集
// 与 mls_interpolate_velocity / mls_spread_force 使用完全相同的参数（2025 JCP Eq.14）
static void build_mls_shape_functions(const lbm::LatticeGrid& grid,
                                       const MarkerSet& ms,
                                       double dx,
                                       std::vector<MlsSupportSet>& phi_data)
{
    const int Nl = ms.size();
    const int nx = grid.nx;
    const int ny = grid.ny;

    // 2025 JCP 参数（与 mls_interpolate_velocity 完全一致）
    const double H_k  = 1.5 * dx;
    const double eps  = 0.3;
    const double h2   = (H_k * eps) * (H_k * eps);
    const int    iR   = static_cast<int>(std::ceil(H_k / dx));

    phi_data.resize(Nl);

    for (int k = 0; k < Nl; ++k) {
        const auto& mk = ms.markers[k];

        const double xm = mk.x / dx;
        const double ym = mk.y / dx;
        const int    i0 = static_cast<int>(std::round(xm));
        const int    j0 = static_cast<int>(std::round(ym));

        // MPI 归属过滤
        if (i0 < ms.owner_i_lo || i0 >= ms.owner_i_hi) continue;
        if (j0 < ms.owner_j_lo || j0 >= ms.owner_j_hi) continue;

        // 第一遍：构建 3×3 MLS 矩阵 M
        double M[3][3] = {};
        int n_contrib = 0;

        for (int dj = -iR; dj <= iR; ++dj) {
            for (int di = -iR; di <= iR; ++di) {
                const int ii = i0 + di;
                const int jj = j0 + dj;
                if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;
                const int node = grid.idx(ii, jj);
                if (!grid.solid.empty() && grid.solid[node]) continue;
                const double ddx = ii * dx - mk.x;
                const double ddy = jj * dx - mk.y;
                if (std::abs(ddx) > H_k || std::abs(ddy) > H_k) continue;
                const double w = std::exp(-(ddx*ddx + ddy*ddy) / h2);
                const double p[3] = {1.0, ddx / dx, ddy / dx};
                for (int r = 0; r < 3; ++r)
                    for (int c = 0; c < 3; ++c)
                        M[r][c] += w * p[r] * p[c];
                ++n_contrib;
            }
        }

        if (n_contrib < 3) continue;

        // 求解 M·c = e_0（e_0=[1,0,0]^T），得到形状函数系数
        double e0[3] = {1.0, 0.0, 0.0};
        double c[3];
        if (!solve3x3(M, e0, c)) continue;

        // 第二遍：计算并存储每个支撑 Euler 节点的形状函数值
        phi_data[k].idx.clear();
        phi_data[k].phi.clear();

        for (int dj = -iR; dj <= iR; ++dj) {
            for (int di = -iR; di <= iR; ++di) {
                const int ii = i0 + di;
                const int jj = j0 + dj;
                if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;
                const int node = grid.idx(ii, jj);
                if (!grid.solid.empty() && grid.solid[node]) continue;
                const double ddx = ii * dx - mk.x;
                const double ddy = jj * dx - mk.y;
                if (std::abs(ddx) > H_k || std::abs(ddy) > H_k) continue;
                const double w   = std::exp(-(ddx*ddx + ddy*ddy) / h2);
                const double p[3] = {1.0, ddx / dx, ddy / dx};
                const double phi  = w * (c[0]*p[0] + c[1]*p[1] + c[2]*p[2]);
                phi_data[k].idx.push_back(node);
                phi_data[k].phi.push_back(phi);
            }
        }
    }
}

// ===========================================================================
// 内部辅助：稠密 N×N LU 分解（含行主元选取），原地存储 L\U，置换存入 piv
// 返回 false 若矩阵奇异
// ===========================================================================
static bool lu_factor_dense(std::vector<double>& A_lu,
                             std::vector<int>& piv,
                             int N)
{
    piv.resize(N);
    for (int k = 0; k < N; ++k) {
        // 寻找最大主元
        int p = k;
        double max_val = std::abs(A_lu[k*N + k]);
        for (int i = k + 1; i < N; ++i) {
            double v = std::abs(A_lu[i*N + k]);
            if (v > max_val) { max_val = v; p = i; }
        }
        piv[k] = p;
        if (p != k)
            for (int j = 0; j < N; ++j)
                std::swap(A_lu[k*N + j], A_lu[p*N + j]);
        if (std::abs(A_lu[k*N + k]) < 1e-30) return false;   // 奇异

        const double inv_akk = 1.0 / A_lu[k*N + k];
        for (int i = k + 1; i < N; ++i) {
            A_lu[i*N + k] *= inv_akk;
            for (int j = k + 1; j < N; ++j)
                A_lu[i*N + j] -= A_lu[i*N + k] * A_lu[k*N + j];
        }
    }
    return true;
}

// 前代 + 后代求解（in-place，x 含初始右端向量，返回时为解）
static void lu_solve_dense(const std::vector<double>& A_lu,
                            const std::vector<int>& piv,
                            double* x,
                            int N)
{
    // 行置换
    for (int k = 0; k < N; ++k)
        if (piv[k] != k) std::swap(x[k], x[piv[k]]);
    // 前代 L·y = Pb
    for (int i = 1; i < N; ++i)
        for (int j = 0; j < i; ++j)
            x[i] -= A_lu[i*N + j] * x[j];
    // 后代 U·x = y
    for (int i = N - 1; i >= 0; --i) {
        for (int j = i + 1; j < N; ++j)
            x[i] -= A_lu[i*N + j] * x[j];
        if (std::abs(A_lu[i*N + i]) > 1e-30)
            x[i] /= A_lu[i*N + i];
    }
}

// ===========================================================================
// 内部辅助：稠密 N×N GMRES 求解（无重启，Arnoldi + Givens 旋转）
// 使用对角 Jacobi 预处理（M = diag(A)）
// tol：相对残差收敛判据（|r|/|r0| < tol）
// 返回实际执行的迭代次数
// ===========================================================================
static int gmres_dense_jacobi(const std::vector<double>& A,
                               const std::vector<double>& b,
                               std::vector<double>& x,
                               int N,
                               double tol,
                               int max_iter)
{
    if (N == 0) return 0;
    max_iter = std::min(max_iter, N);   // 不超过维数

    // 对角 Jacobi 预处理子 D⁻¹
    std::vector<double> D_inv(N, 1.0);
    for (int i = 0; i < N; ++i) {
        double aii = A[i*N + i];
        if (std::abs(aii) > 1e-30) D_inv[i] = 1.0 / aii;
    }

    // 初始残差 r = D⁻¹·(b − A·x)
    std::vector<double> r(N, 0.0);
    for (int i = 0; i < N; ++i) {
        double ax = 0.0;
        for (int j = 0; j < N; ++j) ax += A[i*N + j] * x[j];
        r[i] = D_inv[i] * (b[i] - ax);
    }
    double beta = 0.0;
    for (int i = 0; i < N; ++i) beta += r[i] * r[i];
    beta = std::sqrt(beta);
    if (beta < tol) return 0;   // 初始猜测已收敛

    const int m = max_iter;

    // Krylov 基 V（最多 m+1 个向量，各长 N）
    std::vector<std::vector<double>> V(m + 1, std::vector<double>(N, 0.0));
    // 上 Hessenberg 矩阵 H（(m+1)×m）
    std::vector<std::vector<double>> H(m + 1, std::vector<double>(m, 0.0));
    // Givens 旋转系数
    std::vector<double> cs(m, 0.0), sn(m, 0.0);
    // 右端小向量 g（初始为 [beta, 0, …, 0]^T）
    std::vector<double> g(m + 1, 0.0);
    g[0] = beta;

    // v_0 = r / ||r||
    for (int i = 0; i < N; ++i) V[0][i] = r[i] / beta;

    int j_done = m;
    for (int j = 0; j < m; ++j) {
        // w = D⁻¹·A·v_j（预处理矩阵向量乘积）
        std::vector<double> w(N, 0.0);
        for (int i = 0; i < N; ++i) {
            double av = 0.0;
            for (int k = 0; k < N; ++k) av += A[i*N + k] * V[j][k];
            w[i] = D_inv[i] * av;
        }

        // 改进 Gram-Schmidt 正交化
        for (int i = 0; i <= j; ++i) {
            double h = 0.0;
            for (int k = 0; k < N; ++k) h += w[k] * V[i][k];
            H[i][j] = h;
            for (int k = 0; k < N; ++k) w[k] -= h * V[i][k];
        }
        double norm_w = 0.0;
        for (int k = 0; k < N; ++k) norm_w += w[k] * w[k];
        norm_w = std::sqrt(norm_w);
        H[j + 1][j] = norm_w;
        if (norm_w > 1e-50 && j + 1 <= m)
            for (int k = 0; k < N; ++k) V[j + 1][k] = w[k] / norm_w;

        // 应用已有 Givens 旋转到 Hessenberg 新列
        for (int i = 0; i < j; ++i) {
            double t       =  cs[i]*H[i][j] + sn[i]*H[i + 1][j];
            H[i + 1][j]    = -sn[i]*H[i][j] + cs[i]*H[i + 1][j];
            H[i][j]        =  t;
        }

        // 计算本步新 Givens 旋转
        const double denom = std::hypot(H[j][j], H[j + 1][j]);
        cs[j] = (denom > 1e-50) ? H[j][j]     / denom : 1.0;
        sn[j] = (denom > 1e-50) ? H[j + 1][j] / denom : 0.0;
        H[j][j]     = cs[j]*H[j][j] + sn[j]*H[j + 1][j];
        H[j + 1][j] = 0.0;

        // 旋转 g 向量并检查收敛
        g[j + 1] = -sn[j] * g[j];
        g[j]     =  cs[j] * g[j];

        if (std::abs(g[j + 1]) < tol * beta) {   // 相对残差已达收敛
            j_done = j + 1;
            break;
        }
    }

    // 求解上三角系统 H[0..js-1, 0..js-1]·y = g[0..js-1]（后代）
    const int js = j_done;
    std::vector<double> y(js, 0.0);
    for (int i = js - 1; i >= 0; --i) {
        y[i] = g[i];
        for (int k = i + 1; k < js; ++k) y[i] -= H[i][k] * y[k];
        if (std::abs(H[i][i]) > 1e-50) y[i] /= H[i][i];
    }

    // 更新解：x += Σ_k y[k] · v_k
    for (int k = 0; k < js; ++k)
        for (int i = 0; i < N; ++i)
            x[i] += V[k][i] * y[k];

    return js;
}

// ===========================================================================
// 内部辅助：用预建形状函数集插值速度到 Lagrangian 点
// ===========================================================================
static void interpolate_with_phi(const lbm::LatticeGrid& grid,
                                  MarkerSet& ms,
                                  const std::vector<MlsSupportSet>& phi_data)
{
    const int Nl = ms.size();
    for (int k = 0; k < Nl; ++k) {
        const auto& ss = phi_data[k];
        if (ss.idx.empty()) continue;
        double ux_sum = 0.0, uy_sum = 0.0;
        for (int s = 0; s < static_cast<int>(ss.idx.size()); ++s) {
            const int node = ss.idx[s];
            ux_sum += ss.phi[s] * grid.u[node * 2 + 0];
            uy_sum += ss.phi[s] * grid.u[node * 2 + 1];
        }
        ms.markers[k].ux = ux_sum;
        ms.markers[k].uy = uy_sum;
    }
}

// ===========================================================================
// 内部辅助：用预建形状函数集展布力到 Eulerian 网格
// f_j = Σ_k c_k · φ_j^k · F_k^b，其中 c_k = mk.ds（Eq.16+18）
// ===========================================================================
static void spread_with_phi(lbm::LatticeGrid& grid,
                             const MarkerSet& ms,
                             const std::vector<MlsSupportSet>& phi_data)
{
    std::fill(grid.force.begin(), grid.force.end(), 0.0);
    const int Nl = ms.size();
    for (int k = 0; k < Nl; ++k) {
        const auto& mk = ms.markers[k];
        const auto& ss = phi_data[k];
        const double fxk = mk.fx * mk.ds;
        const double fyk = mk.fy * mk.ds;
        for (int s = 0; s < static_cast<int>(ss.idx.size()); ++s) {
            const int node = ss.idx[s];
            grid.force[node * 2 + 0] += ss.phi[s] * fxk;
            grid.force[node * 2 + 1] += ss.phi[s] * fyk;
        }
    }
}

// ===========================================================================
// 内部辅助：构建相关矩阵 A（稠密 Nl×Nl，行主序）
//
// 论文 Eq.(28)（优化版）：
//   A_{ki} = Σ_{j ∈ S(k) ∩ S(i)} φ_j^k · c_i · φ_j^i
//   c_i = ds_i（守恒因子，Eq.18，uniform lattice）
//
// 利用反向索引 euler_to_lag[j] = [(Lag_idx, phi_value), …]
// 高效枚举所有共享 Euler 支撑节点的 (k,i) 对。
// 总复杂度：O(Σ_k |S(k)| · max_j |euler_to_lag[j]|) ≈ O(N_l · N_e · N_i_avg)，
// 其中 N_e 为每个 Lagrangian 点的支撑 Euler 节点数（≈ 9，矩形 3dx 支撑域），
// N_i_avg 为每个 Euler 节点被多少 Lagrangian 点共享（= N_l · N_e / N_euler）。
// ===========================================================================
static void build_correlation_matrix(const std::vector<MlsSupportSet>& phi_data,
                                      const MarkerSet& ms,
                                      int grid_size,            // nx*ny
                                      std::vector<double>& A_mat)
{
    const int Nl = ms.size();
    A_mat.assign(static_cast<std::size_t>(Nl) * Nl, 0.0);

    // 构建反向索引：euler_to_lag[j] = {(k, φ_j^k), …}
    std::vector<std::vector<std::pair<int, double>>> euler_to_lag(grid_size);
    for (int k = 0; k < Nl; ++k) {
        const auto& ss = phi_data[k];
        for (int s = 0; s < static_cast<int>(ss.idx.size()); ++s)
            euler_to_lag[ss.idx[s]].emplace_back(k, ss.phi[s]);
    }

    // 累积 A_{ki} += φ_j^k · c_i · φ_j^i
    for (int k = 0; k < Nl; ++k) {
        const auto& ss = phi_data[k];
        for (int s = 0; s < static_cast<int>(ss.idx.size()); ++s) {
            const int    euler_j = ss.idx[s];
            const double phi_jk  = ss.phi[s];
            for (const auto& [i, phi_ji] : euler_to_lag[euler_j]) {
                // c_i = ms.markers[i].ds（守恒因子）
                A_mat[k * Nl + i] += phi_jk * ms.markers[i].ds * phi_ji;
            }
        }
    }
}

// ===========================================================================
// 隐式 MLS-IBM 力计算（Algorithm 3，Scheme II：GMRES 求解）
//
// 参考：2025 JCP Wu & Fu §4，Algorithm 3，Scheme II
//
// 实现步骤：
//   A1: 计算传递算子 Φ（所有 Lagrangian 点的 MLS 形状函数）
//   A2: 重建 Lagrangian 速度 U* = J·u*（MLS 插值）
//   C2: 构建相关矩阵 A（Eq.28）和右端向量 B（Eq.24c），
//       用 GMRES 求解 A·X = B（Scheme II，moving 和 stationary 均适用）
//   A4: 用 Eq.(16) 展布 Lagrangian 力到 Eulerian 网格
//   A5: 速度更新 u^{n+1} = u* + Δt·f/ρ（由调用方执行）
//
// 参数 gmres_max_iter：GMRES 最大迭代次数（等于原 n_iter 参数位置）。
//   - 设为 N_l 可保证精确解（理论上 Krylov 维数不超过 N_l）。
//   - 默认值 3 保留后向兼容；对于机器精度结果，建议设 ≥ N_l（或至少 50）。
//   - 收敛判据：相对残差 < 1e-14（接近论文要求的 1e-17）。
//
// 本函数取代了原 Richardson 迭代实现，可消除无滑移误差（≈ 机器精度），
// 并保持力和力矩守恒（2025 JCP Table 1）。
// ===========================================================================
void compute_ibm_forces_mls_implicit(lbm::LatticeGrid& fluid,
                                      MarkerSet& ms,
                                      double dx,
                                      double dt,
                                      int    gmres_max_iter,
                                      double u_target_x,
                                      double u_target_y)
{
    if (fluid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("Implicit MLS-IBM: only D2Q9 supported currently");
    }

    const int Nl = ms.size();
    if (Nl == 0) return;

    // A1: 计算每个 Lagrangian 点的 MLS 形状函数（一次构建，后续共用）
    std::vector<MlsSupportSet> phi_data;
    build_mls_shape_functions(fluid, ms, dx, phi_data);

    // A2: 用 MLS 形状函数重建 Lagrangian 速度 U* = J·u*
    interpolate_with_phi(fluid, ms, phi_data);

    // 构建右端向量 B（Eq.24c，ρ=1 格子单位）
    //   Bx[k] = (u_target_x − U*_k,x) / dt
    //   By[k] = (u_target_y − U*_k,y) / dt
    std::vector<double> Bx(Nl, 0.0), By(Nl, 0.0);
    for (int k = 0; k < Nl; ++k) {
        const auto& mk = ms.markers[k];
        // MPI 归属过滤（与 build_mls_shape_functions 一致）
        const int i0 = static_cast<int>(std::round(mk.x / dx));
        const int j0 = static_cast<int>(std::round(mk.y / dx));
        if (i0 < ms.owner_i_lo || i0 >= ms.owner_i_hi) continue;
        if (j0 < ms.owner_j_lo || j0 >= ms.owner_j_hi) continue;
        Bx[k] = (mk.ux_target - mk.ux) / dt;
        By[k] = (mk.uy_target - mk.uy) / dt;
    }

    // C1/C2: 构建相关矩阵 A（Eq.28）并用 GMRES 求解（Scheme II）
    std::vector<double> A_mat;
    build_correlation_matrix(phi_data, ms, fluid.nx * fluid.ny, A_mat);

#ifdef LBM_ENABLE_MPI
    // MPI：各进程仅持有局部 phi_data（归属过滤），需全局归约得到完整 A 和 B
    MPI_Allreduce(MPI_IN_PLACE, A_mat.data(), Nl * Nl, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, Bx.data(),    Nl,      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, By.data(),    Nl,      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    // GMRES 求解 A·Fx = Bx 和 A·Fy = By（各速度分量独立，矩阵相同）
    const double gmres_tol = 1e-14;   // 相对收敛判据（论文 10⁻¹⁷ 绝对）
    const int    actual_max = std::max(gmres_max_iter, 1);

    std::vector<double> Fx(Nl, 0.0), Fy(Nl, 0.0);   // 初始猜测为零
    gmres_dense_jacobi(A_mat, Bx, Fx, Nl, gmres_tol, actual_max);
    gmres_dense_jacobi(A_mat, By, Fy, Nl, gmres_tol, actual_max);

    // 写入 Lagrangian 还原力（供 FSI 反作用力计算：compute_ibm_body_force）
    for (int k = 0; k < Nl; ++k) {
        ms.markers[k].fx = Fx[k];
        ms.markers[k].fy = Fy[k];
    }

    // A4: 展布 Lagrangian 力到 Eulerian 网格（Eq.16：f_j = Σ_k c_k φ_j^k F_k^b）
    spread_with_phi(fluid, ms, phi_data);
    // A5: 速度更新由调用方执行（solver.step() 中 collide/stream 步骤）
}

// ===========================================================================
// 隐式 MLS-IBM 力计算（Algorithm 3，Scheme I：固定物体直接矩阵求逆）
//
// 参考：2025 JCP Wu & Fu §4，Algorithm 3，Scheme I
//
// 对于几何固定（stationary）的物体，传递算子 Φ 和相关矩阵 A 不随时间变化。
// 本函数在首次调用时（A_lu_cache 为空）构建 A 并完成 LU 分解（相当于 A⁻¹），
// 后续步骤（A_lu_cache 非空）直接用 LU 代换求解 X = A⁻¹·B，
// 从而避免每步重复构建矩阵，大幅降低计算量（论文 Table 2，Scheme I）。
//
// @param fluid         Eulerian 流体网格
// @param ms            Lagrangian 标记点集（需固定不动）
// @param dx            格子间距
// @param dt            时间步长
// @param A_lu_cache    LU 因子缓存（首次调用时填充，后续复用；由调用方持久化）
// @param piv_cache     LU 行主元缓存（与 A_lu_cache 配套）
// @param u_target_x/y  目标速度（对静止固体通常为 0）
// ===========================================================================
void compute_ibm_forces_mls_implicit_stationary(lbm::LatticeGrid& fluid,
                                                 MarkerSet& ms,
                                                 double dx,
                                                 double dt,
                                                 std::vector<double>& A_lu_cache,
                                                 std::vector<int>&    piv_cache,
                                                 std::vector<MlsSupportSet>& phi_cache,
                                                 double u_target_x,
                                                 double u_target_y)
{
    if (fluid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("Stationary MLS-IBM: only D2Q9 supported currently");
    }

    const int Nl = ms.size();
    if (Nl == 0) return;

    // A1: 计算传递算子 Φ（只在首次调用时构建；后续每步直接复用 phi_cache）
    if (phi_cache.empty()) {
        build_mls_shape_functions(fluid, ms, dx, phi_cache);
    }

    // C1: 首次调用时构建相关矩阵 A 并完成 LU 分解（缓存 A⁻¹ 于 A_lu_cache）
    if (A_lu_cache.empty()) {
        build_correlation_matrix(phi_cache, ms, fluid.nx * fluid.ny, A_lu_cache);
#ifdef LBM_ENABLE_MPI
        // MPI：各进程仅持有局部 phi_cache，需全局归约得到完整 A
        MPI_Allreduce(MPI_IN_PLACE, A_lu_cache.data(), Nl * Nl,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        // in-place LU 分解（A_lu_cache 被改写为 L\U）
        if (!lu_factor_dense(A_lu_cache, piv_cache, Nl)) {
            // 矩阵奇异：回退到 GMRES（此时 A_lu_cache 部分覆盖，需清空）
            A_lu_cache.clear();
            piv_cache.clear();
            phi_cache.clear();
            compute_ibm_forces_mls_implicit(fluid, ms, dx, dt,
                                             /*gmres_max_iter=*/Nl,
                                             u_target_x, u_target_y);
            return;
        }
    }

    // A2: 重建 Lagrangian 速度 U* = J·u*（使用缓存的 phi_cache）
    interpolate_with_phi(fluid, ms, phi_cache);

    // 构建右端向量 B（Eq.24c）
    std::vector<double> Bx(Nl, 0.0), By(Nl, 0.0);
    for (int k = 0; k < Nl; ++k) {
        const auto& mk = ms.markers[k];
        const int i0 = static_cast<int>(std::round(mk.x / dx));
        const int j0 = static_cast<int>(std::round(mk.y / dx));
        if (i0 < ms.owner_i_lo || i0 >= ms.owner_i_hi) continue;
        if (j0 < ms.owner_j_lo || j0 >= ms.owner_j_hi) continue;
        Bx[k] = (mk.ux_target - mk.ux) / dt;
        By[k] = (mk.uy_target - mk.uy) / dt;
    }

#ifdef LBM_ENABLE_MPI
    // MPI：归属过滤后各进程 Bx/By 仅含本地贡献，全局归约得到完整 B
    MPI_Allreduce(MPI_IN_PLACE, Bx.data(), Nl, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, By.data(), Nl, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    // C2: Scheme I — X = A⁻¹·B（LU 代换，O(N_l²)，比 GMRES 更快）
    lu_solve_dense(A_lu_cache, piv_cache, Bx.data(), Nl);   // Bx → Fx
    lu_solve_dense(A_lu_cache, piv_cache, By.data(), Nl);   // By → Fy

    // 写入 Lagrangian 还原力
    for (int k = 0; k < Nl; ++k) {
        ms.markers[k].fx = Bx[k];
        ms.markers[k].fy = By[k];
    }

    // A4: 展布到 Eulerian 网格（Eq.16）——使用缓存的 phi_cache
    spread_with_phi(fluid, ms, phi_cache);
}

// ===========================================================================
// 原始 MLS-IBM（Original MLS）—— MLS 插值 + MLS 形状函数展布（含守恒因子 c_i）
//
// 参考：2025 JCP §3.1 "Original MLS-IBM"（Algorithm 1）
//       Vanella & Balaras (2009) J. Comput. Phys. 228:2366-2391
//
// 算法（单步直接力法）：
//   1. 计算 MLS 形状函数 Φ（调用 mls_interpolate_velocity）
//   2. MLS 速度插值：U_m = Φ^T · u*
//   3. 直接力：F_m = ρ(u_target − U_m) / dt（ρ=1 格子单位）
//   4. MLS 形状函数展布（含守恒因子 c_i = ds_i/ΔV_j = ds_i，Eq.16 + Eq.18）：
//        f_j = Σ_m c_m φ_j^m F_m  ≡  mls_spread_force（其中 c_m = ds_m / ΔV_j，uniform lattice 下 ΔV_j=dx²→c_m=ds_m）
//   5. 更新速度：u^{n+1} = u* + dt·f/ρ
//
// 注意：展布算子与插值算子非完全伴随（φ_j^m ≠ φ_m^j），因此
// 原始 MLS-IBM 存在无滑移误差（见 2025 JCP Fig.3a）。
// ===========================================================================
void compute_ibm_forces_mls_original(lbm::LatticeGrid& fluid,
                                      MarkerSet& ms,
                                      double dx,
                                      double dt,
                                      DeltaKernel /*kernel*/,
                                      double u_target_x,
                                      double u_target_y)
{
    if (fluid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("Original MLS-IBM: only D2Q9 supported currently");
    }

    // 1. MLS 速度插值：U_m = J · u（更新 mk.ux/uy）
    mls_interpolate_velocity(fluid, ms, dx);

    // 2. 直接力：F_m = (mk.ux_target − U_m) / dt（逐点目标速度，支持柔性体）
    for (auto& mk : ms.markers) {
        mk.fx = (mk.ux_target - mk.ux) / dt;
        mk.fy = (mk.uy_target - mk.uy) / dt;
    }

    // 3. MLS 形状函数展布（Eq.16；mls_spread_force 含 ds_m 守恒因子）
    mls_spread_force(fluid, ms, dx);
}

// ===========================================================================
// 显式 MLS-IBM（Explicit MLS）—— MLS 插值 + MLS 形状函数展布 + Z 全局修正
//
// 参考：2025 JCP §3.2 "Explicit variant MLS-IBM"（Algorithm 2）
//       Chen et al. (2022) Phys. Rev. E 106:015307
//
// 算法：
//   1. 计算 MLS 形状函数 Φ；MLS 速度插值：U_m = Φ^T · u*
//   2. 直接力：F_m = (u_target − U_m) / dt
//   3. 第一次 MLS 展布：f = J^T · F
//   4. 重插值：g_k = J · f（用展布后力场重新插值到 Lagrangian 点）
//   5. 全局修正因子（最小化 ||Z·g_k − F_k||² 的最小二乘解，Eq.21）：
//        Z = Σ_k (F_k · g_k) / Σ_k |g_k|²
//   6. 修正展布：f → Z · f
//
// 注意：Z 修正因子会破坏力和力矩守恒（见 2025 JCP Table 1），
// 且无滑移残差仍显著（见 Fig.3b）。推荐使用隐式 MLS（compute_ibm_forces_mls_implicit）。
// ===========================================================================
void compute_ibm_forces_mls_explicit(lbm::LatticeGrid& fluid,
                                      MarkerSet& ms,
                                      double dx,
                                      double dt,
                                      double u_target_x,
                                      double u_target_y)
{
    if (fluid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("Explicit MLS-IBM: only D2Q9 supported currently");
    }

    // 1. MLS 速度插值：U_m = J · u（更新 mk.ux/uy）
    mls_interpolate_velocity(fluid, ms, dx);

    // 2. 直接力：F_m = (mk.ux_target − U_m) / dt（逐点目标速度，支持柔性体）
    for (auto& mk : ms.markers) {
        mk.fx = (mk.ux_target - mk.ux) / dt;
        mk.fy = (mk.uy_target - mk.uy) / dt;
    }

    // 3. 第一次 MLS 展布：f = J^T · F（写入 fluid.force）
    mls_spread_force(fluid, ms, dx);

    // 4. 重插值：g_k = J · f（将展布后的力场 fluid.force 插值回 Lagrangian 点）
    //    利用 swap 技巧：临时令 fluid.u = fluid.force，用 mls_interpolate_velocity 读取
    std::swap(fluid.u, fluid.force);
    mls_interpolate_velocity(fluid, ms, dx);  // 写入 mk.ux = g_kx, mk.uy = g_ky
    std::swap(fluid.u, fluid.force);          // 恢复 fluid.u

    // 5. 全局修正因子 Z（最小化 ||Z·g − F||²，Eq.21）：
    //    Z = Σ_k (F_k · g_k) / Σ_k |g_k|²
    //    mk.fx/fy 仍持有步骤 2 的 F_b 值；mk.ux/uy 现在持有步骤 4 的 g_k
    double num = 0.0, den = 0.0;
    for (int m = 0; m < ms.size(); ++m) {
        const auto& mk = ms.markers[m];
        // MPI 归属过滤：仅统计本进程拥有的标记点贡献
        const int i0 = static_cast<int>(std::round(mk.x / dx));
        const int j0 = static_cast<int>(std::round(mk.y / dx));
        if (i0 < ms.owner_i_lo || i0 >= ms.owner_i_hi) continue;
        if (j0 < ms.owner_j_lo || j0 >= ms.owner_j_hi) continue;
        num += mk.fx * mk.ux + mk.fy * mk.uy;
        den += mk.ux * mk.ux + mk.uy * mk.uy;
    }
#ifdef LBM_ENABLE_MPI
    // MPI：各进程仅持有局部标记点贡献，全局归约得到完整 num/den
    {
        double buf[2] = {num, den};
        MPI_Allreduce(MPI_IN_PLACE, buf, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        num = buf[0]; den = buf[1];
    }
#endif
    const double Z = (den > 1e-30) ? num / den : 1.0;

    // 6. 将 fluid.force 乘以 Z（原地修正）
    for (auto& f : fluid.force) f *= Z;
}


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

        // 逐点目标速度（支持柔性体；静止体 mk.ux_target=0）
        const double ex = mk.ux_target - mk.ux;
        const double ey = mk.uy_target - mk.uy;

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
// 隐式速度校正 IBM（IVC-IBM）内部辅助：构建基于 Peskin δ 的相关矩阵
//
// 参考：Wu & Shu (2009) Eq.(27)–(29)
//
// A_{lk} = Δs_k · Σ_{i,j} D_ij^l · D_ij^k · Δx²
//
// 其中 D_ij^l = delta_phi(x_ij - X_l, dx) · delta_phi(y_ij - X_l, dx)（2D Peskin δ）
// 利用反向索引（euler_to_lag[j] = [(Lag_idx, δ值), ...]）高效枚举共享 Euler 节点对。
// ===========================================================================
static void build_ivc_matrix(const lbm::LatticeGrid& grid,
                               const MarkerSet& ms,
                               double dx,
                               DeltaKernel kernel,
                               std::vector<double>& A_mat)
{
    const int Nl = ms.size();
    const int nx = grid.nx;
    const int ny = grid.ny;
    const int support = (kernel == DeltaKernel::TwoPoint) ? 1 : 2;

    A_mat.assign(static_cast<std::size_t>(Nl) * Nl, 0.0);

    // 反向索引：euler_to_lag[node] = {(Lagrangian_idx, δ值), …}
    std::vector<std::vector<std::pair<int, double>>> euler_to_lag(nx * ny);

    for (int l = 0; l < Nl; ++l) {
        const auto& mk = ms.markers[l];
        const double xm = mk.x / dx;
        const double ym = mk.y / dx;
        const int    i0 = static_cast<int>(std::floor(xm));
        const int    j0 = static_cast<int>(std::floor(ym));

        // MPI 归属过滤（与 interpolate_velocity / spread_force 保持一致）
        if (i0 < ms.owner_i_lo || i0 >= ms.owner_i_hi) continue;
        if (j0 < ms.owner_j_lo || j0 >= ms.owner_j_hi) continue;

        for (int dj = -support; dj <= support + 1; ++dj) {
            for (int di = -support; di <= support + 1; ++di) {
                const int ii = i0 + di;
                const int jj = j0 + dj;
                if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;
                const int node = grid.idx(ii, jj);
                if (!grid.solid.empty() && grid.solid[node]) continue;

                const double phi_x = delta_phi(mk.x - ii * dx, dx, kernel);
                const double phi_y = delta_phi(mk.y - jj * dx, dx, kernel);
                const double D_l   = phi_x * phi_y;   // D(x_ij - X_l)
                if (D_l != 0.0)
                    euler_to_lag[node].emplace_back(l, D_l);
            }
        }
    }

    // 累积 A_{lk} = Σ_j D_l(j) · D_k(j) · ds_k · dx²
    for (int node = 0; node < nx * ny; ++node) {
        const auto& entries = euler_to_lag[node];
        if (entries.empty()) continue;
        for (const auto& [l, D_l] : entries) {
            for (const auto& [k, D_k] : entries) {
                A_mat[l * Nl + k] += D_l * D_k * ms.markers[k].ds * dx * dx;
            }
        }
    }
}

// ===========================================================================
// 隐式速度校正 IBM（IVC-IBM）— 通用版（每步重建矩阵 A）
//
// 参考：Wu J. & Shu C. (2009) J. Comput. Phys. 228:1963–1979
//
// 算法步骤（每时间步调用一次）：
//   1. 插值：u*(X_B^l) = Σ_{i,j} u*(x_ij) D_ij^l Δx²  （Eq.26 中间速度项）
//   2. 构建右端向量 B：B^l = U_target^l − u*(X_B^l)   （Eq.29 速度亏量）
//   3. 构建矩阵 A：A_{lk} = Δs_k Σ_{i,j} D_ij^l D_ij^k Δx²  （Eq.27–28）
//   4. LU 分解并求解 A · δu_B = B（x/y 分量独立求解）
//   5. 写入 Lagrangian 力密度：f_B^l = (2/dt) · δu_B^l  （Eq.30，ρ=1）
//   6. 展布到 Eulerian 网格：f(x_ij) = Σ_l f_B^l D_ij^l Δs_l
// ===========================================================================
void compute_ibm_forces_ivc(lbm::LatticeGrid& fluid,
                              MarkerSet& ms,
                              double dx,
                              double dt,
                              DeltaKernel kernel,
                              double u_target_x,
                              double u_target_y)
{
    if (fluid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("IVC-IBM: only D2Q9 supported currently");
    }

    const int Nl = ms.size();
    if (Nl == 0) return;

    // 步骤 1：用 Peskin δ 将中间速度 u* 插值到各边界点
    //   u*(X_B^l) = Σ_{i,j} u*(x_ij) · D_ij^l · Δx²  （写入 mk.ux/uy）
    interpolate_velocity(fluid, ms, dx, kernel);

    // 步骤 2：构建右端向量 B（速度亏量，Eq.29）
    //   B^l = U_target^l − u*(X_B^l)（x/y 分量独立）
    std::vector<double> Bx(Nl, 0.0), By(Nl, 0.0);
    for (int l = 0; l < Nl; ++l) {
        const auto& mk = ms.markers[l];
        const int i0 = static_cast<int>(std::floor(mk.x / dx));
        const int j0 = static_cast<int>(std::floor(mk.y / dx));
        if (i0 < ms.owner_i_lo || i0 >= ms.owner_i_hi) continue;
        if (j0 < ms.owner_j_lo || j0 >= ms.owner_j_hi) continue;
        Bx[l] = mk.ux_target - mk.ux;
        By[l] = mk.uy_target - mk.uy;
    }

    // 步骤 3：构建相关矩阵 A（Eq.27–28）
    //   A_{lk} = Δs_k · Σ_{i,j} D_ij^l · D_ij^k · Δx²
    std::vector<double> A_mat;
    build_ivc_matrix(fluid, ms, dx, kernel, A_mat);

#ifdef LBM_ENABLE_MPI
    // MPI：各进程仅持有局部归属标记点的贡献，全局归约后得到完整 A 和 B
    MPI_Allreduce(MPI_IN_PLACE, A_mat.data(), Nl * Nl, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, Bx.data(),    Nl,      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, By.data(),    Nl,      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    // 步骤 4：LU 分解并求解（x/y 分量独立，矩阵相同，Eq.28）
    std::vector<int> piv;
    if (!lu_factor_dense(A_mat, piv, Nl)) {
        // 奇异系统（极端情况）：退化为显式直接力法
        for (int l = 0; l < Nl; ++l) {
            ms.markers[l].fx = Bx[l] * (2.0 / dt);
            ms.markers[l].fy = By[l] * (2.0 / dt);
        }
        spread_force(fluid, ms, dx, kernel);
        return;
    }
    lu_solve_dense(A_mat, piv, Bx.data(), Nl);   // Bx → δu_Bx
    lu_solve_dense(A_mat, piv, By.data(), Nl);   // By → δu_By

    // 步骤 5：Lagrangian 力密度：f_B^l = (2ρ/dt) · δu_B^l  （Eq.30，ρ=1）
    for (int l = 0; l < Nl; ++l) {
        ms.markers[l].fx = Bx[l] * (2.0 / dt);
        ms.markers[l].fy = By[l] * (2.0 / dt);
    }

    // 步骤 6：展布力到 Eulerian 网格：f(x_ij) = Σ_l f_B^l · D_ij^l · Δs_l
    //   等价：f(x_ij) = (2/dt) · δu(x_ij)，其中 δu(x_ij) = Σ_l δu_B^l D_ij^l Δs_l
    //   spread_force 使用 mk.fx * phi_x * phi_y * mk.ds（与 Eq.24 一致）
    spread_force(fluid, ms, dx, kernel);
}

// ===========================================================================
// 隐式速度校正 IBM（IVC-IBM）— 固定物体优化版（LU 缓存）
//
// 参考：Wu & Shu (2009) §3，算法步骤 (1)
//
// 矩阵 A 仅取决于边界点位置与 δ 核，对固定物体不随时间变化。
// 首次调用时构建 A 并完成 LU 分解（缓存于 A_lu_cache/piv_cache），
// 后续每步只需插值 u* → 构建 B → LU 代换，节省重复矩阵构建开销。
// ===========================================================================
void compute_ibm_forces_ivc_stationary(lbm::LatticeGrid& fluid,
                                        MarkerSet& ms,
                                        double dx,
                                        double dt,
                                        std::vector<double>& A_lu_cache,
                                        std::vector<int>&    piv_cache,
                                        DeltaKernel kernel,
                                        double u_target_x,
                                        double u_target_y)
{
    if (fluid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("IVC-IBM stationary: only D2Q9 supported currently");
    }

    const int Nl = ms.size();
    if (Nl == 0) return;

    // 首次调用：构建矩阵 A 并 LU 分解（后续复用缓存）
    if (A_lu_cache.empty()) {
        build_ivc_matrix(fluid, ms, dx, kernel, A_lu_cache);
#ifdef LBM_ENABLE_MPI
        MPI_Allreduce(MPI_IN_PLACE, A_lu_cache.data(), Nl * Nl,
                      MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif
        if (!lu_factor_dense(A_lu_cache, piv_cache, Nl)) {
            // 奇异矩阵：清空缓存，退化为逐步重建
            A_lu_cache.clear();
            piv_cache.clear();
            compute_ibm_forces_ivc(fluid, ms, dx, dt, kernel,
                                    u_target_x, u_target_y);
            return;
        }
    }

    // 每步：插值 u* → X_B
    interpolate_velocity(fluid, ms, dx, kernel);

    // 每步：构建 B（速度亏量）
    std::vector<double> Bx(Nl, 0.0), By(Nl, 0.0);
    for (int l = 0; l < Nl; ++l) {
        const auto& mk = ms.markers[l];
        const int i0 = static_cast<int>(std::floor(mk.x / dx));
        const int j0 = static_cast<int>(std::floor(mk.y / dx));
        if (i0 < ms.owner_i_lo || i0 >= ms.owner_i_hi) continue;
        if (j0 < ms.owner_j_lo || j0 >= ms.owner_j_hi) continue;
        Bx[l] = mk.ux_target - mk.ux;
        By[l] = mk.uy_target - mk.uy;
    }
#ifdef LBM_ENABLE_MPI
    MPI_Allreduce(MPI_IN_PLACE, Bx.data(), Nl, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, By.data(), Nl, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#endif

    // 每步：LU 代换求解（复用缓存）
    lu_solve_dense(A_lu_cache, piv_cache, Bx.data(), Nl);
    lu_solve_dense(A_lu_cache, piv_cache, By.data(), Nl);

    // 写入 Lagrangian 力密度：f_B^l = (2/dt) · δu_B^l
    for (int l = 0; l < Nl; ++l) {
        ms.markers[l].fx = Bx[l] * (2.0 / dt);
        ms.markers[l].fy = By[l] * (2.0 / dt);
    }

    // 展布力到 Eulerian 网格
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

// ===========================================================================
// MPI 分区适配：坐标系转换 + 归属边界设置
// ===========================================================================
void ibm_marker_set_adapt_to_partition(MarkerSet& ms,
                                        int x_start, int y_start,
                                        int phys_x0, int phys_y0,
                                        int local_nx, int local_ny)
{
    // 从全局坐标到本地坐标的偏移量
    const double off_x = static_cast<double>(phys_x0 - x_start);
    const double off_y = static_cast<double>(phys_y0 - y_start);

    for (auto& mk : ms.markers) {
        mk.x  += off_x;
        mk.y  += off_y;
        mk.x0 += off_x;
        mk.y0 += off_y;
    }

    // 设置本进程"归属"的物理行/列范围（本地坐标）
    ms.owner_i_lo = phys_x0;
    ms.owner_i_hi = phys_x0 + local_nx;
    ms.owner_j_lo = phys_y0;
    ms.owner_j_hi = phys_y0 + local_ny;
}

// ===========================================================================
// MPI 幽灵层 u 场交换（IBM 插值前调用）
//
// solver.step() → stream() 完成后，幽灵行/列的 u 已由 Solver::stream() 自动修正
// （stream() 内在 compute_macroscopic() 后调用 lbm::halo_exchange_u_2d()）。
// 本函数保留为外部调用接口，直接代理到底层通用实现。
// ===========================================================================
void ibm_halo_exchange_u_2d(lbm::LatticeGrid& grid,
                              const lbm::MpiDecomp2D& decomp)
{
#ifdef LBM_ENABLE_MPI
    lbm::halo_exchange_u_2d(grid, decomp);
#else
    (void)grid; (void)decomp;
#endif
}

// ===========================================================================
// MPI 幽灵层力场归并（IBM 力展布后调用）
//
// spread_force() 可能向幽灵行/列写入力贡献，这些贡献属于邻居进程物理区域。
// 本函数通过 MPI_Sendrecv 把幽灵行/列力发回对应邻居的物理行/列并累加（+=），
// 然后清零本地幽灵行/列，保证跨 MPI 边界的 IBM 力展布物理上完整。
//
// n_ghost（= decomp.n_ghost）决定每侧归并的幽灵层数：
//   n_ghost=1（默认）：归并 1 层幽灵行力，满足 TwoPoint 核。
//   n_ghost=2       ：归并 2 层幽灵行力，适用于 FourPoint 核在 MPI
//                     边界附近的标记点（需要 MpiDecomp2D 以 n_ghost=2 创建）。
// ===========================================================================
void ibm_halo_reduce_force_2d(lbm::LatticeGrid& grid,
                                const lbm::MpiDecomp2D& decomp)
{
#ifdef LBM_ENABLE_MPI
    if (decomp.nprocs == 1) return;

    const int d    = grid.dim();
    const int gnx  = grid.nx;
    const int gny  = grid.ny;
    const int lnx  = decomp.local_nx;
    const int lny  = decomp.local_ny;
    const int px0  = decomp.phys_x0();
    const int py0  = decomp.phys_y0();
    const int n_gh = decomp.n_ghost;

    MPI_Status st;

    // -----------------------------------------------------------------------
    // S/N 方向：归并 n_ghost 层幽灵行力贡献
    //
    // 第 k 层（k=0..n_ghost-1）：
    //   南幽灵行 j=k 的力 → 发给南邻，累加到其北物理行 j=py0+lny-n_ghost+k
    //   北幽灵行 j=py0+lny+k 的力 → 发给北邻，累加到其南物理行 j=py0+k
    //
    // 利用 MPI_Sendrecv 对称交换：
    //   Sendrecv 1：发送北幽灵 k 层 → 北邻，接收来自南邻北幽灵 k 层 → 累加到本进程南物理 j=py0+k
    //   Sendrecv 2：发送南幽灵 k 层 → 南邻，接收来自北邻南幽灵 k 层 → 累加到本进程北物理 j=py0+lny-n_ghost+k
    // -----------------------------------------------------------------------
    {
        const int row_size = gnx * d;
        std::vector<double> recv_buf(row_size, 0.0);

        for (int k = 0; k < n_gh; ++k) {
            double* north_ghost_k = &grid.force[static_cast<std::size_t>(
                grid.idx(0, py0 + lny + k)) * d];
            double* bot_phys_k    = &grid.force[static_cast<std::size_t>(
                grid.idx(0, py0 + k)) * d];
            // 向北邻发送北幽灵行 k，从南邻接收其对应层贡献累加到本进程南物理行 k
            std::fill(recv_buf.begin(), recv_buf.end(), 0.0);
            MPI_Sendrecv(north_ghost_k,    row_size, MPI_DOUBLE, decomp.rank_north, 5000 + k,
                         recv_buf.data(),  row_size, MPI_DOUBLE, decomp.rank_south, 5000 + k,
                         MPI_COMM_WORLD, &st);
            for (int c = 0; c < row_size; ++c) bot_phys_k[c] += recv_buf[c];
            std::fill(north_ghost_k, north_ghost_k + row_size, 0.0);

            double* south_ghost_k = &grid.force[static_cast<std::size_t>(
                grid.idx(0, k)) * d];
            double* top_phys_k    = &grid.force[static_cast<std::size_t>(
                grid.idx(0, py0 + lny - n_gh + k)) * d];
            // 向南邻发送南幽灵行 k，从北邻接收其对应层贡献累加到本进程北物理行 k
            std::fill(recv_buf.begin(), recv_buf.end(), 0.0);
            MPI_Sendrecv(south_ghost_k,    row_size, MPI_DOUBLE, decomp.rank_south, 5100 + k,
                         recv_buf.data(),  row_size, MPI_DOUBLE, decomp.rank_north, 5100 + k,
                         MPI_COMM_WORLD, &st);
            for (int c = 0; c < row_size; ++c) top_phys_k[c] += recv_buf[c];
            std::fill(south_ghost_k, south_ghost_k + row_size, 0.0);
        }
    }

    // -----------------------------------------------------------------------
    // W/E 方向：归并 n_ghost 列幽灵列力贡献（列不连续，需打包）
    // 第 k 列（k=0..n_ghost-1）：
    //   西幽灵列 i=k 的力 → 发给西邻，累加到其最东物理列 i=px0+lnx-n_ghost+k
    //   东幽灵列 i=px0+lnx+k 的力 → 发给东邻，累加到其最西物理列 i=px0+k
    // -----------------------------------------------------------------------
    if (decomp.has_west_ghost() || decomp.has_east_ghost()) {
        const int col_size = gny * d;
        std::vector<double> send_buf(col_size, 0.0);
        std::vector<double> recv_buf(col_size, 0.0);

        for (int k = 0; k < n_gh; ++k) {
            // 打包并发送东幽灵列 k，接收西邻贡献累加到本进程最西物理列 k
            std::fill(send_buf.begin(), send_buf.end(), 0.0);
            if (decomp.has_east_ghost()) {
                for (int j = 0; j < gny; ++j) {
                    const double* src = &grid.force[static_cast<std::size_t>(
                        grid.idx(px0 + lnx + k, j)) * d];
                    for (int c = 0; c < d; ++c) send_buf[j*d+c] = src[c];
                }
            }
            std::fill(recv_buf.begin(), recv_buf.end(), 0.0);
            MPI_Sendrecv(send_buf.data(), col_size, MPI_DOUBLE, decomp.rank_east, 5200 + k,
                         recv_buf.data(), col_size, MPI_DOUBLE, decomp.rank_west, 5200 + k,
                         MPI_COMM_WORLD, &st);
            for (int j = 0; j < gny; ++j) {
                double* dst = &grid.force[static_cast<std::size_t>(
                    grid.idx(px0 + k, j)) * d];
                for (int c = 0; c < d; ++c) dst[c] += recv_buf[j*d+c];
            }
            if (decomp.has_east_ghost()) {
                for (int j = 0; j < gny; ++j) {
                    double* dst = &grid.force[static_cast<std::size_t>(
                        grid.idx(px0 + lnx + k, j)) * d];
                    std::fill(dst, dst + d, 0.0);
                }
            }

            // 打包并发送西幽灵列 k，接收东邻贡献累加到本进程最东物理列 k
            std::fill(send_buf.begin(), send_buf.end(), 0.0);
            if (decomp.has_west_ghost()) {
                for (int j = 0; j < gny; ++j) {
                    const double* src = &grid.force[static_cast<std::size_t>(
                        grid.idx(k, j)) * d];
                    for (int c = 0; c < d; ++c) send_buf[j*d+c] = src[c];
                }
            }
            std::fill(recv_buf.begin(), recv_buf.end(), 0.0);
            MPI_Sendrecv(send_buf.data(), col_size, MPI_DOUBLE, decomp.rank_west, 5300 + k,
                         recv_buf.data(), col_size, MPI_DOUBLE, decomp.rank_east, 5300 + k,
                         MPI_COMM_WORLD, &st);
            for (int j = 0; j < gny; ++j) {
                double* dst = &grid.force[static_cast<std::size_t>(
                    grid.idx(px0 + lnx - n_gh + k, j)) * d];
                for (int c = 0; c < d; ++c) dst[c] += recv_buf[j*d+c];
            }
            if (decomp.has_west_ghost()) {
                for (int j = 0; j < gny; ++j) {
                    double* dst = &grid.force[static_cast<std::size_t>(
                        grid.idx(k, j)) * d];
                    std::fill(dst, dst + d, 0.0);
                }
            }
        }
    }
#else
    (void)grid; (void)decomp;
#endif
}

} // namespace ibm
