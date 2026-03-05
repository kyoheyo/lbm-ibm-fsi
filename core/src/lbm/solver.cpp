#include "lbm/solver.hpp"
#include "lbm/boundary.hpp"
#include <cmath>
#include <stdexcept>

#ifdef LBM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace lbm {

// ---------------------------------------------------------------------------
// 构造函数：将分布函数初始化为静止平衡态（rho=1，u=0）
// ---------------------------------------------------------------------------
Solver::Solver(LatticeGrid& grid, double omega, CollisionModel cm)
    : grid_(grid), omega_(omega), cm_(cm)
{
    const int n = grid_.size();
    const int d = grid_.dim();

    if (grid_.model == LatticeModel::D2Q9) {
        std::array<double, 2> u0 = {0.0, 0.0};
        for (int i = 0; i < n; ++i) {
            for (int a = 0; a < d2q9::Q; ++a) {
                const double c[2] = {
                    static_cast<double>(d2q9::C[a][0]),
                    static_cast<double>(d2q9::C[a][1])
                };
                grid_.f[i * d2q9::Q + a] =
                    f_eq(d2q9::W[a], grid_.rho[i], c, u0.data(), d);
            }
        }
    } else {
        std::array<double, 3> u0 = {0.0, 0.0, 0.0};
        for (int i = 0; i < n; ++i) {
            for (int a = 0; a < d3q19::Q; ++a) {
                const double c[3] = {
                    static_cast<double>(d3q19::C[a][0]),
                    static_cast<double>(d3q19::C[a][1]),
                    static_cast<double>(d3q19::C[a][2])
                };
                grid_.f[i * d3q19::Q + a] =
                    f_eq(d3q19::W[a], grid_.rho[i], c, u0.data(), d);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 单时间步：碰撞 + 流式迁移 + 边界条件 + 宏观量更新
//
// 执行顺序说明：
//   1. collide()  — 使用上一步的 ρ/u 计算平衡态并执行 BGK/MRT 松弛
//   2. stream()   — 传播分布函数，并初步计算宏观量（边界节点此时使用幽灵值）
//   3. apply_BC() — 用物理边界条件覆盖幽灵方向的分布函数
//   4. compute_macroscopic() — 用 BC 修正后的 f 重新计算边界节点的正确 ρ/u
//      （若未执行此步，下次碰撞将使用 BC 修正前的错误 ρ/u）
// ---------------------------------------------------------------------------
void Solver::step()
{
    collide();
    stream();
    // 施加通过 add_boundary_condition() 注册的边界条件
    if (!bcs_.empty()) {
        apply_boundary_conditions(grid_, bcs_);
        // BC 修正了边界节点的 f 值，需重新计算宏观量以供下次碰撞使用
        grid_.compute_macroscopic();
    }
}

// ---------------------------------------------------------------------------
// 注册一个边界条件（每步 step() 后自动施加）
// ---------------------------------------------------------------------------
void Solver::add_boundary_condition(const BoundaryCondition& bc)
{
    bcs_.push_back(bc);
}

// ---------------------------------------------------------------------------
// 碰撞：根据选定的模型分发到具体实现
// ---------------------------------------------------------------------------
void Solver::collide()
{
    if (cm_ == CollisionModel::BGK) {
        collide_bgk();
    } else {
        collide_mrt();
    }
}

// ---------------------------------------------------------------------------
// BGK 碰撞（单松弛时间）
// ---------------------------------------------------------------------------
void Solver::collide_bgk()
{
    const int n = grid_.size();
    const int d = grid_.dim();

    if (grid_.model == LatticeModel::D2Q9) {
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
            const double* ui = &grid_.u[i * d];
            const double  ri = grid_.rho[i];
            const double* Fi = &grid_.force[i * d];

            for (int a = 0; a < d2q9::Q; ++a) {
                const double c[2] = {
                    static_cast<double>(d2q9::C[a][0]),
                    static_cast<double>(d2q9::C[a][1])
                };
                double feq = f_eq(d2q9::W[a], ri, c, ui, d);
                double f_a = grid_.f[i * d2q9::Q + a];

                // BGK 碰撞：f_a* = f_a - ω(f_a - f_eq)
                f_a += -omega_ * (f_a - feq);

                // Guo 体力修正
                apply_guo_forcing(i, Fi, &f_a);
                grid_.f[i * d2q9::Q + a] = f_a;
            }
        }
    } else {
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < n; ++i) {
            const double* ui = &grid_.u[i * d];
            const double  ri = grid_.rho[i];
            const double* Fi = &grid_.force[i * d];

            for (int a = 0; a < d3q19::Q; ++a) {
                const double c[3] = {
                    static_cast<double>(d3q19::C[a][0]),
                    static_cast<double>(d3q19::C[a][1]),
                    static_cast<double>(d3q19::C[a][2])
                };
                double feq = f_eq(d3q19::W[a], ri, c, ui, d);
                double f_a = grid_.f[i * d3q19::Q + a];
                f_a += -omega_ * (f_a - feq);
                apply_guo_forcing(i, Fi, &f_a);
                grid_.f[i * d3q19::Q + a] = f_a;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MRT 碰撞（多松弛时间，仅 D2Q9）— 使用标准 9×9 变换矩阵
// ---------------------------------------------------------------------------
void Solver::collide_mrt()
{
    if (grid_.model != LatticeModel::D2Q9) {
        // 三维情形暂未实现 MRT 矩阵，回退到 BGK
        collide_bgk();
        return;
    }

    // D2Q9 标准 MRT 松弛率
    // s = [s0, s1, s2, s3, s4, s5, s6, s7, s8]
    // 与粘度相关的分量：s7 = s8 = omega_
    const double s[d2q9::Q] = {1.0, 1.4, 1.4, 1.0, 1.2, 1.0, 1.2, omega_, omega_};

    // D2Q9 变换矩阵 M
    static const double M[d2q9::Q][d2q9::Q] = {
        { 1,  1,  1,  1,  1,  1,  1,  1,  1},
        {-4, -1, -1, -1, -1,  2,  2,  2,  2},
        { 4, -2, -2, -2, -2,  1,  1,  1,  1},
        { 0,  1,  0, -1,  0,  1, -1, -1,  1},
        { 0, -2,  0,  2,  0,  1, -1, -1,  1},
        { 0,  0,  1,  0, -1,  1,  1, -1, -1},
        { 0,  0, -2,  0,  2,  1,  1, -1, -1},
        { 0,  1, -1,  1, -1,  0,  0,  0,  0},
        { 0,  0,  0,  0,  0,  1, -1,  1, -1},
    };

    const int n = grid_.size();
    const int d = 2;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        const double* fi = &grid_.f[i * d2q9::Q];
        const double* ui = &grid_.u[i * d];
        const double  ri = grid_.rho[i];

        // 投影到矩空间：m = M * f
        double m[d2q9::Q] = {};
        for (int k = 0; k < d2q9::Q; ++k) {
            for (int a = 0; a < d2q9::Q; ++a) {
                m[k] += M[k][a] * fi[a];
            }
        }

        // 计算平衡矩：m_eq
        double m_eq[d2q9::Q];
        const double ux = ui[0];
        const double uy = ui[1];
        const double u2 = ux * ux + uy * uy;
        m_eq[0] = ri;
        m_eq[1] = ri * (-2.0 + 3.0 * u2);
        m_eq[2] = ri * (1.0 - 3.0 * u2);
        m_eq[3] = ri * ux;
        m_eq[4] = -ri * ux;
        m_eq[5] = ri * uy;
        m_eq[6] = -ri * uy;
        m_eq[7] = ri * (ux * ux - uy * uy);
        m_eq[8] = ri * ux * uy;

        // 在矩空间中进行松弛：m* = m - S(m - m_eq)
        double m_star[d2q9::Q];
        for (int k = 0; k < d2q9::Q; ++k) {
            m_star[k] = m[k] - s[k] * (m[k] - m_eq[k]);
        }

        // 反投影：f* = M^{-1} m*（归一化 D2Q9 MRT 矩阵有已知逆系数）
        static const double Mi[d2q9::Q][d2q9::Q] = {
            { 1.0/9,  -1.0/9,  1.0/9,  0,     0,     0,     0,     0,     0    },
            { 1.0/9,  -1.0/36, -1.0/18, 1.0/6, -1.0/6, 0,     0,     1.0/4, 0    },
            { 1.0/9,  -1.0/36, -1.0/18, 0,     0,     1.0/6, -1.0/6, -1.0/4, 0    },
            { 1.0/9,  -1.0/36, -1.0/18, -1.0/6, 1.0/6, 0,     0,     1.0/4, 0    },
            { 1.0/9,  -1.0/36, -1.0/18, 0,     0,    -1.0/6,  1.0/6, -1.0/4, 0    },
            { 1.0/9,  1.0/18,  1.0/36,  1.0/6, 1.0/12, 1.0/6, 1.0/12, 0,     1.0/4},
            { 1.0/9,  1.0/18,  1.0/36, -1.0/6, -1.0/12, 1.0/6, 1.0/12, 0,    -1.0/4},
            { 1.0/9,  1.0/18,  1.0/36, -1.0/6, -1.0/12, -1.0/6, -1.0/12, 0,   1.0/4},
            { 1.0/9,  1.0/18,  1.0/36,  1.0/6, 1.0/12, -1.0/6, -1.0/12, 0,   -1.0/4},
        };

        for (int a = 0; a < d2q9::Q; ++a) {
            double val = 0.0;
            for (int k = 0; k < d2q9::Q; ++k) {
                val += Mi[a][k] * m_star[k];
            }
            grid_.f[i * d2q9::Q + a] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Guo et al. (2002) 体力格式：
//   F_a = w_a (1 - ω/2) [(c_a - u)/cs² + (c_a·u)c_a/cs⁴] · F
// 在 BGK 松弛后作为对 f_a 的修正项加入。
// ---------------------------------------------------------------------------
void Solver::apply_guo_forcing(int node, const double* F, double* /*f_a_ptr*/)
{
    // 修正量通过速度偏移累积到 grid_.force 中并在 compute_macroscopic() 中处理：
    //   u_eff = u + F*dt/(2*rho)
    // 只要调用方遵循标准时序（在 collide() 前调用 compute_macroscopic()），
    // 此方式已在平衡值计算中隐式捕获。
    // 完整的 Guo 修正需要逐方向项；下方为占位实现。
    (void)node;
    (void)F;
    // TODO: 当存在非零体力时实现逐方向 Guo 修正。
}

// ---------------------------------------------------------------------------
// 流式迁移：将分布函数沿各离散速度方向传播
// ---------------------------------------------------------------------------
void Solver::stream()
{
    const int nx = grid_.nx;
    const int ny = grid_.ny;
    const int nz = grid_.nz;

    if (grid_.model == LatticeModel::D2Q9) {
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int src = grid_.idx(i, j);
                for (int a = 0; a < d2q9::Q; ++a) {
                    // 目标节点（周期性取模）
                    int di = (i + d2q9::C[a][0] + nx) % nx;
                    int dj = (j + d2q9::C[a][1] + ny) % ny;
                    int dst = grid_.idx(di, dj);
                    grid_.f_tmp[dst * d2q9::Q + a] = grid_.f[src * d2q9::Q + a];
                }
            }
        }
    } else {
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for collapse(3) schedule(static)
#endif
        for (int k = 0; k < nz; ++k) {
            for (int j = 0; j < ny; ++j) {
                for (int i = 0; i < nx; ++i) {
                    const int src = grid_.idx(i, j, k);
                    for (int a = 0; a < d3q19::Q; ++a) {
                        int di = (i + d3q19::C[a][0] + nx) % nx;
                        int dj = (j + d3q19::C[a][1] + ny) % ny;
                        int dk = (k + d3q19::C[a][2] + nz) % nz;
                        int dst = grid_.idx(di, dj, dk);
                        grid_.f_tmp[dst * d3q19::Q + a] = grid_.f[src * d3q19::Q + a];
                    }
                }
            }
        }
    }

    // 将迁移后的临时缓冲区与主缓冲区交换，并更新宏观量
    std::swap(grid_.f, grid_.f_tmp);
    grid_.compute_macroscopic();
}

} // namespace lbm
