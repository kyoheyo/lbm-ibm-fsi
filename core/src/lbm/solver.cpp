#include "lbm/solver.hpp"
#include "lbm/boundary.hpp"
#include "lbm/mpi_decomp.hpp"
#include "lbm/solid.hpp"
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

    // -----------------------------------------------------------------------
    // 计算本地物理区域范围（MPI 模式下跳过幽灵行/列）
    //
    // pb 同时供固体 BC（apply_solid_bounce_back / apply_solid_ibb）
    // 和面 BC（apply_boundary_conditions）使用，避免重复计算。
    // -----------------------------------------------------------------------
    PhysicalBounds pb;
    pb.j_s = 0;
    pb.j_n = grid_.ny - 1;
    pb.i_w = 0;
    pb.i_e = grid_.nx - 1;
#ifdef LBM_ENABLE_MPI
    if (mpi_decomp_ && mpi_decomp_->nprocs > 1) {
        // 1D Y-切片：幽灵在 j=0（南）和 j=local_ny+1（北）
        pb.j_s = 1;
        pb.j_n = grid_.ny - 2;   // = local_ny
        pb.has_south_wall = mpi_decomp_->has_south_wall();
        pb.has_north_wall = mpi_decomp_->has_north_wall();
        pb.has_west_wall  = true;
        pb.has_east_wall  = true;
    }
    if (mpi_decomp2d_ && mpi_decomp2d_->nprocs > 1) {
        pb.j_s = mpi_decomp2d_->phys_y0();
        pb.j_n = mpi_decomp2d_->phys_y0() + mpi_decomp2d_->local_ny - 1;
        pb.i_w = mpi_decomp2d_->phys_x0();
        pb.i_e = mpi_decomp2d_->phys_x0() + mpi_decomp2d_->local_nx - 1;
        pb.has_south_wall = mpi_decomp2d_->has_south_wall();
        pb.has_north_wall = mpi_decomp2d_->has_north_wall();
        pb.has_west_wall  = mpi_decomp2d_->has_west_wall();
        pb.has_east_wall  = mpi_decomp2d_->has_east_wall();
    }
    if (mpi_decomp3d_ && mpi_decomp3d_->nprocs > 1) {
        pb.j_s = mpi_decomp3d_->phys_y0();
        pb.j_n = mpi_decomp3d_->phys_y0() + mpi_decomp3d_->local_ny - 1;
        pb.i_w = mpi_decomp3d_->phys_x0();
        pb.i_e = mpi_decomp3d_->phys_x0() + mpi_decomp3d_->local_nx - 1;
        pb.has_south_wall = mpi_decomp3d_->has_south_wall();
        pb.has_north_wall = mpi_decomp3d_->has_north_wall();
        pb.has_west_wall  = mpi_decomp3d_->has_west_wall();
        pb.has_east_wall  = mpi_decomp3d_->has_east_wall();
    }
#endif

    // 施加固体节点反弹边界条件（BB 或 IBB）
    // 固体 BC 在面 BC 之前施加，以确保面 BC（Zou-He 等）具有更高优先级（最后写入）。
    // MPI 模式下仅对物理区域 [pb.i_w, pb.i_e] × [pb.j_s, pb.j_n] 施加，
    // 跳过幽灵节点，防止幽灵行的 f 被错误覆盖。
    if (solid_bc_type_ == SolidBCType::BounceBack) {
        apply_solid_bounce_back(grid_, pb.i_w, pb.j_s, pb.i_e, pb.j_n);
    } else if (solid_bc_type_ == SolidBCType::InterpolatedBounceBack) {
        apply_solid_ibb(grid_, pb.i_w, pb.j_s, pb.i_e, pb.j_n);
    }

    // 施加通过 add_boundary_condition() 注册的边界条件
    if (!bcs_.empty()) {
        apply_boundary_conditions(grid_, bcs_, pb);
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
// 幽灵层跳过参数计算（供 collide_bgk / collide_mrt 共用）
// ---------------------------------------------------------------------------
Solver::CollideGuard Solver::make_collide_guard() const
{
    const int n = grid_.size();
    CollideGuard g;
    g.n_start = 0;
    g.n_end   = n;
#ifdef LBM_ENABLE_MPI
    // 一维（Y 方向）幽灵行
    // 1D MpiDecomp 的布局约定（nprocs>1 时 grid_ny = local_ny + 2）：
    //   j=0          : 南幽灵行（所有 rank 均有，rank-0 对应 MPI_PROC_NULL）
    //   j=1..local_ny: 物理行（rank-0 的 j=1 是全局南物理壁）
    //   j=local_ny+1 : 北幽灵行（所有 rank 均有，rank-(N-1) 对应 MPI_PROC_NULL）
    // 因此对所有 rank 均跳过 j=0（n_start = nx）和 j=local_ny+1（n_end = n-nx）。
    // rank-0 的物理南壁在 j=1，已包含在 [n_start, n_end) 范围内。
    if (mpi_decomp_ && mpi_decomp_->nprocs > 1) {
        g.n_start = grid_.nx;       // 跳过南幽灵行 j=0（所有 rank 均有）
        g.n_end   = n - grid_.nx;   // 跳过北幽灵行 j=local_ny+1（所有 rank 均有）
    }
    // 二维（XY 方向）幽灵层
    g.use_mpi2d = (mpi_decomp2d_ && mpi_decomp2d_->nprocs > 1);
    if (g.use_mpi2d) {
        g.gnx2d = grid_.nx;
        g.gny2d = grid_.ny;
        g.sg2d  = mpi_decomp2d_->has_south_ghost();
        g.ng2d  = mpi_decomp2d_->has_north_ghost();
        g.wg2d  = mpi_decomp2d_->has_west_ghost();
        g.eg2d  = mpi_decomp2d_->has_east_ghost();
        // 二维模式下 n_start/n_end 不适用，交由 is_ghost() 逐节点检查
        g.n_start = 0;
        g.n_end   = n;
    }
    // 三维（XYZ 方向）幽灵层
    g.use_mpi3d = (mpi_decomp3d_ && mpi_decomp3d_->nprocs > 1);
    if (g.use_mpi3d) {
        g.gnx3d = grid_.nx;
        g.gny3d = grid_.ny;
        g.gnz3d = grid_.nz;
        g.sg3d  = mpi_decomp3d_->has_south_ghost();
        g.ng3d  = mpi_decomp3d_->has_north_ghost();
        g.wg3d  = mpi_decomp3d_->has_west_ghost();
        g.eg3d  = mpi_decomp3d_->has_east_ghost();
        g.bg3d  = mpi_decomp3d_->has_bottom_ghost();
        g.tg3d  = mpi_decomp3d_->has_top_ghost();
        // 三维模式下 n_start/n_end 不适用，交由 is_ghost() 逐节点检查
        g.n_start = 0;
        g.n_end   = n;
    }
#endif
    return g;
}

// ---------------------------------------------------------------------------
// BGK 碰撞（单松弛时间）
// ---------------------------------------------------------------------------
void Solver::collide_bgk()
{
    const int d = grid_.dim();

    // 计算幽灵层跳过参数（避免幽灵节点二次碰撞）
    const CollideGuard guard = make_collide_guard();

    if (grid_.model == LatticeModel::D2Q9) {
#ifdef LBM_ENABLE_OPENMP
// schedule(guided): decreasing chunk sizes give better load balance when
// some iterations are skipped via is_ghost() (2-D / 3-D MPI ghost layers).
// For 1-D MPI and non-MPI runs the work is already uniform, so guided
// behaves nearly identically to static with negligible scheduling overhead.
#pragma omp parallel for schedule(guided)
#endif
        for (int i = guard.n_start; i < guard.n_end; ++i) {
#ifdef LBM_ENABLE_MPI
            if (guard.is_ghost(i)) continue;
#endif
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
                apply_guo_forcing(i, d2q9::W[a], c, Fi, &f_a);
                grid_.f[i * d2q9::Q + a] = f_a;
            }
        }
    } else {
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(guided)
#endif
        for (int i = guard.n_start; i < guard.n_end; ++i) {
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
                apply_guo_forcing(i, d3q19::W[a], c, Fi, &f_a);
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

    const int d = 2;

    // 计算幽灵层跳过参数（与 collide_bgk 使用同一辅助方法）
    const CollideGuard guard = make_collide_guard();

#ifdef LBM_ENABLE_OPENMP
// schedule(guided) instead of schedule(static): handles load imbalance from
// is_ghost() early-continue in 2-D / 3-D MPI decompositions.
#pragma omp parallel for schedule(guided)
#endif
    for (int i = guard.n_start; i < guard.n_end; ++i) {
#ifdef LBM_ENABLE_MPI
        if (guard.is_ghost(i)) continue;
#endif
        const double* fi = &grid_.f[i * d2q9::Q];
        const double* ui = &grid_.u[i * d];
        const double  ri = grid_.rho[i];
        const double* Fi = &grid_.force[i * d];

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
            const double c_a[2] = {
                static_cast<double>(d2q9::C[a][0]),
                static_cast<double>(d2q9::C[a][1])
            };
            apply_guo_forcing(i, d2q9::W[a], c_a, Fi, &val);
            grid_.f[i * d2q9::Q + a] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// Guo et al. (2002) 体力格式：
//   F_a = w_a (1 - ω/2) [(c_a - u)/cs² + (c_a·u)c_a/cs⁴] · F
// 在碰撞（BGK 或 MRT 反投影）后作为对 f_a 的修正项加入。
// ---------------------------------------------------------------------------
void Solver::apply_guo_forcing(int node, double w_a, const double* c_a,
                                const double* F, double* f_a_ptr)
{
    const int      d   = grid_.dim();
    const double*  u   = &grid_.u[node * d];
    constexpr double cs2 = 1.0 / 3.0;
    constexpr double cs4 = 1.0 / 9.0;

    // c_a · u
    double cu = 0.0;
    for (int alpha = 0; alpha < d; ++alpha)
        cu += c_a[alpha] * u[alpha];

    // sum_α [(c_aα - u_α)/cs² + (c_a·u)*c_aα/cs⁴] * F_α
    double term = 0.0;
    for (int alpha = 0; alpha < d; ++alpha)
        term += ((c_a[alpha] - u[alpha]) / cs2 + cu * c_a[alpha] / cs4) * F[alpha];

    *f_a_ptr += w_a * (1.0 - 0.5 * omega_) * term;
}

// ---------------------------------------------------------------------------
// 流式迁移：将分布函数沿各离散速度方向传播
//
// MPI 幽灵层交换时序说明（关键正确性）
// ─────────────────────────────────────
// 幽灵层交换必须在流式迁移之前完成（即 collide() 之后、stream 循环之前），
// 而非流式迁移之后。原因如下：
//
// 在 D2Q9/D3Q19 推送（PUSH）方案中，每个源节点 (i,j) 将其碰后分布函数
// f*[i,j,a] 推送到目标节点 (i+cx_a, j+cy_a)。对于北物理边界 j=lny 上的节点，
// 推送 cy=+1 方向时目标为 j=lny+1（北幽灵行）：
//   f_tmp[i, lny+1, North] = fc[i, lny, North]
//
// 南邻进程（rank+1）在其 stream 步需要将本进程物理行 j=lny 的碰后值作为
// 幽灵行数据使用。这些值尚未经历 stream，仍位于 grid_.f（碰后值）。
//
// 若在 stream 之后交换，则发送的是已流式迁移的数据（即来自 j=lny-1 的
// cy=+1 方向分量），而非 j=lny 的碰后值，导致接口处一阶误差，
// 表现为交界处速度/密度不连续（与单进程结果不一致）。
//
// 正确时序：
//   1. collide()           — 碰后分布函数保存在 grid_.f
//   2. halo_exchange()     — 发送 grid_.f 中的物理边界行（碰后值）到邻居幽灵行
//   3. stream loop (PUSH)  — 以正确的幽灵数据推送跨进程方向
//   4. swap f ↔ f_tmp
//   5. compute_macroscopic()
// ---------------------------------------------------------------------------
void Solver::stream()
{
    const int nx = grid_.nx;
    const int ny = grid_.ny;
    const int nz = grid_.nz;

#ifdef LBM_ENABLE_MPI
    // 幽灵层交换：在 stream 之前完成（使用碰后分布函数 grid_.f）
    // 这样邻居幽灵行在 stream 推送时已包含正确的对端碰后值，
    // 保证跨进程边界数据与串行结果一致，消除交界处不连续。
    if (mpi_decomp_ && mpi_decomp_->nprocs > 1) {
        halo_exchange_d2q9(grid_, *mpi_decomp_);
    } else if (mpi_decomp2d_ && mpi_decomp2d_->nprocs > 1) {
        halo_exchange_d2q9_2d(grid_, *mpi_decomp2d_);
    } else if (mpi_decomp3d_ && mpi_decomp3d_->nprocs > 1) {
        halo_exchange_d3q19_3d(grid_, *mpi_decomp3d_);
    }
#endif

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
