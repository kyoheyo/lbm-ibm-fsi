// core/src/lbm/solid.cpp — 固体节点几何标记与反弹边界条件实现
//
// 包含：
//   mark_solid_cylinder()   — 圆柱标记 + 精确 q 计算
//   mark_solid_rectangle()  — 矩形标记（q=0.5）
//   compute_ibb_distances() — 通用 q 计算（对任意几何退化为 q=0.5）
//   apply_solid_bounce_back() — 半步长反弹（Ladd 1994）
//   apply_solid_ibb()          — Bouzidi 插值反弹（2001）
//
// 参考文献：
//   Ladd A.J.C. (1994) J. Fluid Mech. 271, 285-309.
//   Bouzidi M. et al. (2001) Phys. Fluids 13(11):3452-3459.
//   Yu D. et al. (2003) Phys. Fluids 15(8):2462-2469.

#include "lbm/solid.hpp"
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef LBM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace lbm {

// ===========================================================================
// 内部辅助
// ===========================================================================

// 对圆柱计算精确的 q_ibb：
// 从流体节点 (i,j) 沿方向 a = (ca, cb) 出发，求与圆心 (cx,cy)、半径 r 的圆
// 的最近交点参数 t（0 < t ≤ 1），其中 t=1 对应相邻固体节点位置。
// 返回 t（即 q），若无有效交点则返回 0.5f（退化到 halfway BB）。
static float cylinder_ibb_q(int i, int j, int ca, int cb,
                              double cx, double cy, double r)
{
    const double ddx = i - cx;
    const double ddy = j - cy;

    const double A = static_cast<double>(ca * ca + cb * cb);
    const double B = 2.0 * (ddx * ca + ddy * cb);
    const double C = ddx * ddx + ddy * ddy - r * r;

    const double disc = B * B - 4.0 * A * C;
    if (disc < 0.0) return 0.5f;

    const double sq = std::sqrt(disc);
    // 两个根（可能为负或超出 [0,1]）
    const double t1 = (-B - sq) / (2.0 * A);
    const double t2 = (-B + sq) / (2.0 * A);

    // 取最小正数且 ≤ 1 的根（正确选取最近交点）
    double t = -1.0;
    if (t1 > 1e-9 && t1 <= 1.0 + 1e-9) {
        t = t1;
        // 若 t2 也有效且更小，选 t2
        if (t2 > 1e-9 && t2 <= 1.0 + 1e-9 && t2 < t1) t = t2;
    } else if (t2 > 1e-9 && t2 <= 1.0 + 1e-9) {
        t = t2;
    }

    if (t > 0.0) {
        return static_cast<float>(std::min(std::max(t, 1e-5), 1.0));
    }
    return 0.5f;  // 未找到有效交点，退化为 halfway BB
}

// ===========================================================================
// 公共接口实现
// ===========================================================================

void mark_solid_cylinder(LatticeGrid& grid, double cx, double cy, double radius)
{
    if (grid.model != LatticeModel::D2Q9) return;

    const int nx = grid.nx;
    const int ny = grid.ny;
    const int Q  = d2q9::Q;
    const int n  = grid.size();

    // 确保数组已分配（构造函数已分配，此处为防御性检查）
    if (static_cast<int>(grid.solid.size()) != n) {
        grid.solid.assign(n, 0);
        grid.q_ibb.assign(n * Q, 0.5f);
    }

    // ---- 第一遍：标记内部固体节点 ----
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const double dx = i - cx;
            const double dy = j - cy;
            if (dx * dx + dy * dy < radius * radius) {
                grid.solid[grid.idx(i, j)] = 1;
            }
        }
    }

    // ---- 第二遍：对流体节点计算 IBB 壁面距离分数 q ----
    // 对每个流体节点，检查 8 个非静止方向，若邻居为固体则计算精确 q
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int nf = grid.idx(i, j);
            if (grid.solid[nf]) continue;

            for (int a = 1; a < Q; ++a) {
                const int ca = d2q9::C[a][0];
                const int cb = d2q9::C[a][1];
                const int ni = i + ca;
                const int nj = j + cb;
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                if (!grid.solid[grid.idx(ni, nj)]) continue;

                // 流体节点 (i,j) 方向 a 指向固体 — 计算圆柱交点参数 t = q
                grid.q_ibb[nf * Q + a] = cylinder_ibb_q(i, j, ca, cb, cx, cy, radius);
            }
        }
    }
}

void mark_solid_rectangle(LatticeGrid& grid, int i0, int j0, int i1, int j1)
{
    if (grid.model != LatticeModel::D2Q9) return;

    const int nx = grid.nx;
    const int ny = grid.ny;
    const int n  = grid.size();

    if (static_cast<int>(grid.solid.size()) != n) {
        grid.solid.assign(n, 0);
        grid.q_ibb.assign(n * d2q9::Q, 0.5f);
    }

    const int ia = std::max(i0, 0), ib = std::min(i1, nx - 1);
    const int ja = std::max(j0, 0), jb = std::min(j1, ny - 1);

    for (int j = ja; j <= jb; ++j) {
        for (int i = ia; i <= ib; ++i) {
            grid.solid[grid.idx(i, j)] = 1;
        }
    }
    // 矩形面上的 q 保留默认 0.5（halfway BB）
}

void compute_ibb_distances(LatticeGrid& grid)
{
    // 通用版本：对任意已标记固体的网格，将 q 设为 0.5（halfway BB 退化）。
    // 如需精确 q，应调用 mark_solid_cylinder() 等专用函数。
    if (grid.model != LatticeModel::D2Q9) return;

    const int n = grid.size();
    const int Q = d2q9::Q;

    if (static_cast<int>(grid.q_ibb.size()) != n * Q) {
        grid.q_ibb.assign(n * Q, 0.5f);
    } else {
        // 仅重置流-固方向对上的 q，其余方向保持 0.5
        std::fill(grid.q_ibb.begin(), grid.q_ibb.end(), 0.5f);
    }
}

// ===========================================================================
// 半步长反弹（Halfway BounceBack，q = 0.5）—— 物理区域限定版（MPI 适配）
//
// 施加时机：stream() 之后
// 公式：f[opp(a)](x_f, t+dt) = f_tmp[a](x_f, t)
//
// g.f_tmp 在 stream() 末尾（swap 之后）保存碰后（post-collision）f 值。
//
// 参数 phys_i0/phys_j0/phys_i1/phys_j1 指定物理区域（本地坐标，含端点），
// 循环仅覆盖此范围，跳过幽灵行/列。非 MPI 模式下传入 0, 0, nx-1, ny-1。
// ===========================================================================
void apply_solid_bounce_back(LatticeGrid& grid,
                              int phys_i0, int phys_j0,
                              int phys_i1, int phys_j1)
{
    if (grid.model != LatticeModel::D2Q9) return;
    if (grid.solid.empty()) return;

    const int nx = grid.nx;
    const int ny = grid.ny;
    const int Q  = d2q9::Q;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int j = phys_j0; j <= phys_j1; ++j) {
        for (int i = phys_i0; i <= phys_i1; ++i) {
            const int nf = grid.idx(i, j);
            if (grid.solid[nf]) continue;  // 跳过固体节点

            for (int a = 1; a < Q; ++a) {  // 静止方向 a=0 无需处理
                const int ni = i + d2q9::C[a][0];
                const int nj = j + d2q9::C[a][1];
                // 边界检查（邻居可为幽灵节点或网格外）
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                if (!grid.solid[grid.idx(ni, nj)]) continue;  // 邻居非固体

                // 半步长反弹：f[opp(a)](x_f) ← f_tmp[a](x_f)
                const int oa = d2q9::OPP[a];
                grid.f[nf * Q + oa] = grid.f_tmp[nf * Q + a];
            }
        }
    }
}

// 向后兼容版：不限定物理区域，覆盖整个本地网格（适用于非 MPI 模式）。
void apply_solid_bounce_back(LatticeGrid& grid)
{
    apply_solid_bounce_back(grid, 0, 0, grid.nx - 1, grid.ny - 1);
}

// ===========================================================================
// Bouzidi 插值反弹（Interpolated BounceBack, IBB）—— 物理区域限定版（MPI 适配）
//
// 施加时机：stream() 之后
//
// 对于壁面距离分数 q = q_ibb[n_f * Q + a]（0 < q ≤ 1）：
//
//   q ≤ 0.5：
//     f[opp(a)](x_f) = 2q · f_tmp[a](x_f) + (1-2q) · f_tmp[a](x_f - c_a)
//     (x_f - c_a 为上游流体侧邻居；若为固体则退化为 halfway BB)
//     注意：x_f - c_a 可以是幽灵节点，此时 f_tmp 中已含正确幽灵碰后值（halo 交换后）。
//
//   q > 0.5：
//     f[opp(a)](x_f) = (1/2q) · f_tmp[a](x_f) + (1-1/2q) · f_tmp[opp(a)](x_f)
//     (使用本节点 opp(a) 方向的碰后值；来自 f_tmp)
//
// 参数 phys_i0/phys_j0/phys_i1/phys_j1 指定物理区域（本地坐标，含端点），
// 循环仅写物理节点，但上游节点读取允许超出物理范围（幽灵行有效）。
//
// 参考：Bouzidi et al. (2001); Yu et al. (2003) 的改进版本。
// ===========================================================================
void apply_solid_ibb(LatticeGrid& grid,
                     int phys_i0, int phys_j0,
                     int phys_i1, int phys_j1)
{
    if (grid.model != LatticeModel::D2Q9) return;
    if (grid.solid.empty()) return;

    const int nx = grid.nx;
    const int ny = grid.ny;
    const int Q  = d2q9::Q;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int j = phys_j0; j <= phys_j1; ++j) {
        for (int i = phys_i0; i <= phys_i1; ++i) {
            const int nf = grid.idx(i, j);
            if (grid.solid[nf]) continue;

            for (int a = 1; a < Q; ++a) {
                const int ca = d2q9::C[a][0];
                const int cb = d2q9::C[a][1];
                const int ni = i + ca;
                const int nj = j + cb;
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                if (!grid.solid[grid.idx(ni, nj)]) continue;

                const int    oa = d2q9::OPP[a];
                const double q  = static_cast<double>(grid.q_ibb[nf * Q + a]);

                if (q <= 0.5) {
                    // 上游流体侧邻居 x_f - c_a = (i-ca, j-cb)
                    // 允许访问幽灵节点（halo_exchange 已填入正确碰后值）
                    const int nni = i - ca;
                    const int nnj = j - cb;
                    if (nni >= 0 && nni < nx && nnj >= 0 && nnj < ny) {
                        const int nnn = grid.idx(nni, nnj);
                        if (!grid.solid[nnn]) {
                            // 线性插值（Bouzidi 2001 低 q 分支）
                            grid.f[nf * Q + oa] = 2.0 * q * grid.f_tmp[nf * Q + a]
                                                + (1.0 - 2.0 * q) * grid.f_tmp[nnn * Q + a];
                            continue;
                        }
                    }
                    // 流体侧邻居为固体或越界：退化为 halfway BB
                    grid.f[nf * Q + oa] = grid.f_tmp[nf * Q + a];
                } else {
                    // q > 0.5：Yu et al. (2003) 高 q 分支
                    const double inv2q = 1.0 / (2.0 * q);
                    grid.f[nf * Q + oa] = inv2q * grid.f_tmp[nf * Q + a]
                                        + (1.0 - inv2q) * grid.f_tmp[nf * Q + oa];
                }
            }
        }
    }
}

// 向后兼容版：不限定物理区域，覆盖整个本地网格（适用于非 MPI 模式）。
void apply_solid_ibb(LatticeGrid& grid)
{
    apply_solid_ibb(grid, 0, 0, grid.nx - 1, grid.ny - 1);
}

// ===========================================================================
// 清除所有固体节点标记（用于每步重新标记的运动固体）
// ===========================================================================
void clear_solid(LatticeGrid& grid)
{
    std::fill(grid.solid.begin(), grid.solid.end(), 0);
    if (!grid.q_ibb.empty())
        std::fill(grid.q_ibb.begin(), grid.q_ibb.end(), 0.5f);
    if (!grid.solid_bc_node.empty())
        std::fill(grid.solid_bc_node.begin(), grid.solid_bc_node.end(), 0);
}

// ===========================================================================
// 运动刚体反弹（Ladd 1994 移动壁面修正版 BB）
//
// 公式（Ladd 1994, Eq.3.5）：
//   f_ᾱ(x_f) = f_α*(x_f) - 2*w_α*ρ*(c_α·U_wall)/cs²
//
// 其中：
//   c_α    — 速度方向向量（格子单位）
//   w_α    — D2Q9 权重
//   cs²    = 1/3 （格子单位）
//   U_wall = U_cm + omega × r_w  （r_w = x_f + 0.5*c_α - x_cm）
//          = (ux_cm - omega * ry_w, uy_cm + omega * rx_w)
//
// 注意：ρ 在流体节点处已知；为简便取 ρ=1（格子单位）以与现有 IBM 一致。
// ===========================================================================
void apply_solid_bounce_back_moving_rigid(LatticeGrid& grid,
                                          double cx, double cy,
                                          double ux_cm, double uy_cm, double omega,
                                          int phys_i0, int phys_j0,
                                          int phys_i1, int phys_j1)
{
    if (grid.model != LatticeModel::D2Q9) return;
    if (grid.solid.empty()) return;

    const int nx = grid.nx;
    const int ny = grid.ny;
    const int Q  = d2q9::Q;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int j = phys_j0; j <= phys_j1; ++j) {
        for (int i = phys_i0; i <= phys_i1; ++i) {
            const int nf = grid.idx(i, j);
            if (grid.solid[nf]) continue;

            for (int a = 1; a < Q; ++a) {
                const int ca = d2q9::C[a][0];
                const int cb = d2q9::C[a][1];
                const int ni = i + ca;
                const int nj = j + cb;
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                if (!grid.solid[grid.idx(ni, nj)]) continue;

                const int oa = d2q9::OPP[a];

                // 壁面位置（流体节点沿 c_α 方向的半步处）
                const double rx_w = (i + 0.5 * ca) - cx;
                const double ry_w = (j + 0.5 * cb) - cy;

                // 壁面速度（刚体旋转）
                const double uw_x = ux_cm - omega * ry_w;
                const double uw_y = uy_cm + omega * rx_w;

                // c_α · U_wall
                const double cu = ca * uw_x + cb * uw_y;

                // Ladd 修正（cs² = 1/3，ρ = 1 格子单位）
                const double correction = 6.0 * d2q9::W[a] * cu;  // 6 = 2/cs²

                grid.f[nf * Q + oa] = grid.f_tmp[nf * Q + a] - correction;
            }
        }
    }
}

void apply_solid_bounce_back_moving_rigid(LatticeGrid& grid,
                                          double cx, double cy,
                                          double ux_cm, double uy_cm, double omega)
{
    apply_solid_bounce_back_moving_rigid(grid, cx, cy, ux_cm, uy_cm, omega,
                                         0, 0, grid.nx - 1, grid.ny - 1);
}

// ===========================================================================
// 运动刚体 Bouzidi IBB（Ladd 移动壁面修正版 IBB）
//
// 在 IBB 基础上叠加 Ladd 壁面速度修正项（同 BB 移动版）：
//   f_ᾱ(x_f) = [IBB 插值结果] - 2*w_α*ρ*(c_α·U_wall)/cs²
// ===========================================================================
void apply_solid_ibb_moving_rigid(LatticeGrid& grid,
                                  double cx, double cy,
                                  double ux_cm, double uy_cm, double omega,
                                  int phys_i0, int phys_j0,
                                  int phys_i1, int phys_j1)
{
    if (grid.model != LatticeModel::D2Q9) return;
    if (grid.solid.empty()) return;

    const int nx = grid.nx;
    const int ny = grid.ny;
    const int Q  = d2q9::Q;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int j = phys_j0; j <= phys_j1; ++j) {
        for (int i = phys_i0; i <= phys_i1; ++i) {
            const int nf = grid.idx(i, j);
            if (grid.solid[nf]) continue;

            for (int a = 1; a < Q; ++a) {
                const int ca = d2q9::C[a][0];
                const int cb = d2q9::C[a][1];
                const int ni = i + ca;
                const int nj = j + cb;
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                if (!grid.solid[grid.idx(ni, nj)]) continue;

                const int    oa = d2q9::OPP[a];
                const double q  = static_cast<double>(grid.q_ibb[nf * Q + a]);

                // Bouzidi IBB 无壁面速度项
                double f_ibb;
                if (q <= 0.5) {
                    const int nni = i - ca;
                    const int nnj = j - cb;
                    if (nni >= 0 && nni < nx && nnj >= 0 && nnj < ny) {
                        const int nnn = grid.idx(nni, nnj);
                        if (!grid.solid[nnn]) {
                            f_ibb = 2.0 * q * grid.f_tmp[nf * Q + a]
                                  + (1.0 - 2.0 * q) * grid.f_tmp[nnn * Q + a];
                        } else {
                            f_ibb = grid.f_tmp[nf * Q + a];
                        }
                    } else {
                        f_ibb = grid.f_tmp[nf * Q + a];
                    }
                } else {
                    const double inv2q = 1.0 / (2.0 * q);
                    f_ibb = inv2q * grid.f_tmp[nf * Q + a]
                          + (1.0 - inv2q) * grid.f_tmp[nf * Q + oa];
                }

                // Ladd 壁面速度修正（r_w 从质心到壁面）
                const double rx_w = (i + 0.5 * ca) - cx;
                const double ry_w = (j + 0.5 * cb) - cy;
                const double uw_x = ux_cm - omega * ry_w;
                const double uw_y = uy_cm + omega * rx_w;
                const double cu   = ca * uw_x + cb * uw_y;
                const double correction = 6.0 * d2q9::W[a] * cu;

                grid.f[nf * Q + oa] = f_ibb - correction;
            }
        }
    }
}

void apply_solid_ibb_moving_rigid(LatticeGrid& grid,
                                  double cx, double cy,
                                  double ux_cm, double uy_cm, double omega)
{
    apply_solid_ibb_moving_rigid(grid, cx, cy, ux_cm, uy_cm, omega,
                                  0, 0, grid.nx - 1, grid.ny - 1);
}
// ===========================================================================
void assign_solid_bc_unmarked(LatticeGrid& grid, int bc_mode)
{
    if (bc_mode == 0) return;
    if (grid.solid.empty()) return;

    const int n = grid.size();
    if (static_cast<int>(grid.solid_bc_node.size()) < n)
        grid.solid_bc_node.assign(n, 0);

    const auto bc = static_cast<int8_t>(bc_mode);
    for (int idx = 0; idx < n; ++idx) {
        if (grid.solid[idx] && grid.solid_bc_node[idx] == 0)
            grid.solid_bc_node[idx] = bc;
    }
}

// ===========================================================================
// 混合反弹边界条件施加器（BB/IBB 逐节点选择）
// ===========================================================================
void apply_solid_bc_mixed(LatticeGrid& grid, int default_bc,
                           int phys_i0, int phys_j0,
                           int phys_i1, int phys_j1)
{
    if (grid.model != LatticeModel::D2Q9) return;
    if (grid.solid.empty()) return;

    const int nx = grid.nx;
    const int ny = grid.ny;
    const int Q  = d2q9::Q;

    const bool have_per_node = !grid.solid_bc_node.empty();

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
    for (int j = phys_j0; j <= phys_j1; ++j) {
        for (int i = phys_i0; i <= phys_i1; ++i) {
            const int nf = grid.idx(i, j);
            if (grid.solid[nf]) continue;  // 跳过固体节点

            for (int a = 1; a < Q; ++a) {
                const int ca = d2q9::C[a][0];
                const int cb = d2q9::C[a][1];
                const int ni = i + ca;
                const int nj = j + cb;
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                const int ns = grid.idx(ni, nj);
                if (!grid.solid[ns]) continue;  // 邻居非固体

                // 确定该链接使用的反弹方案
                int bc = default_bc;
                if (have_per_node && grid.solid_bc_node[ns] != 0)
                    bc = grid.solid_bc_node[ns];

                const int oa = d2q9::OPP[a];

                if (bc == 1) {
                    // 半步长反弹（BounceBack）
                    grid.f[nf * Q + oa] = grid.f_tmp[nf * Q + a];
                } else if (bc == 2) {
                    // Bouzidi 插值反弹（IBB）
                    const double q = static_cast<double>(grid.q_ibb[nf * Q + a]);
                    if (q <= 0.5) {
                        const int nni = i - ca;
                        const int nnj = j - cb;
                        if (nni >= 0 && nni < nx && nnj >= 0 && nnj < ny) {
                            const int nnn = grid.idx(nni, nnj);
                            if (!grid.solid[nnn]) {
                                grid.f[nf * Q + oa] = 2.0 * q * grid.f_tmp[nf * Q + a]
                                                    + (1.0 - 2.0 * q) * grid.f_tmp[nnn * Q + a];
                                continue;
                            }
                        }
                        // 退化为 halfway BB
                        grid.f[nf * Q + oa] = grid.f_tmp[nf * Q + a];
                    } else {
                        const double inv2q = 1.0 / (2.0 * q);
                        grid.f[nf * Q + oa] = inv2q * grid.f_tmp[nf * Q + a]
                                            + (1.0 - inv2q) * grid.f_tmp[nf * Q + oa];
                    }
                }
                // bc == 0：无方案，跳过
            }
        }
    }
}

void apply_solid_bc_mixed(LatticeGrid& grid, int default_bc)
{
    apply_solid_bc_mixed(grid, default_bc, 0, 0, grid.nx - 1, grid.ny - 1);
}

// ===========================================================================
// 固体受力统计：动量交换法（Momentum Exchange Algorithm）—— 物理区域限定版
//
// 调用时机：apply_solid_bounce_back() / apply_solid_ibb() 之后，
// 此时 f_tmp 中保存碰后（post-collision）分布函数值。
//
// 动量交换公式（静止固体，Ladd 1994 / Aidun 1995）：
//   F_x = Σ_{流-固链接 (n_f, a)} 2 · f_tmp[n_f·Q+a] · c_a[x]
//   F_y = Σ_{流-固链接 (n_f, a)} 2 · f_tmp[n_f·Q+a] · c_a[y]
//
// MPI 模式下仅统计本进程物理区域内的贡献，调用方需在所有进程上对结果求和。
// ===========================================================================
void compute_solid_body_force(const LatticeGrid& grid,
                               double& out_fx, double& out_fy,
                               int phys_i0, int phys_j0,
                               int phys_i1, int phys_j1)
{
    out_fx = 0.0;
    out_fy = 0.0;

    if (grid.model != LatticeModel::D2Q9) return;
    if (grid.solid.empty() || grid.f_tmp.empty()) return;

    const int nx = grid.nx;
    const int ny = grid.ny;
    const int Q  = d2q9::Q;

    double fx = 0.0, fy = 0.0;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static) collapse(2) reduction(+:fx,fy)
#endif
    for (int j = phys_j0; j <= phys_j1; ++j) {
        for (int i = phys_i0; i <= phys_i1; ++i) {
            const int nf = grid.idx(i, j);
            if (grid.solid[nf]) continue;  // 跳过固体节点

            for (int a = 1; a < Q; ++a) {
                const int ca = d2q9::C[a][0];
                const int cb = d2q9::C[a][1];
                const int ni = i + ca;
                const int nj = j + cb;
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                if (!grid.solid[grid.idx(ni, nj)]) continue;  // 邻居非固体

                // 动量交换法（MEA）：固体受力 = 流体在流-固链接上传递的动量之和
                // 推导：在位置 x_f（流体节点），方向 a 指向固体。
                //   f_tmp[a](x_f)：碰后沿 +c_a 方向（流体→固体）的分布函数
                //   标准 MEA 力分量：Δp_a = 2 · f_tmp[a] · c_a
                //   （因为 f[a] 被固体"反射"，流体失去动量 2·f·c_a，固体得到 +2·f·c_a）
                // 符号验证：来流 ux > 0，a 沿 +x 方向时 ca>0，f_tmp[a] > 0 → fx > 0。
                //   这表示固体在来流方向受正力（阻力，流体推动固体）。
                const double f_post = grid.f_tmp[nf * Q + a];
                fx += 2.0 * f_post * ca;
                fy += 2.0 * f_post * cb;
            }
        }
    }

    out_fx = fx;
    out_fy = fy;
}

// 向后兼容版：覆盖整个本地网格（适用于非 MPI 模式）。
void compute_solid_body_force(const LatticeGrid& grid,
                               double& out_fx, double& out_fy)
{
    compute_solid_body_force(grid, out_fx, out_fy,
                              0, 0, grid.nx - 1, grid.ny - 1);
}

// ===========================================================================
// 从外部 CSV 网格文件加载固体边界（第三方网格接口）
//
// 文件格式（每行一个边界点，以逗号分隔）：
//   x, y [, q]
//   - x, y：边界点坐标（浮点数，格子单位）
//   - q：IBB 壁面距离分数（可选；缺省 0.5）
//
// 实现：
//   1. 解析文件中所有边界采样点
//   2. 对每个采样点，将最近格子节点标记为固体
//   3. 对每个 (流体节点, 方向) 对指向固体节点时，设定 q_ibb 值
//      若 q 列存在，使用文件中最近采样点的 q；否则使用 0.5。
//
// MPI 说明：
//   本函数使用本地网格坐标（含幽灵层）。在 MPI 模式下，
//   调用方需先将全局坐标转换为本地坐标后再调用本函数（与
//   mark_solid_cylinder() 相同的约定）。
// ===========================================================================
void mark_solid_from_mesh_file(LatticeGrid& grid, const std::string& filename)
{
    if (grid.model != LatticeModel::D2Q9) return;

    const int nx = grid.nx;
    const int ny = grid.ny;
    const int Q  = d2q9::Q;
    const int n  = grid.size();

    // 确保数组已分配
    if (static_cast<int>(grid.solid.size()) != n) {
        grid.solid.assign(n, 0);
        grid.q_ibb.assign(n * Q, 0.5f);
    }

    // ---- 1. 解析 CSV 文件 ----
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        throw std::runtime_error(
            "mark_solid_from_mesh_file: cannot open file: " + filename);
    }

    struct BoundaryPoint { double x, y, q; };
    std::vector<BoundaryPoint> pts;

    std::string line;
    int line_no = 0;
    while (std::getline(ifs, line)) {
        ++line_no;
        const auto tp = line.find_first_not_of(" \t\r\n");
        if (tp == std::string::npos) continue;
        if (line[tp] == '#') continue;

        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        double x = 0.0, y = 0.0, q = 0.5;
        if (!(iss >> x >> y)) {
            throw std::runtime_error(
                "mark_solid_from_mesh_file: parse error at line " +
                std::to_string(line_no) + " in " + filename);
        }
        double q_tmp;
        if (iss >> q_tmp) q = q_tmp;
        pts.push_back({x, y, q});
    }

    if (pts.empty()) {
        throw std::runtime_error(
            "mark_solid_from_mesh_file: no boundary points loaded from " + filename);
    }

    // ---- 2. 将采样点标记为固体节点（最近格子节点）----
    // 同时记录每个节点关联的最小 q（用于后续 IBB 距离设定）
    std::vector<double> node_q(n, 0.5);

    for (const auto& pt : pts) {
        // 最近格子节点（四舍五入）
        const int i = static_cast<int>(std::round(pt.x));
        const int j = static_cast<int>(std::round(pt.y));
        if (i < 0 || i >= nx || j < 0 || j >= ny) continue;
        const int idx = grid.idx(i, j);
        grid.solid[idx] = 1;
        node_q[idx] = pt.q;  // 记录该节点关联的 q 值
    }

    // ---- 3. 对流体节点的流-固方向对设定 q_ibb ----
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int nf = grid.idx(i, j);
            if (grid.solid[nf]) continue;

            for (int a = 1; a < Q; ++a) {
                const int ca = d2q9::C[a][0];
                const int cb = d2q9::C[a][1];
                const int ni = i + ca;
                const int nj = j + cb;
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                const int ns = grid.idx(ni, nj);
                if (!grid.solid[ns]) continue;

                // 使用固体节点关联的 q（文件中提供的距离分数）
                const double q = std::min(std::max(node_q[ns], 1e-5), 1.0);
                grid.q_ibb[nf * Q + a] = static_cast<float>(q);
            }
        }
    }
}

} // namespace lbm
