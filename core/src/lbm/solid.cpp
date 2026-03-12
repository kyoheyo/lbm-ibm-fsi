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

} // namespace lbm
