// core/src/lbm/mg_tree.cpp — 多重网格嵌套关系树（MgTree / MgNode）实现
//
// 本文件实现 mg_tree.hpp 中声明的 MgTree 和辅助函数。
// 详细文档请参阅头文件 include/lbm/mg_tree.hpp 和 docs/MPI并行详解.md。

#include "lbm/mg_tree.hpp"

#include <queue>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#ifdef LBM_ENABLE_OPENMP
#  include <omp.h>
#endif

namespace lbm {

// ---------------------------------------------------------------------------
// MgTree 实现
// ---------------------------------------------------------------------------

MgTree::MgTree(MgExtent root_extent, MgDim dim)
{
    auto node           = std::make_unique<MgNode>();
    node->extent        = root_extent;
    node->level         = 0;
    node->refine_ratio  = 1;   // 根节点无加密（relative to itself）
    node->dim           = dim;
    node->parent        = nullptr;
    root_ = node.get();
    all_nodes_.push_back(std::move(node));
}

MgNode* MgTree::add_level(MgNode* parent, MgExtent child_extent, int refine_ratio)
{
    if (!parent) {
        throw std::invalid_argument("MgTree::add_level: parent must not be nullptr");
    }
    if (!parent->extent.contains(child_extent)) {
        throw std::invalid_argument(
            "MgTree::add_level: child_extent must be within parent->extent. "
            "parent=[" + std::to_string(parent->extent.x_start) + ","
                       + std::to_string(parent->extent.x_end) + "  "
                       + std::to_string(parent->extent.y_start) + ","
                       + std::to_string(parent->extent.y_end) + "] "
            "child=[" + std::to_string(child_extent.x_start) + ","
                      + std::to_string(child_extent.x_end) + "  "
                      + std::to_string(child_extent.y_start) + ","
                      + std::to_string(child_extent.y_end) + "]"
        );
    }
    if (refine_ratio < 1 || refine_ratio > 1024) {
        throw std::invalid_argument("MgTree::add_level: refine_ratio must be in [1, 1024]");
    }

    auto node           = std::make_unique<MgNode>();
    node->extent        = child_extent;
    node->level         = parent->level + 1;
    node->refine_ratio  = refine_ratio;
    node->dim           = parent->dim;   // 继承父节点维度
    node->parent        = parent;

    // -----------------------------------------------------------------------
    // 设置域边界标志：检查子节点的每侧是否与全局域（根节点）的边界重合。
    // 若重合，mg_apply_fringe_bc* 会跳过该侧的 fringe 插值，
    // 让 fine.solver 注册的域 BC 独立控制那些节点；
    // mg_couple_fine_to_coarse 也跳过向粗网格对应 fringe 区写入。
    // -----------------------------------------------------------------------
    node->west_is_domain_wall  = (child_extent.x_start == root_->extent.x_start);
    node->east_is_domain_wall  = (child_extent.x_end   == root_->extent.x_end);
    node->south_is_domain_wall = (child_extent.y_start == root_->extent.y_start);
    node->north_is_domain_wall = (child_extent.y_end   == root_->extent.y_end);

    MgNode* ptr = node.get();
    parent->children.push_back(ptr);
    all_nodes_.push_back(std::move(node));
    return ptr;
}

void MgTree::traverse_coarse_to_fine(const std::function<void(MgNode*)>& fn)
{
    std::queue<MgNode*> q;
    q.push(root_);
    while (!q.empty()) {
        MgNode* node = q.front();
        q.pop();
        fn(node);
        for (MgNode* child : node->children) {
            q.push(child);
        }
    }
}

void MgTree::traverse_fine_to_coarse(const std::function<void(MgNode*)>& fn)
{
    // 先用 BFS 收集所有节点（由粗到细），再逆序调用回调（由细到粗）
    std::vector<MgNode*> order;
    order.reserve(all_nodes_.size());
    std::queue<MgNode*> q;
    q.push(root_);
    while (!q.empty()) {
        MgNode* node = q.front();
        q.pop();
        order.push_back(node);
        for (MgNode* child : node->children) {
            q.push(child);
        }
    }
    for (auto it = order.rbegin(); it != order.rend(); ++it) {
        fn(*it);
    }
}

int MgTree::max_level() const
{
    int max_lv = 0;
    for (const auto& node : all_nodes_) {
        if (node->level > max_lv) max_lv = node->level;
    }
    return max_lv;
}

std::vector<MgNode*> MgTree::nodes_at_level(int level) const
{
    std::vector<MgNode*> result;
    for (const auto& node : all_nodes_) {
        if (node->level == level) {
            result.push_back(node.get());
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// AMR 延拓算子（Prolongation）：粗 → 细，双线性插值 ρ/u
//
// 坐标映射说明（2D，D2Q9）：
//   细节点 (if, jf)（0-based 本地细格索引）在粗坐标系中的节点位置：
//     px = fine.extent.x_start + if / r
//     py = fine.extent.y_start + jf / r
//   双线性插值：取 floor(px) 和 floor(py) 确定左下粗节点，alpha = px - floor(px)。
//   这与 LBM 节点布局一致（节点在整数坐标处，不是格心）。
// ---------------------------------------------------------------------------
void mg_prolong_rho_u(const MgNode& coarse, MgNode& fine)
{
    if (!coarse.grid || !fine.grid) {
        throw std::invalid_argument("mg_prolong_rho_u: both coarse and fine nodes must have grids");
    }

    const LatticeGrid& cg = *coarse.grid;
    LatticeGrid&       fg = *fine.grid;
    const int r  = fine.refine_ratio;
    const int d  = cg.dim();  // 2 for D2Q9

    // 在 x/y 方向上循环细节点（fine LatticeGrid 的本地尺寸）
    const int fn_x = fg.nx;
    const int fn_y = fg.ny;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int jf = 0; jf < fn_y; ++jf) {
        for (int ix = 0; ix < fn_x; ++ix) {
            // 细节点在粗坐标系中的浮点位置（节点位于整数坐标处）
            const double px = fine.extent.x_start + static_cast<double>(ix) / r;
            const double py = fine.extent.y_start + static_cast<double>(jf)  / r;

            // 双线性插值的左下粗节点（全局粗坐标系）
            const int i0 = static_cast<int>(std::floor(px));
            const int j0 = static_cast<int>(std::floor(py));

            // 双线性权重（alpha=0 → 完全使用左节点；alpha=1 → 完全使用右节点）
            const double alpha = px - i0;  // [0, 1]
            const double beta  = py - j0;

            // 四角粗节点的本地索引（夹持到粗格有效范围）
            auto clamp_ci = [&](int ci) -> int {
                return std::max(coarse.extent.x_start,
                                std::min(coarse.extent.x_end, ci));
            };
            auto clamp_cj = [&](int cj) -> int {
                return std::max(coarse.extent.y_start,
                                std::min(coarse.extent.y_end, cj));
            };

            const int ci00 = clamp_ci(i0)     - coarse.extent.x_start;
            const int ci10 = clamp_ci(i0 + 1) - coarse.extent.x_start;
            const int cj00 = clamp_cj(j0)     - coarse.extent.y_start;
            const int cj10 = clamp_cj(j0 + 1) - coarse.extent.y_start;

            // 粗节点索引
            const int c00 = cg.idx(ci00, cj00);
            const int c10 = cg.idx(ci10, cj00);
            const int c01 = cg.idx(ci00, cj10);
            const int c11 = cg.idx(ci10, cj10);

            // 双线性插值权重
            const double w00 = (1.0 - alpha) * (1.0 - beta);
            const double w10 = alpha          * (1.0 - beta);
            const double w01 = (1.0 - alpha)  * beta;
            const double w11 = alpha           * beta;

            const int fi = fg.idx(ix, jf);

            // 插值 ρ
            fg.rho[fi] = w00 * cg.rho[c00]
                       + w10 * cg.rho[c10]
                       + w01 * cg.rho[c01]
                       + w11 * cg.rho[c11];

            // 插值 u（每个空间分量）
            for (int k = 0; k < d; ++k) {
                fg.u[fi * d + k] = w00 * cg.u[c00 * d + k]
                                 + w10 * cg.u[c10 * d + k]
                                 + w01 * cg.u[c01 * d + k]
                                 + w11 * cg.u[c11 * d + k];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AMR 限制算子（Restriction）：细 → 粗，体积平均 ρ/u
//
// 对粗格 (ic_local, jc_local)，覆盖细格范围 if ∈ [ic_local*r, (ic_local+1)*r)，
// 取这 r×r 个细格的简单平均写入粗格。
// ---------------------------------------------------------------------------
void mg_restrict_rho_u(const MgNode& fine, MgNode& coarse)
{
    if (!fine.grid || !coarse.grid) {
        throw std::invalid_argument("mg_restrict_rho_u: both fine and coarse nodes must have grids");
    }

    const LatticeGrid& fg = *fine.grid;
    LatticeGrid&       cg = *coarse.grid;
    const int r  = fine.refine_ratio;
    const int d  = fg.dim();

    // 遍历细网格覆盖的粗格范围（以本地粗格索引）
    const int cx_lo = fine.extent.x_start - coarse.extent.x_start;
    const int cy_lo = fine.extent.y_start - coarse.extent.y_start;
    const int cx_hi = fine.extent.x_end   - coarse.extent.x_start;
    const int cy_hi = fine.extent.y_end   - coarse.extent.y_start;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int jc = cy_lo; jc <= cy_hi; ++jc) {
        for (int ic = cx_lo; ic <= cx_hi; ++ic) {
            // 对应细格的起始本地索引
            const int if0 = (ic - cx_lo) * r;
            const int jf0 = (jc - cy_lo) * r;

            double rho_sum = 0.0;
            double u_sum[3] = {0.0, 0.0, 0.0};
            int    count    = 0;

            for (int djf = 0; djf < r && jf0 + djf < fg.ny; ++djf) {
                for (int dif = 0; dif < r && if0 + dif < fg.nx; ++dif) {
                    const int fi = fg.idx(if0 + dif, jf0 + djf);
                    rho_sum += fg.rho[fi];
                    for (int k = 0; k < d; ++k) {
                        u_sum[k] += fg.u[fi * d + k];
                    }
                    ++count;
                }
            }

            if (count > 0) {
                const double inv = 1.0 / count;
                const int ci = cg.idx(ic, jc);
                cg.rho[ci] = rho_sum * inv;
                for (int k = 0; k < d; ++k) {
                    cg.u[ci * d + k] = u_sum[k] * inv;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// f 分布函数延拓（平衡态重建）
// ---------------------------------------------------------------------------
void mg_prolong_f(const MgNode& coarse, MgNode& fine)
{
    if (!coarse.grid || !fine.grid) {
        throw std::invalid_argument("mg_prolong_f: both coarse and fine nodes must have grids");
    }
    if (fine.grid->model != LatticeModel::D2Q9) {
        throw std::invalid_argument("mg_prolong_f: only D2Q9 is currently supported");
    }

    // 第一步：延拓宏观量 ρ/u（双线性插值）
    mg_prolong_rho_u(coarse, fine);

    // 第二步：在细网格每个节点用延拓后的 (ρ, u) 重建平衡分布函数 f
    LatticeGrid& fg = *fine.grid;
    const int fn = fg.size();

    for (int i = 0; i < fn; ++i) {
        const double rho_f = fg.rho[i];
        const double ux_f  = fg.u[i * 2 + 0];
        const double uy_f  = fg.u[i * 2 + 1];

        for (int a = 0; a < d2q9::Q; ++a) {
            const double c[2] = {
                static_cast<double>(d2q9::C[a][0]),
                static_cast<double>(d2q9::C[a][1])
            };
            const double u[2] = {ux_f, uy_f};
            fg.f[i * d2q9::Q + a] = f_eq(d2q9::W[a], rho_f, c, u, 2);
        }
        // f_tmp 同步初始化为相同的平衡值（避免首步流式时使用未初始化数据）
        for (int a = 0; a < d2q9::Q; ++a) {
            fg.f_tmp[i * d2q9::Q + a] = fg.f[i * d2q9::Q + a];
        }
    }
}

// ---------------------------------------------------------------------------
// 覆盖网格（Overset/Fringe）耦合：C→F 耦合（论文 Eqs. 9–10）
//
// 算法：对细网格 fringe 区域的每个节点 (ix, jy)：
//   1. 计算该节点在粗坐标系中的浮点位置 (px, py)
//   2. 双线性插值粗网格 ρ、u、f_neq（在四角粗节点上计算 f_neq = f - f_eq，再插值）
//   3. fi,f = f_eq(ρ_interp, u_interp) + (ωc/2ωf) * f_neq_interp
//
// 当 (ix, jy) 与粗节点重合（alpha=β=0）时，双线性插值退化为精确取值（对应 Eq. 9）；
// 否则为空间插值（对应 Eq. 10）。
// ---------------------------------------------------------------------------
void mg_apply_fringe_bc(const MgNode& coarse, MgNode& fine,
                        int fringe_width, double omega_c)
{
    if (!coarse.grid || !fine.grid) {
        throw std::invalid_argument("mg_apply_fringe_bc: both coarse and fine nodes must have grids");
    }
    if (fine.grid->model != LatticeModel::D2Q9) {
        throw std::invalid_argument("mg_apply_fringe_bc: only D2Q9 is currently supported");
    }
    if (omega_c <= 0.0 || omega_c >= 4.0) {
        throw std::invalid_argument(
            "mg_apply_fringe_bc: omega_c must be in (0, 4)");
    }
    if (fringe_width < 1) fringe_width = 1;

    const LatticeGrid& cg = *coarse.grid;
    LatticeGrid&       fg = *fine.grid;

    const int r   = fine.refine_ratio;
    const int d   = fg.dim();          // 2 for D2Q9
    const int Q   = d2q9::Q;          // 9
    const int fnx = fg.nx;
    const int fny = fg.ny;

    // 粗网格累积空间加密比：根节点到 coarse 节点所有 refine_ratio 的乘积。
    // 例：root(r=1)→L1(r=2)→L2(r=2)，L1 as coarse → coarse_scale=2，L2 as coarse → 4。
    // 用于将根坐标系下的整数坐标映射到 coarse 本地数组索引：
    //   coarse_local = (root_coord − coarse.extent.x_start) * coarse_scale
    const int coarse_scale = mg_total_subcycle_steps(coarse);

    // 松弛频率缩放（Eq. 4），C→F 缩放比 = ωc / (2ωf)
    const double omega_f = mg_omega_rescale(omega_c);
    const double scale   = omega_c / (2.0 * omega_f);

    // 域边界标志：若某侧与全局域边界重合，则跳过该侧 fringe 的 C→F 插值，
    // 保留该侧 fine.solver 的域 BC（Zou-He/BB 等）独立控制边界节点。
    const bool skip_west  = fine.west_is_domain_wall;
    const bool skip_east  = fine.east_is_domain_wall;
    const bool skip_south = fine.south_is_domain_wall;
    const bool skip_north = fine.north_is_domain_wall;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int jy = 0; jy < fny; ++jy) {
        for (int ix = 0; ix < fnx; ++ix) {
            // 判断是否为 fringe 节点（外边界 fringe_width 层内）
            const bool in_fringe = (ix < fringe_width || ix >= fnx - fringe_width ||
                                    jy < fringe_width || jy >= fny - fringe_width);
            if (!in_fringe) continue;

            // 若该 fringe 节点位于与全局域边界重合的一侧，跳过 C→F 插值。
            // 该侧的边界条件由 fine.solver 注册的域 BC 独立施加。
            const bool on_dom = (skip_west  && ix < fringe_width)
                             || (skip_east  && ix >= fnx - fringe_width)
                             || (skip_south && jy < fringe_width)
                             || (skip_north && jy >= fny - fringe_width);
            if (on_dom) continue;

            // 细节点在根坐标系中的浮点位置
            // fine.extent.x_start 为根坐标（所有层的 extent 均以根坐标存储）
            const double px = fine.extent.x_start + static_cast<double>(ix) / r;
            const double py = fine.extent.y_start + static_cast<double>(jy) / r;

            const int i0 = static_cast<int>(std::floor(px));
            const int j0 = static_cast<int>(std::floor(py));

            const double alpha = px - i0;
            const double beta  = py - j0;

            auto clamp_ci = [&](int ci) -> int {
                return std::max(coarse.extent.x_start,
                                std::min(coarse.extent.x_end, ci));
            };
            auto clamp_cj = [&](int cj) -> int {
                return std::max(coarse.extent.y_start,
                                std::min(coarse.extent.y_end, cj));
            };

            // 将根坐标转换为粗网格本地数组索引：乘以 coarse_scale
            const int ci00 = (clamp_ci(i0)     - coarse.extent.x_start) * coarse_scale;
            const int ci10 = (clamp_ci(i0 + 1) - coarse.extent.x_start) * coarse_scale;
            const int cj00 = (clamp_cj(j0)     - coarse.extent.y_start) * coarse_scale;
            const int cj10 = (clamp_cj(j0 + 1) - coarse.extent.y_start) * coarse_scale;

            const int c00 = cg.idx(ci00, cj00);
            const int c10 = cg.idx(ci10, cj00);
            const int c01 = cg.idx(ci00, cj10);
            const int c11 = cg.idx(ci10, cj10);

            const double w00 = (1.0 - alpha) * (1.0 - beta);
            const double w10 = alpha          * (1.0 - beta);
            const double w01 = (1.0 - alpha)  * beta;
            const double w11 = alpha           * beta;

            // 插值粗网格 ρ 和 u
            double rho_f = w00 * cg.rho[c00] + w10 * cg.rho[c10]
                         + w01 * cg.rho[c01] + w11 * cg.rho[c11];
            double u_f[2];
            for (int k = 0; k < d; ++k) {
                u_f[k] = w00 * cg.u[c00 * d + k] + w10 * cg.u[c10 * d + k]
                       + w01 * cg.u[c01 * d + k] + w11 * cg.u[c11 * d + k];
            }

            const int fi = fg.idx(ix, jy);
            fg.rho[fi] = rho_f;
            for (int k = 0; k < d; ++k) fg.u[fi * d + k] = u_f[k];

            // 对每个方向 a：计算插值后的 f_neq，应用 Eqs. 9–10
            for (int a = 0; a < Q; ++a) {
                const double ca[2] = {
                    static_cast<double>(d2q9::C[a][0]),
                    static_cast<double>(d2q9::C[a][1])
                };

                // 在各粗角点计算 f_neq_a = f_a - f_eq_a(ρ, u)
                auto f_neq_corner = [&](int cidx) -> double {
                    const double rc = cg.rho[cidx];
                    const double uc[2] = {cg.u[cidx * d], cg.u[cidx * d + 1]};
                    return cg.f[cidx * Q + a] - f_eq(d2q9::W[a], rc, ca, uc, d);
                };

                // 双线性插值 f_neq
                const double f_neq_interp =
                    w00 * f_neq_corner(c00) + w10 * f_neq_corner(c10)
                  + w01 * f_neq_corner(c01) + w11 * f_neq_corner(c11);

                // fi,f = f_eq(ρ_interp, u_interp) + (ωc/2ωf) * f_neq_interp
                fg.f[fi * Q + a] = f_eq(d2q9::W[a], rho_f, ca, u_f, d)
                                 + scale * f_neq_interp;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 粗→细（C→F）时间+空间双重插值耦合（论文 Algorithm 步骤 3）
//
// 算法：
//   1. 对 fringe 区域每个细节点 (ix, jy)：
//      a. 确定四角粗节点 (c00, c10, c01, c11) 及双线性权重 (w00..w11)
//      b. 对每个粗角点 cidx：线性时间插值 f_a_half = (1-t_alpha)*f_prev[cidx] + t_alpha*f_next[cidx]
//         → 从 f_a_half 计算 ρ_half, u_half, f_neq_half = f_a_half - f_eq(ρ_half, u_half)
//      c. 双线性空间插值 ρ_interp, u_interp, f_neq_interp（与 mg_apply_fringe_bc 相同）
//      d. 应用 Eqs. 9–10：fi,f = f_eq(ρ_interp, u_interp) + (ωc/2ωf) * f_neq_interp
// ---------------------------------------------------------------------------
void mg_apply_fringe_bc_temporal(const MgNode& coarse_prev,
                                  const MgNode& coarse,
                                  MgNode& fine,
                                  double t_alpha,
                                  int fringe_width,
                                  double omega_c)
{
    if (!coarse_prev.grid || !coarse.grid || !fine.grid) {
        throw std::invalid_argument(
            "mg_apply_fringe_bc_temporal: coarse_prev, coarse, and fine nodes must all have grids");
    }
    if (fine.grid->model != LatticeModel::D2Q9) {
        throw std::invalid_argument(
            "mg_apply_fringe_bc_temporal: only D2Q9 is currently supported");
    }
    if (omega_c <= 0.0 || omega_c >= 4.0) {
        throw std::invalid_argument(
            "mg_apply_fringe_bc_temporal: omega_c must be in (0, 4)");
    }
    if (t_alpha < 0.0 || t_alpha > 1.0) {
        throw std::invalid_argument(
            "mg_apply_fringe_bc_temporal: t_alpha must be in [0, 1]");
    }
    if (coarse_prev.grid->nx != coarse.grid->nx ||
        coarse_prev.grid->ny != coarse.grid->ny) {
        throw std::invalid_argument(
            "mg_apply_fringe_bc_temporal: coarse_prev and coarse grids must have identical dimensions");
    }
    if (fringe_width < 1) fringe_width = 1;

    const LatticeGrid& cg_prev = *coarse_prev.grid;
    const LatticeGrid& cg      = *coarse.grid;
    LatticeGrid&       fg      = *fine.grid;

    const int r   = fine.refine_ratio;
    const int d   = fg.dim();
    const int Q   = d2q9::Q;
    const int fnx = fg.nx;
    const int fny = fg.ny;

    const double alpha1 = 1.0 - t_alpha;   // weight for t (prev)
    const double alpha2 = t_alpha;           // weight for t+δtc (next)

    // 粗网格累积空间加密比（同 mg_apply_fringe_bc 注释）
    const int coarse_scale = mg_total_subcycle_steps(coarse);

    // 松弛频率缩放（Eq. 4），C→F 缩放比 = ωc / (2ωf)
    const double omega_f = mg_omega_rescale(omega_c);
    const double scale   = omega_c / (2.0 * omega_f);

    // 域边界标志：与 mg_apply_fringe_bc 相同，跳过与全局域边界重合侧的 fringe 插值。
    const bool skip_west  = fine.west_is_domain_wall;
    const bool skip_east  = fine.east_is_domain_wall;
    const bool skip_south = fine.south_is_domain_wall;
    const bool skip_north = fine.north_is_domain_wall;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int jy = 0; jy < fny; ++jy) {
        for (int ix = 0; ix < fnx; ++ix) {
            // fringe 判断
            const bool in_fringe = (ix < fringe_width || ix >= fnx - fringe_width ||
                                    jy < fringe_width || jy >= fny - fringe_width);
            if (!in_fringe) continue;

            // 若该 fringe 节点位于域边界侧，跳过 C→F 时间插值；
            // 该侧由 fine.solver 的域 BC 控制。
            const bool on_dom = (skip_west  && ix < fringe_width)
                             || (skip_east  && ix >= fnx - fringe_width)
                             || (skip_south && jy < fringe_width)
                             || (skip_north && jy >= fny - fringe_width);
            if (on_dom) continue;

            // 细节点在根坐标系中的浮点位置
            const double px = fine.extent.x_start + static_cast<double>(ix) / r;
            const double py = fine.extent.y_start + static_cast<double>(jy) / r;

            const int i0 = static_cast<int>(std::floor(px));
            const int j0 = static_cast<int>(std::floor(py));

            const double bx = px - i0;
            const double by = py - j0;

            auto clamp_ci = [&](int ci) -> int {
                return std::max(coarse.extent.x_start,
                                std::min(coarse.extent.x_end, ci));
            };
            auto clamp_cj = [&](int cj) -> int {
                return std::max(coarse.extent.y_start,
                                std::min(coarse.extent.y_end, cj));
            };

            // 将根坐标转换为粗网格本地数组索引（乘以 coarse_scale）
            const int ci00 = (clamp_ci(i0)     - coarse.extent.x_start) * coarse_scale;
            const int ci10 = (clamp_ci(i0 + 1) - coarse.extent.x_start) * coarse_scale;
            const int cj00 = (clamp_cj(j0)     - coarse.extent.y_start) * coarse_scale;
            const int cj10 = (clamp_cj(j0 + 1) - coarse.extent.y_start) * coarse_scale;

            const int c00 = cg.idx(ci00, cj00);
            const int c10 = cg.idx(ci10, cj00);
            const int c01 = cg.idx(ci00, cj10);
            const int c11 = cg.idx(ci10, cj10);

            const double w00 = (1.0 - bx) * (1.0 - by);
            const double w10 = bx          * (1.0 - by);
            const double w01 = (1.0 - bx)  * by;
            const double w11 = bx           * by;

            // 双线性权重数组便于统一循环
            const double ws[4]  = {w00, w10, w01, w11};
            const int    cs[4]  = {c00, c10, c01, c11};

            // 空间插值 ρ 和 u（对每个粗角点先做时间插值，再做空间双线性插值）
            double rho_f    = 0.0;
            double u_f[2]   = {0.0, 0.0};
            for (int corner = 0; corner < 4; ++corner) {
                const int cidx = cs[corner];
                // 时间插值 ρ
                const double rho_c = alpha1 * cg_prev.rho[cidx] + alpha2 * cg.rho[cidx];
                // 时间插值 u
                const double uc[2] = {
                    alpha1 * cg_prev.u[cidx * d + 0] + alpha2 * cg.u[cidx * d + 0],
                    alpha1 * cg_prev.u[cidx * d + 1] + alpha2 * cg.u[cidx * d + 1]
                };
                rho_f    += ws[corner] * rho_c;
                u_f[0]   += ws[corner] * uc[0];
                u_f[1]   += ws[corner] * uc[1];
            }

            const int fi = fg.idx(ix, jy);
            fg.rho[fi] = rho_f;
            for (int k = 0; k < d; ++k) fg.u[fi * d + k] = u_f[k];

            // 对每个方向 a：计算时间+空间插值后的 f_neq，应用 Eqs. 9–10
            for (int a = 0; a < Q; ++a) {
                const double ca[2] = {
                    static_cast<double>(d2q9::C[a][0]),
                    static_cast<double>(d2q9::C[a][1])
                };

                // 对每个粗角点：先做时间插值 f_a_half，再计算 f_neq_half
                double f_neq_interp = 0.0;
                for (int corner = 0; corner < 4; ++corner) {
                    const int cidx = cs[corner];
                    // 时间插值 f_a
                    const double f_a_half = alpha1 * cg_prev.f[cidx * Q + a]
                                          + alpha2 * cg.f[cidx * Q + a];
                    // 时间插值 ρ 和 u（各角点）
                    const double rho_c = alpha1 * cg_prev.rho[cidx] + alpha2 * cg.rho[cidx];
                    const double uc[2] = {
                        alpha1 * cg_prev.u[cidx * d + 0] + alpha2 * cg.u[cidx * d + 0],
                        alpha1 * cg_prev.u[cidx * d + 1] + alpha2 * cg.u[cidx * d + 1]
                    };
                    // f_neq_half 在该粗角点
                    const double f_neq_c = f_a_half - f_eq(d2q9::W[a], rho_c, ca, uc, d);
                    f_neq_interp += ws[corner] * f_neq_c;
                }

                // Eqs. 9–10：fi,f = f_eq(ρ_interp, u_interp) + (ωc/2ωf) * f_neq_interp
                fg.f[fi * Q + a] = f_eq(d2q9::W[a], rho_f, ca, u_f, d)
                                 + scale * f_neq_interp;
            }
        }
    }
}


// 算法：对细网格域边界（fringe_width 细格 ↔ coarse_fringe 粗格）内的每个粗节点 (ic, jc)：
//   1. 映射到对应细本地索引 if = (ic − fx_s)*r, jf = (jc − fy_s)*r
//   2. 从细网格 f 计算 ρf 和 uf
//   3. 对每个方向 a：
//      - 按 Eq. 8 对 Q 邻格（细坐标偏移 ej）的 f_neq_a 取平均（越界处夹取）
//      - 按 Eq. 7 更新 fi,c = f_eq_a(ρf, uf) + (2ωf/ωc) * f_neq_filtered_a
//   4. 同步粗网格 ρc 和 uc
// ---------------------------------------------------------------------------
void mg_couple_fine_to_coarse(const MgNode& fine, MgNode& coarse,
                               int fringe_width, double omega_c)
{
    if (!fine.grid || !coarse.grid) {
        throw std::invalid_argument(
            "mg_couple_fine_to_coarse: both fine and coarse nodes must have grids");
    }
    if (fine.grid->model != LatticeModel::D2Q9) {
        throw std::invalid_argument(
            "mg_couple_fine_to_coarse: only D2Q9 is currently supported");
    }
    if (omega_c <= 0.0 || omega_c >= 4.0) {
        throw std::invalid_argument(
            "mg_couple_fine_to_coarse: omega_c must be in (0, 4)");
    }
    if (fringe_width < 1) fringe_width = 1;

    const LatticeGrid& fg = *fine.grid;
    LatticeGrid&       cg = *coarse.grid;

    const int r  = fine.refine_ratio;
    const int d  = fg.dim();   // 2 for D2Q9
    const int Q  = d2q9::Q;   // 9

    // 松弛频率缩放（Eq. 4），F→C 缩放比 = 2ωf / ωc
    const double omega_f = mg_omega_rescale(omega_c);
    const double scale   = 2.0 * omega_f / omega_c;

    // 粗网格累积空间加密比（同 mg_apply_fringe_bc 注释）
    const int coarse_scale = mg_total_subcycle_steps(coarse);

    // 粗网格 fringe 宽度（细 fringe_width 对应的粗格数，至少 1）
    const int coarse_fringe = std::max(1, (fringe_width + r - 1) / r);

    // 细网格范围（根坐标系）
    const int fx_s = fine.extent.x_start;
    const int fx_e = fine.extent.x_end;
    const int fy_s = fine.extent.y_start;
    const int fy_e = fine.extent.y_end;

    // 域边界标志：若细网格某侧与全局域边界重合，跳过该侧的 F→C 写回，
    // 保留粗网格由域 BC 设置的边界值（下次 coarse.solver->step() 会重新施加域 BC）。
    const bool skip_west  = fine.west_is_domain_wall;
    const bool skip_east  = fine.east_is_domain_wall;
    const bool skip_south = fine.south_is_domain_wall;
    const bool skip_north = fine.north_is_domain_wall;

    // 循环计数（闭区间 [fx_s, fx_e] 转为 0-based 以支持 OpenMP collapse）
    const int nx_count = fx_e - fx_s + 1;
    const int ny_count = fy_e - fy_s + 1;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int jj = 0; jj < ny_count; ++jj) {
        for (int ii = 0; ii < nx_count; ++ii) {
            const int jc_g = fy_s + jj;   // 根坐标
            const int ic_g = fx_s + ii;   // 根坐标
            // 粗网格本地数组索引：乘以 coarse_scale 以适应空间加密后的粗层
            const int jc_l = (jc_g - coarse.extent.y_start) * coarse_scale;
            const int ic_l = (ic_g - coarse.extent.x_start) * coarse_scale;

            // 判断是否在 fringe 内
            const bool in_j_fringe = (jc_g <= fy_s + coarse_fringe - 1 ||
                                       jc_g >= fy_e - coarse_fringe + 1);
            const bool in_i_fringe = (ic_g <= fx_s + coarse_fringe - 1 ||
                                       ic_g >= fx_e - coarse_fringe + 1);
            if (!in_i_fringe && !in_j_fringe) continue;

            // 若该粗 fringe 节点对应细网格的域边界侧，跳过 F→C 写回。
            // 粗网格在该侧的域 BC 在下一次 coarse.solver->step() 中重新施加。
            const bool on_dom = (skip_west  && ic_g <= fx_s + coarse_fringe - 1)
                             || (skip_east  && ic_g >= fx_e - coarse_fringe + 1)
                             || (skip_south && jc_g <= fy_s + coarse_fringe - 1)
                             || (skip_north && jc_g >= fy_e - coarse_fringe + 1);
            if (on_dom) continue;

            // 本粗节点在细网格中的本地索引
            const int if_loc = (ic_g - fx_s) * r;
            const int jf_loc = (jc_g - fy_s) * r;

            // 越界保护（正常配置不应触发）
            if (if_loc < 0 || if_loc >= fg.nx || jf_loc < 0 || jf_loc >= fg.ny) continue;

            // 从细网格 f 计算 ρf 和 uf（直接求和，不依赖 rho/u 缓存）
            double rho_f    = 0.0;
            double u_f[2]   = {0.0, 0.0};
            {
                const int fi0 = fg.idx(if_loc, jf_loc);
                for (int a = 0; a < Q; ++a) {
                    const double fa = fg.f[fi0 * Q + a];
                    rho_f  += fa;
                    u_f[0] += fa * d2q9::C[a][0];
                    u_f[1] += fa * d2q9::C[a][1];
                }
                if (rho_f > 1e-15) { u_f[0] /= rho_f; u_f[1] /= rho_f; }
            }

            const int ci = cg.idx(ic_l, jc_l);

            // 对每个方向 a，计算空间滤波后的 f_neq（Eq. 8）并更新粗节点（Eq. 7）
            for (int a = 0; a < Q; ++a) {
                const double ca[2] = {
                    static_cast<double>(d2q9::C[a][0]),
                    static_cast<double>(d2q9::C[a][1])
                };

                // Eq. 8：f_neq_filtered_a = (1/Q) * Σ_j f_neq_a,f(fine_site + ej)
                double f_neq_sum = 0.0;
                for (int j = 0; j < Q; ++j) {
                    // 邻格细坐标（夹持到细网格范围）
                    const int if_nb = std::max(0, std::min(fg.nx - 1,
                                        if_loc + d2q9::C[j][0]));
                    const int jf_nb = std::max(0, std::min(fg.ny - 1,
                                        jf_loc + d2q9::C[j][1]));
                    const int fi_nb = fg.idx(if_nb, jf_nb);

                    const double rho_nb   = fg.rho[fi_nb];
                    const double u_nb[2]  = {fg.u[fi_nb * d], fg.u[fi_nb * d + 1]};
                    const double f_neq_nb = fg.f[fi_nb * Q + a]
                                          - f_eq(d2q9::W[a], rho_nb, ca, u_nb, d);
                    f_neq_sum += f_neq_nb;
                }
                const double f_neq_filtered = f_neq_sum / Q;

                // Eq. 7：fi,c = f_eq_a(ρf, uf) + (2ωf/ωc) * f_neq_filtered_a
                cg.f[ci * Q + a] = f_eq(d2q9::W[a], rho_f, ca, u_f, d)
                                  + scale * f_neq_filtered;
            }

            // 从更新后的 f 同步粗网格宏观量 ρ 和 u
            cg.rho[ci]       = 0.0;
            cg.u[ci * d]     = 0.0;
            cg.u[ci * d + 1] = 0.0;
            for (int a = 0; a < Q; ++a) {
                const double fa = cg.f[ci * Q + a];
                cg.rho[ci]       += fa;
                cg.u[ci * d]     += fa * d2q9::C[a][0];
                cg.u[ci * d + 1] += fa * d2q9::C[a][1];
            }
            if (cg.rho[ci] > 1e-15) {
                cg.u[ci * d]     /= cg.rho[ci];
                cg.u[ci * d + 1] /= cg.rho[ci];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 累积时间步细化倍数
// ---------------------------------------------------------------------------
int mg_total_subcycle_steps(const MgNode& node)
{
    int total = 1;
    const MgNode* cur = &node;
    while (cur->parent != nullptr) {
        total *= cur->refine_ratio;
        cur = cur->parent;
    }
    return total;
}

// ---------------------------------------------------------------------------
// AMR 细化指标：密度梯度范数
// ---------------------------------------------------------------------------
void mg_compute_refinement_indicator(
    const MgNode& node,
    std::vector<double>& indicator)
{
    if (!node.grid) {
        throw std::invalid_argument("mg_compute_refinement_indicator: node.grid must not be nullptr");
    }

    const LatticeGrid& g = *node.grid;
    const int nx = g.nx;
    const int ny = g.ny;
    const int n  = g.size();

    indicator.assign(n, 0.0);

    // 二阶中心差分 |∇ρ|（内部节点），单侧差分（边界节点）
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int c  = g.idx(i, j);

            // dρ/dx（中心差分 or 单侧）
            double drho_dx;
            if (i == 0) {
                const int r = g.idx(i + 1, j);
                drho_dx = g.rho[r] - g.rho[c];
            } else if (i == nx - 1) {
                const int l = g.idx(i - 1, j);
                drho_dx = g.rho[c] - g.rho[l];
            } else {
                const int l = g.idx(i - 1, j);
                const int r = g.idx(i + 1, j);
                drho_dx = 0.5 * (g.rho[r] - g.rho[l]);
            }

            // dρ/dy
            double drho_dy;
            if (j == 0) {
                const int u = g.idx(i, j + 1);
                drho_dy = g.rho[u] - g.rho[c];
            } else if (j == ny - 1) {
                const int d = g.idx(i, j - 1);
                drho_dy = g.rho[c] - g.rho[d];
            } else {
                const int d = g.idx(i, j - 1);
                const int u = g.idx(i, j + 1);
                drho_dy = 0.5 * (g.rho[u] - g.rho[d]);
            }

            indicator[c] = std::sqrt(drho_dx * drho_dx + drho_dy * drho_dy);
        }
    }
}

// ---------------------------------------------------------------------------
// 递归多重网格时间步推进（Lagrava 2012 五步算法）
// ---------------------------------------------------------------------------
//
// 算法（对加密比 r 的内部节点）：
//   1. 保存当前 f/ρ/u 到暂存缓冲区（时间插值参考 t 时刻）
//   2. node.solver->step()           推进 t → t+δtc（含 MPI 幽灵层交换）
//   3. 创建 prev_node：借用暂存缓冲区表示 t 时刻状态（O(1) swap，无深拷贝）
//   4. 对每个子节点 child（r = child.refine_ratio）：
//        for k = 1..r:
//          if k > 1: mg_apply_fringe_bc_temporal(prev_node, node, child, (k-1)/r)
//          mg_step_recursive(child)     （递归处理子树）
//        mg_couple_fine_to_coarse(child, node)   （F→C 更新粗 fringe 区域）
//   5. 归还暂存缓冲区（swap 回）
//
// MPI 并行透明性：
//   Solver::step() 内部已处理 halo_exchange，mg_step_recursive 不需要额外 MPI 调用。
//   每层独立地绑定 MpiDecomp，通过 solver.attach_mpi2d()/attach_mpi1d() 设置。
// ---------------------------------------------------------------------------
void mg_step_recursive(MgNode& node, int fringe_width)
{
    if (!node.solver) {
        throw std::invalid_argument("mg_step_recursive: node.solver must not be nullptr");
    }
    if (!node.grid) {
        throw std::invalid_argument("mg_step_recursive: node.grid must not be nullptr");
    }

    // 叶节点：直接推进，无需耦合
    if (node.is_leaf()) {
        node.solver->step();
        return;
    }

    LatticeGrid& g = *node.grid;

    // 步骤 1：保存 t 时刻状态（懒分配暂存缓冲区）
    node.f_scratch  .assign(g.f  .begin(), g.f  .end());
    node.rho_scratch.assign(g.rho.begin(), g.rho.end());
    node.u_scratch  .assign(g.u  .begin(), g.u  .end());

    // 步骤 2：推进当前节点 t → t+δtc（MPI halo exchange 在 step() 内部完成）
    node.solver->step();

    // 步骤 3：构建 prev_node（代表 t 时刻状态），通过 swap 借用暂存缓冲区（O(1)，无拷贝）
    // 临时 LatticeGrid 仅用于在 mg_apply_fringe_bc_temporal 中提供 t 时刻 f/ρ/u 读取
    LatticeGrid prev_g;
    prev_g.nx    = g.nx;
    prev_g.ny    = g.ny;
    prev_g.nz    = g.nz;
    prev_g.q     = g.q;
    prev_g.model = g.model;
    std::swap(prev_g.f,   node.f_scratch);   // prev_g 持有 t 时刻 f
    std::swap(prev_g.rho, node.rho_scratch);
    std::swap(prev_g.u,   node.u_scratch);

    MgNode prev_node;
    prev_node.extent       = node.extent;
    prev_node.level        = node.level;
    prev_node.refine_ratio = node.refine_ratio;
    prev_node.dim          = node.dim;
    prev_node.grid         = &prev_g;

    const double omega_c = node.solver->omega();

    // 步骤 4：对每个子节点执行 refine_ratio 次子循环 + F→C 更新
    for (MgNode* child : node.children) {
        if (!child->solver || !child->grid) {
            // 安全跳过未完全初始化的子节点（用户仅构建结构不运行的情况）
            continue;
        }

        const int r = child->refine_ratio;

        // k 次细步子循环
        for (int k = 1; k <= r; ++k) {
            // 在第 2..r 次细步之前，用时间+空间双重插值更新 fringe BC
            // t_alpha = (k-1)/r：在 t 时刻（k=1 之前=0）和 t+δtc（k=r 之后=1）之间插值
            if (k > 1) {
                const double t_alpha = static_cast<double>(k - 1) / static_cast<double>(r);
                mg_apply_fringe_bc_temporal(prev_node, node, *child,
                                            t_alpha, fringe_width, omega_c);
            }
            // 递归推进子节点（若子节点也有子节点，会再次递归）
            mg_step_recursive(*child, fringe_width);
        }

        // 步骤 5：F→C 耦合——用细网格 f 更新粗网格 fringe 区域（论文 Eqs. 7-8）
        mg_couple_fine_to_coarse(*child, node, fringe_width, omega_c);
    }

    // 步骤 5（收尾）：归还暂存缓冲区（swap 回，保留已分配内存以供下次使用）
    std::swap(prev_g.f,   node.f_scratch);
    std::swap(prev_g.rho, node.rho_scratch);
    std::swap(prev_g.u,   node.u_scratch);
}

} // namespace lbm
