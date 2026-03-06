// core/src/lbm/mg_tree.cpp — 多重网格嵌套关系树（MgTree / MgNode）实现
//
// 本文件实现 mg_tree.hpp 中声明的 MgTree 和辅助函数。
// 详细文档请参阅头文件 include/lbm/mg_tree.hpp 和 docs/MPI并行详解.md。

#include "lbm/mg_tree.hpp"

#include <queue>
#include <algorithm>
#include <cmath>
#include <stdexcept>

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

} // namespace lbm
