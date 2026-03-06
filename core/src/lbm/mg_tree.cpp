// core/src/lbm/mg_tree.cpp — 多重网格嵌套关系树（MgTree / MgNode）实现
//
// 本文件实现 mg_tree.hpp 中声明的 MgTree 和辅助函数。
// 详细文档请参阅头文件 include/lbm/mg_tree.hpp 和 docs/MPI并行详解.md。

#include "lbm/mg_tree.hpp"

#include <queue>
#include <algorithm>
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
    if (refine_ratio < 1) {
        throw std::invalid_argument("MgTree::add_level: refine_ratio must be >= 1");
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

} // namespace lbm
