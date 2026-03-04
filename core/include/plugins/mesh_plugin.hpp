#pragma once
/// @file plugins/mesh_plugin.hpp
/// @brief 自定义网格处理（例如自适应细化）的扩展接口。
///
/// ## 如何添加新的网格处理方法
///
/// LBM 核心使用均匀笛卡尔欧拉网格，因此此处的"网格自适应"
/// 指在运行时**修改网格拓扑、节点布局或网格间距信息**的算法。
/// 典型使用场景：
///
/// - 拉伸 / 非均匀格子间距
/// - 动态块细化（在边界层附近局部细化网格格子）
/// - 覆盖网格 / Chimera 网格耦合
///
/// ### 步骤
/// 1. 继承 `IMeshPlugin`。
/// 2. 重写 `initialize()` 以在仿真循环前建立数据结构。
/// 3. 重写 `adapt()` 以在每步（或每 N 步）演化网格。
/// 4. 注册：
///    ```cpp
///    MyMeshPlugin plugin;
///    PluginRegistry::instance().set_mesh_plugin(&plugin);
///    ```
///
/// ### 骨架示例
/// ```cpp
/// #include "plugins/mesh_plugin.hpp"
/// class StretchedZMesh : public lbm::IMeshPlugin {
/// public:
///     void initialize(lbm::LatticeGrid& grid, const void* /*cfg*/) override {
///         // 预计算作为用户数据存储的拉伸坐标
///     }
///     void adapt(lbm::LatticeGrid& grid, int step) override {
///         // 每 1000 步后可选地重新平衡
///         if (step % 1000 == 0) rebalance(grid);
///     }
///     const char* name() const override { return "stretched_z"; }
/// };
/// ```

#include "lbm/lattice.hpp"

namespace lbm {

/// 自定义网格细化 / 网格处理插件的抽象接口。
///
/// 实现此接口可在不修改核心求解器的情况下添加新的网格划分策略
/// 或自适应细化方案。
class IMeshPlugin {
public:
    virtual ~IMeshPlugin() = default;

    /// 时间积分循环**开始前**调用的一次性初始化。
    ///
    /// @param grid  格子网格（可在此处调整大小或重构）。
    /// @param cfg   指向插件特定配置数据的可选指针。
    ///              在实现内部转换为具体类型。
    virtual void initialize(LatticeGrid& grid, const void* cfg = nullptr) = 0;

    /// 每步网格自适应钩子，在 `Solver::step()` 中流式迁移和
    /// 宏观量更新**之后**调用。
    ///
    /// @param grid  格子网格。
    /// @param step  当前仿真步骤索引（从 0 开始）。
    virtual void adapt(LatticeGrid& grid, int step) = 0;

    /// 短标识符（例如 `"adaptive_uniform"`、`"stretched_z"`）。
    virtual const char* name() const = 0;
};

} // namespace lbm
