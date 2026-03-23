#pragma once
/// @file plugins/boundary_plugin.hpp
/// @brief 自定义边界条件的扩展接口。
///
/// ## 如何添加新的边界条件
///
/// 1. 创建 `IBoundaryPlugin` 的子类。
/// 2. 重写 `apply()` 以实现新的边界逻辑。该方法在所有标准边界条件
///    （反弹、Zou-He）应用**之后**调用，因此可以访问已更新的分布函数
///    和宏观场量。
/// 3. 在仿真循环前注册插件：
///    ```cpp
///    MyBC plugin;
///    PluginRegistry::instance().set_boundary_plugin(&plugin);
///    ```
/// 4. 供 Rust / C 用户使用的 C ABI 封装见
///    `core/src/capi/lbm_capi.cpp`（`lbm_set_plugins`）。
///
/// ### 骨架示例
/// ```cpp
/// #include "plugins/boundary_plugin.hpp"
/// class OpenPressureBC : public lbm::IBoundaryPlugin {
/// public:
///     void apply(lbm::LatticeGrid& grid, int /*step*/) override {
///         // 例：东面对流出口
///         for (int j = 0; j < grid.ny; ++j) { ... }
///     }
///     const char* name() const override { return "open_pressure"; }
/// };
/// ```

#include "lbm/lattice.hpp"

namespace lbm {

/// 自定义边界条件插件的抽象接口。
///
/// 实现此接口可在不修改任何已有求解器代码的情况下添加新的边界条件。
/// 该插件在每个时间步的内置边界条件应用**之后**被调用一次。
class IBoundaryPlugin {
public:
    virtual ~IBoundaryPlugin() = default;

    /// 在给定 *step* 时刻将自定义边界条件应用到 *grid*。
    ///
    /// @param grid  欧拉格子网格（分布函数、宏观场量和体力数组均可直接访问和修改）。
    /// @param step  当前仿真步骤索引（从 0 开始）。
    virtual void apply(LatticeGrid& grid, int step) = 0;

    /// 在日志信息中返回的短标识符（例如 `"convective_outlet"`）。
    virtual const char* name() const = 0;
};

} // namespace lbm
