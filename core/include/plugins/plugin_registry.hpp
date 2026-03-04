#pragma once
/// @file plugins/plugin_registry.hpp
/// @brief 汇聚所有四类插件钩子的中央注册表。
///
/// `PluginRegistry` 是一个普通结构体，持有指向四种插件接口的可空指针。
/// 全局单例可通过 `PluginRegistry::instance()` 访问。
///
/// ## 生命周期
///
/// ```
/// PluginRegistry& reg = PluginRegistry::instance();
///
/// // 在仿真循环前注册插件（只需一次）
/// reg.set_boundary_plugin(&my_bc);
/// reg.set_mesh_plugin(&my_mesh);
/// reg.set_motion_plugin(&my_motion);
/// reg.set_flexible_plugin(&my_flex);
///
/// // C ABI 层（lbm_capi.cpp）会在 lbm_solver_step() 中自动调用 apply_* 方法。
/// ```
///
/// 当插件指针为 `nullptr`（默认值）时，对应的 `apply_*` 调用是廉价的空操作。
/// 在运行时删除插件只需 `reg.set_boundary_plugin(nullptr)`。
///
/// ## 线程安全
///
/// 插件的**注册**不是线程安全的——插件必须在仿真循环开始前由单一线程注册。
/// `apply_*` 派发方法是只读的，可以并发调用。

#include "boundary_plugin.hpp"
#include "mesh_plugin.hpp"
#include "motion_plugin.hpp"
#include "flexible_plugin.hpp"

namespace lbm {

/// 汇聚求解器使用的所有可选插件钩子。
class PluginRegistry {
public:
    // ------------------------------------------------------------------
    // 单例访问
    // ------------------------------------------------------------------

    /// 返回进程级单例实例。
    static PluginRegistry& instance() {
        static PluginRegistry reg;
        return reg;
    }

    // ------------------------------------------------------------------
    // 注册（存储非拥有指针；调用方管理生命周期）
    // ------------------------------------------------------------------

    void set_boundary_plugin (IBoundaryPlugin*      p) { boundary_  = p; }
    void set_mesh_plugin     (IMeshPlugin*          p) { mesh_       = p; }
    void set_motion_plugin   (IMotionPlugin*        p) { motion_     = p; }
    void set_flexible_plugin (IFlexibleSolverPlugin* p){ flexible_   = p; }

    // ------------------------------------------------------------------
    // 查询
    // ------------------------------------------------------------------

    [[nodiscard]] IBoundaryPlugin*       boundary_plugin()  const { return boundary_; }
    [[nodiscard]] IMeshPlugin*           mesh_plugin()      const { return mesh_;     }
    [[nodiscard]] IMotionPlugin*         motion_plugin()    const { return motion_;   }
    [[nodiscard]] IFlexibleSolverPlugin* flexible_plugin()  const { return flexible_; }

    /// 若至少一个插件处于激活状态则返回 true。
    [[nodiscard]] bool any_active() const {
        return boundary_ || mesh_ || motion_ || flexible_;
    }

    // ------------------------------------------------------------------
    // 派发辅助方法 — 由 lbm_capi.cpp 在 lbm_solver_step() 前后调用
    // ------------------------------------------------------------------

    /// 调用已注册的边界插件（若为 nullptr 则为空操作）。
    void apply_boundary(LatticeGrid& grid, int step) const {
        if (boundary_) boundary_->apply(grid, step);
    }

    /// 调用已注册的网格自适应插件（若为 nullptr 则为空操作）。
    void adapt_mesh(LatticeGrid& grid, int step) const {
        if (mesh_) mesh_->adapt(grid, step);
    }

    /// 调用已注册的运动插件（若为 nullptr 则为空操作）。
    void update_motion(LatticeGrid& grid,
                       ibm::MarkerSet* markers,
                       double dt,
                       int step) const {
        if (motion_) motion_->update(grid, markers, dt, step);
    }

    /// 调用已注册的柔性体插件（若为 nullptr 或 *markers* 为 nullptr 则为空操作）。
    ///
    /// 柔性体插件需要有效的 `MarkerSet`，因为它们以 IBM 拉格朗日标记点
    /// 作为自由度来推进结构。当 IBM 在本次运行中未激活（`markers == nullptr`）
    /// 时，没有结构需要推进，调用将被静默跳过。
    void step_flexible(ibm::MarkerSet* markers, double dt, int step) const {
        if (flexible_ && markers) flexible_->step(*markers, dt, step);
    }

private:
    IBoundaryPlugin*       boundary_ = nullptr;
    IMeshPlugin*           mesh_     = nullptr;
    IMotionPlugin*         motion_   = nullptr;
    IFlexibleSolverPlugin* flexible_ = nullptr;

    // 禁止外部构造 / 拷贝 — 使用 instance()
    PluginRegistry()  = default;
    ~PluginRegistry() = default;
    PluginRegistry(const PluginRegistry&)            = delete;
    PluginRegistry& operator=(const PluginRegistry&) = delete;
};

} // namespace lbm
