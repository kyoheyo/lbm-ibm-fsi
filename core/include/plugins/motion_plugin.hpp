#pragma once
/// @file plugins/motion_plugin.hpp
/// @brief 运动网格 / 刚体运动或规定运动的扩展接口。
///
/// ## 如何添加流体运动网格或固体结构运动
///
/// 该插件钩子在**每步碰撞前调用一次**，使更新后的拉格朗日标记点
/// 位置和/或欧拉网格速度能够在流体求解前应用。典型使用场景：
///
/// - 浸入边界的规定刚体运动（旋转、平移）
/// - 运动壁面边界条件（例如旋转圆柱）
/// - 任意拉格朗日-欧拉（ALE）网格运动
/// - 与流体动载荷耦合的 6 自由度刚体动力学
///
/// ### 步骤
/// 1. 继承 `IMotionPlugin`。
/// 2. 重写 `update()` 以将位置和速度推进 *dt*。
/// 3. 注册：
///    ```cpp
///    MySineMotion plugin;
///    PluginRegistry::instance().set_motion_plugin(&plugin);
///    ```
///
/// ### 骨架示例
/// ```cpp
/// #include "plugins/motion_plugin.hpp"
/// #include "ibm/marker.hpp"
/// class RotatingCylinder : public lbm::IMotionPlugin {
/// public:
///     void update(lbm::LatticeGrid& grid,
///                 ibm::MarkerSet* markers,    // 可能为 nullptr
///                 double dt, int step) override
///     {
///         if (!markers) return;
///         const double angle = 0.01 * step;   // 规定旋转角度
///         for (auto& m : markers->markers) {
///             double r = std::hypot(m.x - cx_, m.y - cy_);
///             double theta = std::atan2(m.y - cy_, m.x - cx_) + 0.01 * dt;
///             m.x = cx_ + r * std::cos(theta);
///             m.y = cy_ + r * std::sin(theta);
///         }
///     }
///     const char* name() const override { return "rotating_cylinder"; }
/// private:
///     double cx_ = 50.0, cy_ = 40.0;
/// };
/// ```

#include "lbm/lattice.hpp"
#include "ibm/marker.hpp"

namespace lbm {

/// 运动插件（运动网格 / 运动固体）的抽象接口。
///
/// 实现此接口可在不修改核心 LBM 或 IBM 代码的情况下，
/// 将规定运动或动态刚体运动与流体求解器耦合。
class IMotionPlugin {
public:
    virtual ~IMotionPlugin() = default;

    /// 将固体/标记点位置推进 *dt* 秒。
    ///
    /// @param grid     欧拉格子网格（壁面速度可按需写入 `force` 或 `u` 数组）。
    /// @param markers  拉格朗日 IBM 标记点集，若本次运行未启用 IBM 则为 `nullptr`。
    /// @param dt       物理时间步长。
    /// @param step     当前仿真步骤索引（从 0 开始）。
    virtual void update(LatticeGrid& grid,
                        ibm::MarkerSet* markers,
                        double dt,
                        int step) = 0;

    /// 短标识符（例如 `"prescribed_sine"`、`"rigid_body_6dof"`）。
    virtual const char* name() const = 0;
};

} // namespace lbm
