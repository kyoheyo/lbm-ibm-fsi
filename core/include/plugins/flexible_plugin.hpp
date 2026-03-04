#pragma once
/// @file plugins/flexible_plugin.hpp
/// @brief 替代 / 附加柔性体求解器的扩展接口。
///
/// ## 如何添加新的柔性体求解器
///
/// 内置结构求解器是线性 Euler-Bernoulli 梁（`fsi::BeamSolver`）。
/// 该插件允许在不修改已有耦合代码的情况下，将完全不同的结构模型
/// 插入到 FSI 耦合循环中。典型使用场景：
///
/// - 用于薄壳结构的 **Kirchhoff-Love 板**求解器
/// - 用于大挠度问题的**非线性协转梁**
/// - 用于三维柔性面的**有限体积薄膜**
/// - 快速近似响应的**降阶模型**（POD/ROM）
/// - 通过共享内存或 MPI 耦合的**外部有限元代码**
///   （例如 OpenFOAM solid、Calculix）
///
/// ### 步骤
/// 1. 继承 `IFlexibleSolverPlugin`。
/// 2. 重写 `step()` 以将结构自由度推进 *dt*。
///    该方法接收当前 IBM 标记点位置（速度插值的输出），
///    并需要更新标记点位置和力以供下一步展布使用。
/// 3. 注册：
///    ```cpp
///    MyPlatePlugin plugin;
///    PluginRegistry::instance().set_flexible_plugin(&plugin);
///    ```
///
/// ### FSI 循环中的数据流
/// ```
/// interpolate_velocity(fluid, markers) →  IMotionPlugin::update()
///                                       →  IFlexibleSolverPlugin::step()
///                                       →  spread_force(fluid, markers)
/// ```
///
/// ### 骨架示例
/// ```cpp
/// #include "plugins/flexible_plugin.hpp"
/// class KirchhoffPlate : public lbm::IFlexibleSolverPlugin {
/// public:
///     void step(ibm::MarkerSet& markers, double dt, int step) override {
///         // 1. 从 markers.fx / .fy / .fz 提取节点力
///         // 2. 求解板方程 M*a + K*u = F
///         // 3. 根据新挠度更新标记点位置（m.x, m.y）
///     }
///     const char* name() const override { return "kirchhoff_love_plate"; }
/// };
/// ```

#include "ibm/marker.hpp"

namespace lbm {

/// 柔性体求解器插件的抽象接口。
///
/// 实现此接口可用任意结构模型替代或补充内置的
/// Euler-Bernoulli 梁求解器。
class IFlexibleSolverPlugin {
public:
    virtual ~IFlexibleSolverPlugin() = default;

    /// 将柔性体推进一个时间步 *dt*。
    ///
    /// 进入时：`markers` 包含插值得到的流体速度
    ///         （`m.ux`、`m.uy`、`m.uz`）以及耦合层设置的 IBM 力
    ///         （`m.fx`、`m.fy`、`m.fz`）。
    ///
    /// 退出时：必须更新 `markers` 的位置（`m.x`、`m.y`、`m.z`）
    ///         和力（`m.fx`、`m.fy`、`m.fz`），
    ///         以便 `spread_force()` 能正确将其分配回欧拉网格。
    ///
    /// @param markers  拉格朗日 IBM 标记点集。
    /// @param dt       物理时间步长。
    /// @param step     当前仿真步骤索引（从 0 开始）。
    virtual void step(ibm::MarkerSet& markers, double dt, int step) = 0;

    /// 短标识符（例如 `"kirchhoff_love_plate"`、`"corotational_beam"`）。
    virtual const char* name() const = 0;
};

} // namespace lbm
