#pragma once
#include "lattice.hpp"

namespace lbm {

// ---------------------------------------------------------------------------
// 边界条件类型
// ---------------------------------------------------------------------------
enum class BCType {
    // ---- 反弹类 ----
    BounceBack,        ///< 半步长反弹（halfway bounce-back）：壁面位于节点间半格处，2阶精度
    BounceBackFullWay, ///< 全步长反弹（full-way / on-node bounce-back）：壁面位于节点处，1阶精度

    // ---- Zou-He 非平衡反弹类 ----
    ZouHe_Velocity,    ///< Zou-He 速度边界条件（进/出口规定速度）
    ZouHe_Pressure,    ///< Zou-He 压力边界条件（进/出口规定密度/压力）

    // ---- 出口类 ----
    FullyDeveloped,    ///< 充分发展出口（零法向梯度，从上游一层拷贝 f）
    FreeOutlet,        ///< 自由出口（与 FullyDeveloped 等价，别名保留）

    // ---- 非平衡外推类 ----
    Guo_Extrapolation, ///< 郭照立非平衡外推格式（Guo et al., Chinese Physics 2002）

    // ---- 周期类 ----
    Periodic,          ///< 周期边界（在流式迁移中隐式处理，此枚举仅作记录）
};

// ---------------------------------------------------------------------------
// 面（Face）枚举 — 矩形计算域的六个面
// ---------------------------------------------------------------------------
enum class Face { West, East, South, North, Bottom, Top };

// ---------------------------------------------------------------------------
// 边界条件描述符
// ---------------------------------------------------------------------------
struct BoundaryCondition {
    BCType type;
    Face   face;

    /// 指定速度（用于 ZouHe_Velocity；Guo_Extrapolation 在速度模式下也使用）
    double ux = 0.0;
    double uy = 0.0;
    double uz = 0.0;

    /// 指定密度（用于 ZouHe_Pressure 和 Guo_Extrapolation 压力模式）
    ///
    /// 对 Guo_Extrapolation 的特殊约定：
    ///   - rho > 0：压力模式 — 以 rho 为壁面密度，壁面速度从相邻内部节点外推；
    ///   - rho = 0（明确设置）：速度模式 — 以 ux/uy 为壁面速度（含 ux=uy=0 的无滑移），
    ///                          壁面密度从相邻内部节点外推。
    ///
    /// 注意：默认值 1.0 适合 ZouHe_Pressure；使用 Guo_Extrapolation 速度模式时
    ///       需显式设置 rho = 0.0。
    double rho = 1.0;
};

// ---------------------------------------------------------------------------
// 将所有已注册的边界条件应用到网格
// ---------------------------------------------------------------------------
void apply_boundary_conditions(LatticeGrid& grid,
                                const std::vector<BoundaryCondition>& bcs);

} // namespace lbm
