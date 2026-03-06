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
// 物理边界范围描述符（MPI 模式下本地网格含幽灵层，须显式指定物理行/列范围）
//
// 非 MPI 模式（默认）：j_s=0, j_n=ny-1, i_w=0, i_e=nx-1（整个网格均为物理域）
// 1D Y-切片 MPI 模式 ：j_s=1, j_n=local_ny, i_w=0, i_e=nx-1
// 2D XY 块分解 MPI  ：j_s=phys_y0, j_n=phys_y0+local_ny-1,
//                     i_w=phys_x0, i_e=phys_x0+local_nx-1
//
// apply_boundary_conditions() 使用此结构确保 BC 只施加到物理边界节点，
// 防止在幽灵行/列上错误地覆盖 MPI 幽灵层交换数据。
//
// has_*_wall 标志（默认 true）：
//   MPI 模式下内部分区（非全局壁面所在分区）必须将对应标志设为 false，
//   防止 BC 错误施加到非壁面的内部物理行/列上，导致分块边界处速度出现阶跃。
//   例：4 进程 1D Y 分解时，只有 rank 0 的 has_south_wall=true，
//       rank 1/2/3 必须设 has_south_wall=false。
// ---------------------------------------------------------------------------
struct PhysicalBounds {
    int j_s = 0;   ///< 物理南边界行索引（含，本地坐标）
    int j_n = 0;   ///< 物理北边界行索引（含，本地坐标）
    int i_w = 0;   ///< 物理西边界列索引（含，本地坐标）
    int i_e = 0;   ///< 物理东边界列索引（含，本地坐标）

    /// 本进程是否实际拥有全局南物理壁（仅 has_south_wall=true 时才施加 South 面 BC）
    bool has_south_wall = true;
    /// 本进程是否实际拥有全局北物理壁
    bool has_north_wall = true;
    /// 本进程是否实际拥有全局西物理壁
    bool has_west_wall  = true;
    /// 本进程是否实际拥有全局东物理壁
    bool has_east_wall  = true;
};

// ---------------------------------------------------------------------------
// 将所有已注册的边界条件应用到网格
//
// pb  — 物理边界范围。非 MPI 模式可忽略（使用默认值），MPI 模式下须由
//       Solver::step() 根据当前分解信息计算后传入，以跳过幽灵行/列。
// ---------------------------------------------------------------------------
void apply_boundary_conditions(LatticeGrid& grid,
                                const std::vector<BoundaryCondition>& bcs,
                                PhysicalBounds pb);

} // namespace lbm
