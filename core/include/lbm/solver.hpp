#pragma once
#include "lattice.hpp"
#include "boundary.hpp"
#include "mpi_decomp.hpp"
#include <vector>

namespace lbm {

// ---------------------------------------------------------------------------
// 碰撞算子选择
// ---------------------------------------------------------------------------
enum class CollisionModel {
    BGK,  ///< 单松弛时间碰撞算子（Bhatnagar-Gross-Krook）
    MRT,  ///< 多松弛时间碰撞算子（低粘度下更稳定）
};

// ---------------------------------------------------------------------------
// LBM 求解器 — 持有 LatticeGrid 引用，逐步推进仿真
// ---------------------------------------------------------------------------
class Solver {
public:
    Solver(LatticeGrid& grid, double omega, CollisionModel cm = CollisionModel::BGK);

    /// 单时间步：碰撞 + 流式迁移 + 边界条件应用
    void step();

    /// 仅执行碰撞步骤（原地修改 f）
    void collide();

    /// 流式迁移步骤：将 f 传播到 f_tmp，然后交换，并更新宏观量
    void stream();

    double omega() const { return omega_; }
    void   set_omega(double w) { omega_ = w; }

    /// 注册一个边界条件，在每步流式迁移后自动施加。
    /// 可多次调用以注册多个边界条件（按注册顺序依次施加）。
    void add_boundary_condition(const BoundaryCondition& bc);

    /// 返回当前已注册的边界条件列表（只读）。
    [[nodiscard]] const std::vector<BoundaryCondition>& boundary_conditions() const {
        return bcs_;
    }

    /// 绑定 MPI 域分解描述符，之后每次 step() 的 stream() 末尾自动执行幽灵行交换。
    /// 传入 nullptr 可解除绑定。
    /// 仅在 LBM_ENABLE_MPI 编译宏定义时有实际效果；否则为空操作。
    void attach_mpi(const MpiDecomp* decomp) { mpi_decomp_ = decomp; mpi_decomp2d_ = nullptr; }

    /// 绑定二维 MPI 块分解描述符（XY 方向），之后每次 step() 自动执行 2D 幽灵层交换。
    /// 传入 nullptr 可解除绑定。
    /// 仅在 LBM_ENABLE_MPI 编译宏定义时有实际效果；否则为空操作。
    void attach_mpi2d(const MpiDecomp2D* decomp) { mpi_decomp2d_ = decomp; mpi_decomp_ = nullptr; }

private:
    LatticeGrid&   grid_;
    double         omega_;       ///< 松弛频率  ω = 1/τ
    CollisionModel cm_;
    std::vector<BoundaryCondition> bcs_;  ///< 每步自动施加的边界条件列表
    const MpiDecomp* mpi_decomp_ = nullptr; ///< 可选 MPI 一维域分解（nullptr = 单进程模式）
    const MpiDecomp2D* mpi_decomp2d_ = nullptr; ///< 可选 MPI 二维块分解（nullptr = 未使用）

    void collide_bgk();
    void collide_mrt();

    // Guo 体力格式 — 在碰撞过程中添加体力修正项
    void apply_guo_forcing(int node, const double* F, double* f_post);
};

} // namespace lbm
