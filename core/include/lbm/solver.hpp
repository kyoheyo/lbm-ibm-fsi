#pragma once
#include "lattice.hpp"

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

    /// 流式迁移步骤：将 f 传播到 f_tmp，然后交换
    void stream();

    double omega() const { return omega_; }
    void   set_omega(double w) { omega_ = w; }

private:
    LatticeGrid&   grid_;
    double         omega_;       ///< 松弛频率  ω = 1/τ
    CollisionModel cm_;

    void collide_bgk();
    void collide_mrt();

    // Guo 体力格式 — 在碰撞过程中添加体力修正项
    void apply_guo_forcing(int node, const double* F, double* f_post);
};

} // namespace lbm
