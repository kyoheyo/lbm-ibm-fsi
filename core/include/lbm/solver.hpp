#pragma once
#include "lattice.hpp"
#include "boundary.hpp"
#include "mpi_decomp.hpp"
#include "solid.hpp"
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

    /// 设置固体节点边界条件类型。
    /// 若设置为 BounceBack 或 InterpolatedBounceBack，则每次 step() 的 stream() 之后
    /// 自动对 grid_.solid 中标记的固体节点施加相应反弹 BC。
    /// 默认 SolidBCType::None（不施加，全流体模式）。
    void set_solid_bc_type(SolidBCType t) { solid_bc_type_ = t; }

    /// 返回当前固体 BC 类型。
    [[nodiscard]] SolidBCType solid_bc_type() const { return solid_bc_type_; }

    /// 绑定 MPI 域分解描述符，之后每次 step() 的 stream() 末尾自动执行幽灵行交换。
    /// 传入 nullptr 可解除绑定。
    /// 仅在 LBM_ENABLE_MPI 编译宏定义时有实际效果；否则为空操作。
    void attach_mpi(const MpiDecomp* decomp) { mpi_decomp_ = decomp; mpi_decomp2d_ = nullptr; mpi_decomp3d_ = nullptr; }

    /// 绑定二维 MPI 块分解描述符（XY 方向），之后每次 step() 自动执行 2D 幽灵层交换。
    /// 传入 nullptr 可解除绑定。
    /// 仅在 LBM_ENABLE_MPI 编译宏定义时有实际效果；否则为空操作。
    void attach_mpi2d(const MpiDecomp2D* decomp) { mpi_decomp2d_ = decomp; mpi_decomp_ = nullptr; mpi_decomp3d_ = nullptr; }

    /// 绑定三维 MPI 块分解描述符（XYZ 方向），之后每次 step() 自动执行 3D 幽灵层交换。
    /// 传入 nullptr 可解除绑定。
    /// 仅在 LBM_ENABLE_MPI 编译宏定义时有实际效果；否则为空操作。
    void attach_mpi3d(const MpiDecomp3D* decomp) { mpi_decomp3d_ = decomp; mpi_decomp_ = nullptr; mpi_decomp2d_ = nullptr; }

private:
    LatticeGrid&   grid_;
    double         omega_;       ///< 松弛频率  ω = 1/τ
    CollisionModel cm_;
    std::vector<BoundaryCondition> bcs_;  ///< 每步自动施加的边界条件列表
    const MpiDecomp*   mpi_decomp_    = nullptr;
    const MpiDecomp2D* mpi_decomp2d_  = nullptr;
    const MpiDecomp3D* mpi_decomp3d_  = nullptr;
    SolidBCType        solid_bc_type_ = SolidBCType::None;

    void collide_bgk();
    void collide_mrt();

    // Guo 体力格式 — 在碰撞过程中添加体力修正项
    // node   : 节点索引（用于读取 grid_.u）
    // w_a    : 方向 a 的格子权重
    // c_a    : 方向 a 的格子速度向量（维度分量，double 类型）
    // F      : 该节点的体力密度（维度分量）
    // f_a_ptr: 指向碰后分布函数 f_a 的指针（原地修正）
    void apply_guo_forcing(int node, double w_a, const double* c_a,
                           const double* F, double* f_a_ptr);

    /// MPI 幽灵层跳过控制（供 collide_bgk/collide_mrt 共用）
    struct CollideGuard {
        int  n_start    = 0;
        int  n_end      = 0;
        bool use_mpi2d  = false;
        int  gnx2d      = 0;
        int  gny2d      = 0;
        int  n_ghost_2d = 1;    ///< 每侧幽灵层数（2D 模式）
        bool sg2d = false, ng2d = false, wg2d = false, eg2d = false;

        // 三维 MPI 扩展
        bool use_mpi3d = false;
        int  gnx3d     = 0;
        int  gny3d     = 0;
        int  gnz3d     = 0;
        bool sg3d = false, ng3d = false;   ///< 南/北幽灵（Y 方向）
        bool wg3d = false, eg3d = false;   ///< 西/东幽灵（X 方向）
        bool bg3d = false, tg3d = false;   ///< 底/顶幽灵（Z 方向）

        /// 判断节点索引 i 是否属于幽灵层（二维或三维模式时使用）
        bool is_ghost(int i) const {
            if (use_mpi3d) {
                const int slice = gnx3d * gny3d;
                const int iz = i / slice;
                const int iy = (i % slice) / gnx3d;
                const int ix = i % gnx3d;
                if ((bg3d && iz == 0) || (tg3d && iz == gnz3d - 1)) return true;
                if ((sg3d && iy == 0) || (ng3d && iy == gny3d - 1)) return true;
                if ((wg3d && ix == 0) || (eg3d && ix == gnx3d - 1)) return true;
                return false;
            }
            if (!use_mpi2d) return false;
            const int ix = i % gnx2d;
            const int iy = i / gnx2d;
            if (sg2d && iy < n_ghost_2d) return true;
            if (ng2d && iy >= gny2d - n_ghost_2d) return true;
            if (wg2d && ix < n_ghost_2d) return true;
            if (eg2d && ix >= gnx2d - n_ghost_2d) return true;
            return false;
        }
    };

    /// 计算碰撞时的幽灵层跳过参数（同时处理 1D 和 2D MPI 模式）
    CollideGuard make_collide_guard() const;
};

} // namespace lbm
