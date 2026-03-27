#pragma once
// lbm/mg_tree.hpp — 多重网格嵌套关系树数据结构
//
// ============================================================
// 概述
// ============================================================
// 本头文件定义了 AMR（Adaptive Mesh Refinement）风格的多重网格层次结构，
// 支持细网格嵌套在疏网格上，可实现任意嵌套深度和顺序。
//
// 数据结构：N 叉树（每个节点可有多个细网格子节点）
//   - 根节点（root）: 最粗网格（level=0）
//   - 子节点（children）: 细网格，嵌套在父节点空间范围内
//   - 叶节点（无子节点）: 最细网格，实际运行 LBM 仿真
//
// ============================================================
// 三维预留
// ============================================================
// MgExtent 包含 x/y/z 三个方向的范围：
//   - 2D 仿真（D2Q9）：z_start=z_end=0（默认，nz()=1）
//   - 3D 仿真（D3Q19/D3Q27）：z_start/z_end 表示 Z 方向物理范围
//
// 枚举值 MgDim::D2 / MgDim::D3 可在树级别标记是否为三维，
// 影响加密比（D2 下每格分为 refine_ratio^2 个细格，D3 为 refine_ratio^3）。
//
// ============================================================
// 与 MpiDecomp2D（或 MpiDecomp3D）的关系
// ============================================================
// 每个 MgNode 持有可选的 MPI 域分解指针（void* decomp），
// 支持并行多重网格：每一层的 LBM 网格可独立并行分解。
// 若 decomp==nullptr，该层以单进程模式运行。
//
// ============================================================
// 典型用法（双层 2D 嵌套示例）
// ============================================================
//   // 创建粗网格（256×256）
//   MgTree tree({0,255, 0,255, 0,0});
//
//   // 在粗网格中嵌套一个细网格（中心 64×64 区域，加密比 2）
//   auto* fine = tree.add_level(tree.root(), {96,159, 96,159, 0,0}, 2);
//
//   // 将 LatticeGrid 实例绑定到节点
//   tree.root()->grid = &coarse_grid;
//   fine->grid = &fine_grid;
//
//   // 从粗到细遍历（LBM 步骤）
//   tree.traverse_coarse_to_fine([](MgNode* node) {
//       if (node->has_grid()) node->solver->step();
//   });
//
// ============================================================
// 参考文献
// ============================================================
// Yu D. et al. (2002) "A unified formulation for incompressible and
//   weakly compressible fluid dynamics", Phys. Fluids 14(6):2007-2022.
// Rohde M. et al. (2006) "A generic, mass conservative local grid refinement
//   technique for lattice Boltzmann schemes", Int. J. Numer. Meth. Fluids 51.

#include <vector>
#include <memory>
#include <functional>
#include <stdexcept>
#include <variant>

// 引入 LatticeGrid 及 MPI 域分解结构（MgNode 需要 MpiDecomp2D/3D 完整类型以使用 std::variant）
#include "lattice.hpp"
#include "mpi_decomp.hpp"
// Solver 供 MgNode::solver 字段使用（Solver 拥有 attach_mpi*() + step() 接口）
#include "solver.hpp"

namespace lbm {

// ---------------------------------------------------------------------------
/// 空间维度枚举（标识网格是 2D 还是 3D）
// ---------------------------------------------------------------------------
enum class MgDim { D2, D3 };

// ---------------------------------------------------------------------------
/// 网格空间范围描述符（支持 2D 和 3D）
///
/// 坐标系：全局格子坐标（0-based），与 LatticeGrid::idx(i,j,k) 一致。
///
/// 2D 仿真：z_start=z_end=0，nz()=1。
/// 3D 仿真：z_start/z_end 给出 Z 方向的格子范围。
// ---------------------------------------------------------------------------
struct MgExtent {
    int x_start = 0, x_end = 0;   ///< 全局 X 坐标范围（格子单位，含端点）
    int y_start = 0, y_end = 0;   ///< 全局 Y 坐标范围
    int z_start = 0, z_end = 0;   ///< 全局 Z 坐标范围（2D 时设 0，nz()=1）

    /// 该范围的 X 方向格子数
    int nx() const { return x_end - x_start + 1; }
    /// 该范围的 Y 方向格子数
    int ny() const { return y_end - y_start + 1; }
    /// 该范围的 Z 方向格子数（2D 时为 1）
    int nz() const { return (z_end > z_start) ? z_end - z_start + 1 : 1; }

    /// 是否为三维范围（z_end > z_start）
    bool is_3d() const { return z_end > z_start; }

    /// 总节点数（nx × ny × nz）
    long long total_nodes() const { return static_cast<long long>(nx()) * ny() * nz(); }

    /// 判断本范围是否完全包含 other（用于验证子节点合法性）
    bool contains(const MgExtent& other) const {
        return x_start <= other.x_start && x_end >= other.x_end
            && y_start <= other.y_start && y_end >= other.y_end
            && z_start <= other.z_start && z_end >= other.z_end;
    }

    /// 判断两范围是否有重叠（用于检测子节点间的重叠冲突）
    bool overlaps(const MgExtent& other) const {
        return x_start <= other.x_end && x_end >= other.x_start
            && y_start <= other.y_end && y_end >= other.y_start
            && z_start <= other.z_end && z_end >= other.z_start;
    }
};

// ---------------------------------------------------------------------------
/// 多重网格树节点
///
/// 每个节点代表层次多重网格中的一个"网格块"（grid patch），可以是：
///   - 整个全局网格（顶层根节点）
///   - 嵌套在父节点（粗网格）中的某个细化子区域
///
/// 节点形成 N 叉树（N-ary tree）：
///   - 根节点：level=0，最粗网格
///   - 每个节点可有任意多个子节点（细化网格块）
///   - 子节点的空间范围是父节点空间范围的子集
///
/// 与 MpiDecomp2D 的关系：
///   - `decomp` 字段可选绑定本层的 MPI 域分解描述符
///   - nullptr 表示本层以单进程运行
///   - 未来 3D 时可绑定 MpiDecomp3D*（类型安全需调用方保证）
// ---------------------------------------------------------------------------
struct MgNode {
    MgExtent extent;                   ///< 本节点在全局坐标中的空间范围
    int      level        = 0;         ///< 网格层级（0=最粗，1,2,...=逐级加密）
    int      refine_ratio = 1;         ///< 相对于父节点的线性加密比（通常 2 或 4；根节点为 1）
    MgDim    dim          = MgDim::D2; ///< 网格维度（D2 或 D3，影响加密体积比）

    LatticeGrid* grid  = nullptr;  ///< 指向该层 LatticeGrid（nullptr=仅记录结构）
    Solver*      solver = nullptr; ///< 可选：指向该层的 LBM 求解器（用于 mg_step_recursive）
                                   ///< MgNode 不持有 Solver 的所有权，调用方负责生命周期
    MgNode*      parent = nullptr; ///< 父节点（最粗节点的 parent==nullptr）
    std::vector<MgNode*> children; ///< 子节点（更细的嵌套网格块）

    /// 可选 MPI 域分解句柄（monostate=单进程）
    /// MpiDecomp2D*：二维 XY 块分解；MpiDecomp3D*：三维 XYZ 块分解。
    /// MgNode 不持有分解对象的所有权，调用方负责管理生命周期。
    std::variant<std::monostate, MpiDecomp2D*, MpiDecomp3D*> decomp;

    /// 时间步暂存缓冲区（供 mg_step_recursive 保存时刻 t 的状态以用于时间插值）
    /// 由 mg_step_recursive 首次调用时自动分配（大小 = grid->f.size()）。
    /// 调用方不应直接读写这些字段。
    std::vector<double> f_scratch;
    std::vector<double> rho_scratch;
    std::vector<double> u_scratch;

    // ---- 便捷查询方法 ----

    /// 是否为根节点（无父节点）
    bool is_root()  const { return parent == nullptr; }
    /// 是否为叶节点（无子节点，即最细层）
    bool is_leaf()  const { return children.empty(); }
    /// 是否已绑定 LatticeGrid 实例
    bool has_grid() const { return grid != nullptr; }
    /// 是否为二维节点（z_start == z_end）
    bool is_2d()    const { return !extent.is_3d(); }
    /// 是否已绑定 MPI 域分解
    bool has_decomp() const { return !std::holds_alternative<std::monostate>(decomp); }

    /// 本层相对于全局根网格的体积加密比（返回 long long 防止溢出）
    /// D2: refine_ratio^2；D3: refine_ratio^3
    long long volume_ratio() const {
        if (refine_ratio <= 1) return 1LL;
        const long long r = refine_ratio;
        return (dim == MgDim::D3) ? r * r * r : r * r;
    }
};

// ---------------------------------------------------------------------------
/// 多重网格树
///
/// 管理整个多重网格层次结构的所有节点（通过 unique_ptr 持有所有权）。
/// 用户通过返回的裸指针（MgNode*）访问/修改节点，**不应手动 delete**。
///
/// 遍历模式：
///   - `traverse_coarse_to_fine`：广度优先（BFS），从粗层到细层
///     典型用途：每步先推进粗网格，再推进细网格
///   - `traverse_fine_to_coarse`：BFS 逆序，从细层到粗层
///     典型用途：残差传递（restriction）、粗网格修正（prolongation）
///
/// 线程安全：不保证（由调用方负责同步）。
// ---------------------------------------------------------------------------
class MgTree {
public:
    /// 创建多重网格树，根节点（最粗网格）的空间范围为 root_extent。
    /// root_extent 通常是全局网格的 {0, nx-1, 0, ny-1, 0, nz-1}（2D: nz=0）。
    explicit MgTree(MgExtent root_extent, MgDim dim = MgDim::D2);

    ~MgTree() = default;

    /// 禁止拷贝（树内部存储裸指针，拷贝会产生悬垂指针）
    MgTree(const MgTree&)            = delete;
    MgTree& operator=(const MgTree&) = delete;

    /// 允许移动
    MgTree(MgTree&&)            = default;
    MgTree& operator=(MgTree&&) = default;

    // ---- 节点访问 ----

    /// 获取根节点（最粗网格）
    MgNode*       root()       { return root_; }
    const MgNode* root() const { return root_; }

    // ---- 树构建 ----

    /// 在父节点（粗网格）内添加一个细化子区域节点。
    ///
    /// @param parent        父节点（细网格嵌套于其空间范围内，不可为 nullptr）
    /// @param child_extent  细网格在**全局坐标**中的空间范围（须是 parent->extent 的子集）
    /// @param refine_ratio  线性加密比（默认 2：每个粗格对应 2×2（2D）或 2×2×2（3D）个细格）
    ///                      取值范围 [1, 1024]；超出范围抛出 std::invalid_argument。
    /// @return  新创建的子节点指针（由树管理，调用方不得 delete）
    ///
    /// @throws std::invalid_argument 若 parent==nullptr、child_extent 不在 parent->extent 内，
    ///         或 refine_ratio 超出 [1, 1024] 范围
    MgNode* add_level(MgNode* parent, MgExtent child_extent, int refine_ratio = 2);

    // ---- 遍历 ----

    /// 从粗到细广度优先遍历（根节点 → 所有子节点，按 BFS 层次顺序）。
    /// 回调 fn 接收每个节点的指针。
    void traverse_coarse_to_fine(const std::function<void(MgNode*)>& fn);

    /// 从细到粗广度优先遍历（BFS 逆序）。
    /// 回调 fn 接收每个节点的指针。
    void traverse_fine_to_coarse(const std::function<void(MgNode*)>& fn);

    // ---- 查询 ----

    /// 返回树中最深的层级（根节点为 0）
    int max_level() const;

    /// 返回树中所有节点数（包括根节点）
    std::size_t node_count() const { return all_nodes_.size(); }

    /// 按层级收集节点（level 0 = 根节点，1 = 第一层细化，...）
    std::vector<MgNode*> nodes_at_level(int level) const;

private:
    MgNode* root_ = nullptr;
    std::vector<std::unique_ptr<MgNode>> all_nodes_;  ///< 所有节点的所有权
};

// ---------------------------------------------------------------------------
// AMR 粗-细网格数据传递算子
//
// 这些函数在多重网格步骤中传递宏观量（密度 ρ 和速度 u），用于：
//   - 时间步细化（每步开始时，用粗网格 ρ/u 初始化细化边界层）
//   - 粗网格修正（限制）：从细网格返回更新的 ρ/u 到粗网格
// ---------------------------------------------------------------------------

/// 延拓算子（Prolongation）：从粗网格双线性插值 ρ/u 到细网格。
///
/// 对 fine 节点覆盖范围内的每个细节点（if, jf），在粗网格上进行双线性插值：
///   ρ_f(if,jf) = bilinear(ρ_c, px, py)
///   u_f(if,jf) = bilinear(u_c, px, py)
/// 其中 px = fine.extent.x_start + if/r（细节点在粗坐标系中的位置，
/// 节点位于整数坐标处，与 LBM 节点布局一致）。
///
/// @throws std::invalid_argument 若 grid 指针为 nullptr
void mg_prolong_rho_u(const MgNode& coarse, MgNode& fine);

/// 限制算子（Restriction）：从细网格体积平均 ρ/u 到粗网格。
///
/// 对 coarse 节点中与 fine 重叠区域的每个粗格（ic, jc），
/// 计算覆盖它的 r×r 个细格的简单平均：
///   ρ_c(ic,jc) = mean(ρ_f[if0..if0+r-1][jf0..jf0+r-1])
///   u_c(ic,jc) = mean(u_f[if0..if0+r-1][jf0..jf0+r-1])  （逐分量）
///
/// @throws std::invalid_argument 若 grid 指针为 nullptr
void mg_restrict_rho_u(const MgNode& fine, MgNode& coarse);

// ---------------------------------------------------------------------------
// 分布函数延拓（平衡态重建）
// ---------------------------------------------------------------------------

/// @brief f 分布函数延拓：从粗网格初始化细网格的分布函数 f。
///
/// 算法（"平衡态重建"法，overset/AMR 初始化时推荐）：
///   1. 调用 mg_prolong_rho_u 将粗网格 ρ/u 双线性插值到细网格
///   2. 对细网格每个节点，用插值得到的 (ρ, u) 重建局部平衡分布函数：
///        f_α(if,jf) = f_eq(W_α, ρ, c_α, u)
///
/// 物理依据：
///   - 细网格被首次激活（或粗化后重新细化）时，无先验分布函数信息
///   - 用粗网格宏观量插值并重建平衡态 f 是最自然的初始化方式
///   - 经过少量时间步松弛后，非平衡部分将由细网格自身动力学建立
///
/// 与 mg_prolong_rho_u 的关系：
///   mg_prolong_f 隐含调用 mg_prolong_rho_u，然后对每个细节点执行 f_eq 重建。
///   若仅需更新 ρ/u（例如 fringe BC），使用 mg_prolong_rho_u 即可。
///
/// @pre coarse.has_grid() && fine.has_grid()
/// @pre fine.grid->model == D2Q9（仅支持 D2Q9）
/// @throws std::invalid_argument 若 grid 指针为 nullptr 或模型非 D2Q9
void mg_prolong_f(const MgNode& coarse, MgNode& fine);

// ---------------------------------------------------------------------------
// ω 松弛频率缩放（粗 → 细，论文 Eq. 4）
// ---------------------------------------------------------------------------

/// @brief 由粗网格松弛频率计算细网格松弛频率（论文 2015-multigrid.pdf Eq. 4）。
///
/// 在对流缩放（convective scaling）下，细网格与粗网格具有相同的运动粘度，
/// 而细网格空间步长 δxf = δxc/2（加密比 r=2）。此时：
///   ωf = 2ωc / (4 − ωc)
///
/// 推导：ν = cs² * (1/ω − 0.5)，对流缩放要求 νf/νc = (δxf/δxc)²：
///   (1/ωf − 0.5) = (δxf/δxc)² * (1/ωc − 0.5)
///   当 δxf = δxc/2 时解出 ωf = 2ωc/(4−ωc)。
///
/// @param omega_c  粗网格松弛频率；数学有效范围 (0, 4)（4 为公式奇点），
///                 LBM 稳定性要求 omega_c ∈ (0, 2)
/// @return         细网格松弛频率 ωf
/// @throws std::invalid_argument 若 omega_c ∉ (0, 4)（4 为公式奇点）
inline double mg_omega_rescale(double omega_c)
{
    if (omega_c <= 0.0 || omega_c >= 4.0) {
        throw std::invalid_argument(
            "mg_omega_rescale: omega_c must be in (0, 4); "
            "for LBM stability use omega_c in (0, 2)");
    }
    return 2.0 * omega_c / (4.0 - omega_c);
}

// ---------------------------------------------------------------------------
// 覆盖网格（Overset/Fringe）耦合 — 粗 → 细（C→F, 论文 Eqs. 9–10）
// ---------------------------------------------------------------------------

/// @brief 粗→细（C→F）耦合（**纯空间插值**）：在细网格 fringe 区域施加粗网格边界条件。
///
/// 实现论文 2015-multigrid.pdf（Lagrava 2012）Eqs. (9)–(10)：
///   对与粗节点重合的细 fringe 节点（xc_c→f）——(Eq. 9)：
///     fi,f = f_eq_i(ρc, uc) + (ωc / 2ωf) × f_neq_i,c(xc)
///   对非重合的细 fringe 节点（xf_c→f）——(Eq. 10)：
///     fi,f = f_eq_i(ρc_interp, uc_interp) + (ωc / 2ωf) × f_neq_i,c_interp
///
/// 两种情形统一由双线性插值处理（重合时 alpha=β=0，退化为精确取值）：
///   f_neq_i,c = fi,c − f_eq_i(ρc, uc) 在四角粗节点计算后做双线性插值。
///   ωf 由 mg_omega_rescale(omega_c) 计算（Eq. 4）。
///
/// @note 本函数仅做空间插值（使用 coarse.grid 的当前时刻值）。
///       若需要按照论文 Algorithm 步骤 3 正确处理"先时间插值再空间插值"，
///       请使用 mg_apply_fringe_bc_temporal，它在 t 和 t+δtc 两个时刻之间插值。
///
/// 耦合时序（论文 Algorithm）：
///   步骤 1：粗步 collide+stream(t → t+δtc)
///   步骤 2：细步 collide+stream(t → t+δtc/2)
///   步骤 3：C→F 耦合 — 调用 mg_apply_fringe_bc_temporal(t, t+δtc, t_alpha=0.5) ← 推荐
///              或       mg_apply_fringe_bc（简化版，仅使用 t+δtc）
///   步骤 4：细步 collide+stream(t+δtc/2 → t+δtc)
///   步骤 5：F→C 耦合 — 调用 mg_couple_fine_to_coarse
///
/// @param coarse        粗网格节点（含有效 f、ρ、u；提供 C→F 数据）
/// @param fine          细网格节点（fringe BC 施加目标）
/// @param fringe_width  fringe 区域宽度（细网格格子数，默认 2；建议 ≥ 1）
/// @param omega_c       粗网格松弛频率（默认 1.0；用于 Eq. 4 和缩放比 ωc/2ωf）
///
/// @pre coarse.has_grid() && fine.has_grid()
/// @pre fine.grid->model == D2Q9（仅支持 D2Q9）
/// @throws std::invalid_argument 若 grid 为 nullptr 或 omega_c ∉ (0,4)
void mg_apply_fringe_bc(const MgNode& coarse, MgNode& fine,
                        int fringe_width = 2, double omega_c = 1.0);

/// @brief 粗→细（C→F）耦合（**时间+空间双重插值**，论文 Algorithm 步骤 3）。
///
/// 实现论文 2015-multigrid.pdf Algorithm Step 3 中的"double interpolation"：
///   "First the values of ρc, uc and f_neq_i,c of the coarse sites in xc→f
///    are interpolated at time t + δtc/2. Then the values of the fine sites
///    ρc(t + δt/2), uc(t + δt/2) and f_neq_i,c(t + δt/2) are interpolated
///    in space."
///
/// 算法（对 fringe 区域每个细节点）：
///   1. **时间插值**：在四角粗节点处，线性混合 t 和 t+δtc 时刻的值：
///        f_a_half = (1-t_alpha)*f_a_prev + t_alpha*f_a_next  （a = 0..Q-1）
///        ρ_half   = Σ_a f_a_half；  u_half = Σ_a e_a·f_a_half / ρ_half
///        f_neq_half = f_a_half - f_eq(ρ_half, u_half)
///   2. **空间插值**：双线性插值 ρ_half、u_half、f_neq_half 到细 fringe 节点
///   3. 应用 Eqs. 9–10：fi,f = f_eq(ρ_interp, u_interp) + (ωc/2ωf)*f_neq_interp
///
/// @param coarse_prev   粗网格节点持有的**时刻 t** 的分布函数（即步骤 1 之前保存的状态）
/// @param coarse        粗网格节点（步骤 1 后处于时刻 t+δtc，提供 coarse.grid → f_next）
/// @param fine          细网格节点（fringe BC 施加目标，处于时刻 t+δtc/2）
/// @param t_alpha       时间插值因子（0.0=纯 prev，0.5=中点，1.0=纯 next）
/// @param fringe_width  fringe 区域宽度（细网格格子数，默认 2）
/// @param omega_c       粗网格松弛频率（默认 1.0）
///
/// @pre coarse_prev.has_grid() && coarse.has_grid() && fine.has_grid()
/// @pre coarse_prev.grid->nx == coarse.grid->nx（两个粗网格尺寸必须相同）
/// @pre fine.grid->model == D2Q9
/// @throws std::invalid_argument 若 grid 为 nullptr 或 t_alpha ∉ [0,1]
void mg_apply_fringe_bc_temporal(const MgNode& coarse_prev,
                                  const MgNode& coarse,
                                  MgNode& fine,
                                  double t_alpha    = 0.5,
                                  int fringe_width  = 2,
                                  double omega_c    = 1.0);

// ---------------------------------------------------------------------------
// 细 → 粗分布函数耦合（F→C coupling，论文 Eqs. 7–8）
// ---------------------------------------------------------------------------

/// @brief 细→粗（F→C）耦合：用细网格 f 更新粗网格耦合区域的分布函数。
///
/// 实现论文 2015-multigrid.pdf（Lagrava 2012）Eqs. (7)–(8)：
///   fi,c(xc_f→c, t) = f_eq_i(ρf, uf) + (2ωf/ωc) × f_neq_filtered_i    (Eq. 7)
///   f_neq_filtered_i = (1/Q) × Σ_{j=0}^{Q−1} f_neq_i,f(xc + ej, t)   (Eq. 8)
///
/// 算法步骤（对细网格域边界处的每个粗网格节点）：
///   1. 将粗节点 (ic, jc) 映射到对应的细本地索引 if = (ic − fx_s)×r, jf = (jc − fy_s)×r
///   2. 从细网格 f 计算 ρf = Σ_a fi,f 和 uf = Σ_a ei·fi,f / ρf
///   3. 对每个方向 a：按 Eq. 8 对 Q 个邻格（细坐标 ej-偏移处）的 f_neq_a 求平均
///   4. 按 Eq. 7 更新粗网格 fi,c，并同步 ρc、uc
///
/// 用于论文 Algorithm 步骤 5（每个粗步的末尾）：
///   细步 × r 完成 → 调用 mg_couple_fine_to_coarse → 粗网格 fringe 区域更新
///
/// @param fine          细网格节点（F→C 的信息源，需含有效 f）
/// @param coarse        粗网格节点（待更新目标）
/// @param fringe_width  fringe 宽度（细网格格子数，默认 2；转换为粗格宽度 max(1,w/r)）
/// @param omega_c       粗网格松弛频率（默认 1.0）
///
/// @pre fine.has_grid() && coarse.has_grid()
/// @pre fine.grid->model == D2Q9（仅支持 D2Q9）
/// @throws std::invalid_argument 若 grid 为 nullptr 或 omega_c ∉ (0,4)
void mg_couple_fine_to_coarse(const MgNode& fine, MgNode& coarse,
                               int fringe_width = 2, double omega_c = 1.0);

// ---------------------------------------------------------------------------
// 时间步细化辅助
// ---------------------------------------------------------------------------

/// @brief 计算细网格相对于粗网格的时间步细化倍数。
///
/// 对加密比为 r 的细网格，为维持相同的 CFL 条件，细网格的时间步应为：
///   Δt_fine = Δt_coarse / r
/// 即每个粗步对应 r 个细步。
///
/// 对树中多层细化，层 l 相对于根节点（层 0）的总细化倍数为：
///   r_total = r_1 * r_2 * ... * r_l（每层加密比的乘积）
///
/// @param node  目标细网格节点（需有 parent）
/// @return      细网格相对于其直接父节点的时间步细化倍数
///              （若为根节点，返回 1）
inline int mg_subcycle_steps(const MgNode& node) {
    return node.parent ? node.refine_ratio : 1;
}

/// @brief 计算某节点相对于根节点的累积时间步细化倍数
///
/// 遍历从 node 到根节点的路径，将所有 refine_ratio 相乘。
///
/// @param node  目标节点
/// @return      累积细化倍数（根节点返回 1）
int mg_total_subcycle_steps(const MgNode& node);

// ---------------------------------------------------------------------------
// AMR 自适应网格细化标记与判据
// ---------------------------------------------------------------------------

/// @brief 计算网格节点的密度梯度范数（用于 AMR 细化判据）
///
/// 使用二阶中心差分近似 |∇ρ| 在每个内部节点处的值。
/// 边界节点使用单侧差分。结果存入 refinement_indicator（大小 = grid.size()）。
///
/// 典型用法：
///   std::vector<double> indicator(grid.size());
///   mg_compute_refinement_indicator(node, indicator);
///   // indicator[n] > threshold → 标记节点 n 为需要细化区域
///
/// @param node              含 LatticeGrid 的多重网格节点
/// @param indicator         输出：每节点的细化指标值（大小 = node.grid->size()）
/// @throws std::invalid_argument 若 node.grid == nullptr
void mg_compute_refinement_indicator(
    const MgNode& node,
    std::vector<double>& indicator);

// ---------------------------------------------------------------------------
// GPU 平台扩展预留（LBM_ENABLE_CUDA）
// ---------------------------------------------------------------------------
//
// 当前所有多重网格耦合算子（mg_apply_fringe_bc / mg_apply_fringe_bc_temporal /
// mg_couple_fine_to_coarse / mg_prolong_rho_u / mg_restrict_rho_u）均为
// CPU 端纯 C++ 实现，支持 OpenMP 多核并行（#ifdef LBM_ENABLE_OPENMP）。
//
// 未来 GPU 扩展路径（参考论文 Section III GPU Implementation）：
//
// 1. 数据布局：
//    - 细网格和粗网格各自使用 SoA（Structure-of-Arrays）内存布局（论文 "Coalesced" 策略）
//      以最大化全局内存合并访问（coalesced memory access）。
//    - LatticeGrid 的 f/f_tmp/rho/u 已是连续数组，适合直接 cudaMemcpy 或统一内存映射。
//
// 2. CUDA 核函数设计（每个 CPU 函数对应一个 .cu 核函数）：
//    - mg_gpu_apply_fringe_bc: 每线程处理一个细 fringe 节点；
//      线程数 = fringe 区域节点总数，线程块尺寸建议 128 或 256。
//    - mg_gpu_couple_fine_to_coarse: 每线程处理一个粗 fringe 节点；
//      利用 __ldg() 从只读缓存读取细网格 f，减少全局内存延迟。
//    - mg_gpu_prolong_rho_u / mg_gpu_restrict_rho_u: 每线程一个目标节点。
//
// 3. CUDA Dynamic Parallelism（论文核心 GPU 策略）：
//    - 粗网格主核函数完成 collide+stream 后，由子核函数直接在 GPU 端执行
//      C→F 时间+空间插值，无需回到 CPU 调度，避免 PCIe 同步开销。
//    - 对应接口：mg_gpu_apply_fringe_bc_temporal<<<grid,block,0,stream>>>(...)
//
// 4. 异构调度（论文 "Heterogeneous" 策略）：
//    - 粗网格在 GPU 计算，细网格在 CPU（OpenMP）并行；
//      或反之（Top-GPU / Top-Multicore 两种模式）。
//    - 边界数据通过 cudaMemcpyAsync + CUDA stream 与 CPU OpenMP 计算流水线重叠。
//
// 5. 条件编译占位（待实现时取消注释）：
// #ifdef LBM_ENABLE_CUDA
//   void mg_gpu_apply_fringe_bc(const MgNode& coarse, MgNode& fine,
//                                int fringe_width, double omega_c,
//                                cudaStream_t stream = nullptr);
//   void mg_gpu_couple_fine_to_coarse(const MgNode& fine, MgNode& coarse,
//                                      int fringe_width, double omega_c,
//                                      cudaStream_t stream = nullptr);
// #endif

} // namespace lbm
