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

// 前向声明，避免在仅使用树结构时引入完整 LatticeGrid 头
namespace lbm { struct LatticeGrid; }

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
    MgNode*      parent = nullptr; ///< 父节点（最粗节点的 parent==nullptr）
    std::vector<MgNode*> children; ///< 子节点（更细的嵌套网格块）

    /// 可选 MPI 域分解句柄（nullptr=单进程）
    /// 调用方负责管理生命周期；MgNode 不持有分解对象的所有权。
    /// - 2D 时绑定 `MpiDecomp2D*`（从 `lbm_mpi_decomp2d_new()` 获取）
    /// - 3D 时绑定 `MpiDecomp3D*`（预留，暂未实现幽灵交换）
    void* decomp = nullptr;

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
    bool has_decomp() const { return decomp != nullptr; }

    /// 本层相对于全局根网格的体积加密比
    /// D2: refine_ratio^2；D3: refine_ratio^3
    int volume_ratio() const {
        if (refine_ratio <= 1) return 1;
        return (dim == MgDim::D3)
            ? refine_ratio * refine_ratio * refine_ratio
            : refine_ratio * refine_ratio;
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
    /// @return  新创建的子节点指针（由树管理，调用方不得 delete）
    ///
    /// @throws std::invalid_argument 若 parent==nullptr 或 child_extent 不在 parent->extent 内
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

} // namespace lbm
