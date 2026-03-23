#pragma once
// core/include/lbm/solid.hpp — 固体节点边界条件（BB / IBB）
//
// ============================================================
// 固体边界处理方案
// ============================================================
// 本模块实现了两种用于**内部浸入固体**的反弹边界条件：
//
//   1. 半步长反弹（Halfway BounceBack, BB）
//      Ladd (1994)。壁面位于流体节点与固体节点之间的半格处（q=0.5）。
//      实现简单、一阶精度，是 LBM 固体边界的经典方案。
//
//   2. Bouzidi 插值反弹（Interpolated BounceBack, IBB）
//      Bouzidi et al. (2001)。壁面距离 q = |流体节点→壁面| / 格子间距
//      可为任意值（0 < q ≤ 1），通过线性插值达到二阶精度：
//
//        q ≤ 0.5：f[opp(a)](x_f) = 2q·f*[a](x_f) + (1−2q)·f*[a](x_f−c_a)
//        q > 0.5：f[opp(a)](x_f) = (1/2q)·f*[a](x_f) + (1−1/2q)·f*[opp(a)](x_f)
//
//      其中 f*[a] 为碰后（pre-stream）分布函数，存储在 g.f_tmp 中。
//
// ============================================================
// 尖锐界面 IBM 框架（Sharp-Interface IBM）
// ============================================================
// 上述 BB/IBB 方案本身即构成完整的"尖锐界面 IBM"框架：
//   - mark_solid_cylinder() / mark_solid_rectangle()：将节点分为固体/流体
//   - compute_ibb_distances()：计算各流固链接的壁面距离分数 q
//   - apply_solid_bounce_back() / apply_solid_ibb()：在流式迁移后施加反弹
//
// 与扩散界面 IBM（基于 Peskin δ 函数）相比，尖锐界面方案无需显式标记点集，
// 边界完全由 solid[] 标志和 q_ibb[] 数组确定，适用于固定刚体（圆柱绕流等）。
//
// ============================================================
// 使用示例（圆柱绕流）
// ============================================================
//   lbm::LatticeGrid grid(nx, ny, 1, lbm::LatticeModel::D2Q9);
//   lbm::Solver solver(grid, omega);
//   // 注册进/出口边界条件...
//   lbm::mark_solid_cylinder(grid, cx, cy, radius);   // 标记圆柱固体节点 + 计算 q
//   solver.set_solid_bc_type(lbm::SolidBCType::InterpolatedBounceBack);  // 开启 IBB
//   for (int t = 0; t < N; ++t) solver.step();
//
// ============================================================
// 参考文献
// ============================================================
//   Ladd A.J.C. (1994) J. Fluid Mech. 271, 285-309.
//   Bouzidi M. et al. (2001) Phys. Fluids 13(11):3452-3459.
//   Yu D. et al. (2003) Phys. Fluids 15(8):2462-2469.

#include "lattice.hpp"
#include <string>

namespace lbm {

// ---------------------------------------------------------------------------
// 固体边界条件类型（供 Solver::set_solid_bc_type() 使用）
// ---------------------------------------------------------------------------
enum class SolidBCType {
    None,                   ///< 不施加固体边界条件（默认，仅流体）
    BounceBack,             ///< 半步长反弹（Ladd 1994；q=0.5，一阶精度）
    InterpolatedBounceBack, ///< Bouzidi 插值反弹（2001；精确 q，二阶精度）
};

// ---------------------------------------------------------------------------
// 几何标记：将圆柱（圆形截面）内部节点标记为固体
//
// 同时对每个"流体节点↔固体节点"方向对计算 IBB 壁面距离分数 q_ibb，
// 通过求流体节点出发的射线与圆柱表面的交点获得。
//
// @param grid    格子网格（D2Q9，须已构造）
// @param cx, cy  圆柱中心（格子单位）
// @param radius  圆柱半径（格子单位）
// ---------------------------------------------------------------------------
void mark_solid_cylinder(LatticeGrid& grid, double cx, double cy, double radius);

// ---------------------------------------------------------------------------
// 几何标记：将矩形区域 [i0,i1] × [j0,j1]（含边界）标记为固体
//
// 矩形面上的 q_ibb 保留默认值 0.5（即退化为标准半步长反弹）。
//
// @param grid      格子网格
// @param i0,j0    西南角格子坐标（含）
// @param i1,j1    东北角格子坐标（含）
// ---------------------------------------------------------------------------
void mark_solid_rectangle(LatticeGrid& grid, int i0, int j0, int i1, int j1);

// ---------------------------------------------------------------------------
// 计算 IBB 壁面距离分数 q_ibb（通用版本）
//
// 对于每个流体节点 x_f 以及每个指向相邻固体节点的方向 a，
// 将 q_ibb[n*Q + a] 设为流体节点到固体节点距离的 0.5 倍（即 q=0.5，halfway BB）。
//
// 如需针对特定几何体精确计算 q，应使用 mark_solid_cylinder() 等专用函数
// （内部已自动计算精确 q），而非调用本通用版本。
//
// @param grid  格子网格（须已通过 mark_solid_* 设置 solid 标志）
// ---------------------------------------------------------------------------
void compute_ibb_distances(LatticeGrid& grid);

// ---------------------------------------------------------------------------
// 对固体节点施加半步长反弹（BounceBack，q=0.5）
//
// 应在 stream() 之后调用。
// 内核：f[opp(a)](x_f) = f_tmp[a](x_f)（g.f_tmp 含碰后值）。
//
// @param grid         格子网格（须已设置 solid 标志）
// @param phys_i0      物理列起始索引（本地坐标，含）；非 MPI 时为 0
// @param phys_j0      物理行起始索引（本地坐标，含）；非 MPI 时为 0
// @param phys_i1      物理列结束索引（本地坐标，含）；非 MPI 时为 nx-1
// @param phys_j1      物理行结束索引（本地坐标，含）；非 MPI 时为 ny-1
//
// MPI 模式下须传入物理区域范围以跳过幽灵节点，否则传入全网格范围。
// 无参版本（向后兼容）自动使用全网格范围 [0, nx-1] × [0, ny-1]。
// ---------------------------------------------------------------------------
void apply_solid_bounce_back(LatticeGrid& grid);

/// MPI 适配版：仅对物理区域 [phys_i0, phys_i1] × [phys_j0, phys_j1] 内的节点施加 BB。
/// 幽灵节点（超出物理范围的行/列）将被跳过，避免幽灵数据被错误覆盖。
void apply_solid_bounce_back(LatticeGrid& grid,
                              int phys_i0, int phys_j0,
                              int phys_i1, int phys_j1);

// ---------------------------------------------------------------------------
// 对固体节点施加 Bouzidi 插值反弹（IBB）
//
// 应在 stream() 之后调用。利用 q_ibb 数组中预计算的壁面距离分数 q 进行插值。
// 若 q_ibb 未填充（仍为默认 0.5），则退化为标准半步长反弹。
//
// @param grid  格子网格（须已设置 solid 标志，且 q_ibb 已由 mark_solid_*() 填充）
//
// MPI 版本同 apply_solid_bounce_back，传入物理区域范围以跳过幽灵节点。
// 上游节点的 f_tmp 值允许来自幽灵行（幽灵行在 halo_exchange 后含正确碰后值）。
// ---------------------------------------------------------------------------
void apply_solid_ibb(LatticeGrid& grid);

/// MPI 适配版：仅对物理区域 [phys_i0, phys_i1] × [phys_j0, phys_j1] 内的节点施加 IBB。
void apply_solid_ibb(LatticeGrid& grid,
                     int phys_i0, int phys_j0,
                     int phys_i1, int phys_j1);

// ---------------------------------------------------------------------------
// 逐固体节点标记反弹方案（用于多固体混合 BC 场景）
//
// 扫描当前已标记的所有固体节点（solid[idx]==1），对其中 solid_bc_node[idx]==0
// 的节点（尚未分配方案的节点）写入 bc_mode，从而实现"按最近标记的固体体"分配
// 方案的效果——在每个固体体 mark_solid_*() 调用后立即调用本函数即可。
//
// @param grid     格子网格（须已设置 solid 标志）
// @param bc_mode  1 = BounceBack（BB）| 2 = InterpolatedBounceBack（IBB）| 0 = 跳过
// ---------------------------------------------------------------------------
void assign_solid_bc_unmarked(LatticeGrid& grid, int bc_mode);

// ---------------------------------------------------------------------------
// 混合反弹边界条件施加器（支持同一仿真中不同固体使用不同方案）
//
// 对于每个流-固界面链接 (x_f, a)，根据对应固体节点的 solid_bc_node 值选择方案：
//   solid_bc_node[solid_idx] == 1  → 半步长反弹（BB）
//   solid_bc_node[solid_idx] == 2  → Bouzidi 插值反弹（IBB）
//   solid_bc_node[solid_idx] == 0  → 使用 default_bc 参数指定的全局方案
//
// 当所有节点的 solid_bc_node 相同时，性能与纯 BB/IBB 版本等价（只增加一次整数比较）。
//
// @param grid        格子网格
// @param default_bc  0=跳过 | 1=BB | 2=IBB（供无逐节点标记时使用的全局方案）
// @param phys_i0/j0/i1/j1  物理区域（本地坐标，含端点）；非 MPI 模式传全网格范围
// ---------------------------------------------------------------------------
void apply_solid_bc_mixed(LatticeGrid& grid, int default_bc,
                           int phys_i0, int phys_j0,
                           int phys_i1, int phys_j1);

/// 向后兼容版（非 MPI，覆盖全网格）。
void apply_solid_bc_mixed(LatticeGrid& grid, int default_bc);

// ---------------------------------------------------------------------------
// 固体受力统计：动量交换法（Momentum Exchange Algorithm, MEA）
//
// 参考：Ladd A.J.C. (1994) J. Fluid Mech. 271, 285-309.
//       Aidun C.K. & Lu Y. (1995) J. Stat. Phys. 81(1-2):49-61.
//
// 原理：在每个流-固界面链接 (x_f, a) 处，流体传递给固体的动量为：
//
//   ΔP_a = 2 · f_post[a](x_f) · c_a
//
// 其中 f_post[a] 为 f_tmp 中存储的碰后（post-collision）分布函数值，
// c_a 为该方向的格子速度向量。
//
// 总力（格子单位，ρ_0=1）：
//   F_x = Σ_{所有流-固链接 (n_f, a)} 2 · f_tmp[n_f·Q+a] · c_a[x]
//   F_y = Σ_{所有流-固链接 (n_f, a)} 2 · f_tmp[n_f·Q+a] · c_a[y]
//
// 调用时机：apply_solid_bounce_back() 或 apply_solid_ibb() 之后（两者均在
//   stream() 之后调用），此时 f_tmp 中仍保存碰后分布函数。
//
// 注意：在 MPI 模式下，本函数仅统计本进程物理区域内的贡献，
//   调用方需通过 MPI_Allreduce 将各进程的 (fx, fy) 求和，以得到全局力。
//
// @param grid    格子网格（须已完成 mark_solid_*() + stream() + apply_solid_*()）
// @param out_fx  输出 x 方向合力（格子单位）
// @param out_fy  输出 y 方向合力（格子单位）
// @param phys_i0  物理区域起始列（含）；非 MPI 模式传 0
// @param phys_j0  物理区域起始行（含）；非 MPI 模式传 0
// @param phys_i1  物理区域结束列（含）；非 MPI 模式传 nx-1
// @param phys_j1  物理区域结束行（含）；非 MPI 模式传 ny-1
// ---------------------------------------------------------------------------
void compute_solid_body_force(const LatticeGrid& grid,
                               double& out_fx, double& out_fy,
                               int phys_i0, int phys_j0,
                               int phys_i1, int phys_j1);

/// 非 MPI 版本（覆盖整个本地网格）。
void compute_solid_body_force(const LatticeGrid& grid,
                               double& out_fx, double& out_fy);

// ---------------------------------------------------------------------------
// MPI 模式下固体幽灵层标记的设计说明
//
// 在 MPI 块分解中，mark_solid_cylinder() / mark_solid_rectangle() 以本地坐标
// （含幽灵行/列）完整迭代，因此幽灵层的 solid[] 和 q_ibb[] 已被正确设置——
// 无需额外的 MPI 通信即可处理固体跨块边界的情况。
//
// 若将来引入运动固体（每步更新几何位置），需在每步更新几何后重新调用标记函数。
// 此时标记函数本身不需要 MPI 通信，因为几何参数通过广播在所有进程保持一致。
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// 固体边界：从外部 CSV 网格文件加载（第三方网格接口）
//
// 文件格式（每行一个边界点，以逗号分隔，格子坐标）：
//   x, y [, q]
//   - x, y：边界点坐标（浮点数，格子单位）
//   - q：IBB 壁面距离分数（可选；缺省 0.5，退化为半步长反弹）
//
// 算法：
//   1. 解析文件中所有边界采样点
//   2. 对每个采样点，找到最近的格子节点，将其标记为固体
//   3. 对每个流体节点↔固体节点的方向对，若提供了 q 值则设定 q_ibb
//
// 注意：此函数标记的是固体"内部"节点；若边界点恰在格线上，建议用
//   mark_solid_cylinder() / mark_solid_rectangle() 等专用函数。
//   本函数适合从外部 CAD/FEM 软件生成的任意形状固体边界网格。
//
// @param grid     格子网格（须已构造）
// @param filename CSV 文件路径
// ---------------------------------------------------------------------------
void mark_solid_from_mesh_file(LatticeGrid& grid, const std::string& filename);

} // namespace lbm
