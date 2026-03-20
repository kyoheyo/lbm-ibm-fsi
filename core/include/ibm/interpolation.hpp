#pragma once
// core/include/ibm/interpolation.hpp — IBM 速度插值与力展布
//
// 提供四种 IBM 方案：
//   1. 标准 Peskin δ 函数插值 + 直接力法（Uhlmann 2005）
//   2. 多重直接力法（MDF-IBM，Luo et al. 2007）：迭代修正，提高无滑移精度
//   3. 移动最小二乘插值（MLS-IBM）：高阶速度重构，适用于非均匀标记点分布
//   4. 罚函数法（Penalty-IBM）：Goldstein 反馈力 + 积分项，稳定性好
//
// MPI 跨块注意事项
// ----------------
// interpolate_velocity() / mls_interpolate_velocity()：
//   读取本地网格（含幽灵层）的速度值。幽灵层在 stream() 前由 halo_exchange()
//   填充了邻居进程的 f 值，故 macroscopic 计算后幽灵节点速度也是正确的。
//   因此，标记点的支撑域只要不超出幽灵层宽度（1 格），插值就是正确的。
//
//   【重要】当使用 FourPoint 核（support=2）时，支撑宽度为 2 格；
//   若标记点距分区边界 < 2 格，插值将遗漏超出幽灵层的贡献，引入误差。
//   建议：FourPoint 核下，IBM 标记点应距 MPI 分区边界 ≥ 2 格，
//   或在配置中启用 2 层幽灵交换（设置 [mpi] ibm_halo_width = 2）。
//
// spread_force()：
//   向本地网格（含幽灵层）写入体力。写入幽灵层的贡献在下一次 halo_exchange()
//   时不会自动传递给邻居进程——调用方需在 spread_force() 后显式执行力场的
//   幽灵层归并（reduce-scatter）。若 IBM 力仅用于 Guo 体力格式，且标记点始终
//   位于物理区域内部（距块边界 ≥ 2 格），则可安全忽略此项。
//
//   spread_force_with_halo_reduce() 提供了自动 MPI 幽灵层力场归并功能：
//   展布后对幽灵层力贡献执行 MPI_Allreduce，将跨块力正确累加到邻居物理层。

#include "marker.hpp"
#include "../lbm/lattice.hpp"

namespace ibm {

// ---------------------------------------------------------------------------
// Peskin 正则化 δ 函数  φ(r) = (1/h)*φ̂(r/h)
// 支持的核函数：
//   2 点（线性）— 支撑宽度 2h
//   4 点（Peskin）— 支撑宽度 4h
// ---------------------------------------------------------------------------
enum class DeltaKernel { TwoPoint, FourPoint };

// ---------------------------------------------------------------------------
// 速度插值（标准 Peskin δ 函数）：
//   u_IB(X) = Σ_{x} u(x) δ(x − X) Δx²
// 将欧拉流体速度映射到拉格朗日标记点速度（写入 mk.ux, mk.uy）。
// ---------------------------------------------------------------------------
void interpolate_velocity(const lbm::LatticeGrid& grid,
                          MarkerSet& ms,
                          double dx,
                          DeltaKernel kernel = DeltaKernel::FourPoint);

// ---------------------------------------------------------------------------
// 力展布（标准 Peskin δ 函数）：
//   f(x) = Σ_{X} F(X) δ(x − X) ΔS
// 将拉格朗日 IBM 力密度展布到欧拉体力场（写入 grid.force）。
// 每次调用前自动将 grid.force 清零。
// ---------------------------------------------------------------------------
void spread_force(lbm::LatticeGrid& grid,
                  const MarkerSet& ms,
                  double dx,
                  DeltaKernel kernel = DeltaKernel::FourPoint);

// ---------------------------------------------------------------------------
// 一维 δ 函数核值（公开供测试使用）
// ---------------------------------------------------------------------------
double delta_phi(double r, double h, DeltaKernel kernel);

// ===========================================================================
// 多重直接力法（MDF-IBM, Multi-Direct Forcing）
//
// 参考：Luo K. et al. (2007) J. Comput. Phys. 227:454-483.
//
// 算法（对刚体边界，目标速度 u_target = 0）：
//   初始化：F = 0，u_work = grid.u（流体速度工作副本）
//   对 iter = 0..n_iter-1：
//     1. 用 u_work 插值标记点速度 u_k
//     2. 计算增量力 δF = (0 − u_k) / dt
//     3. 展布 δF 到 Eulerian 网格（临时场）
//     4. 更新 u_work += dt · J[δF]（Lagrangian→Eulerian 展布的速度修正）
//     5. 累积总力 F += δF
//   结束：将最终总力 F 写入 grid.force（供 Guo 体力格式使用）
//
// 相比单次直接力，MDF 通过子迭代逐步逼近无滑移条件，显著改善界面处误差。
//
// @param fluid   Eulerian 流体网格（grid.force 将被覆盖为最终 IBM 体力）
// @param ms      拉格朗日标记点集（mk.fx/fy 将被写入最终力）
// @param dx      格子间距
// @param dt      时间步长（格子单位 = 1）
// @param n_iter  子迭代次数（建议 2–4，默认 3）
// @param kernel  δ 函数核
// ===========================================================================
void compute_ibm_forces_mdf(lbm::LatticeGrid& fluid,
                             MarkerSet& ms,
                             double dx,
                             double dt   = 1.0,
                             int    n_iter  = 3,
                             DeltaKernel kernel = DeltaKernel::FourPoint);

// ===========================================================================
// 罚函数法 IBM（Penalty-IBM / Feedback Forcing）
//
// 参考：Goldstein D. et al. (1993) J. Comput. Phys. 105:354-366.
//       Fadlun E.A. et al. (2000) J. Comput. Phys. 161:35-60.
//
// 物理原理：
//   通过在 Lagrangian 标记点处施加刚度反馈力，强迫流体速度趋向目标速度
//   u_target（对静止边界 = 0）。力由两部分组成：
//
//   原始形式（令 e = u_target − u_IB 为无滑移残差，α、β 均为正数）：
//   F(X, t) = α · e(t) + β · ∫₀ᵗ e(τ) dτ
//
//   等价展开（验证符号一致性）：
//   F(X, t) = −α · (u_IB − u_target) − β · ∫₀ᵗ (u_IB − u_target) dτ
//
//   当 u_IB > u_target（流体速度超过目标），e < 0，F < 0，力阻减流体：
//   例：α=8, u_IB=0.05, u_target=0 → F = 8·(-0.05) = -0.4（正确方向）。
//
//   其中 α 为比例增益（应为大正数，推荐 2/dt² 至 10/dt²），
//   β 为积分增益（可选，可设 0；β > 0 时收敛更稳定）。
//
// 算法（每步调用）：
//   1. 插值流体速度 → u_IB
//   2. 计算残差 e = u_target − u_IB
//   3. 更新积分 integral += dt · e（含抗饱和限幅）
//   4. F = α · e + β · integral
//   5. 展布力 F 到欧拉网格（写入 grid.force）
//
// 参数选择：
//   α = K_p（"比例系数"，正数，通常取 K_p = O(1/dt²)）
//   β = K_i（"积分系数"，非负数，通常取 K_i = 0 或 K_i ≈ K_p/100）
//
//   对 dt=1（格子单位），推荐 α ∈ [2, 10]。
//   过大的 α 可能导致数值不稳定。
//
// @param fluid      Eulerian 流体网格（grid.force 被覆盖）
// @param ms         Lagrangian 标记点集（mk.fx/fy 被写入最终力；mk.ux/uy 写入插值速度）
// @param dx         格子间距
// @param dt         时间步长（格子单位通常 = 1）
// @param alpha      比例增益（大正数，如 8.0/dt/dt）
// @param beta       积分增益（非负数，可设 0；mk.fz 被临时征用存储积分）
// @param integral_x 各标记点 x 方向速度误差积分（长度 = ms.size()，需在外部持久化）
// @param integral_y 各标记点 y 方向速度误差积分（长度 = ms.size()，需在外部持久化）
// @param kernel     δ 函数核
// @param u_target_x  目标 x 速度（如刚体壁面速度；常取 0.0）
// @param u_target_y  目标 y 速度
// ===========================================================================
void compute_ibm_forces_penalty(lbm::LatticeGrid& fluid,
                                 MarkerSet& ms,
                                 double dx,
                                 double dt,
                                 double alpha,
                                 double beta,
                                 std::vector<double>& integral_x,
                                 std::vector<double>& integral_y,
                                 DeltaKernel kernel = DeltaKernel::FourPoint,
                                 double u_target_x = 0.0,
                                 double u_target_y = 0.0);

// ===========================================================================
// 移动最小二乘速度插值（MLS-IBM）
//
// 参考：Liu W.K. et al. (1997) Int. J. Numer. Meth. Fluids 25:1387-1407.
//       Wang Z. et al. (2009) J. Comput. Phys. 228:1963-1978.
//
// 使用线性多项式基 {1, Δx, Δy}（3 个自由度）和 Gaussian 权函数
//   w(r) = exp(−r² / h²)，h = 2.5 · dx，支撑半径 R_s = 2.5 · dx
// 对每个标记点，在其邻域内构建加权最小二乘系统（3×3），求解后
// 以多项式插值取代 Peskin δ 函数，得到更精确的速度估计。
//
// 与标准 interpolate_velocity() 接口相同：写入 mk.ux, mk.uy。
// 适用场景：标记点密度不均匀、支撑域节点数目有限的情形。
//
// @param grid    Eulerian 流体网格
// @param ms      拉格朗日标记点集（写入 mk.ux, mk.uy）
// @param dx      格子间距
// ===========================================================================
void mls_interpolate_velocity(const lbm::LatticeGrid& grid,
                               MarkerSet& ms,
                               double dx);

// ===========================================================================
// IBM 固体受力统计：合力计算
//
// 通过对 Lagrangian 标记点的力密度加权求和，计算浸入固体所受的总合力：
//
//   F_x = Σ_m  mk.fx * mk.ds
//   F_y = Σ_m  mk.fy * mk.ds
//
// 其中 mk.fx/fy 为 IBM 力计算（compute_ibm_forces_mdf / compute_ibm_forces_penalty
// / MLS 方案）在各标记点处得到的力密度，mk.ds 为该标记点对应的弧长/面积元素。
//
// 调用时机：任一 IBM 力计算函数（compute_ibm_forces_mdf/penalty 或 MLS 方案）之后。
// 此时 mk.fx/mk.fy 已包含本步的 IBM 力密度值。
//
// MPI 说明：
//   在 MPI 模式下，若标记点已按进程分配（每进程仅持有部分标记），
//   本函数仅统计本进程持有的标记点贡献；调用方需通过 MPI_Allreduce 求全局和。
//   若所有进程持有完整标记点集（复制模式），则调用方无需额外归约。
//
// @param ms      拉格朗日标记点集
// @param out_fx  输出 x 方向合力（格子单位）
// @param out_fy  输出 y 方向合力（格子单位）
// ===========================================================================
void compute_ibm_body_force(const MarkerSet& ms,
                             double& out_fx, double& out_fy);

} // namespace ibm
