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
//   读取本地网格（含幽灵层）的速度值。
//
//   【重要】solver.step() 内部调用 stream() 时会先执行 halo_exchange()（交换 f），
//   然后执行 PUSH 流式迁移。PUSH 迁移会将物理边界行的 f 推送进幽灵行，
//   从而 **覆盖** halo_exchange 写入的邻居数据。其后 compute_macroscopic() 对所有
//   节点（含幽灵行）计算 u，但此时幽灵行 u 来自本地边界行的外推值，
//   **并非邻居进程的真实速度**。
//
//   因此，在 step_ibm() 调用 interpolate_velocity() 之前，必须先通过
//   ibm_halo_exchange_u_2d() 显式交换 u 场幽灵行，才能得到正确的邻居速度。
//   ibm_halo_width=1 交换 1 层（满足 TwoPoint 核）；FourPoint 核（support=2）
//   仍需标记点距分区边界 ≥ 2 格（当前结构仅分配 1 层幽灵行，无法完全支持 2 层）。
//
// spread_force()：
//   向本地网格（含幽灵层）写入体力。写入幽灵层的贡献在下一次 halo_exchange()
//   时不会自动传递给邻居进程——调用方需在 spread_force() 后显式执行力场的
//   幽灵层归并（reduce-scatter）。
//
//   ibm_halo_reduce_force_2d() 提供了正确的 MPI 幽灵层力场归并：
//   将本进程幽灵行中的力贡献通过 MPI_Sendrecv 发送回邻居的对应物理行并累加，
//   然后清零本地幽灵行，确保跨块力展布的完整性。

#include "marker.hpp"
#include "../lbm/lattice.hpp"
#include "../lbm/mpi_decomp.hpp"
#include <vector>

namespace ibm {

// ---------------------------------------------------------------------------
// Peskin 正则化 δ 函数  φ(r) = (1/h)*φ̂(r/h)
// 支持的核函数：
//   2 点（线性）— 支撑宽度 2h
//   4 点（Peskin）— 支撑宽度 4h
// ---------------------------------------------------------------------------
enum class DeltaKernel { TwoPoint, FourPoint };

// ===========================================================================
// MLS 形状函数支撑集（隐式 MLS 方案内部数据结构）
//
// 用于缓存每个 Lagrangian 标记点的 MLS 形状函数值，避免固定物体每步重复构建。
// 由 compute_ibm_forces_mls_implicit_stationary 的 phi_cache 参数持久化。
//
// 字段说明：
//   idx  — 该标记点支撑域内各 Euler 节点的全局索引（行主序 grid.idx(i,j)）
//   phi  — 对应的 MLS 形状函数值 φ_j^k（2025 JCP Eq.10–14）
//
// 注意：对非本进程拥有的标记点，idx 和 phi 均为空（MPI 归属过滤）。
// ===========================================================================
struct MlsSupportSet {
    std::vector<int>    idx;   ///< Euler 节点全局索引
    std::vector<double> phi;   ///< 对应 MLS 形状函数值 φ_j^k
};

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
// 在任意位置插值流体速度（使用标准 Peskin δ 函数）。
//
// 将欧拉流体速度场 u 插值到给定的 n 个点 (x[], y[]) 处，
// 结果写入 out_ux[], out_uy[]（调用方须保证长度 ≥ n）。
//
// 用途：为刚体内部拉格朗日点（方案 C 内部质量）插值流体速度，
//       无需为此构造完整 MarkerSet。
//
// @param grid     流体网格
// @param x        点的 x 坐标数组（格子单位）
// @param y        点的 y 坐标数组（格子单位）
// @param n        点的数量
// @param dx       格子间距
// @param out_ux   输出 x 速度（长度 ≥ n）
// @param out_uy   输出 y 速度（长度 ≥ n）
// @param kernel   δ 核类型（默认 TwoPoint，内部点一般分布均匀，TwoPoint 足够）
// ---------------------------------------------------------------------------
void interpolate_velocity_at_points(const lbm::LatticeGrid& grid,
                                    const double* x, const double* y, int n,
                                    double dx,
                                    double* out_ux, double* out_uy,
                                    DeltaKernel kernel = DeltaKernel::TwoPoint);

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
// 移动最小二乘速度插值（MLS 插值算子 J）
//
// 参考：Liu W.K. et al. (1997) Int. J. Numer. Meth. Fluids 25:1387-1407.
//       Vanella & Balaras (2009) J. Comput. Phys. 228:2366-2391
//       2025 JCP Wu & Fu §3.1, Eq.(10)–(14)
//
// 使用线性多项式基 {1, Δx/dx, Δy/dx}（3 个自由度）和 Gaussian 权函数
//   w(r) = exp(−r²/(H_k·ε)²)，H_k = 1.5·dx，ε = 0.3（2025 JCP Eq.14）
//   支撑域：矩形 |Δx|≤H_k 且 |Δy|≤H_k（2025 JCP Fig.2）
// 对每个标记点，在其邻域内构建 3×3 加权最小二乘系统，求解后
// 以多项式插值取代 Peskin δ 函数，得到更精确的速度估计。
//
// 写入 mk.ux, mk.uy。供三种 MLS-IBM 方案共用：
//   - 原始 MLS（compute_ibm_forces_mls_original）
//   - 显式 MLS（compute_ibm_forces_mls_explicit）
//   - 隐式 MLS（compute_ibm_forces_mls_implicit）
// ===========================================================================
void mls_interpolate_velocity(const lbm::LatticeGrid& grid,
                               MarkerSet& ms,
                               double dx);

// ===========================================================================
// MLS 力展布（MLS 伴随算子 J^T）
//
// 参考：2025 JCP "An implicit moving-least-squares immersed boundary method
//       for high fidelity fluid-structure interaction simulations"
//
// 使用 MLS 插值算子的转置进行力展布，保证与 mls_interpolate_velocity()
// 的离散伴随一致性（J^T 伴随属性），从而满足动量守恒条件。
//
// 算法（对每个标记点 X_m）：
//   1. 构造与插值相同的 MLS 矩阵 M = Σ_i w_i p_i ⊗ p_i^T
//   2. 求解 M · c = e_0，e_0 = [1,0,0]^T（即求第一行的 MLS 系数）
//   3. 对每个支撑节点 x_i：
//        φ_i = w_i · (c_0 + c_1·Δx/dx + c_2·Δy/dx)  ← MLS 形状函数
//        f[x_i] += φ_i · F_m · ds_m
//
// 此展布与 mls_interpolate_velocity 互为伴随，满足离散恒等式：
//   Σ_m F_m · (J·u)_m = Σ_i u_i · (J^T·F)_i
//
// 供显式 MLS（compute_ibm_forces_mls_explicit）和
// 隐式 MLS（compute_ibm_forces_mls_implicit）使用。
//
// 每次调用前自动将 grid.force 清零。
//
// @param grid    Eulerian 流体网格（grid.force 将被覆盖为最终 IBM 体力）
// @param ms      拉格朗日标记点集（读取 mk.fx, mk.fy, mk.ds）
// @param dx      格子间距
// ===========================================================================
void mls_spread_force(lbm::LatticeGrid& grid,
                      const MarkerSet& ms,
                      double dx);

// ===========================================================================
// 原始 MLS-IBM（Original MLS）—— MLS 插值 + MLS 形状函数展布（含守恒因子 c_i）
//
// 参考：2025 JCP Wu & Fu §3.1（Algorithm 1）
//       Vanella & Balaras (2009) J. Comput. Phys. 228:2366-2391
//
// 算法（单步直接力法）：
//   1. MLS 速度插值：U_m = J · u*
//   2. 直接力：F_m = (u_target − U_m) / dt
//   3. MLS 形状函数展布（Eq.16，含 c_m=ds_m 守恒因子）：
//        f_j = Σ_m c_m φ_j^m F_m
//
// 注意：展布与插值算子非完全伴随，存在无滑移误差（见 Fig.3a）。
// `kernel` 参数保留供向后兼容但不再使用（展布已改为 MLS 形状函数）。
// ===========================================================================
void compute_ibm_forces_mls_original(lbm::LatticeGrid& fluid,
                                      MarkerSet& ms,
                                      double dx,
                                      double dt          = 1.0,
                                      DeltaKernel kernel = DeltaKernel::FourPoint);

// ===========================================================================
// 显式 MLS-IBM（Explicit MLS）—— MLS 插值 + MLS 展布 + 全局 Z 修正
//
// 参考：2025 JCP Wu & Fu §3.2（Algorithm 2）
//       Chen et al. (2022) Phys. Rev. E 106:015307
//
// 算法：
//   1. MLS 速度插值：U_m = J · u*
//   2. 直接力：F_m = (u_target − U_m) / dt
//   3. 第一次 MLS 展布：f = J^T · F
//   4. 重插值：g_k = J · f（再次 MLS 插值展布后的力场）
//   5. Z 修正因子（Eq.21，最小化 ||Z·g − F||²）：
//        Z = Σ_k(F_k·g_k) / Σ_k|g_k|²
//   6. 最终展布：f → Z · f
//
// 注意：Z 因子破坏力和力矩守恒（见 Table 1），无滑移残差仍显著。
// 推荐使用隐式 MLS（compute_ibm_forces_mls_implicit）。
// ===========================================================================
void compute_ibm_forces_mls_explicit(lbm::LatticeGrid& fluid,
                                      MarkerSet& ms,
                                      double dx,
                                      double dt         = 1.0);

// ===========================================================================
// 隐式 MLS-IBM 力计算（Implicit MLS）—— Algorithm 3，Scheme II：GMRES 求解
//
// 参考：2025 JCP Wu & Fu §4，Algorithm 3，Eq.(24)–(28)
//
// 完整实现论文 Algorithm 3：
//   A1: 计算传递算子 Φ（每个 Lagrangian 点的 MLS 形状函数 φ_j^k，Eq.10–14）
//   A2: 重建 Lagrangian 速度 U* = J·u*（MLS 插值）
//   C2: 构建 N_l×N_l 相关矩阵 A（Eq.28）和右端向量 B（Eq.24c），
//       用 GMRES（对角预处理，相对收敛判据 10⁻¹⁴）求解 A·X = B（Scheme II）
//   A4: 展布 Lagrangian 还原力到 Eulerian 网格（Eq.16）
//   A5: 速度更新由调用方执行
//
// 与原 Richardson 迭代不同，本实现精确求解 N_l×N_l 线性系统，可将无滑移
// 边界速度误差降至机器精度（2025 JCP Fig.3d），同时保持力和力矩守恒（Table 1）。
//
// @param fluid          Eulerian 流体网格（fluid.force 将被写入 IBM 体力）
// @param ms             Lagrangian 标记点集（mk.fx/fy 写入 Lagrangian 还原力）
// @param dx             格子间距
// @param dt             时间步长（格子单位通常 = 1）
// @param gmres_max_iter GMRES 最大迭代次数（原 n_iter 参数；默认 3 保留后向兼容）
//                       注：对于机器精度结果，建议设 ≥ ms.size()（至少 50）；
//                       对小 N_l（≤ 50）N 步内即可精确求解
// @param u_target_x/y   边界目标速度（静止固体取 0；移动固体取壁面速度）
// ===========================================================================
void compute_ibm_forces_mls_implicit(lbm::LatticeGrid& fluid,
                                      MarkerSet& ms,
                                      double dx,
                                      double dt             = 1.0,
                                      int    gmres_max_iter = 3,
                                      double u_target_x     = 0.0,
                                      double u_target_y     = 0.0);

// ===========================================================================
// 隐式 MLS-IBM 力计算（Algorithm 3，Scheme I：固定物体直接矩阵求逆）
//
// 参考：2025 JCP Wu & Fu §4，Algorithm 3，Scheme I
//
// 对于几何固定（stationary）的物体，传递算子 Φ（phi_cache）和相关矩阵 A 不随时间变化。
// 本函数将 LU 分解结果缓存于 A_lu_cache / piv_cache，将 MLS 形状函数集缓存于
// phi_cache（调用方持久化），后续每步仅需：
//   1. 用 phi_cache 插值速度（O(N_l·N_e)）
//   2. 构建右端向量 B（O(N_l)）
//   3. LU 代换求解（O(N_l²)）
// 与 Scheme II（每步 GMRES）相比，消除了 phi_data 重建和矩阵构建两处重复开销。
//
// 用法示例（在时间循环外声明缓存，在循环内每步调用）：
// @code
//   std::vector<double>           A_lu_cache;
//   std::vector<int>              piv_cache;
//   std::vector<ibm::MlsSupportSet> phi_cache;
//   for (int step = 0; step < n_steps; ++step) {
//       solver.step();
//       compute_ibm_forces_mls_implicit_stationary(
//           fluid, ms, dx, dt, A_lu_cache, piv_cache, phi_cache);
//   }
// @endcode
//
// @param fluid        Eulerian 流体网格
// @param ms           Lagrangian 标记点集（必须固定不动；位置每步不变）
// @param dx           格子间距
// @param dt           时间步长
// @param A_lu_cache   LU 分解缓存（首次调用时填充，后续复用；传入空 vector 即自动初始化）
// @param piv_cache    LU 行主元缓存（与 A_lu_cache 配套）
// @param phi_cache    MLS 形状函数集缓存（首次调用时填充，后续复用；传入空 vector 即自动初始化）
// @param u_target_x/y 目标速度（静止固体通常为 0.0）
// ===========================================================================
void compute_ibm_forces_mls_implicit_stationary(lbm::LatticeGrid& fluid,
                                                 MarkerSet& ms,
                                                 double dx,
                                                 double dt,
                                                 std::vector<double>& A_lu_cache,
                                                 std::vector<int>&    piv_cache,
                                                 std::vector<MlsSupportSet>& phi_cache,
                                                 double u_target_x = 0.0,
                                                 double u_target_y = 0.0);

// ===========================================================================
// 隐式速度校正 IBM（IVC-IBM，Implicit Velocity Correction）
//
// 参考：Wu J. & Shu C. (2009) J. Comput. Phys. 228:1963–1979
//   "Implicit velocity correction-based immersed boundary-lattice Boltzmann
//    method and its applications"
//
// 核心思想（Wu & Shu 2009 §3）：
//   在 Guo 体力 LBM 框架下，流体速度可分解为中间速度 u* 与速度修正 δu：
//     ρu  = Σ_α e_α f_α + (1/2) f dt   （Eq.17）
//     ρu* = Σ_α e_α f_α                （Eq.18，中间速度）
//     ρδu = (1/2) f dt                  （Eq.19，速度修正量）
//     u   = u* + δu                     （Eq.20）
//   以边界点速度修正 δu_B^l 为未知量，通过 Peskin δ 展布与插值封闭方程组，
//   强制满足非滑移边界条件 u(X_B^l) = U_B^l。
//
// 方程组（AX=B，Eq.28–29）：
//   A_{lk} = Σ_{i,j} D_ij^l · D_ij^k · Δs_k · Δx²
//   B^l    = U_B^l − Σ_{i,j} u*(x_ij) · D_ij^l · Δx²
//   X      = {δu_B^1, δu_B^2, …, δu_B^m}   （m 为边界点数）
//   其中 D_ij^l = d(x_ij − X_B^l) · d(y_ij − X_B^l)（2D Peskin δ 核）
//
// 力密度（Eq.30）：
//   f(x_ij) = (2ρ/dt) · δu(x_ij)  = (2ρ/dt) · Σ_l δu_B^l · D_ij^l · Δs_l
//   f_B^l   = (2ρ/dt) · δu_B^l    （Lagrangian 力密度，供 FSI 合力计算）
//
// 求解步骤（algorithm §3 overview）：
//   (1) 初始化：计算矩阵 A 并 LU 分解（固定物体只需做一次）。
//   (2) LBM 步：collide+stream → 中间速度 u*。
//   (3) 插值：u*(X_B^l) = Σ_{i,j} u*(x_ij) D_ij^l Δx²。
//   (4) 构建 B：B^l = U_target^l − u*(X_B^l)。
//   (5) 求解：A · δu_B = B（LU 代换，x/y 分量独立求解）。
//   (6) 展布：f(x_ij) += (2/dt) · Σ_l δu_B^l · D_ij^l · Δs_l。
//   (7) 写入 Lagrangian 力：f_B^l = (2/dt) · δu_B^l（供 compute_ibm_body_force）。
//
// 与其他方法的关系：
//   ─ MDF-IBM：迭代修正，只近似满足非滑移，每步 O(n_iter · N_l)
//   ─ IVC-IBM：精确满足非滑移，每步 O(N_l²)（LU 代换），固定物体更快
//   ─ MLS-IBM（Scheme I/II）：使用 MLS 形状函数而非 Peskin δ，2025 JCP 精度更高
//
// @param fluid        Eulerian 流体网格（fluid.force 写入 IBM 体力）
// @param ms           Lagrangian 标记点集（mk.fx/fy 写入 Lagrangian 力密度）
// @param dx           格子间距（通常 = 1.0）
// @param dt           时间步长（通常 = 1.0）
// @param kernel       δ 核类型（默认 FourPoint，对应论文 Eq.22）
// @param u_target_x/y 边界目标速度（静止固体取 0；移动固体取壁面速度）
// ===========================================================================
void compute_ibm_forces_ivc(lbm::LatticeGrid& fluid,
                              MarkerSet& ms,
                              double dx,
                              double dt             = 1.0,
                              DeltaKernel kernel    = DeltaKernel::FourPoint);

// ===========================================================================
// 隐式速度校正 IBM（IVC-IBM）— 固定物体优化版（LU 缓存）
//
// 参考：Wu & Shu (2009) §3，算法步骤 (1)
//
// 对于几何固定（stationary）的浸入边界，矩阵 A 只取决于边界点位置和 δ 核，
// 与流场无关，因此只需在首次调用时构建 A 并完成 LU 分解，后续每步仅执行：
//   插值 u*(X_B^l) → 构建 B → LU 代换 → 展布力
// 从而节省 O(N_l² · N_e) 的矩阵构建开销（N_e ≈ 16 为每个标记的支撑 Euler 节点数）。
//
// 用法示例（在时间循环外声明缓存，在循环内每步调用）：
// @code
//   std::vector<double> A_lu_cache;
//   std::vector<int>    piv_cache;
//   for (int step = 0; step < n_steps; ++step) {
//       solver.step();
//       compute_ibm_forces_ivc_stationary(fluid, ms, 1.0, 1.0,
//                                          A_lu_cache, piv_cache);
//   }
// @endcode
//
// @param A_lu_cache  LU 分解缓存（首次调用时填充；传入空 vector 自动初始化）
// @param piv_cache   行主元缓存（与 A_lu_cache 配套）
// ===========================================================================
void compute_ibm_forces_ivc_stationary(lbm::LatticeGrid& fluid,
                                        MarkerSet& ms,
                                        double dx,
                                        double dt,
                                        std::vector<double>& A_lu_cache,
                                        std::vector<int>&    piv_cache,
                                        DeltaKernel kernel    = DeltaKernel::FourPoint);

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

// ===========================================================================
// MPI 分区适配：坐标系转换 + 归属边界设置
//
// 将标记点坐标从 **全局格子坐标系** 转换到本进程的 **本地格子坐标系**，
// 并在 MarkerSet 中记录本进程"拥有"的行/列范围：
//
//   offset_x = phys_x0 - x_start
//   offset_y = phys_y0 - y_start
//   mk.x  += offset_x,  mk.x0 += offset_x
//   mk.y  += offset_y,  mk.y0 += offset_y
//   ms.owner_i_lo = phys_x0,   ms.owner_i_hi = phys_x0 + local_nx
//   ms.owner_j_lo = phys_y0,   ms.owner_j_hi = phys_y0 + local_ny
//
// 此后 interpolate_velocity / spread_force / mls_interpolate_velocity 仅
// 处理标记中心落入 [owner_i_lo,owner_i_hi) × [owner_j_lo,owner_j_hi) 的点，
// 防止跨块重复计算导致的 IBM 力双重计数。
//
// 对于近分区边界的"归属"标记点，其 δ 核支撑域可跨入幽灵行；幽灵行速度和
// 力分量由 ibm_halo_exchange_u_2d / ibm_halo_reduce_force_2d 正确同步。
//
// 【注意】此函数只应在 MPI 初始化并获知分区信息后调用一次；
//         非 MPI 或单进程模式下无需调用（默认 owner 范围涵盖全域）。
//
// @param ms        标记点集（将被原地修改：坐标偏移 + owner 范围设置）
// @param x_start   本分区物理域在全局 X 方向的起始格点索引
// @param y_start   本分区物理域在全局 Y 方向的起始格点索引
// @param phys_x0   本地网格中物理列的起始列索引（含幽灵列时 ≥ 1）
// @param phys_y0   本地网格中物理行的起始行索引（含幽灵行时 ≥ 1）
// @param local_nx  本分区物理列数
// @param local_ny  本分区物理行数
// ===========================================================================
void ibm_marker_set_adapt_to_partition(MarkerSet& ms,
                                        int x_start, int y_start,
                                        int phys_x0, int phys_y0,
                                        int local_nx, int local_ny);

// ===========================================================================
// MPI 幽灵层 u 场交换（IBM 插值前调用）
//
// solver.step() 中 PUSH 流式迁移会覆盖幽灵行的 f 值，导致 compute_macroscopic()
// 后幽灵行 u 来自本地边界行外推，而非邻居实际速度。
// 本函数在 step_ibm() 之前显式交换 u 场幽灵行，使 interpolate_velocity()
// 能读到邻居进程的真实速度。
//
// @param grid    本地流体网格（含幽灵层；u 场将被更新）
// @param decomp  二维 MPI 域分解描述符
// ===========================================================================
void ibm_halo_exchange_u_2d(lbm::LatticeGrid& grid,
                              const lbm::MpiDecomp2D& decomp);

// ===========================================================================
// MPI 幽灵层力场归并（IBM 力展布后调用）
//
// spread_force() 可能向幽灵行写入力贡献，这些贡献属于邻居进程物理行的一部分。
// 本函数通过 MPI_Sendrecv 将幽灵行力贡献发回各自的邻居物理行并累加，
// 然后清零本地幽灵行，确保跨 MPI 边界的 IBM 力展布物理上完整。
//
// @param grid    本地流体网格（含幽灵层；force 场将被修改）
// @param decomp  二维 MPI 域分解描述符
// ===========================================================================
void ibm_halo_reduce_force_2d(lbm::LatticeGrid& grid,
                                const lbm::MpiDecomp2D& decomp);

} // namespace ibm
