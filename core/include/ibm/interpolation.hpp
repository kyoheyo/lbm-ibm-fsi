#pragma once
// core/include/ibm/interpolation.hpp — IBM 速度插值与力展布
//
// 提供三种 IBM 方案：
//   1. 标准 Peskin δ 函数插值 + 直接力法（Uhlmann 2005）
//   2. 多重直接力法（MDF-IBM，Luo et al. 2007）：迭代修正，提高无滑移精度
//   3. 移动最小二乘插值（MLS-IBM）：高阶速度重构，适用于非均匀标记点分布

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

} // namespace ibm
