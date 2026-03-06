#pragma once
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
// 速度插值：
//   u_IB(X) = Σ_{x} u(x) δ(x - X) Δx^dim
// 将欧拉流体速度映射到拉格朗日标记点速度。
// ---------------------------------------------------------------------------
void interpolate_velocity(const lbm::LatticeGrid& grid,
                          MarkerSet& ms,
                          double dx,
                          DeltaKernel kernel = DeltaKernel::FourPoint);

// ---------------------------------------------------------------------------
// 力展布：
//   f(x) = Σ_{X} F(X) δ(x - X) ΔS
// 将拉格朗日 IBM 力密度展布到欧拉体力场。
// ---------------------------------------------------------------------------
void spread_force(lbm::LatticeGrid& grid,
                  const MarkerSet& ms,
                  double dx,
                  DeltaKernel kernel = DeltaKernel::FourPoint);

// ---------------------------------------------------------------------------
// 一维 δ 函数核值
// ---------------------------------------------------------------------------
double delta_phi(double r, double h, DeltaKernel kernel);

} // namespace ibm
