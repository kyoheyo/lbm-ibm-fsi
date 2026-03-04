#pragma once
#include "marker.hpp"
#include "../lbm/lattice.hpp"

namespace ibm {

// ---------------------------------------------------------------------------
// Peskin's regularised delta function  φ(r) = (1/h)*φ̂(r/h)
// Supported kernels:
//   2-point (linear) — width 2h
//   4-point (Peskin) — width 4h
// ---------------------------------------------------------------------------
enum class DeltaKernel { TwoPoint, FourPoint };

// ---------------------------------------------------------------------------
// Velocity interpolation:
//   u_IB(X) = Σ_{x} u(x) δ(x - X) Δx^dim
// Maps Eulerian fluid velocity → Lagrangian marker velocity.
// ---------------------------------------------------------------------------
void interpolate_velocity(const lbm::LatticeGrid& grid,
                          MarkerSet& ms,
                          double dx,
                          DeltaKernel kernel = DeltaKernel::FourPoint);

// ---------------------------------------------------------------------------
// Force spreading:
//   f(x) = Σ_{X} F(X) δ(x - X) ΔS
// Spreads Lagrangian IBM force density → Eulerian body force field.
// ---------------------------------------------------------------------------
void spread_force(lbm::LatticeGrid& grid,
                  const MarkerSet& ms,
                  double dx,
                  DeltaKernel kernel = DeltaKernel::FourPoint);

// ---------------------------------------------------------------------------
// 1-D delta-function kernel value
// ---------------------------------------------------------------------------
double delta_phi(double r, double h, DeltaKernel kernel);

} // namespace ibm
