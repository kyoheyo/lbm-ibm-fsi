#pragma once
#include "lattice.hpp"

namespace lbm {

// ---------------------------------------------------------------------------
// Boundary condition types
// ---------------------------------------------------------------------------
enum class BCType {
    BounceBack,     ///< No-slip wall (simple bounce-back)
    ZouHe_Velocity, ///< Zou-He velocity inlet/outlet
    ZouHe_Pressure, ///< Zou-He pressure inlet/outlet
    Periodic,       ///< Handled implicitly during streaming
};

// ---------------------------------------------------------------------------
// Face enum — which face of the rectangular domain
// ---------------------------------------------------------------------------
enum class Face { West, East, South, North, Bottom, Top };

// ---------------------------------------------------------------------------
// BoundaryCondition descriptor
// ---------------------------------------------------------------------------
struct BoundaryCondition {
    BCType type;
    Face   face;

    /// Prescribed velocity (ZouHe_Velocity)
    double ux = 0.0;
    double uy = 0.0;
    double uz = 0.0;

    /// Prescribed density (ZouHe_Pressure)
    double rho = 1.0;
};

// ---------------------------------------------------------------------------
// Apply all registered boundary conditions to the grid
// ---------------------------------------------------------------------------
void apply_boundary_conditions(LatticeGrid& grid,
                                const std::vector<BoundaryCondition>& bcs);

} // namespace lbm
