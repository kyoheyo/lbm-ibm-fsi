#pragma once
#include "structure.hpp"
#include "../ibm/marker.hpp"
#include "../ibm/interpolation.hpp"
#include "../lbm/lattice.hpp"

namespace fsi {

// ---------------------------------------------------------------------------
// FSI coupling parameters
// ---------------------------------------------------------------------------
struct CouplingParams {
    double dx;                    ///< Eulerian grid spacing
    double dt;                    ///< Time step
    ibm::DeltaKernel delta_kernel; ///< IBM delta-function kernel
    int    sub_iterations;        ///< Inner iterations for implicit coupling (1 = explicit)
};

// ---------------------------------------------------------------------------
// Partitioned (explicit) FSI coupling step
//
//  Given the current state of the fluid grid and beam, one FSI sub-cycle is:
//   1. Interpolate fluid velocity at IBM markers (velocity coupling)
//   2. Move markers to match the interpolated velocity
//   3. Compute IBM restoring force (penalty / direct-forcing)
//   4. Spread IBM force back onto the Eulerian grid
//   5. Advance the structural solver by dt
// ---------------------------------------------------------------------------
void fsi_step(lbm::LatticeGrid& fluid,
              ibm::MarkerSet&   markers,
              BeamSolver&       beam,
              const CouplingParams& params);

// ---------------------------------------------------------------------------
// Synchronise Lagrangian marker positions with the beam node positions
// (call after the structural solver has advanced)
// ---------------------------------------------------------------------------
void sync_markers_from_beam(ibm::MarkerSet& markers,
                             const BeamSolver& beam);

} // namespace fsi
