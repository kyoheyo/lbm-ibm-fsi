#include "fsi/coupling.hpp"
#include <cstring>

namespace fsi {

// ---------------------------------------------------------------------------
void fsi_step(lbm::LatticeGrid& fluid,
              ibm::MarkerSet&   markers,
              BeamSolver&       beam,
              const CouplingParams& params)
{
    for (int iter = 0; iter < params.sub_iterations; ++iter) {
        // 1. Interpolate fluid velocity at Lagrangian markers
        ibm::interpolate_velocity(fluid, markers,
                                  params.dx, params.delta_kernel);

        // 2. Compute IBM direct-forcing:
        //    F_IB = rho * (U_IB_target - U_interp) / dt
        //    For rigid IBM: U_IB_target is the structure velocity
        //    For flexible IBM: it's the beam node velocity
        const auto& beam_dofs = beam.dofs();
        const int n_markers = markers.size();
        for (int m = 0; m < n_markers; ++m) {
            auto& mk = markers.markers[m];
            // Find nearest beam node (simple linear search; use kd-tree for large beams)
            double best_dist2 = 1e30;
            int    best_node  = 0;
            for (int k = 0; k < beam.n_nodes(); ++k) {
                const double dx = mk.x - beam_dofs[k].x;
                const double dy = mk.y - beam_dofs[k].y;
                const double d2 = dx * dx + dy * dy;
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_node  = k;
                }
            }

            const double target_ux = beam_dofs[best_node].vx;
            const double target_uy = beam_dofs[best_node].vy;

            // Penalty force (stiffness coefficient κ = 1/dt for direct forcing)
            const double kappa = 1.0 / params.dt;
            mk.fx = kappa * (target_ux - mk.ux);
            mk.fy = kappa * (target_uy - mk.uy);
        }

        // 3. Spread IBM force to Eulerian grid
        ibm::spread_force(fluid, markers, params.dx, params.delta_kernel);
    }

    // 4. Transfer IBM reaction force to beam (Newton's 3rd law)
    auto& beam_dofs = beam.dofs();
    // Zero structural forces first
    for (auto& dof : beam_dofs) {
        dof.fx = 0.0;
        dof.fy = 0.0;
        dof.m  = 0.0;
    }

    const int n_markers = markers.size();
    for (int m = 0; m < n_markers; ++m) {
        const auto& mk = markers.markers[m];
        // Find nearest beam node
        double best_dist2 = 1e30;
        int    best_node  = 0;
        for (int k = 0; k < beam.n_nodes(); ++k) {
            const double dx = mk.x - beam_dofs[k].x;
            const double dy = mk.y - beam_dofs[k].y;
            const double d2 = dx * dx + dy * dy;
            if (d2 < best_dist2) {
                best_dist2 = d2;
                best_node  = k;
            }
        }
        // Reaction force on structure: opposite sign, weighted by arc length
        beam_dofs[best_node].fx -= mk.fx * mk.ds;
        beam_dofs[best_node].fy -= mk.fy * mk.ds;
    }

    // 5. Advance structural solver
    beam.step(params.dt);

    // 6. Update marker positions from beam
    sync_markers_from_beam(markers, beam);
}

// ---------------------------------------------------------------------------
void sync_markers_from_beam(ibm::MarkerSet& markers,
                             const BeamSolver& beam)
{
    const auto& dofs = beam.dofs();
    const int n_markers = markers.size();
    const int n_nodes   = beam.n_nodes();

    for (int m = 0; m < n_markers; ++m) {
        auto& mk = markers.markers[m];

        // Interpolate marker position from beam nodes using linear shape functions
        // Find element containing this marker (based on reference coordinate)
        const double L = beam.dofs().back().x0 - beam.dofs().front().x0;
        if (L < 1e-15) {
            // Zero-length beam — cannot interpolate; leave markers unchanged
            continue;
        }

        // Normalised reference coordinate [0,1]
        const double xi0 = (mk.x0 - dofs[0].x0) / L;
        const double s   = xi0 * (n_nodes - 1);
        const int    e   = std::min(static_cast<int>(s), n_nodes - 2);
        const double t   = s - e;  // local coordinate in [0,1]

        mk.x = (1.0 - t) * dofs[e].x + t * dofs[e + 1].x;
        mk.y = (1.0 - t) * dofs[e].y + t * dofs[e + 1].y;
    }
}

} // namespace fsi
