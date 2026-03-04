#include "lbm/boundary.hpp"
#include "lbm/lattice.hpp"
#include <stdexcept>

#ifdef LBM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace lbm {

// ---------------------------------------------------------------------------
// Simple bounce-back on a face
// ---------------------------------------------------------------------------
static void apply_bounce_back(LatticeGrid& g, Face face)
{
    if (g.model != LatticeModel::D2Q9) {
        // 3-D bounce-back omitted for brevity — extend similarly
        return;
    }

    const int nx = g.nx;
    const int ny = g.ny;

    auto bb_node = [&](int i, int j) {
        const int n = g.idx(i, j);
        for (int a = 1; a < d2q9::Q; ++a) {
            // Reverse the direction
            g.f[n * d2q9::Q + a] = g.f[n * d2q9::Q + d2q9::OPP[a]];
        }
    };

    switch (face) {
        case Face::West:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 0; j < ny; ++j) bb_node(0, j);
            break;
        case Face::East:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 0; j < ny; ++j) bb_node(nx - 1, j);
            break;
        case Face::South:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nx; ++i) bb_node(i, 0);
            break;
        case Face::North:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nx; ++i) bb_node(i, ny - 1);
            break;
        default:
            break;
    }
}

// ---------------------------------------------------------------------------
// Zou-He velocity boundary condition (2-D, North face example)
// Reference: Zou & He, Phys. Fluids 9(6), 1997
// ---------------------------------------------------------------------------
static void apply_zou_he_velocity(LatticeGrid& g, const BoundaryCondition& bc)
{
    if (g.model != LatticeModel::D2Q9) return;

    const int nx = g.nx;
    const int ny = g.ny;

    if (bc.face == Face::North) {
        const double ux = bc.ux;
        const double uy = bc.uy;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < nx; ++i) {
            const int n = g.idx(i, ny - 1);
            double* f = &g.f[n * d2q9::Q];
            // Known incoming: f[4], f[7], f[8]
            // Unknown: f[2], f[5], f[6]
            double rho_w = (f[0] + f[1] + f[3]
                          + 2.0 * (f[4] + f[7] + f[8]))
                         / (1.0 + uy);
            f[2] = f[4] + (2.0 / 3.0) * rho_w * uy;
            f[5] = f[7] - 0.5 * (f[1] - f[3])
                        + (1.0 / 6.0) * rho_w * uy
                        + 0.5 * rho_w * ux;
            f[6] = f[8] + 0.5 * (f[1] - f[3])
                        + (1.0 / 6.0) * rho_w * uy
                        - 0.5 * rho_w * ux;
            g.rho[n] = rho_w;
            g.u[n * 2 + 0] = ux;
            g.u[n * 2 + 1] = uy;
        }
    }
    // Additional faces (West, East, South) follow the same pattern —
    // extend as needed for specific problem setups.
}

// ---------------------------------------------------------------------------
void apply_boundary_conditions(LatticeGrid& grid,
                                const std::vector<BoundaryCondition>& bcs)
{
    for (const auto& bc : bcs) {
        switch (bc.type) {
            case BCType::BounceBack:
                apply_bounce_back(grid, bc.face);
                break;
            case BCType::ZouHe_Velocity:
                apply_zou_he_velocity(grid, bc);
                break;
            case BCType::ZouHe_Pressure:
                // TODO: implement pressure BC
                break;
            case BCType::Periodic:
                // Handled by periodic wrap in streaming
                break;
        }
    }
}

} // namespace lbm
