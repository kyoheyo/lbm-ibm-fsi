#include "ibm/interpolation.hpp"
#include <cmath>
#include <stdexcept>

#ifdef LBM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace ibm {

static constexpr double PI = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// 1-D kernel functions
// ---------------------------------------------------------------------------
double delta_phi(double r, double h, DeltaKernel kernel)
{
    const double roh = r / h;   // r/h
    if (kernel == DeltaKernel::TwoPoint) {
        // Linear (hat) kernel — support width 2h
        const double absr = std::abs(roh);
        if (absr < 1.0) return (1.0 - absr) / h;
        return 0.0;
    } else {
        // Peskin 4-point cosine kernel — support width 4h
        const double absr = std::abs(roh);
        if (absr < 2.0) {
            return (1.0 + std::cos(PI * roh / 2.0)) / (4.0 * h);
        }
        return 0.0;
    }
}

// ---------------------------------------------------------------------------
// Velocity interpolation (2-D, D2Q9 lattice assumed)
// ---------------------------------------------------------------------------
void interpolate_velocity(const lbm::LatticeGrid& grid,
                          MarkerSet& ms,
                          double dx,
                          DeltaKernel kernel)
{
    if (grid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("IBM interpolation: only D2Q9 supported currently");
    }

    const int nx = grid.nx;
    const int ny = grid.ny;

    // Support radius in grid cells
    const int support = (kernel == DeltaKernel::TwoPoint) ? 1 : 2;

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
    for (int m = 0; m < ms.size(); ++m) {
        auto& mk = ms.markers[m];
        // Marker position in grid units
        const double xm = mk.x / dx;
        const double ym = mk.y / dx;

        // Nearest grid node
        const int i0 = static_cast<int>(std::floor(xm));
        const int j0 = static_cast<int>(std::floor(ym));

        double ux_sum = 0.0, uy_sum = 0.0;

        for (int dj = -support; dj <= support + 1; ++dj) {
            for (int di = -support; di <= support + 1; ++di) {
                const int ii = i0 + di;
                const int jj = j0 + dj;
                // Periodic clamp
                if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;

                const int node = grid.idx(ii, jj);
                const double phi_x = delta_phi(mk.x - ii * dx, dx, kernel);
                const double phi_y = delta_phi(mk.y - jj * dx, dx, kernel);
                const double phi = phi_x * phi_y * dx * dx;

                ux_sum += grid.u[node * 2 + 0] * phi;
                uy_sum += grid.u[node * 2 + 1] * phi;
            }
        }

        mk.ux = ux_sum;
        mk.uy = uy_sum;
        mk.uz = 0.0;
    }
}

// ---------------------------------------------------------------------------
// Force spreading (2-D, D2Q9)
// ---------------------------------------------------------------------------
void spread_force(lbm::LatticeGrid& grid,
                  const MarkerSet& ms,
                  double dx,
                  DeltaKernel kernel)
{
    if (grid.model != lbm::LatticeModel::D2Q9) {
        throw std::runtime_error("IBM spread_force: only D2Q9 supported currently");
    }

    const int nx = grid.nx;
    const int ny = grid.ny;
    const int support = (kernel == DeltaKernel::TwoPoint) ? 1 : 2;

    // Zero the force field first
    std::fill(grid.force.begin(), grid.force.end(), 0.0);

    // Spreading is a scatter operation — need atomic or serialised loop
    for (int m = 0; m < ms.size(); ++m) {
        const auto& mk = ms.markers[m];
        const double xm = mk.x / dx;
        const double ym = mk.y / dx;

        const int i0 = static_cast<int>(std::floor(xm));
        const int j0 = static_cast<int>(std::floor(ym));

        for (int dj = -support; dj <= support + 1; ++dj) {
            for (int di = -support; di <= support + 1; ++di) {
                const int ii = i0 + di;
                const int jj = j0 + dj;
                if (ii < 0 || ii >= nx || jj < 0 || jj >= ny) continue;

                const int node = grid.idx(ii, jj);
                const double phi_x = delta_phi(mk.x - ii * dx, dx, kernel);
                const double phi_y = delta_phi(mk.y - jj * dx, dx, kernel);
                const double phi = phi_x * phi_y * mk.ds;

#ifdef LBM_ENABLE_OPENMP
#pragma omp atomic
#endif
                grid.force[node * 2 + 0] += mk.fx * phi;
#ifdef LBM_ENABLE_OPENMP
#pragma omp atomic
#endif
                grid.force[node * 2 + 1] += mk.fy * phi;
            }
        }
    }
}

} // namespace ibm
