#include "lbm/lattice.hpp"
#include <cmath>
#include <stdexcept>

#ifdef LBM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace lbm {

// ---------------------------------------------------------------------------
LatticeGrid::LatticeGrid(int nx_, int ny_, int nz_, LatticeModel model_)
    : nx(nx_), ny(ny_), nz(nz_), model(model_)
{
    q = (model == LatticeModel::D2Q9) ? d2q9::Q : d3q19::Q;
    const int n = size();
    const int d = dim();
    f.assign(n * q, 0.0);
    f_tmp.assign(n * q, 0.0);
    rho.assign(n, 1.0);
    u.assign(n * d, 0.0);
    force.assign(n * d, 0.0);
}

// ---------------------------------------------------------------------------
void LatticeGrid::compute_macroscopic()
{
    const int n = size();
    const int d = dim();

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        double r = 0.0;
        double ux_loc = 0.0;
        double uy_loc = 0.0;
        double uz_loc = 0.0;

        if (model == LatticeModel::D2Q9) {
            for (int a = 0; a < d2q9::Q; ++a) {
                const double fa = f[i * d2q9::Q + a];
                r      += fa;
                ux_loc += fa * d2q9::C[a][0];
                uy_loc += fa * d2q9::C[a][1];
            }
            rho[i]         = r;
            u[i * d + 0]   = ux_loc / r;
            u[i * d + 1]   = uy_loc / r;
        } else {
            for (int a = 0; a < d3q19::Q; ++a) {
                const double fa = f[i * d3q19::Q + a];
                r      += fa;
                ux_loc += fa * d3q19::C[a][0];
                uy_loc += fa * d3q19::C[a][1];
                uz_loc += fa * d3q19::C[a][2];
            }
            rho[i]         = r;
            u[i * d + 0]   = ux_loc / r;
            u[i * d + 1]   = uy_loc / r;
            u[i * d + 2]   = uz_loc / r;
        }
    }
}

// ---------------------------------------------------------------------------
double f_eq(double w, double rho,
            const double* c,
            const double* u_vec,
            int d)
{
    // f_eq = w * rho * [1 + (c·u)/cs² + (c·u)²/(2cs⁴) - u²/(2cs²)]
    // cs² = 1/3 for standard lattice
    constexpr double cs2 = 1.0 / 3.0;
    constexpr double cs4 = cs2 * cs2;

    double cu = 0.0;
    double u2 = 0.0;
    for (int k = 0; k < d; ++k) {
        cu += c[k] * u_vec[k];
        u2 += u_vec[k] * u_vec[k];
    }
    return w * rho * (1.0 + cu / cs2 + cu * cu / (2.0 * cs4) - u2 / (2.0 * cs2));
}

} // namespace lbm
