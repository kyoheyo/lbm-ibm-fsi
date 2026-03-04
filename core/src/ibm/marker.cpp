#include "ibm/marker.hpp"
#include <cmath>

namespace ibm {

static constexpr double PI = 3.14159265358979323846;

// ---------------------------------------------------------------------------
MarkerSet MarkerSet::make_circle(double cx, double cy,
                                 double radius, int n_markers)
{
    MarkerSet ms;
    ms.markers.resize(n_markers);
    const double dtheta = 2.0 * PI / n_markers;
    const double ds = radius * dtheta;   // arc-length element

    for (int k = 0; k < n_markers; ++k) {
        const double theta = k * dtheta;
        auto& m = ms.markers[k];
        m.x  = m.x0 = cx + radius * std::cos(theta);
        m.y  = m.y0 = cy + radius * std::sin(theta);
        m.z  = m.z0 = 0.0;
        m.fx = m.fy = m.fz = 0.0;
        m.ux = m.uy = m.uz = 0.0;
        m.ds = ds;
    }
    return ms;
}

// ---------------------------------------------------------------------------
MarkerSet MarkerSet::make_filament(double x0, double y0,
                                   double length, int n_markers)
{
    MarkerSet ms;
    ms.markers.resize(n_markers);
    const double dl = length / (n_markers - 1);

    for (int k = 0; k < n_markers; ++k) {
        auto& m = ms.markers[k];
        m.x  = m.x0 = x0 + k * dl;
        m.y  = m.y0 = y0;
        m.z  = m.z0 = 0.0;
        m.fx = m.fy = m.fz = 0.0;
        m.ux = m.uy = m.uz = 0.0;
        m.ds = dl;
    }
    return ms;
}

} // namespace ibm
