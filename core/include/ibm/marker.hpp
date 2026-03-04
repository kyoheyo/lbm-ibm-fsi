#pragma once
#include <vector>

namespace ibm {

// ---------------------------------------------------------------------------
// A Lagrangian marker point on the immersed boundary
// ---------------------------------------------------------------------------
struct Marker {
    double x, y, z;            ///< Current position
    double x0, y0, z0;         ///< Reference (undeformed) position
    double fx, fy, fz;         ///< IBM force density at this marker
    double ux, uy, uz;         ///< Interpolated fluid velocity at marker
    double ds;                 ///< Arc-length element (surface area in 3-D)
};

// ---------------------------------------------------------------------------
// A collection of markers forming one immersed boundary
// ---------------------------------------------------------------------------
struct MarkerSet {
    std::vector<Marker> markers;

    /// Total number of markers
    [[nodiscard]] int size() const {
        return static_cast<int>(markers.size());
    }

    /// Build a uniform circular ring of markers in 2-D
    static MarkerSet make_circle(double cx, double cy,
                                 double radius, int n_markers);

    /// Build a uniform straight filament along the x-axis in 2-D
    static MarkerSet make_filament(double x0, double y0,
                                   double length, int n_markers);
};

} // namespace ibm
