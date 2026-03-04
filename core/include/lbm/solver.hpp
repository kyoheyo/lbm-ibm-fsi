#pragma once
#include "lattice.hpp"

namespace lbm {

// ---------------------------------------------------------------------------
// Collision operator selection
// ---------------------------------------------------------------------------
enum class CollisionModel {
    BGK,  ///< Single-relaxation-time (Bhatnagar-Gross-Krook)
    MRT,  ///< Multiple-relaxation-time (more stable at low viscosity)
};

// ---------------------------------------------------------------------------
// LBM Solver — owns a LatticeGrid and advances the simulation in time
// ---------------------------------------------------------------------------
class Solver {
public:
    Solver(LatticeGrid& grid, double omega, CollisionModel cm = CollisionModel::BGK);

    /// Single time step: collision + streaming + BC application
    void step();

    /// Collision step only (modifies f in place)
    void collide();

    /// Streaming step: propagate f → f_tmp, then swap
    void stream();

    double omega() const { return omega_; }
    void   set_omega(double w) { omega_ = w; }

private:
    LatticeGrid&   grid_;
    double         omega_;       ///< Relaxation frequency  ω = 1/τ
    CollisionModel cm_;

    void collide_bgk();
    void collide_mrt();

    // Guo's forcing scheme — adds body-force correction during collision
    void apply_guo_forcing(int node, const double* F, double* f_post);
};

} // namespace lbm
