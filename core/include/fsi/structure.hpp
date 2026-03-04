#pragma once
#include <vector>

namespace fsi {

// ---------------------------------------------------------------------------
// Material parameters for a flexible Euler-Bernoulli beam
// ---------------------------------------------------------------------------
struct BeamParams {
    double E;           ///< Young's modulus
    double I;           ///< Second moment of area
    double rho_s;       ///< Structural density
    double A;           ///< Cross-sectional area
    double length;      ///< Rest length
    int    n_elements;  ///< Number of beam elements
};

// ---------------------------------------------------------------------------
// Degrees of freedom for a single beam node: (x, y, theta)
// ---------------------------------------------------------------------------
struct BeamDOF {
    double x, y;     ///< Current position
    double x0, y0;   ///< Reference (undeformed) position
    double theta;    ///< Rotation angle
    double vx, vy;   ///< Velocity
    double omega;    ///< Angular velocity
    double fx, fy;   ///< External force (from IBM)
    double m;        ///< External moment
};

// ---------------------------------------------------------------------------
// Euler-Bernoulli beam FEM structural solver
//   Uses a Newmark-β time integrator (β=0.25, γ=0.5 — unconditionally stable)
// ---------------------------------------------------------------------------
class BeamSolver {
public:
    explicit BeamSolver(const BeamParams& params);

    /// Advance the beam one time step dt under the current external forces
    void step(double dt);

    /// Access nodal DOFs
    [[nodiscard]] const std::vector<BeamDOF>& dofs() const { return dofs_; }
    [[nodiscard]]       std::vector<BeamDOF>& dofs()       { return dofs_; }

    /// Number of nodes  = n_elements + 1
    [[nodiscard]] int n_nodes() const {
        return params_.n_elements + 1;
    }

private:
    BeamParams           params_;
    std::vector<BeamDOF> dofs_;

    /// Global mass matrix (lumped diagonal, stored as vector)
    std::vector<double> mass_diag_;
    /// Global stiffness matrix (sparse CSR — stored dense for small beams)
    std::vector<double> K_;

    void assemble_mass();
    void assemble_stiffness();

    /// Solve  K_eff * x = rhs  via dense Gaussian elimination with partial pivoting
    void solve_newmark(const std::vector<double>& K_eff,
                       const std::vector<double>& rhs,
                       std::vector<double>& x);
};

} // namespace fsi
