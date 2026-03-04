#include "fsi/structure.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace fsi {

// ---------------------------------------------------------------------------
BeamSolver::BeamSolver(const BeamParams& params)
    : params_(params)
{
    const int n = n_nodes();
    dofs_.resize(n);

    // Set reference positions along x-axis
    const double dl = params_.length / params_.n_elements;
    for (int i = 0; i < n; ++i) {
        dofs_[i].x = dofs_[i].x0 = i * dl;
        dofs_[i].y = dofs_[i].y0 = 0.0;
        dofs_[i].theta = 0.0;
        dofs_[i].vx    = 0.0;
        dofs_[i].vy    = 0.0;
        dofs_[i].omega = 0.0;
        dofs_[i].fx    = 0.0;
        dofs_[i].fy    = 0.0;
        dofs_[i].m     = 0.0;
    }

    assemble_mass();
    assemble_stiffness();
}

// ---------------------------------------------------------------------------
// Lumped mass matrix:  m_i = ρ_s * A * l_element / 2  (half to each node)
// ---------------------------------------------------------------------------
void BeamSolver::assemble_mass()
{
    const int n = n_nodes();
    const double dl = params_.length / params_.n_elements;
    const double m_node = params_.rho_s * params_.A * dl;

    mass_diag_.assign(n * 3, 0.0);   // 3 DOF per node: x, y, theta
    for (int i = 0; i < n; ++i) {
        mass_diag_[i * 3 + 0] = m_node;   // translational x
        mass_diag_[i * 3 + 1] = m_node;   // translational y
        // Rotary inertia: I_rot ≈ ρ_s * I * dl  (per element, halved to each node)
        mass_diag_[i * 3 + 2] = params_.rho_s * params_.I * dl;
    }
}

// ---------------------------------------------------------------------------
// Global stiffness matrix for Euler-Bernoulli beam (dense, size 3n x 3n)
// Only bending (EI) terms are assembled; axial (EA) terms are omitted for
// an inextensible beam assumption.
// ---------------------------------------------------------------------------
void BeamSolver::assemble_stiffness()
{
    const int n = n_nodes();
    const int dof = 3 * n;
    K_.assign(dof * dof, 0.0);

    const double dl = params_.length / params_.n_elements;
    const double EI = params_.E * params_.I;
    const double l  = dl;
    const double l2 = l * l;
    const double l3 = l2 * l;

    // Local 4x4 Euler-Bernoulli bending stiffness (v, θ per node)
    // Mapped to DOFs [y_i, theta_i, y_j, theta_j]
    const double ke[4][4] = {
        { 12*EI/l3,  6*EI/l2, -12*EI/l3,  6*EI/l2},
        {  6*EI/l2,  4*EI/l,   -6*EI/l2,  2*EI/l },
        {-12*EI/l3, -6*EI/l2,  12*EI/l3, -6*EI/l2},
        {  6*EI/l2,  2*EI/l,   -6*EI/l2,  4*EI/l },
    };

    for (int e = 0; e < params_.n_elements; ++e) {
        // Local DOF indices: [y_i, theta_i, y_j, theta_j]
        const int local_dof[4] = {
            e * 3 + 1,    // y_i
            e * 3 + 2,    // theta_i
            (e+1)*3 + 1,  // y_j
            (e+1)*3 + 2,  // theta_j
        };

        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                K_[local_dof[r] * dof + local_dof[c]] += ke[r][c];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Newmark-β time integrator  (β=0.25, γ=0.5 — average acceleration, stable)
// Solves: M*a_{n+1} + K*d_{n+1} = F_{n+1}
// where: d_{n+1} = d_n + dt*v_n + dt²[(0.5-β)*a_n + β*a_{n+1}]
//        v_{n+1} = v_n + dt[(1-γ)*a_n + γ*a_{n+1}]
// ---------------------------------------------------------------------------
void BeamSolver::step(double dt)
{
    const int n  = n_nodes();
    const int dof = 3 * n;

    constexpr double beta  = 0.25;
    constexpr double gamma = 0.5;

    // Gather current state into flat vectors
    std::vector<double> d(dof), v(dof), a(dof), F(dof);
    for (int i = 0; i < n; ++i) {
        d[i*3+0] = dofs_[i].x;
        d[i*3+1] = dofs_[i].y;
        d[i*3+2] = dofs_[i].theta;
        v[i*3+0] = dofs_[i].vx;
        v[i*3+1] = dofs_[i].vy;
        v[i*3+2] = dofs_[i].omega;
        a[i*3+0] = 0.0;  // acceleration from previous step (simplified)
        a[i*3+1] = 0.0;
        a[i*3+2] = 0.0;
        F[i*3+0] = dofs_[i].fx;
        F[i*3+1] = dofs_[i].fy;
        F[i*3+2] = dofs_[i].m;
    }

    // Effective stiffness: K_eff = M/(β*dt²) + K
    // RHS: F_eff = F + M/β/dt² * (d + dt*v + dt²*(0.5-β)*a)
    std::vector<double> K_eff(dof * dof);
    for (int r = 0; r < dof; ++r) {
        for (int c = 0; c < dof; ++c) {
            K_eff[r * dof + c] = K_[r * dof + c];
        }
        // Add mass contribution to diagonal
        K_eff[r * dof + r] += mass_diag_[r] / (beta * dt * dt);
    }

    std::vector<double> d_pred(dof);
    std::vector<double> rhs(dof);
    for (int i = 0; i < dof; ++i) {
        d_pred[i] = d[i] + dt * v[i] + dt * dt * (0.5 - beta) * a[i];
        rhs[i] = F[i] + mass_diag_[i] / (beta * dt * dt) * d_pred[i];
    }

    // Solve K_eff * d_{n+1} = rhs  — dense Gaussian elimination
    std::vector<double> d_new(dof);
    solve_newmark(K_eff, rhs, d_new);

    // Update accelerations and velocities
    std::vector<double> a_new(dof), v_new(dof);
    for (int i = 0; i < dof; ++i) {
        a_new[i] = (d_new[i] - d_pred[i]) / (beta * dt * dt);
        v_new[i] = v[i] + dt * ((1.0 - gamma) * a[i] + gamma * a_new[i]);
    }

    // Scatter back
    for (int i = 0; i < n; ++i) {
        dofs_[i].x     = d_new[i*3+0];
        dofs_[i].y     = d_new[i*3+1];
        dofs_[i].theta = d_new[i*3+2];
        dofs_[i].vx    = v_new[i*3+0];
        dofs_[i].vy    = v_new[i*3+1];
        dofs_[i].omega = v_new[i*3+2];
    }
}

// ---------------------------------------------------------------------------
// Dense LU solve:  K_eff * x = rhs
// ---------------------------------------------------------------------------
void BeamSolver::solve_newmark(const std::vector<double>& K_eff,
                                const std::vector<double>& rhs,
                                std::vector<double>& x)
{
    const int dof = static_cast<int>(rhs.size());

    // Working copy of K_eff
    std::vector<double> A(K_eff);

    // Copy rhs into x
    x = rhs;

    // Gaussian elimination with partial pivoting
    for (int k = 0; k < dof; ++k) {
        // Find pivot
        int max_row = k;
        double max_val = std::abs(A[k * dof + k]);
        for (int i = k + 1; i < dof; ++i) {
            if (std::abs(A[i * dof + k]) > max_val) {
                max_val = std::abs(A[i * dof + k]);
                max_row = i;
            }
        }
        if (max_row != k) {
            std::swap(x[k], x[max_row]);
            for (int j = 0; j < dof; ++j) {
                std::swap(A[k * dof + j], A[max_row * dof + j]);
            }
        }

        if (std::abs(A[k * dof + k]) < 1e-15) {
            // Singular or near-singular — set displacement to zero.
            // NOTE: This implementation only supports homogeneous (zero) constraints.
            // For non-zero prescribed displacements, apply the constraint via
            // the penalty method or explicit elimination before calling this solver.
            x[k] = 0.0;
            continue;
        }

        for (int i = k + 1; i < dof; ++i) {
            const double factor = A[i * dof + k] / A[k * dof + k];
            x[i] -= factor * x[k];
            for (int j = k; j < dof; ++j) {
                A[i * dof + j] -= factor * A[k * dof + j];
            }
        }
    }

    // Back substitution
    for (int k = dof - 1; k >= 0; --k) {
        if (std::abs(A[k * dof + k]) < 1e-15) continue;
        for (int j = k + 1; j < dof; ++j) {
            x[k] -= A[k * dof + j] * x[j];
        }
        x[k] /= A[k * dof + k];
    }
}

} // namespace fsi
