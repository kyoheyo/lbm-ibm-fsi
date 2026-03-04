#pragma once
#include <array>
#include <vector>
#include <cstdint>

namespace lbm {

// ---------------------------------------------------------------------------
// Supported lattice models
// ---------------------------------------------------------------------------
enum class LatticeModel {
    D2Q9,   ///< 2-D, 9 velocities
    D3Q19,  ///< 3-D, 19 velocities
    D3Q27,  ///< 3-D, 27 velocities
};

// ---------------------------------------------------------------------------
// D2Q9 compile-time constants
// ---------------------------------------------------------------------------
namespace d2q9 {

constexpr int Q = 9;

/// Discrete velocity vectors  [direction][dim]
constexpr std::array<std::array<int, 2>, Q> C = {{
    { 0,  0},  // 0 — rest
    { 1,  0},  // 1 — E
    { 0,  1},  // 2 — N
    {-1,  0},  // 3 — W
    { 0, -1},  // 4 — S
    { 1,  1},  // 5 — NE
    {-1,  1},  // 6 — NW
    {-1, -1},  // 7 — SW
    { 1, -1},  // 8 — SE
}};

/// Lattice weights
constexpr std::array<double, Q> W = {{
    4.0 / 9.0,
    1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
}};

/// Opposite direction look-up  (bounce-back)
constexpr std::array<int, Q> OPP = {{0, 3, 4, 1, 2, 7, 8, 5, 6}};

} // namespace d2q9

// ---------------------------------------------------------------------------
// D3Q19 compile-time constants
// ---------------------------------------------------------------------------
namespace d3q19 {

constexpr int Q = 19;

constexpr std::array<std::array<int, 3>, Q> C = {{
    { 0,  0,  0},  // 0
    { 1,  0,  0},  // 1
    {-1,  0,  0},  // 2
    { 0,  1,  0},  // 3
    { 0, -1,  0},  // 4
    { 0,  0,  1},  // 5
    { 0,  0, -1},  // 6
    { 1,  1,  0},  // 7
    {-1, -1,  0},  // 8
    { 1, -1,  0},  // 9
    {-1,  1,  0},  // 10
    { 1,  0,  1},  // 11
    {-1,  0, -1},  // 12
    { 1,  0, -1},  // 13
    {-1,  0,  1},  // 14
    { 0,  1,  1},  // 15
    { 0, -1, -1},  // 16
    { 0,  1, -1},  // 17
    { 0, -1,  1},  // 18
}};

constexpr std::array<double, Q> W = {{
    1.0 / 3.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
    1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
}};

constexpr std::array<int, Q> OPP = {{
    0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17
}};

} // namespace d3q19

// ---------------------------------------------------------------------------
// Generic LatticeGrid — stores distribution functions f[idx][q]
// ---------------------------------------------------------------------------
struct LatticeGrid {
    int nx;       ///< Grid size in x
    int ny;       ///< Grid size in y
    int nz;       ///< Grid size in z (1 for 2-D)
    int q;        ///< Number of discrete velocities
    LatticeModel model;

    /// Flat storage: f[node_index * q + direction]
    std::vector<double> f;
    /// Temporary buffer for streaming step
    std::vector<double> f_tmp;

    /// Macroscopic density ρ at each node
    std::vector<double> rho;
    /// Macroscopic velocity u at each node [node_index * dim]
    std::vector<double> u;

    /// Body-force density at each node [node_index * dim]
    std::vector<double> force;

    LatticeGrid() = default;
    LatticeGrid(int nx, int ny, int nz, LatticeModel model);

    /// Total number of nodes
    [[nodiscard]] int size() const { return nx * ny * nz; }

    /// Flat node index
    [[nodiscard]] int idx(int i, int j, int k = 0) const {
        return i + nx * (j + ny * k);
    }

    /// Spatial dimension
    [[nodiscard]] int dim() const {
        return (model == LatticeModel::D2Q9) ? 2 : 3;
    }

    /// Compute macroscopic quantities from distribution functions
    void compute_macroscopic();
};

// ---------------------------------------------------------------------------
// Maxwell–Boltzmann equilibrium distribution
// ---------------------------------------------------------------------------
double f_eq(double w, double rho,
            const double* c,   ///< lattice velocity vector (dim)
            const double* u,   ///< fluid velocity (dim)
            int dim);

} // namespace lbm
