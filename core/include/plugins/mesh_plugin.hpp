#pragma once
/// @file plugins/mesh_plugin.hpp
/// @brief Extension interface for custom mesh handling (e.g. adaptive refinement).
///
/// ## How to add a new mesh handling method
///
/// The LBM core uses a uniform Cartesian Eulerian grid, so "mesh adaptation"
/// in this context means any algorithm that **modifies the grid topology,
/// node layout, or grid-spacing information** at run time.  Typical use-cases:
///
/// - Stretched / non-uniform lattice spacing
/// - Dynamic block refinement (locally refine cells near a boundary layer)
/// - Overset / Chimera grid coupling
///
/// ### Steps
/// 1. Subclass `IMeshPlugin`.
/// 2. Override `initialize()` to set up data structures before the loop.
/// 3. Override `adapt()` to evolve the mesh each step (or every N steps).
/// 4. Register:
///    ```cpp
///    MyMeshPlugin plugin;
///    PluginRegistry::instance().set_mesh_plugin(&plugin);
///    ```
///
/// ### Example skeleton
/// ```cpp
/// #include "plugins/mesh_plugin.hpp"
/// class StretchedZMesh : public lbm::IMeshPlugin {
/// public:
///     void initialize(lbm::LatticeGrid& grid, const void* /*cfg*/) override {
///         // Pre-compute stretched coordinates stored as user data
///     }
///     void adapt(lbm::LatticeGrid& grid, int step) override {
///         // Optionally re-balance after every 1000 steps
///         if (step % 1000 == 0) rebalance(grid);
///     }
///     const char* name() const override { return "stretched_z"; }
/// };
/// ```

#include "lbm/lattice.hpp"

namespace lbm {

/// Abstract interface for a custom mesh-refinement / mesh-handling plugin.
///
/// Implement this interface to add a new mesh partitioning strategy or
/// adaptive-refinement scheme without modifying the core solver.
class IMeshPlugin {
public:
    virtual ~IMeshPlugin() = default;

    /// One-time setup called **before** the time-integration loop begins.
    ///
    /// @param grid  The lattice grid (may be re-sized or restructured here).
    /// @param cfg   Optional pointer to plugin-specific configuration data.
    ///              Cast to a concrete type inside the implementation.
    virtual void initialize(LatticeGrid& grid, const void* cfg = nullptr) = 0;

    /// Per-step mesh adaptation hook, called **after** streaming and
    /// macroscopic update in `Solver::step()`.
    ///
    /// @param grid  The lattice grid.
    /// @param step  Current simulation step index (0-based).
    virtual void adapt(LatticeGrid& grid, int step) = 0;

    /// Short identifier (e.g. "adaptive_uniform", "stretched_z").
    virtual const char* name() const = 0;
};

} // namespace lbm
