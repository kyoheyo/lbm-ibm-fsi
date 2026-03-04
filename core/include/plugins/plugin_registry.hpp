#pragma once
/// @file plugins/plugin_registry.hpp
/// @brief Central registry that aggregates all four plugin hooks.
///
/// `PluginRegistry` is a plain struct that holds nullable pointers to the
/// four plugin interfaces.  A global singleton is accessible via
/// `PluginRegistry::instance()`.
///
/// ## Lifecycle
///
/// ```
/// PluginRegistry& reg = PluginRegistry::instance();
///
/// // Register plugins once, before the simulation loop
/// reg.set_boundary_plugin(&my_bc);
/// reg.set_mesh_plugin(&my_mesh);
/// reg.set_motion_plugin(&my_motion);
/// reg.set_flexible_plugin(&my_flex);
///
/// // The C ABI layer (lbm_capi.cpp) calls the apply_* methods
/// // automatically from lbm_solver_step().
/// ```
///
/// When a plugin pointer is `nullptr` (the default) the corresponding
/// `apply_*` call is a cheap no-op.  Removing a plugin at run time is
/// as simple as `reg.set_boundary_plugin(nullptr)`.
///
/// ## Thread safety
///
/// Plugin *registration* is not thread-safe — plugins must be registered
/// from a single thread before the simulation loop starts.  The `apply_*`
/// dispatch methods are read-only and safe to call concurrently.

#include "boundary_plugin.hpp"
#include "mesh_plugin.hpp"
#include "motion_plugin.hpp"
#include "flexible_plugin.hpp"

namespace lbm {

/// Aggregates all optional plugin hooks used by the solver.
class PluginRegistry {
public:
    // ------------------------------------------------------------------
    // Singleton access
    // ------------------------------------------------------------------

    /// Return the process-wide singleton instance.
    static PluginRegistry& instance() {
        static PluginRegistry reg;
        return reg;
    }

    // ------------------------------------------------------------------
    // Registration (store non-owning pointers; caller manages lifetime)
    // ------------------------------------------------------------------

    void set_boundary_plugin (IBoundaryPlugin*      p) { boundary_  = p; }
    void set_mesh_plugin     (IMeshPlugin*          p) { mesh_       = p; }
    void set_motion_plugin   (IMotionPlugin*        p) { motion_     = p; }
    void set_flexible_plugin (IFlexibleSolverPlugin* p){ flexible_   = p; }

    // ------------------------------------------------------------------
    // Inspection
    // ------------------------------------------------------------------

    [[nodiscard]] IBoundaryPlugin*       boundary_plugin()  const { return boundary_; }
    [[nodiscard]] IMeshPlugin*           mesh_plugin()      const { return mesh_;     }
    [[nodiscard]] IMotionPlugin*         motion_plugin()    const { return motion_;   }
    [[nodiscard]] IFlexibleSolverPlugin* flexible_plugin()  const { return flexible_; }

    /// True when at least one plugin is active.
    [[nodiscard]] bool any_active() const {
        return boundary_ || mesh_ || motion_ || flexible_;
    }

    // ------------------------------------------------------------------
    // Dispatch helpers — called by lbm_capi.cpp around lbm_solver_step()
    // ------------------------------------------------------------------

    /// Call the registered boundary plugin (no-op if nullptr).
    void apply_boundary(LatticeGrid& grid, int step) const {
        if (boundary_) boundary_->apply(grid, step);
    }

    /// Call the registered mesh-adaptation plugin (no-op if nullptr).
    void adapt_mesh(LatticeGrid& grid, int step) const {
        if (mesh_) mesh_->adapt(grid, step);
    }

    /// Call the registered motion plugin (no-op if nullptr).
    void update_motion(LatticeGrid& grid,
                       ibm::MarkerSet* markers,
                       double dt,
                       int step) const {
        if (motion_) motion_->update(grid, markers, dt, step);
    }

    /// Call the registered flexible-body plugin (no-op if nullptr or if
    /// *markers* is nullptr).
    ///
    /// Flexible-body plugins require a valid `MarkerSet` because they advance
    /// structural DOFs using the IBM Lagrangian markers as their degrees of
    /// freedom.  When IBM is not active in a given run (`markers == nullptr`)
    /// there is no structure to advance and the call is silently skipped.
    void step_flexible(ibm::MarkerSet* markers, double dt, int step) const {
        if (flexible_ && markers) flexible_->step(*markers, dt, step);
    }

private:
    IBoundaryPlugin*       boundary_ = nullptr;
    IMeshPlugin*           mesh_     = nullptr;
    IMotionPlugin*         motion_   = nullptr;
    IFlexibleSolverPlugin* flexible_ = nullptr;

    // Disallow external construction / copying — use instance()
    PluginRegistry()  = default;
    ~PluginRegistry() = default;
    PluginRegistry(const PluginRegistry&)            = delete;
    PluginRegistry& operator=(const PluginRegistry&) = delete;
};

} // namespace lbm
