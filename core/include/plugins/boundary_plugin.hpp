#pragma once
/// @file plugins/boundary_plugin.hpp
/// @brief Extension interface for custom boundary conditions.
///
/// ## How to add a new boundary condition
///
/// 1. Create a subclass of `IBoundaryPlugin`.
/// 2. Override `apply()` to implement the new boundary logic.  The method
///    is called **after** all standard BCs (bounce-back, Zou-He) have been
///    applied, so you have access to the already-updated distribution
///    functions and macroscopic fields.
/// 3. Register the plugin before the simulation loop:
///    ```cpp
///    MyBC plugin;
///    PluginRegistry::instance().set_boundary_plugin(&plugin);
///    ```
/// 4. For Rust / C users see the C ABI wrapper in
///    `core/src/capi/lbm_capi.cpp` (`lbm_set_plugins`).
///
/// ### Example skeleton
/// ```cpp
/// #include "plugins/boundary_plugin.hpp"
/// class OpenPressureBC : public lbm::IBoundaryPlugin {
/// public:
///     void apply(lbm::LatticeGrid& grid, int /*step*/) override {
///         // e.g. convective outlet on the East face
///         for (int j = 0; j < grid.ny; ++j) { ... }
///     }
///     const char* name() const override { return "open_pressure"; }
/// };
/// ```

#include "lbm/lattice.hpp"

namespace lbm {

/// Abstract interface for a custom boundary-condition plugin.
///
/// Implement this interface to add a new boundary condition without
/// modifying any existing solver code.  The plugin is called once per
/// time step, **after** the built-in boundary conditions have been applied.
class IBoundaryPlugin {
public:
    virtual ~IBoundaryPlugin() = default;

    /// Apply the custom boundary condition to *grid* at the given *step*.
    ///
    /// @param grid  The Eulerian lattice grid (distribution functions,
    ///              macroscopic fields, and body-force arrays are all
    ///              directly accessible and modifiable).
    /// @param step  Current simulation step index (0-based).
    virtual void apply(LatticeGrid& grid, int step) = 0;

    /// Short identifier returned in log messages (e.g. "convective_outlet").
    virtual const char* name() const = 0;
};

} // namespace lbm
