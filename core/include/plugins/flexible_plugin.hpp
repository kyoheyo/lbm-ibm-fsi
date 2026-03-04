#pragma once
/// @file plugins/flexible_plugin.hpp
/// @brief Extension interface for alternative / additional flexible-body solvers.
///
/// ## How to add a new flexible solver
///
/// The built-in structural solver is a linear Euler-Bernoulli beam
/// (`fsi::BeamSolver`).  This plugin allows a completely different
/// structural model to be inserted into the FSI coupling loop without
/// modifying the existing coupling code.  Typical use-cases:
///
/// - **Kirchhoff-Love plate** solver for thin-shell structures
/// - **Nonlinear co-rotational beam** for large-deflection problems
/// - **Finite-volume membrane** for 3-D flexible surfaces
/// - **Reduced-order model** (POD/ROM) for fast approximate responses
/// - **External FEM code** (e.g. OpenFOAM solid, Calculix) coupled via
///   shared memory or MPI
///
/// ### Steps
/// 1. Subclass `IFlexibleSolverPlugin`.
/// 2. Override `step()` to advance the structural DOFs by *dt*.
///    The method receives the current IBM marker positions (output of
///    velocity interpolation) and is expected to update the marker
///    positions and forces for the next spreading step.
/// 3. Register:
///    ```cpp
///    MyPlatePlugin plugin;
///    PluginRegistry::instance().set_flexible_plugin(&plugin);
///    ```
///
/// ### Data flow in the FSI loop
/// ```
/// interpolate_velocity(fluid, markers) →  IMotionPlugin::update()
///                                       →  IFlexibleSolverPlugin::step()
///                                       →  spread_force(fluid, markers)
/// ```
///
/// ### Example skeleton
/// ```cpp
/// #include "plugins/flexible_plugin.hpp"
/// class KirchhoffPlate : public lbm::IFlexibleSolverPlugin {
/// public:
///     void step(ibm::MarkerSet& markers, double dt, int step) override {
///         // 1. Extract nodal forces from markers.fx / .fy / .fz
///         // 2. Solve plate equation M*a + K*u = F
///         // 3. Update marker positions (m.x, m.y) from new deflections
///     }
///     const char* name() const override { return "kirchhoff_love_plate"; }
/// };
/// ```

#include "ibm/marker.hpp"

namespace lbm {

/// Abstract interface for a flexible-body solver plugin.
///
/// Implement this interface to replace or augment the built-in
/// Euler-Bernoulli beam solver with an arbitrary structural model.
class IFlexibleSolverPlugin {
public:
    virtual ~IFlexibleSolverPlugin() = default;

    /// Advance the flexible body by one time step *dt*.
    ///
    /// On entry  : `markers` contains interpolated fluid velocities
    ///             (`m.ux`, `m.uy`, `m.uz`) and possibly IBM forces
    ///             (`m.fx`, `m.fy`, `m.fz`) set by the coupling layer.
    ///
    /// On exit   : `markers` positions (`m.x`, `m.y`, `m.z`) and forces
    ///             (`m.fx`, `m.fy`, `m.fz`) must be updated so that
    ///             `spread_force()` can correctly distribute them back to
    ///             the Eulerian grid.
    ///
    /// @param markers  The Lagrangian IBM marker set.
    /// @param dt       Physical time step size.
    /// @param step     Current simulation step index (0-based).
    virtual void step(ibm::MarkerSet& markers, double dt, int step) = 0;

    /// Short identifier (e.g. "kirchhoff_love_plate", "corotational_beam").
    virtual const char* name() const = 0;
};

} // namespace lbm
