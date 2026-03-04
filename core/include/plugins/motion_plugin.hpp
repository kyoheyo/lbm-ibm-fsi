#pragma once
/// @file plugins/motion_plugin.hpp
/// @brief Extension interface for moving mesh / rigid-body or prescribed motion.
///
/// ## How to add fluid moving mesh or solid structure motion
///
/// This plugin hook is invoked **once per time step, before collision**, so
/// that updated Lagrangian marker positions and/or Eulerian grid velocities
/// can be applied before the fluid solve.  Typical use-cases:
///
/// - Prescribed rigid-body motion (rotation, translation) of an immersed
///   boundary
/// - Moving-wall boundary conditions (e.g. rotating cylinder)
/// - Arbitrary Lagrangian-Eulerian (ALE) mesh motion
/// - 6-DOF rigid-body dynamics coupled to hydrodynamic loads
///
/// ### Steps
/// 1. Subclass `IMotionPlugin`.
/// 2. Override `update()` to advance positions and velocities by *dt*.
/// 3. Register:
///    ```cpp
///    MySineMotion plugin;
///    PluginRegistry::instance().set_motion_plugin(&plugin);
///    ```
///
/// ### Example skeleton
/// ```cpp
/// #include "plugins/motion_plugin.hpp"
/// #include "ibm/marker.hpp"
/// class RotatingCylinder : public lbm::IMotionPlugin {
/// public:
///     void update(lbm::LatticeGrid& grid,
///                 ibm::MarkerSet* markers,    // may be nullptr
///                 double dt, int step) override
///     {
///         if (!markers) return;
///         const double angle = 0.01 * step;   // prescribed rotation
///         for (auto& m : markers->markers) {
///             double r = std::hypot(m.x - cx_, m.y - cy_);
///             double theta = std::atan2(m.y - cy_, m.x - cx_) + 0.01 * dt;
///             m.x = cx_ + r * std::cos(theta);
///             m.y = cy_ + r * std::sin(theta);
///         }
///     }
///     const char* name() const override { return "rotating_cylinder"; }
/// private:
///     double cx_ = 50.0, cy_ = 40.0;
/// };
/// ```

#include "lbm/lattice.hpp"
#include "ibm/marker.hpp"

namespace lbm {

/// Abstract interface for a motion plugin (moving mesh / moving body).
///
/// Implement this interface to couple prescribed or dynamic rigid-body
/// motion to the fluid solver without changing the core LBM or IBM code.
class IMotionPlugin {
public:
    virtual ~IMotionPlugin() = default;

    /// Advance body/marker positions by *dt* seconds.
    ///
    /// @param grid     The Eulerian lattice grid (wall velocities can be
    ///                 written into the `force` or `u` arrays as needed).
    /// @param markers  The Lagrangian IBM marker set, or `nullptr` if IBM
    ///                 is not active for this run.
    /// @param dt       Physical time step size.
    /// @param step     Current simulation step index (0-based).
    virtual void update(LatticeGrid& grid,
                        ibm::MarkerSet* markers,
                        double dt,
                        int step) = 0;

    /// Short identifier (e.g. "prescribed_sine", "rigid_body_6dof").
    virtual const char* name() const = 0;
};

} // namespace lbm
