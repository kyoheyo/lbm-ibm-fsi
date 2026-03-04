/// @file plugins/plugin_registry.cpp
/// @brief C-compatible callback adapters for the PluginRegistry singleton.
///
/// This translation unit provides concrete `IBoundaryPlugin` / `IMeshPlugin` /
/// `IMotionPlugin` / `IFlexibleSolverPlugin` implementations that wrap plain
/// C function pointers.  These adapters allow Rust (and any C consumer) to
/// register plugin callbacks through the C ABI without exposing C++ virtual
/// dispatch details across the boundary.
///
/// The four adapter types are instantiated by the C ABI functions
/// `lbm_set_plugins()` (defined in `lbm_capi.cpp`) and stored in a static
/// slot so that their lifetime spans the entire simulation.

#include "plugins/boundary_plugin.hpp"
#include "plugins/mesh_plugin.hpp"
#include "plugins/motion_plugin.hpp"
#include "plugins/flexible_plugin.hpp"
#include "plugins/plugin_registry.hpp"

#include <cstddef> // nullptr

// ---------------------------------------------------------------------------
// C function-pointer callback types (must match lbm_capi.cpp declarations)
// ---------------------------------------------------------------------------
extern "C" {

typedef void (*lbm_boundary_fn) (void* grid, int step,
                                  void* userdata);
typedef void (*lbm_mesh_adapt_fn)(void* grid, int step,
                                  void* userdata);
typedef void (*lbm_motion_fn)   (void* grid, void* markers,
                                  double dt, int step,
                                  void* userdata);
typedef void (*lbm_flexible_fn) (void* markers,
                                  double dt, int step,
                                  void* userdata);

} // extern "C"

// ---------------------------------------------------------------------------
// Adapter: wraps a C callback as IBoundaryPlugin
// ---------------------------------------------------------------------------
namespace {

class BoundaryAdapter final : public lbm::IBoundaryPlugin {
public:
    lbm_boundary_fn fn   = nullptr;
    void*           data = nullptr;

    void apply(lbm::LatticeGrid& grid, int step) override {
        if (fn) fn(static_cast<void*>(&grid), step, data);
    }
    const char* name() const override { return "c_callback_boundary"; }
};

// ---------------------------------------------------------------------------
// Adapter: wraps a C callback as IMeshPlugin
// ---------------------------------------------------------------------------
class MeshAdapter final : public lbm::IMeshPlugin {
public:
    lbm_mesh_adapt_fn fn   = nullptr;
    void*             data = nullptr;

    void initialize(lbm::LatticeGrid& /*grid*/,
                    const void* /*cfg*/) override {}  // not exposed via C ABI

    void adapt(lbm::LatticeGrid& grid, int step) override {
        if (fn) fn(static_cast<void*>(&grid), step, data);
    }
    const char* name() const override { return "c_callback_mesh"; }
};

// ---------------------------------------------------------------------------
// Adapter: wraps a C callback as IMotionPlugin
// ---------------------------------------------------------------------------
class MotionAdapter final : public lbm::IMotionPlugin {
public:
    lbm_motion_fn fn   = nullptr;
    void*         data = nullptr;

    void update(lbm::LatticeGrid& grid,
                ibm::MarkerSet*  markers,
                double dt, int step) override {
        if (fn) fn(static_cast<void*>(&grid),
                   static_cast<void*>(markers),
                   dt, step, data);
    }
    const char* name() const override { return "c_callback_motion"; }
};

// ---------------------------------------------------------------------------
// Adapter: wraps a C callback as IFlexibleSolverPlugin
// ---------------------------------------------------------------------------
class FlexibleAdapter final : public lbm::IFlexibleSolverPlugin {
public:
    lbm_flexible_fn fn   = nullptr;
    void*           data = nullptr;

    void step(ibm::MarkerSet& markers, double dt, int step_idx) override {
        if (fn) fn(static_cast<void*>(&markers), dt, step_idx, data);
    }
    const char* name() const override { return "c_callback_flexible"; }
};

// ---------------------------------------------------------------------------
// Static adapter slots — live for the entire program lifetime
// ---------------------------------------------------------------------------
static BoundaryAdapter g_boundary_adapter;
static MeshAdapter     g_mesh_adapter;
static MotionAdapter   g_motion_adapter;
static FlexibleAdapter g_flexible_adapter;

} // anonymous namespace

// ---------------------------------------------------------------------------
// Public C ABI — called from lbm_capi.cpp
// ---------------------------------------------------------------------------
extern "C" {

/// Register all four optional plugin callbacks at once.
///
/// Pass `nullptr` for any callback that should remain inactive.
/// Calling this function a second time replaces all previous registrations.
///
/// @param boundary_fn   Called after built-in BCs in each step.
/// @param boundary_data Opaque pointer forwarded to boundary_fn.
/// @param mesh_fn       Called after streaming/macroscopic update.
/// @param mesh_data     Opaque pointer forwarded to mesh_fn.
/// @param motion_fn     Called before collision (position update).
/// @param motion_data   Opaque pointer forwarded to motion_fn.
/// @param flexible_fn   Called in place of / after BeamSolver::step().
/// @param flexible_data Opaque pointer forwarded to flexible_fn.
void lbm_set_plugins(
    lbm_boundary_fn  boundary_fn,  void* boundary_data,
    lbm_mesh_adapt_fn mesh_fn,     void* mesh_data,
    lbm_motion_fn    motion_fn,    void* motion_data,
    lbm_flexible_fn  flexible_fn,  void* flexible_data)
{
    auto& reg = lbm::PluginRegistry::instance();

    if (boundary_fn) {
        g_boundary_adapter.fn   = boundary_fn;
        g_boundary_adapter.data = boundary_data;
        reg.set_boundary_plugin(&g_boundary_adapter);
    } else {
        reg.set_boundary_plugin(nullptr);
    }

    if (mesh_fn) {
        g_mesh_adapter.fn   = mesh_fn;
        g_mesh_adapter.data = mesh_data;
        reg.set_mesh_plugin(&g_mesh_adapter);
    } else {
        reg.set_mesh_plugin(nullptr);
    }

    if (motion_fn) {
        g_motion_adapter.fn   = motion_fn;
        g_motion_adapter.data = motion_data;
        reg.set_motion_plugin(&g_motion_adapter);
    } else {
        reg.set_motion_plugin(nullptr);
    }

    if (flexible_fn) {
        g_flexible_adapter.fn   = flexible_fn;
        g_flexible_adapter.data = flexible_data;
        reg.set_flexible_plugin(&g_flexible_adapter);
    } else {
        reg.set_flexible_plugin(nullptr);
    }
}

} // extern "C"
