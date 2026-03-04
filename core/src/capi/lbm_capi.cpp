// C ABI wrapper — allows Rust (and any C consumer) to call the C++ core
// without C++ name mangling.
#include "lbm/lattice.hpp"
#include "lbm/solver.hpp"
#include "plugins/plugin_registry.hpp"
#include <cstdlib>
#include <new>

// ---------------------------------------------------------------------------
// C-callback typedefs (must match plugin_registry.cpp)
// ---------------------------------------------------------------------------
extern "C" {

typedef void (*lbm_boundary_fn) (void* grid, int step, void* userdata);
typedef void (*lbm_mesh_adapt_fn)(void* grid, int step, void* userdata);
typedef void (*lbm_motion_fn)   (void* grid, void* markers,
                                  double dt, int step, void* userdata);
typedef void (*lbm_flexible_fn) (void* markers, double dt, int step,
                                  void* userdata);

// Declared in plugin_registry.cpp
void lbm_set_plugins(
    lbm_boundary_fn   boundary_fn,  void* boundary_data,
    lbm_mesh_adapt_fn mesh_fn,      void* mesh_data,
    lbm_motion_fn     motion_fn,    void* motion_data,
    lbm_flexible_fn   flexible_fn,  void* flexible_data);

} // extern "C"

extern "C" {

// ---------------------------------------------------------------------------
// LatticeGrid
// ---------------------------------------------------------------------------

lbm::LatticeGrid* lbm_grid_new(int nx, int ny, int nz, int model_id)
{
    // Validate model_id to avoid UB from out-of-range enum cast
    if (model_id < 0 || model_id > 2) return nullptr;
    auto model = static_cast<lbm::LatticeModel>(model_id);
    return new (std::nothrow) lbm::LatticeGrid(nx, ny, nz, model);
}

void lbm_grid_free(lbm::LatticeGrid* g)
{
    delete g;
}

int lbm_grid_nx(const lbm::LatticeGrid* g) { return g ? g->nx : 0; }
int lbm_grid_ny(const lbm::LatticeGrid* g) { return g ? g->ny : 0; }
int lbm_grid_nz(const lbm::LatticeGrid* g) { return g ? g->nz : 0; }

double lbm_grid_rho(const lbm::LatticeGrid* g, int idx)
{
    if (!g || idx < 0 || idx >= g->size()) return 0.0;
    return g->rho[idx];
}

double lbm_grid_ux(const lbm::LatticeGrid* g, int idx)
{
    if (!g || idx < 0 || idx >= g->size()) return 0.0;
    return g->u[idx * g->dim() + 0];
}

double lbm_grid_uy(const lbm::LatticeGrid* g, int idx)
{
    if (!g || idx < 0 || idx >= g->size()) return 0.0;
    return g->u[idx * g->dim() + 1];
}

// ---------------------------------------------------------------------------
// Solver
// ---------------------------------------------------------------------------

lbm::Solver* lbm_solver_new(lbm::LatticeGrid* g, double omega, int cm_id)
{
    if (!g) return nullptr;
    // Validate collision model id
    if (cm_id < 0 || cm_id > 1) return nullptr;
    auto cm = static_cast<lbm::CollisionModel>(cm_id);
    return new (std::nothrow) lbm::Solver(*g, omega, cm);
}

void lbm_solver_free(lbm::Solver* s)
{
    delete s;
}

/// Advance the simulation by one step and invoke any registered plugins.
///
/// This is the simplified overload that passes `step_index = 0` and `dt = 1.0`
/// to all plugins.  If plugins need accurate step count or physical time
/// (e.g. prescribed-motion trajectories, adaptive schedules), prefer
/// `lbm_solver_step_n()` instead, which accepts explicit step_index and dt.
///
/// Plugin call order within a single step:
///   1. `IMotionPlugin::update()`       — update body/mesh positions (pre-collision)
///   2. `Solver::step()`                — standard LBM collision + streaming
///   3. `IBoundaryPlugin::apply()`      — custom boundary condition (post-streaming)
///   4. `IMeshPlugin::adapt()`          — mesh adaptation (post-streaming)
///   5. `IFlexibleSolverPlugin::step()` — flexible body advance (post-streaming)
void lbm_solver_step(lbm::Solver* s, lbm::LatticeGrid* g)
{
    if (!s || !g) return;

    auto& reg = lbm::PluginRegistry::instance();

    // 1. Motion plugin: update body/marker positions before collision
    //    (no MarkerSet handle available via this ABI — pass nullptr)
    reg.update_motion(*g, nullptr, /*dt=*/1.0, /*step=*/0);

    // 2. Standard LBM step (collision + streaming + macroscopic update)
    // The solver holds a reference to the grid passed at construction.
    (void)g;
    s->step();

    // 3. Custom boundary condition plugin
    reg.apply_boundary(*g, /*step=*/0);

    // 4. Mesh adaptation plugin
    reg.adapt_mesh(*g, /*step=*/0);

    // 5. Flexible-body plugin (no MarkerSet via this simplified ABI)
    reg.step_flexible(nullptr, /*dt=*/1.0, /*step=*/0);
}

/// Extended solver step that forwards step index and dt to plugins.
///
/// Prefer this over `lbm_solver_step()` when plugins need accurate
/// time information (motion trajectories, adaptive schedule, etc.).
void lbm_solver_step_n(lbm::Solver* s, lbm::LatticeGrid* g,
                        int step_index, double dt)
{
    if (!s || !g) return;

    auto& reg = lbm::PluginRegistry::instance();

    reg.update_motion(*g, nullptr, dt, step_index);
    s->step();
    reg.apply_boundary(*g, step_index);
    reg.adapt_mesh(*g, step_index);
    reg.step_flexible(nullptr, dt, step_index);
}

} // extern "C"

