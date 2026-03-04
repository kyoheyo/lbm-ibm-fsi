// C ABI wrapper — allows Rust (and any C consumer) to call the C++ core
// without C++ name mangling.
#include "lbm/lattice.hpp"
#include "lbm/solver.hpp"
#include <cstdlib>
#include <new>

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

void lbm_solver_step(lbm::Solver* s, lbm::LatticeGrid* g)
{
    if (!s || !g) return;
    // The solver holds a reference to the grid passed at construction.
    // Re-assign here is safe because g is the same object.
    (void)g;
    s->step();
}

} // extern "C"
