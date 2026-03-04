//! Safe Rust wrappers around the C++ LBM core library.
//!
//! The C++ library exposes a thin C ABI (`extern "C"`) so that Rust's FFI can
//! call it without a C++ name-mangling dependency.  The raw `ffi` module
//! contains the `unsafe` declarations; all public types in this module are
//! safe wrappers.

// ---------------------------------------------------------------------------
// C ABI declarations
// ---------------------------------------------------------------------------
mod ffi {
    use std::ffi::c_int;

    #[repr(C)]
    pub enum LatticeModelC {
        D2Q9  = 0,
        D3Q19 = 1,
        D3Q27 = 2,
    }

    #[repr(C)]
    pub enum CollisionModelC {
        Bgk = 0,
        Mrt = 1,
    }

    /// Opaque handle to a `lbm::LatticeGrid` on the heap
    pub enum LatticeGridHandle {}
    /// Opaque handle to a `lbm::Solver` on the heap
    pub enum SolverHandle {}

    extern "C" {
        // --- LatticeGrid ---
        pub fn lbm_grid_new(nx: c_int, ny: c_int, nz: c_int,
                            model: LatticeModelC) -> *mut LatticeGridHandle;
        pub fn lbm_grid_free(g: *mut LatticeGridHandle);
        pub fn lbm_grid_nx(g: *const LatticeGridHandle) -> c_int;
        pub fn lbm_grid_ny(g: *const LatticeGridHandle) -> c_int;
        pub fn lbm_grid_nz(g: *const LatticeGridHandle) -> c_int;
        pub fn lbm_grid_rho(g: *const LatticeGridHandle, idx: c_int) -> f64;
        pub fn lbm_grid_ux(g: *const LatticeGridHandle, idx: c_int) -> f64;
        pub fn lbm_grid_uy(g: *const LatticeGridHandle, idx: c_int) -> f64;

        // --- Solver ---
        pub fn lbm_solver_new(g: *mut LatticeGridHandle,
                              omega: f64,
                              cm: CollisionModelC) -> *mut SolverHandle;
        pub fn lbm_solver_free(s: *mut SolverHandle);
        pub fn lbm_solver_step(s: *mut SolverHandle,
                               g: *mut LatticeGridHandle);
    }
}

// ---------------------------------------------------------------------------
// Safe public types
// ---------------------------------------------------------------------------

/// Lattice model selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatticeModel {
    D2Q9,
    D3Q19,
    D3Q27,
}

/// Collision model selector
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionModel {
    Bgk,
    Mrt,
}

impl From<LatticeModel> for ffi::LatticeModelC {
    fn from(m: LatticeModel) -> Self {
        match m {
            LatticeModel::D2Q9  => ffi::LatticeModelC::D2Q9,
            LatticeModel::D3Q19 => ffi::LatticeModelC::D3Q19,
            LatticeModel::D3Q27 => ffi::LatticeModelC::D3Q27,
        }
    }
}

impl From<CollisionModel> for ffi::CollisionModelC {
    fn from(c: CollisionModel) -> Self {
        match c {
            CollisionModel::Bgk => ffi::CollisionModelC::Bgk,
            CollisionModel::Mrt => ffi::CollisionModelC::Mrt,
        }
    }
}

// ---------------------------------------------------------------------------
/// Safe wrapper around `lbm::LatticeGrid`
pub struct LbmGrid {
    ptr: *mut ffi::LatticeGridHandle,
}

// SAFETY: the C++ LatticeGrid is heap-allocated and not shared across threads
// unless the caller explicitly synchronises access.
unsafe impl Send for LbmGrid {}

impl LbmGrid {
    pub fn new(nx: i32, ny: i32, nz: i32, model: LatticeModel) -> Self {
        let ptr = unsafe {
            ffi::lbm_grid_new(nx, ny, nz, model.into())
        };
        assert!(!ptr.is_null(), "lbm_grid_new returned null");
        LbmGrid { ptr }
    }

    pub fn nx(&self) -> i32 { unsafe { ffi::lbm_grid_nx(self.ptr) } }
    pub fn ny(&self) -> i32 { unsafe { ffi::lbm_grid_ny(self.ptr) } }
    pub fn nz(&self) -> i32 { unsafe { ffi::lbm_grid_nz(self.ptr) } }

    pub fn rho(&self, idx: i32) -> f64 { unsafe { ffi::lbm_grid_rho(self.ptr, idx) } }
    pub fn ux (&self, idx: i32) -> f64 { unsafe { ffi::lbm_grid_ux (self.ptr, idx) } }
    pub fn uy (&self, idx: i32) -> f64 { unsafe { ffi::lbm_grid_uy (self.ptr, idx) } }

    /// Raw mutable pointer — used only by `LbmSolver::step`
    pub(crate) fn as_mut_ptr(&mut self) -> *mut ffi::LatticeGridHandle {
        self.ptr
    }
}

impl Drop for LbmGrid {
    fn drop(&mut self) {
        unsafe { ffi::lbm_grid_free(self.ptr) };
    }
}

// ---------------------------------------------------------------------------
/// Safe wrapper around `lbm::Solver`
pub struct LbmSolver {
    ptr: *mut ffi::SolverHandle,
}

unsafe impl Send for LbmSolver {}

impl LbmSolver {
    pub fn new(grid: &mut LbmGrid, omega: f64, cm: CollisionModel) -> Self {
        let ptr = unsafe {
            ffi::lbm_solver_new(grid.as_mut_ptr(), omega, cm.into())
        };
        assert!(!ptr.is_null(), "lbm_solver_new returned null");
        LbmSolver { ptr }
    }

    pub fn step(&mut self, grid: &mut LbmGrid) {
        unsafe { ffi::lbm_solver_step(self.ptr, grid.as_mut_ptr()) };
    }
}

impl Drop for LbmSolver {
    fn drop(&mut self) {
        unsafe { ffi::lbm_solver_free(self.ptr) };
    }
}

// ---------------------------------------------------------------------------
// C ABI implementation (defined in a companion .cpp file compiled by build.rs)
// ---------------------------------------------------------------------------
// The actual implementations of lbm_grid_new / lbm_solver_step etc. live in
// core/src/capi/lbm_capi.cpp and are compiled into lbm_core.a.
