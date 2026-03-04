//! C++ LBM 核心库的安全 Rust 封装。
//!
//! C++ 库通过细薄的 C ABI（`extern "C"`）对外暴露接口，
//! 使 Rust 的 FFI 能够在不依赖 C++ 名称修饰（name mangling）的情况下调用它。
//! 内部的 `ffi` 模块包含 `unsafe` 声明；
//! 本模块对外公开的所有类型均为安全封装。

// ---------------------------------------------------------------------------
// C ABI 声明
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

    /// 指向堆上 `lbm::LatticeGrid` 的不透明句柄
    pub enum LatticeGridHandle {}
    /// 指向堆上 `lbm::Solver` 的不透明句柄
    pub enum SolverHandle {}

    /// 插件 ABI 使用的 C 兼容函数指针类型
    pub type BoundaryFn  = unsafe extern "C" fn(*mut std::ffi::c_void, c_int, *mut std::ffi::c_void);
    pub type MeshFn      = unsafe extern "C" fn(*mut std::ffi::c_void, c_int, *mut std::ffi::c_void);
    pub type MotionFn    = unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void,
                                                 f64, c_int, *mut std::ffi::c_void);
    pub type FlexibleFn  = unsafe extern "C" fn(*mut std::ffi::c_void, f64, c_int,
                                                 *mut std::ffi::c_void);

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
        pub fn lbm_solver_step_n(s: *mut SolverHandle,
                                  g: *mut LatticeGridHandle,
                                  step_index: c_int,
                                  dt: f64);

        // --- 插件注册 ---
        pub fn lbm_set_plugins(
            boundary_fn:   Option<BoundaryFn>,  boundary_data:  *mut std::ffi::c_void,
            mesh_fn:       Option<MeshFn>,      mesh_data:      *mut std::ffi::c_void,
            motion_fn:     Option<MotionFn>,    motion_data:    *mut std::ffi::c_void,
            flexible_fn:   Option<FlexibleFn>,  flexible_data:  *mut std::ffi::c_void,
        );
    }
}

// ---------------------------------------------------------------------------
use std::ffi::{c_int, c_void};

// ---------------------------------------------------------------------------
// 安全的公开类型
// ---------------------------------------------------------------------------

/// 格子模型选择器
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LatticeModel {
    D2Q9,
    D3Q19,
    D3Q27,
}

/// 碰撞模型选择器
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
/// `lbm::LatticeGrid` 的安全封装
pub struct LbmGrid {
    ptr: *mut ffi::LatticeGridHandle,
}

// SAFETY: C++ LatticeGrid 在堆上分配，未跨线程共享；
// 若调用方需要多线程访问，需自行保证同步。
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

    /// 原始可变指针 — 仅供 `LbmSolver::step` 内部使用
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
/// `lbm::Solver` 的安全封装
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

    /// 带显式步骤索引和时间步长的推进接口，
    /// 使已注册的插件能够获得准确的时间信息。
    pub fn step_n(&mut self, grid: &mut LbmGrid, step_index: i32, dt: f64) {
        unsafe { ffi::lbm_solver_step_n(self.ptr, grid.as_mut_ptr(), step_index, dt) };
    }
}

impl Drop for LbmSolver {
    fn drop(&mut self) {
        unsafe { ffi::lbm_solver_free(self.ptr) };
    }
}

// ---------------------------------------------------------------------------
// 插件注册
// ---------------------------------------------------------------------------

/// 可选 C 函数指针插件回调的持有结构体。
///
/// 通过 `PluginCallbacks::default()` 构建，然后填入所需的字段，
/// 最后调用 [`register_plugins`] 注册。
///
/// # 示例
///
/// ```no_run
/// use lbm_bindings::PluginCallbacks;
///
/// unsafe extern "C" fn my_bc(grid: *mut std::ffi::c_void, step: i32,
///                             _data: *mut std::ffi::c_void) {
///     // 将 grid 转换为 &mut lbm::LatticeGrid 并应用自定义边界条件
///     let _ = (grid, step);
/// }
///
/// let mut cbs = PluginCallbacks::default();
/// cbs.boundary_fn = Some(my_bc);
/// lbm_bindings::register_plugins(cbs);
/// ```
#[derive(Default, Clone, Copy)]
pub struct PluginCallbacks {
    /// 每步标准边界条件执行完毕后调用的回调函数。
    pub boundary_fn:   Option<unsafe extern "C" fn(*mut c_void, c_int, *mut c_void)>,
    /// 转发给 `boundary_fn` 的不透明用户数据指针。
    pub boundary_data: *mut c_void,

    /// 每步流式迁移/宏观量更新后调用的回调函数（自适应网格）。
    pub mesh_fn:       Option<unsafe extern "C" fn(*mut c_void, c_int, *mut c_void)>,
    /// 转发给 `mesh_fn` 的不透明用户数据指针。
    pub mesh_data:     *mut c_void,

    /// 每步碰撞前调用以更新固体位置的回调函数（运动网格/固体）。
    pub motion_fn:     Option<unsafe extern "C" fn(*mut c_void, *mut c_void, f64, c_int, *mut c_void)>,
    /// 转发给 `motion_fn` 的不透明用户数据指针。
    pub motion_data:   *mut c_void,

    /// 替代/补充柔性体求解器的回调函数，在流式迁移后调用。
    pub flexible_fn:   Option<unsafe extern "C" fn(*mut c_void, f64, c_int, *mut c_void)>,
    /// 转发给 `flexible_fn` 的不透明用户数据指针。
    pub flexible_data: *mut c_void,
}

// SAFETY: PluginCallbacks 内部的原始指针是不透明的用户数据指针，
// 其线程安全性由调用方负责保证。
unsafe impl Send for PluginCallbacks {}
unsafe impl Sync for PluginCallbacks {}

/// 向全局 `PluginRegistry` 单例注册插件回调。
///
/// 这是对 C ABI 中 `lbm_set_plugins` 的薄封装。
/// 对不需要激活的回调传入 `None`（默认值）即可。
///
/// # 安全性
///
/// 若提供了函数指针，它们在仿真期间（即下次调用本函数覆盖它们或
/// 进程退出之前）必须保持有效。
pub fn register_plugins(cbs: PluginCallbacks) {
    unsafe {
        ffi::lbm_set_plugins(
            cbs.boundary_fn, cbs.boundary_data,
            cbs.mesh_fn,     cbs.mesh_data,
            cbs.motion_fn,   cbs.motion_data,
            cbs.flexible_fn, cbs.flexible_data,
        );
    }
}

// ---------------------------------------------------------------------------
// C ABI 实现说明
// ---------------------------------------------------------------------------
// lbm_grid_new / lbm_solver_step 等函数的实际实现位于：
//   core/src/capi/lbm_capi.cpp     — 格子网格与求解器函数
//   core/src/plugins/plugin_registry.cpp — 插件注册函数
// 两者均被编译进 liblbm_core.a，由 build.rs 链接到本 crate。
