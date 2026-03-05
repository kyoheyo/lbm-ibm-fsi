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

    /// 指向堆上 `lbm::LatticeGrid` 的不透明句柄。
    ///
    /// 用空枚举（没有任何变体的 enum）而非 `struct LatticeGridHandle;`，
    /// 是因为空枚举永远无法被实例化——Rust 编译器会在编译期阻止任何试图
    /// 构造 `LatticeGridHandle` 值的代码。这从类型层面强制：只能持有指针，
    /// 不能直接创建或解引用内部内容，与 C++ 的"前向声明不完整类型"等价。
    pub enum LatticeGridHandle {}
    /// 指向堆上 `lbm::Solver` 的不透明句柄（原理同 `LatticeGridHandle`）。
    pub enum SolverHandle {}

    /// 插件 ABI 使用的 C 兼容函数指针类型。
    /// 对应 lbm_capi.cpp / plugin_registry.cpp 中的同名 typedef（lbm_boundary_fn 等）。
    pub type BoundaryFn  = unsafe extern "C" fn(*mut std::ffi::c_void, c_int, *mut std::ffi::c_void);
    pub type MeshFn      = unsafe extern "C" fn(*mut std::ffi::c_void, c_int, *mut std::ffi::c_void);
    pub type MotionFn    = unsafe extern "C" fn(*mut std::ffi::c_void, *mut std::ffi::c_void,
                                                 f64, c_int, *mut std::ffi::c_void);
    pub type FlexibleFn  = unsafe extern "C" fn(*mut std::ffi::c_void, f64, c_int,
                                                 *mut std::ffi::c_void);

    extern "C" {
        // --- LatticeGrid — 实现于 core/src/capi/lbm_capi.cpp ---
        // 对应 C++ 函数: lbm_grid_new / lbm_grid_free / lbm_grid_nx 等
        // Rust 安全封装: LbmGrid（见下方）
        pub fn lbm_grid_new(nx: c_int, ny: c_int, nz: c_int,
                            model: LatticeModelC) -> *mut LatticeGridHandle;
        pub fn lbm_grid_free(g: *mut LatticeGridHandle);
        pub fn lbm_grid_nx(g: *const LatticeGridHandle) -> c_int;
        pub fn lbm_grid_ny(g: *const LatticeGridHandle) -> c_int;
        pub fn lbm_grid_nz(g: *const LatticeGridHandle) -> c_int;
        pub fn lbm_grid_rho(g: *const LatticeGridHandle, idx: c_int) -> f64;
        pub fn lbm_grid_ux(g: *const LatticeGridHandle, idx: c_int) -> f64;
        pub fn lbm_grid_uy(g: *const LatticeGridHandle, idx: c_int) -> f64;

        // --- Solver — 实现于 core/src/capi/lbm_capi.cpp ---
        // 对应 C++ 函数: lbm_solver_new / lbm_solver_free / lbm_solver_step 等
        // Rust 安全封装: LbmSolver（见下方）
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
        /// 向求解器注册一个边界条件，在每步 `step()` 的流式迁移后自动施加。
        /// bc_type: 0=BounceBack 1=BounceBackFullWay 2=ZouHe_Velocity 3=ZouHe_Pressure
        ///          4=FullyDeveloped 5=FreeOutlet 6=Guo_Extrapolation 7=Periodic
        /// face:    0=West 1=East 2=South 3=North 4=Bottom 5=Top
        /// Rust 封装: LbmSolver::add_boundary_condition()
        pub fn lbm_solver_add_bc(s: *mut SolverHandle,
                                  bc_type: c_int, face: c_int,
                                  ux: f64, uy: f64, uz: f64, rho: f64);
        /// 将 MPI 域分解句柄绑定到求解器（启用自动幽灵行交换）。
        pub fn lbm_solver_attach_mpi(s: *mut SolverHandle, h: *mut MpiDecompHandle);

        // --- MPI 并行接口 — 实现于 core/src/capi/lbm_capi.cpp ---
        pub fn lbm_mpi_init() -> c_int;
        pub fn lbm_mpi_finalize();
        pub fn lbm_mpi_rank() -> c_int;
        pub fn lbm_mpi_size() -> c_int;
        /// 返回含幽灵行的本地 ny；并通过指针输出 y_start 和 local_ny（物理行数）。
        pub fn lbm_mpi_local_ny(global_ny: c_int,
                                 out_y_start: *mut c_int,
                                 out_local_ny: *mut c_int) -> c_int;
        pub fn lbm_mpi_decomp_new(global_nx: c_int, global_ny: c_int) -> *mut MpiDecompHandle;
        pub fn lbm_mpi_decomp_free(h: *mut MpiDecompHandle);

        // --- GPU（CUDA）接口 — 实现于 core/src/capi/lbm_capi.cpp ---
        pub fn lbm_gpu_solver_new(g: *mut LatticeGridHandle, omega: f64) -> *mut GpuSolverHandle;
        pub fn lbm_gpu_solver_free(h: *mut GpuSolverHandle);
        pub fn lbm_gpu_collide(h: *mut GpuSolverHandle);
        pub fn lbm_gpu_stream(h: *mut GpuSolverHandle);
        pub fn lbm_gpu_macroscopic(h: *mut GpuSolverHandle);
        pub fn lbm_gpu_download(h: *mut GpuSolverHandle, g: *mut LatticeGridHandle);
        pub fn lbm_gpu_upload(h: *mut GpuSolverHandle, g: *mut LatticeGridHandle);

        // --- 并行状态查询 — 实现于 core/src/capi/lbm_capi.cpp ---
        pub fn lbm_openmp_enabled()     -> c_int;
        pub fn lbm_openmp_max_threads() -> c_int;
        pub fn lbm_cuda_enabled()       -> c_int;
        pub fn lbm_mpi_enabled()        -> c_int;

        // --- 插件注册 — 实现于 core/src/plugins/plugin_registry.cpp ---
        // 对应 C++ 函数: lbm_set_plugins
        // Rust 安全封装: register_plugins()（见本文件底部）
        pub fn lbm_set_plugins(
            boundary_fn:   Option<BoundaryFn>,  boundary_data:  *mut std::ffi::c_void,
            mesh_fn:       Option<MeshFn>,      mesh_data:      *mut std::ffi::c_void,
            motion_fn:     Option<MotionFn>,    motion_data:    *mut std::ffi::c_void,
            flexible_fn:   Option<FlexibleFn>,  flexible_data:  *mut std::ffi::c_void,
        );
    }

    /// MPI 域分解描述符的不透明句柄（仅持有指针，不可实例化）
    pub enum MpiDecompHandle {}
    /// GPU 求解器的不透明句柄
    pub enum GpuSolverHandle {}
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

/// 边界条件类型
///
/// 整数值与 C++ 侧 `lbm::BCType` 枚举成员的顺序**严格一致**。
/// 若在 C++ 侧修改枚举顺序，此处必须同步更新，否则会导致运行时错误。
///
/// C++ BCType 定义位于 `core/include/lbm/boundary.hpp`：
/// ```cpp
/// enum class BCType {
///     BounceBack        = 0,
///     BounceBackFullWay = 1,
///     ZouHe_Velocity    = 2,
///     ZouHe_Pressure    = 3,
///     FullyDeveloped    = 4,
///     FreeOutlet        = 5,
///     Guo_Extrapolation = 6,
///     Periodic          = 7,
/// };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum BcType {
    /// 半步长（halfway）反弹：壁面位于节点间半格处，2 阶精度。
    /// 适用于静止无滑移固壁。
    BounceBack          = 0,
    /// 全步长（on-node）反弹：壁面位于节点处，1 阶精度。
    /// 实现简单，精度较低，适合快速验证。
    BounceBackFullWay   = 1,
    /// Zou-He 速度进/出口：通过质量+动量守恒+非平衡反弹条件，
    /// 由规定速度（ux, uy, uz）确定未知方向的分布函数。
    ZouHeVelocity       = 2,
    /// Zou-He 压力进/出口：类似 ZouHeVelocity，
    /// 但规定的是密度 rho（对应 LBM 压力 p = ρ·cs²）而非速度。
    ZouHePressure       = 3,
    /// 充分发展出口：将出口节点的分布函数替换为上游相邻层的值（零法向梯度）。
    /// 适合通道出口处流动已充分发展的场景。
    FullyDeveloped      = 4,
    /// 自由出口：与 `FullyDeveloped` 等价（别名）。
    FreeOutlet          = 5,
    /// 郭照立非平衡外推格式（Guo et al., Chinese Physics 2002）：
    /// `f_b = f_eq(ρ_b, u_b) + f_neq(interior)`，适用于各种入/出口。
    /// `rho > 0` 时为压力模式（外推速度）；`rho = 0` 时为速度模式（外推密度）。
    GuoExtrapolation    = 6,
    /// 周期边界：在流式迁移的模运算中隐式实现，
    /// 注册此类型为空操作（仅作文档/可视化用途）。
    Periodic            = 7,
}

/// 计算域各面编号
///
/// 坐标系约定：
/// - x 轴从西向东（West → East）
/// - y 轴从南向北（South → North）
/// - z 轴从底到顶（Bottom → Top，三维仿真）
///
/// 整数值与 C++ 侧 `lbm::Face` 枚举成员的顺序一致。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum Face {
    /// x = 0 边界（左壁）
    West   = 0,
    /// x = nx-1 边界（右壁）
    East   = 1,
    /// y = 0 边界（底壁）
    South  = 2,
    /// y = ny-1 边界（顶壁 / 盖板）
    North  = 3,
    /// z = 0 边界（三维仿真前壁）
    Bottom = 4,
    /// z = nz-1 边界（三维仿真后壁）
    Top    = 5,
}

impl From<LatticeModel> for ffi::LatticeModelC {
    fn from(m: LatticeModel) -> Self {
        // match 穷举所有变体，确保未来新增变体时编译期报错而非运行时 bug
        match m {
            LatticeModel::D2Q9  => ffi::LatticeModelC::D2Q9,
            LatticeModel::D3Q19 => ffi::LatticeModelC::D3Q19,
            LatticeModel::D3Q27 => ffi::LatticeModelC::D3Q27,
        }
    }
}

impl From<CollisionModel> for ffi::CollisionModelC {
    fn from(c: CollisionModel) -> Self {
        // 通过 From trait 实现类型转换，调用方只需写 .into()，无需接触 ffi 模块内部类型
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
            // model.into() 调用上方定义的 From<LatticeModel> for ffi::LatticeModelC，
            // 将公开的安全枚举转换为 C ABI 使用的 #[repr(C)] 枚举，再传给 C++ 侧
            ffi::lbm_grid_new(nx, ny, nz, model.into())
        };
        // assert! 在这里充当"安全网"：lbm_grid_new 只在内存不足或参数非法时返回 null，
        // 提前以 panic 捕获失败（这属于编程错误，而非可恢复的运行时条件）。
        // 若需要可恢复的错误处理，可将 new() 改为返回 Result<Self, &'static str>。
        assert!(!ptr.is_null(), "lbm_grid_new returned null");
        LbmGrid { ptr }
    }

    pub fn nx(&self) -> i32 { unsafe { ffi::lbm_grid_nx(self.ptr) } }
    pub fn ny(&self) -> i32 { unsafe { ffi::lbm_grid_ny(self.ptr) } }
    pub fn nz(&self) -> i32 { unsafe { ffi::lbm_grid_nz(self.ptr) } }

    pub fn rho(&self, idx: i32) -> f64 { unsafe { ffi::lbm_grid_rho(self.ptr, idx) } }
    pub fn ux (&self, idx: i32) -> f64 { unsafe { ffi::lbm_grid_ux (self.ptr, idx) } }
    pub fn uy (&self, idx: i32) -> f64 { unsafe { ffi::lbm_grid_uy (self.ptr, idx) } }

    /// 原始可变指针 — 仅供 `LbmSolver::step` 内部使用。
    ///
    /// `pub(crate)` 是 Rust 的受限可见性：此方法只在当前 crate（即 `lbm_bindings`）
    /// 内部可见，外部使用者无法直接拿到裸指针，从而防止绕过安全封装。
    pub(crate) fn as_mut_ptr(&mut self) -> *mut ffi::LatticeGridHandle {
        self.ptr
    }
}

impl Drop for LbmGrid {
    fn drop(&mut self) {
        // C++ 侧: delete g → ~LatticeGrid() 析构，释放所有 std::vector 内存
        // 对应 lbm_capi.cpp: lbm_grid_free()
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
            // cm.into(): From<CollisionModel> → ffi::CollisionModelC，同 LbmGrid::new 中的 model.into()
            // C++ 侧: new (std::nothrow) lbm::Solver(*g, omega, cm) — lbm_capi.cpp: lbm_solver_new()
            ffi::lbm_solver_new(grid.as_mut_ptr(), omega, cm.into())
        };
        assert!(!ptr.is_null(), "lbm_solver_new returned null");
        LbmSolver { ptr }
    }

    pub fn step(&mut self, grid: &mut LbmGrid) {
        // C++ 侧按序执行: update_motion → Solver::step → apply_boundary → adapt_mesh → step_flexible
        // 详见 lbm_capi.cpp: lbm_solver_step()；step_index=0, dt=1.0（简化接口）
        unsafe { ffi::lbm_solver_step(self.ptr, grid.as_mut_ptr()) };
    }

    /// 带显式步骤索引和时间步长的推进接口，
    /// 使已注册的插件能够获得准确的时间信息。
    pub fn step_n(&mut self, grid: &mut LbmGrid, step_index: i32, dt: f64) {
        // C++ 侧同 step()，但将 step_index 和 dt 原样转发给各插件
        // 详见 lbm_capi.cpp: lbm_solver_step_n()
        unsafe { ffi::lbm_solver_step_n(self.ptr, grid.as_mut_ptr(), step_index, dt) };
    }

    /// 向求解器注册一个边界条件，在每步 `step()` 的流式迁移后自动施加。
    ///
    /// 可多次调用以注册多个边界条件（按注册顺序依次施加）。
    ///
    /// # 参数
    /// * `bc_type` — 边界条件类型（[`BcType`]）
    /// * `face`    — 施加边界条件的面（[`Face`]）
    /// * `ux,uy,uz`— 规定速度（用于 [`BcType::ZouHeVelocity`]）
    /// * `rho`     — 规定密度（用于 [`BcType::ZouHePressure`]）
    pub fn add_boundary_condition(
        &mut self,
        bc_type: BcType,
        face: Face,
        ux: f64, uy: f64, uz: f64,
        rho: f64,
    ) {
        // C++ 侧: lbm_solver_add_bc() — core/src/capi/lbm_capi.cpp
        // 将参数转换为整数枚举值并传给 C++ 的 Solver::add_boundary_condition()
        unsafe {
            ffi::lbm_solver_add_bc(self.ptr, bc_type as i32, face as i32, ux, uy, uz, rho);
        }
    }
}

impl Drop for LbmSolver {
    fn drop(&mut self) {
        // C++ 侧: delete s → ~Solver() 析构
        // 对应 lbm_capi.cpp: lbm_solver_free()
        unsafe { ffi::lbm_solver_free(self.ptr) };
    }
}

// ---------------------------------------------------------------------------
/// MPI 并行工具函数（无论是否启用 MPI 均可安全调用）
// ---------------------------------------------------------------------------

/// 初始化 MPI。在创建 Grid/Solver 之前调用（幂等）。
/// 返回 `true` 表示 MPI 已成功初始化（需要编译期启用 LBM_ENABLE_MPI）。
pub fn mpi_init() -> bool {
    unsafe { ffi::lbm_mpi_init() != 0 }
}

/// 终结 MPI（幂等，安全在任何时刻调用）。
pub fn mpi_finalize() {
    unsafe { ffi::lbm_mpi_finalize() };
}

/// 返回当前 MPI 进程编号（未启用或未初始化时返回 0）。
pub fn mpi_rank() -> i32 {
    unsafe { ffi::lbm_mpi_rank() }
}

/// 返回 MPI 进程总数（未启用或未初始化时返回 1）。
pub fn mpi_size() -> i32 {
    unsafe { ffi::lbm_mpi_size() }
}

/// 计算本进程的本地网格 ny（含幽灵行）、全局起始行 y_start 以及物理行数 local_ny。
/// 返回 `(grid_ny, y_start, local_ny)`。
pub fn mpi_local_ny(global_ny: i32) -> (i32, i32, i32) {
    let mut y_start   = 0i32;
    let mut local_ny  = 0i32;
    let grid_ny = unsafe {
        ffi::lbm_mpi_local_ny(global_ny, &mut y_start, &mut local_ny)
    };
    (grid_ny, y_start, local_ny)
}

// ---------------------------------------------------------------------------
/// 打印当前编译时和运行时并行配置摘要。
///
/// 输出示例（OpenMP 已启用，8 线程）：
/// ```text
/// 并行配置：
///   OpenMP : 已启用  线程数 = 8（受 OMP_NUM_THREADS 控制）
///   MPI    : 未启用
///   GPU    : 未启用
/// ```
// ---------------------------------------------------------------------------
pub fn print_parallel_status() {
    let omp_on  = unsafe { ffi::lbm_openmp_enabled() } != 0;
    let cuda_on = unsafe { ffi::lbm_cuda_enabled()   } != 0;
    let mpi_on  = unsafe { ffi::lbm_mpi_enabled()    } != 0;

    println!("并行配置：");
    if omp_on {
        let threads = unsafe { ffi::lbm_openmp_max_threads() };
        println!("  OpenMP : 已启用  线程数 = {}（受 OMP_NUM_THREADS 控制）", threads);
    } else {
        println!("  OpenMP : 未启用");
    }
    if mpi_on {
        let rank  = mpi_rank();
        let procs = mpi_size();
        println!("  MPI    : 已启用  进程数 = {}  当前 rank = {}", procs, rank);
    } else {
        println!("  MPI    : 未启用");
    }
    if cuda_on {
        println!("  GPU    : 已启用（CUDA）");
    } else {
        println!("  GPU    : 未启用");
    }
}

/// `lbm::MpiDecomp` 的安全封装（持有堆上的 C++ MpiDecomp 对象）
pub struct LbmMpiDecomp {
    ptr: *mut ffi::MpiDecompHandle,
}

unsafe impl Send for LbmMpiDecomp {}

impl LbmMpiDecomp {
    /// 创建 MPI 域分解（需先调用 `mpi_init()`）。
    /// 未启用 MPI 时返回 `None`。
    pub fn new(global_nx: i32, global_ny: i32) -> Option<Self> {
        let ptr = unsafe { ffi::lbm_mpi_decomp_new(global_nx, global_ny) };
        if ptr.is_null() { None } else { Some(LbmMpiDecomp { ptr }) }
    }

    /// 原始可变指针（仅供 LbmSolver::attach_mpi 内部使用）
    pub(crate) fn as_mut_ptr(&mut self) -> *mut ffi::MpiDecompHandle { self.ptr }
}

impl Drop for LbmMpiDecomp {
    fn drop(&mut self) {
        unsafe { ffi::lbm_mpi_decomp_free(self.ptr) };
    }
}

impl LbmSolver {
    /// 绑定 MPI 域分解：之后每次 `step()` 的流式迁移后自动执行幽灵行交换。
    /// 传入 `None` 可解除绑定（恢复单进程模式）。
    pub fn attach_mpi(&mut self, decomp: Option<&mut LbmMpiDecomp>) {
        let h = decomp.map_or(std::ptr::null_mut(), |d| d.as_mut_ptr());
        unsafe { ffi::lbm_solver_attach_mpi(self.ptr, h) };
    }
}

// ---------------------------------------------------------------------------
/// GPU（CUDA）求解器封装
// ---------------------------------------------------------------------------

/// `lbm::GpuSolver` 的安全封装（D2Q9 BGK on CUDA）
pub struct LbmGpuSolver {
    ptr: *mut ffi::GpuSolverHandle,
}

unsafe impl Send for LbmGpuSolver {}

impl LbmGpuSolver {
    /// 创建 GPU 求解器，将 CPU 网格数据上传到 GPU。
    /// 未启用 CUDA 或 GPU 初始化失败时返回 `None`。
    pub fn new(grid: &mut LbmGrid, omega: f64) -> Option<Self> {
        let ptr = unsafe { ffi::lbm_gpu_solver_new(grid.as_mut_ptr(), omega) };
        if ptr.is_null() { None } else { Some(LbmGpuSolver { ptr }) }
    }

    /// 在 GPU 上执行 BGK 碰撞。
    pub fn collide(&mut self) {
        unsafe { ffi::lbm_gpu_collide(self.ptr) };
    }

    /// 在 GPU 上执行流式迁移。
    pub fn stream(&mut self) {
        unsafe { ffi::lbm_gpu_stream(self.ptr) };
    }

    /// 在 GPU 上更新宏观量（ρ、u）。
    pub fn compute_macroscopic(&mut self) {
        unsafe { ffi::lbm_gpu_macroscopic(self.ptr) };
    }

    /// 将 GPU 端分布函数 + 宏观量下载到 CPU 网格
    /// （用于在 CPU 上执行边界条件，下载后再调用 `upload`）。
    pub fn download(&self, grid: &mut LbmGrid) {
        unsafe { ffi::lbm_gpu_download(self.ptr, grid.as_mut_ptr()) };
    }

    /// 将 CPU 网格 f 上传到 GPU（CPU 端边界条件修正后调用）。
    pub fn upload(&mut self, grid: &mut LbmGrid) {
        unsafe { ffi::lbm_gpu_upload(self.ptr, grid.as_mut_ptr()) };
    }
}

impl Drop for LbmGpuSolver {
    fn drop(&mut self) {
        unsafe { ffi::lbm_gpu_solver_free(self.ptr) };
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
/// `#[derive(Default)]`：自动生成 `default()` 方法，将所有 `Option` 字段初始化为
/// `None`，将所有 `*mut c_void` 字段初始化为 `std::ptr::null_mut()`。
/// 这样调用方只需填写需要激活的回调，其余字段保持默认值（空操作）。
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
// `unsafe impl Send` 允许将此结构体的所有权转移到其他线程；
// `unsafe impl Sync` 允许多个线程同时持有其不可变引用（读取函数指针）。
// 两者都是"我作为封装作者承诺已手动验证安全性"的显式声明。
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
    // C++ 侧: lbm_set_plugins() — core/src/plugins/plugin_registry.cpp
    // 将各函数指针写入对应的静态适配器槽（BoundaryAdapter / MeshAdapter 等），
    // 再通过 PluginRegistry::set_*_plugin() 注册到单例
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
