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
    use std::ffi::{c_int, c_char, c_void};

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
    /// 指向堆上 `lbm::MpiDecomp` 的不透明句柄（一维 Y 切片，向后兼容）。
    pub enum MpiDecompHandle {}
    /// 指向堆上 `lbm::MpiDecomp2D` 的不透明句柄（二维 XY 块分解，1D 为特例）。
    pub enum MpiDecomp2DHandle {}
    /// 指向堆上 `lbm::MpiDecomp3D` 的不透明句柄（三维 XYZ 块分解，预留接口）。
    pub enum MpiDecomp3DHandle {}
    /// 指向堆上 `ibm::MarkerSet` 的不透明句柄（Lagrangian 标记点集）。
    pub enum IbmMarkerSetHandle {}
    /// 指向堆上 `lbm::GpuSolver` 的不透明句柄（CUDA 加速）。
    pub enum GpuSolverHandle {}
    /// 指向堆上 `lbm::MgTree` 的不透明句柄（多重网格嵌套关系树）。
    pub enum MgTreeHandle {}
    /// 指向 `lbm::MgNode` 的不透明句柄（由树管理所有权，不由 Rust 释放）。
    pub enum MgNodeHandle {}
    /// 指向堆上 `RigidBodySolver2D` 的不透明句柄（2D 刚体动力学求解器）。
    pub enum RigidBody2DHandle {}

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
        pub fn lbm_grid_zero_force(g: *mut LatticeGridHandle);
        pub fn lbm_grid_nx(g: *const LatticeGridHandle) -> c_int;
        pub fn lbm_grid_ny(g: *const LatticeGridHandle) -> c_int;
        pub fn lbm_grid_nz(g: *const LatticeGridHandle) -> c_int;
        pub fn lbm_grid_rho(g: *const LatticeGridHandle, idx: c_int) -> f64;
        pub fn lbm_grid_fill_rho(g: *mut LatticeGridHandle, rho0: f64);
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
        /// 将每个进程的一个 int 值 gather 到 root 进程的 recv_buf。
        /// 非 root 进程的 recv_buf 传 null；未启用 MPI 时直接复制。
        pub fn lbm_mpi_gather_int(send_val: c_int, recv_buf: *mut c_int, root: c_int);
        /// MPI_Gatherv：将各进程变长 double 数组 gather 到 root 进程。
        /// 非 root 进程的 recv_buf/recv_counts/displs 传 null；未启用 MPI 时直接复制。
        pub fn lbm_mpi_gatherv_f64(send_buf: *const f64, send_count: c_int,
                                    recv_buf: *mut f64,
                                    recv_counts: *const c_int,
                                    displs: *const c_int,
                                    root: c_int);
        /// MPI 屏障同步；未启用 MPI 时为空操作。
        pub fn lbm_mpi_barrier();
        /// 对所有进程的一个 f64 执行 MPI_Allreduce(SUM)；未启用 MPI 时将 in_val 复制到 *out_val。
        pub fn lbm_mpi_allreduce_sum_f64(in_val: f64, out_val: *mut f64);

        // --- MPI 二维块分解接口 — 实现于 core/src/capi/lbm_capi.cpp ---
        /// 创建 2D 块分解（px×py）；px*py 必须等于 MPI 进程数，否则返回 null。
        pub fn lbm_mpi_decomp2d_new(global_nx: c_int, global_ny: c_int,
                                     px: c_int, py: c_int) -> *mut MpiDecomp2DHandle;
        /// 同 lbm_mpi_decomp2d_new，但额外指定每侧幽灵层数 n_ghost（≥1）。
        /// n_ghost=2 为 FourPoint IBM 核在 MPI 边界附近提供 2 层幽灵行/列。
        pub fn lbm_mpi_decomp2d_new_n(global_nx: c_int, global_ny: c_int,
                                       px: c_int, py: c_int,
                                       n_ghost: c_int) -> *mut MpiDecomp2DHandle;
        pub fn lbm_mpi_decomp2d_free(h: *mut MpiDecomp2DHandle);
        /// 将 MpiDecomp2D 绑定到求解器（启用 2D 幽灵层自动交换）。
        pub fn lbm_solver_attach_mpi2d(s: *mut SolverHandle, h: *mut MpiDecomp2DHandle);
        /// 返回本进程本地网格含幽灵列的 nx。
        pub fn lbm_mpi_decomp2d_grid_nx(h: *const MpiDecomp2DHandle) -> c_int;
        /// 返回本进程本地网格含幽灵行的 ny。
        pub fn lbm_mpi_decomp2d_grid_ny(h: *const MpiDecomp2DHandle) -> c_int;
        /// 返回本进程物理区域在全局坐标系中的 x 起始坐标。
        pub fn lbm_mpi_decomp2d_x_start(h: *const MpiDecomp2DHandle) -> c_int;
        /// 返回本进程物理区域在全局坐标系中的 y 起始坐标。
        pub fn lbm_mpi_decomp2d_y_start(h: *const MpiDecomp2DHandle) -> c_int;
        /// 返回本进程物理列数（不含幽灵列）。
        pub fn lbm_mpi_decomp2d_local_nx(h: *const MpiDecomp2DHandle) -> c_int;
        /// 返回本进程物理行数（不含幽灵行）。
        pub fn lbm_mpi_decomp2d_local_ny(h: *const MpiDecomp2DHandle) -> c_int;
        /// 返回物理区域在本地网格中的 X 偏移（0 = 无西幽灵列；1 = 有西幽灵列）。
        pub fn lbm_mpi_decomp2d_phys_x0(h: *const MpiDecomp2DHandle) -> c_int;
        /// 返回物理区域在本地网格中的 Y 偏移（0 = 无南幽灵行；1 = 有南幽灵行）。
        pub fn lbm_mpi_decomp2d_phys_y0(h: *const MpiDecomp2DHandle) -> c_int;

        // --- MPI 三维块分解接口（Z 方向幽灵层交换已实现）---
        pub fn lbm_mpi_decomp3d_new(gnx: c_int, gny: c_int, gnz: c_int,
                                    px: c_int, py: c_int, pz: c_int) -> *mut MpiDecomp3DHandle;
        pub fn lbm_mpi_decomp3d_free(h: *mut MpiDecomp3DHandle);
        pub fn lbm_mpi_decomp3d_grid_nx(h: *const MpiDecomp3DHandle) -> c_int;
        pub fn lbm_mpi_decomp3d_grid_ny(h: *const MpiDecomp3DHandle) -> c_int;
        pub fn lbm_mpi_decomp3d_grid_nz(h: *const MpiDecomp3DHandle) -> c_int;
        pub fn lbm_mpi_decomp3d_x_start(h: *const MpiDecomp3DHandle) -> c_int;
        pub fn lbm_mpi_decomp3d_y_start(h: *const MpiDecomp3DHandle) -> c_int;
        pub fn lbm_mpi_decomp3d_z_start(h: *const MpiDecomp3DHandle) -> c_int;
        pub fn lbm_mpi_decomp3d_local_nx(h: *const MpiDecomp3DHandle) -> c_int;
        pub fn lbm_mpi_decomp3d_local_ny(h: *const MpiDecomp3DHandle) -> c_int;
        pub fn lbm_mpi_decomp3d_local_nz(h: *const MpiDecomp3DHandle) -> c_int;
        pub fn lbm_mpi_decomp3d_phys_x0(h: *const MpiDecomp3DHandle) -> c_int;
        pub fn lbm_mpi_decomp3d_phys_y0(h: *const MpiDecomp3DHandle) -> c_int;
        pub fn lbm_mpi_decomp3d_phys_z0(h: *const MpiDecomp3DHandle) -> c_int;
        /// 将 MpiDecomp3D 绑定到求解器（启用 3D 幽灵层自动交换）。
        pub fn lbm_solver_attach_mpi3d(s: *mut SolverHandle, h: *mut MpiDecomp3DHandle);

        // --- 多重网格树接口 — 实现于 core/src/capi/lbm_capi.cpp ---
        /// 创建多重网格树。x0..x1 × y0..y1 × z0..z1 为根节点（最粗网格）空间范围。
        /// 2D 时令 z0=z1=0, is_3d=0；3D 时令 is_3d=1。
        pub fn lbm_mg_tree_new(x0: c_int, x1: c_int, y0: c_int, y1: c_int,
                               z0: c_int, z1: c_int, is_3d: c_int) -> *mut MgTreeHandle;
        pub fn lbm_mg_tree_free(h: *mut MgTreeHandle);
        /// 在父节点内添加细化子区域。refine_ratio=2 表示每格加密为 2×2（2D）或 2×2×2（3D）。
        pub fn lbm_mg_tree_add_level(tree: *mut MgTreeHandle, parent: *mut MgNodeHandle,
                                     x0: c_int, x1: c_int, y0: c_int, y1: c_int,
                                     z0: c_int, z1: c_int,
                                     refine_ratio: c_int) -> *mut MgNodeHandle;
        pub fn lbm_mg_tree_root(h: *mut MgTreeHandle) -> *mut MgNodeHandle;
        pub fn lbm_mg_tree_max_level(h: *const MgTreeHandle) -> c_int;
        pub fn lbm_mg_tree_node_count(h: *const MgTreeHandle) -> c_int;
        pub fn lbm_mg_node_set_grid(node: *mut MgNodeHandle, grid: *mut LatticeGridHandle);
        pub fn lbm_mg_node_x_start(h: *const MgNodeHandle) -> c_int;
        pub fn lbm_mg_node_x_end  (h: *const MgNodeHandle) -> c_int;
        pub fn lbm_mg_node_y_start(h: *const MgNodeHandle) -> c_int;
        pub fn lbm_mg_node_y_end  (h: *const MgNodeHandle) -> c_int;
        pub fn lbm_mg_node_z_start(h: *const MgNodeHandle) -> c_int;
        pub fn lbm_mg_node_z_end  (h: *const MgNodeHandle) -> c_int;
        pub fn lbm_mg_node_level  (h: *const MgNodeHandle) -> c_int;
        pub fn lbm_mg_node_refine_ratio(h: *const MgNodeHandle) -> c_int;
        pub fn lbm_mg_node_child_count (h: *const MgNodeHandle) -> c_int;
        /// 将 Solver 绑定到多重网格节点（mg_step_recursive 需要每层均设置 solver）。
        pub fn lbm_mg_node_set_solver(node: *mut MgNodeHandle, solver: *mut SolverHandle);
        /// 递归多重网格时间步推进（Lagrava 2012 五步算法）。
        /// 返回 0=成功, -1=参数错误（node 为 nullptr 或节点未绑定 solver/grid）。
        pub fn lbm_mg_step_recursive(node: *mut MgNodeHandle, fringe_width: c_int) -> c_int;

        // --- OpenMP 线程数设置 — 实现于 core/src/capi/lbm_capi.cpp ---
        /// 设置 OpenMP 线程数（等价于 omp_set_num_threads()）。
        pub fn lbm_omp_set_num_threads(n: c_int);

        // --- GPU（CUDA）接口 — 实现于 core/src/capi/lbm_capi.cpp ---
        pub fn lbm_gpu_solver_new(g: *mut LatticeGridHandle, omega: f64) -> *mut GpuSolverHandle;
        pub fn lbm_gpu_solver_free(h: *mut GpuSolverHandle);
        pub fn lbm_gpu_collide(h: *mut GpuSolverHandle);
        pub fn lbm_gpu_stream(h: *mut GpuSolverHandle);
        pub fn lbm_gpu_macroscopic(h: *mut GpuSolverHandle);
        // 同步下载（全量 f+f_tmp+rho+u，44 MB @512²）
        pub fn lbm_gpu_download(h: *mut GpuSolverHandle, g: *mut LatticeGridHandle);
        // 同步下载（仅 rho+u，6.3 MB @512²，用于输出快照）
        pub fn lbm_gpu_download_rho_u(h: *mut GpuSolverHandle, g: *mut LatticeGridHandle);
        pub fn lbm_gpu_upload(h: *mut GpuSolverHandle, g: *mut LatticeGridHandle);
        // 异步输出管线
        pub fn lbm_gpu_step_async(h: *mut GpuSolverHandle);
        pub fn lbm_gpu_wait_compute(h: *mut GpuSolverHandle);
        pub fn lbm_gpu_enqueue_async_download_rho_u(h: *mut GpuSolverHandle, buf_idx: c_int);
        pub fn lbm_gpu_sync_async_download(h: *mut GpuSolverHandle);
        pub fn lbm_gpu_pinned_rho(h: *const GpuSolverHandle, buf_idx: c_int) -> *const f64;
        pub fn lbm_gpu_pinned_u  (h: *const GpuSolverHandle, buf_idx: c_int) -> *const f64;
        pub fn lbm_gpu_n         (h: *const GpuSolverHandle) -> c_int;

        // --- 并行状态查询 — 实现于 core/src/capi/lbm_capi.cpp ---
        pub fn lbm_openmp_enabled()     -> c_int;
        pub fn lbm_openmp_max_threads() -> c_int;
        pub fn lbm_cuda_enabled()       -> c_int;
        pub fn lbm_mpi_enabled()        -> c_int;

        // --- 固体边界（BB / IBB）—— 实现于 core/src/capi/lbm_capi.cpp ---
        /// 将圆柱标记为固体并计算 IBB 距离分数 q。
        pub fn lbm_mark_solid_cylinder(g: *mut LatticeGridHandle,
                                       cx: f64, cy: f64, radius: f64);
        /// 将矩形区域标记为固体（q=0.5）。
        pub fn lbm_mark_solid_rectangle(g: *mut LatticeGridHandle,
                                        i0: c_int, j0: c_int,
                                        i1: c_int, j1: c_int);
        /// 从外部 CSV 网格文件加载固体边界（第三方网格接口）。
        /// 文件格式：每行 "x, y [, q]"；x/y 为格子坐标，q 为 IBB 距离分数（可选，缺省 0.5）。
        pub fn lbm_mark_solid_from_mesh_file(g: *mut LatticeGridHandle,
                                              filename: *const c_char);
        /// 设置固体反弹方案：0=None, 1=BB, 2=IBB。
        pub fn lbm_solver_set_solid_bc(s: *mut SolverHandle, bc_mode: c_int);
        /// 将当前所有已标记固体节点（solid==1 且 solid_bc_node==0）的逐节点反弹方案设为 bc_mode。
        /// 应在每个固体体 lbm_mark_solid_*() 调用后立即调用，以实现逐体方案分配。
        /// bc_mode: 1=BB, 2=IBB, 0=无操作。
        pub fn lbm_grid_assign_solid_bc_unmarked(g: *mut LatticeGridHandle, bc_mode: c_int);
        /// 用动量交换法（MEA）计算固体受力。
        /// 调用时机：apply_solid_bounce_back()/apply_solid_ibb() 之后（即 step() 之后）。
        /// phys_i0/j0/i1/j1：本进程物理区域范围（非 MPI 时传 0/0/nx-1/ny-1）。
        /// MPI 模式下本函数仅统计本进程贡献，调用方需额外通过 MPI_Allreduce 求全局和。
        pub fn lbm_compute_solid_force(g: *const LatticeGridHandle,
                                       phys_i0: c_int, phys_j0: c_int,
                                       phys_i1: c_int, phys_j1: c_int,
                                       out_fx: *mut f64, out_fy: *mut f64);

        // --- 运动刚体 BB/IBB ---
        /// 清除所有固体节点标记（运动固体每步重标记前调用）。
        pub fn lbm_clear_solid(g: *mut LatticeGridHandle);
        /// 运动刚体半步长反弹（Ladd 1994 移动壁面修正）。
        pub fn lbm_apply_solid_bb_moving_rigid(g: *mut LatticeGridHandle,
            cx: f64, cy: f64, ux_cm: f64, uy_cm: f64, omega: f64,
            phys_i0: c_int, phys_j0: c_int, phys_i1: c_int, phys_j1: c_int);
        /// 运动刚体 Bouzidi IBB（Ladd 移动壁面修正）。
        pub fn lbm_apply_solid_ibb_moving_rigid(g: *mut LatticeGridHandle,
            cx: f64, cy: f64, ux_cm: f64, uy_cm: f64, omega: f64,
            phys_i0: c_int, phys_j0: c_int, phys_i1: c_int, phys_j1: c_int);

        // --- IBM 浸入边界法 — 实现于 core/src/capi/lbm_capi.cpp (IBM section) ---
        /// 创建圆柱表面标记点集（均匀分布）。
        pub fn lbm_ibm_marker_set_new_circle(cx: f64, cy: f64, radius: f64, n_markers: c_int)
            -> *mut IbmMarkerSetHandle;
        /// 创建直线丝状体标记点集（沿 x 轴均匀分布）。
        pub fn lbm_ibm_marker_set_new_filament(x0: f64, y0: f64, length: f64, n_markers: c_int)
            -> *mut IbmMarkerSetHandle;
        /// 从坐标数组创建标记点集（Python FFI / 外部网格接口）。
        /// ds 为 nullptr 时自动由相邻点距计算弧长元素。
        /// 返回 nullptr 若 n_markers ≤ 0 或 x/y 为 nullptr。
        pub fn lbm_ibm_marker_set_from_coords(x: *const f64, y: *const f64,
                                               ds: *const f64, n_markers: c_int)
            -> *mut IbmMarkerSetHandle;
        /// 从 CSV 文件加载标记点（第三方网格接口）。
        /// 文件格式：每行 "x, y [, z [, ds]]"；忽略 '#' 注释行和空行。
        /// 返回 nullptr 若文件无法打开或格式错误。
        pub fn lbm_ibm_marker_set_from_file(filename: *const c_char)
            -> *mut IbmMarkerSetHandle;
        /// 释放标记点集。
        pub fn lbm_ibm_marker_set_free(h: *mut IbmMarkerSetHandle);
        /// 返回标记点数量。
        pub fn lbm_ibm_marker_set_size(h: *const IbmMarkerSetHandle) -> c_int;
        /// MDF-IBM 一步：多重直接力法（n_iter 子迭代）。
        pub fn lbm_ibm_compute_mdf(g: *mut LatticeGridHandle,
                                   ms: *mut IbmMarkerSetHandle,
                                   dx: f64, dt: f64, n_iter: c_int);
        /// Penalty-IBM 一步：罚函数反馈力。integral_x/y 调用方管理（每步传入同一指针）。
        pub fn lbm_ibm_compute_penalty(g: *mut LatticeGridHandle,
                                       ms: *mut IbmMarkerSetHandle,
                                       dx: f64, dt: f64,
                                       alpha: f64, beta: f64,
                                       integral_x: *mut f64,
                                       integral_y: *mut f64,
                                       u_target_x: f64, u_target_y: f64);
        /// MLS-IBM 隐式一步：MLS 插值 + MLS 伴随展布 + 多步迭代修正（n_iter=3）。
        /// 相比旧接口，使用 MLS 伴随展布替代 Peskin δ，满足离散伴随一致性。
        pub fn lbm_ibm_compute_mls(g: *mut LatticeGridHandle,
                                   ms: *mut IbmMarkerSetHandle,
                                   dx: f64, dt: f64);
        /// MLS-IBM 隐式一步（可指定迭代次数和目标速度）。
        /// n_iter: 迭代次数（建议 2–4）；u_target_x/y: 目标速度（静止固体取 0.0）。
        pub fn lbm_ibm_compute_mls_implicit(g: *mut LatticeGridHandle,
                                            ms: *mut IbmMarkerSetHandle,
                                            dx: f64, dt: f64,
                                            n_iter: c_int,
                                            u_target_x: f64, u_target_y: f64);
        /// 原始 MLS-IBM 一步（MLS 插值 + MLS 形状函数展布，Algorithm 1，JCP 2025）。
        pub fn lbm_ibm_compute_mls_original(g: *mut LatticeGridHandle,
                                             ms: *mut IbmMarkerSetHandle,
                                             dx: f64, dt: f64);
        /// 显式 MLS-IBM 一步（MLS 插值 + MLS 展布 + Z 修正，Algorithm 2，JCP 2025）。
        pub fn lbm_ibm_compute_mls_explicit(g: *mut LatticeGridHandle,
                                             ms: *mut IbmMarkerSetHandle,
                                             dx: f64, dt: f64);
        /// IVC-IBM 一步（隐式速度校正，Wu & Shu 2009）。每步重建矩阵 A，适用于移动物体。
        pub fn lbm_ibm_compute_ivc(g: *mut LatticeGridHandle,
                                    ms: *mut IbmMarkerSetHandle,
                                    dx: f64, dt: f64);
        /// IVC-IBM 一步（固定物体，LU 分解缓存）。首次调用构建并缓存 A 的 LU 分解，
        /// 后续步骤直接 LU 代换，更高效。*cache_handle 首次调用前须为 null。
        pub fn lbm_ibm_compute_ivc_stationary(g: *mut LatticeGridHandle,
                                               ms: *mut IbmMarkerSetHandle,
                                               dx: f64, dt: f64,
                                               cache_handle: *mut *mut c_void);
        /// 释放由 lbm_ibm_compute_ivc_stationary() 分配的缓存。
        pub fn lbm_ibm_ivc_stationary_cache_free(cache_handle: *mut *mut c_void);
        /// 读取 Lagrangian 力 (fx, fy)；out_fx/out_fy 长度须 ≥ size()。
        pub fn lbm_ibm_get_forces(ms: *const IbmMarkerSetHandle,
                                  out_fx: *mut f64, out_fy: *mut f64);
        /// 读取所有标记点当前坐标 (x, y)；out_x/out_y 长度须 ≥ size()。
        pub fn lbm_ibm_get_positions(ms: *const IbmMarkerSetHandle,
                                     out_x: *mut f64, out_y: *mut f64);
        /// 计算 IBM 固体受力合力（力密度与弧长元素的加权和）。
        ///   out_fx = Σ mk.fx * mk.ds,  out_fy = Σ mk.fy * mk.ds
        /// 固体所受流体合力为其负值（牛顿第三定律）。
        /// 调用时机：任一 IBM 力计算函数之后。
        pub fn lbm_ibm_compute_body_force(ms: *const IbmMarkerSetHandle,
                                           out_fx: *mut f64, out_fy: *mut f64);

        /// IBM 标记点集 MPI 分区适配：坐标系从全局转换到本地，设置归属行/列范围。
        /// 调用后 interpolate_velocity/spread_force 仅处理"归属"本进程的标记点，
        /// 避免跨 MPI 块时的 IBM 力双重计数。
        pub fn lbm_ibm_marker_set_adapt_to_partition(ms: *mut IbmMarkerSetHandle,
                                                      x_start: c_int, y_start: c_int,
                                                      phys_x0: c_int, phys_y0: c_int,
                                                      local_nx: c_int, local_ny: c_int);

        /// IBM 幽灵层 u 场交换：step_ibm() 前调用，填充幽灵行/列的真实邻居速度。
        /// 修正 solver.step() 后幽灵行 u 来自本地外推（而非邻居速度）的问题。
        pub fn lbm_ibm_halo_exchange_u_2d(g: *mut LatticeGridHandle,
                                           h: *mut MpiDecomp2DHandle);

        /// IBM 幽灵层力场归并：spread_force() 后调用，将幽灵行/列力贡献归还邻居并累加。
        pub fn lbm_ibm_halo_reduce_force_2d(g: *mut LatticeGridHandle,
                                             h: *mut MpiDecomp2DHandle);

        // --- 刚体求解器 C-API（RigidBodySolver2D）---

        /// 创建 2D 刚体求解器。scheme: 0=None,1=Uhlmann,2=Feng,3=Lagrangian
        pub fn lbm_rigid2d_create(
            mass: f64, inertia: f64,
            rho_b: f64, rho_f: f64,
            cx0: f64, cy0: f64,
            scheme: c_int, is_closed: c_int,
            ref_x: *const f64, ref_y: *const f64, n_bnd: c_int,
            int_ref_x: *const f64, int_ref_y: *const f64, n_int: c_int,
        ) -> *mut RigidBody2DHandle;
        pub fn lbm_rigid2d_free(h: *mut RigidBody2DHandle);
        pub fn lbm_rigid2d_get_state(h: *const RigidBody2DHandle,
            cx: *mut f64, cy: *mut f64,
            ux: *mut f64, uy: *mut f64,
            theta: *mut f64, omega: *mut f64);
        pub fn lbm_rigid2d_set_velocity(h: *mut RigidBody2DHandle,
            ux: f64, uy: f64, omega: f64);
        pub fn lbm_rigid2d_n_boundary(h: *const RigidBody2DHandle) -> c_int;
        pub fn lbm_rigid2d_get_boundary_positions(h: *const RigidBody2DHandle,
            out_x: *mut f64, out_y: *mut f64);
        pub fn lbm_rigid2d_get_boundary_velocities(h: *const RigidBody2DHandle,
            out_ux: *mut f64, out_uy: *mut f64);
        pub fn lbm_rigid2d_n_internal(h: *const RigidBody2DHandle) -> c_int;
        pub fn lbm_rigid2d_get_internal_positions(h: *const RigidBody2DHandle,
            out_x: *mut f64, out_y: *mut f64);
        pub fn lbm_rigid2d_set_internal_velocities(h: *mut RigidBody2DHandle,
            ux: *const f64, uy: *const f64);
        pub fn lbm_rigid2d_compute_internal_momentum(h: *mut RigidBody2DHandle);
        pub fn lbm_rigid2d_advance(h: *mut RigidBody2DHandle,
            total_fx: f64, total_fy: f64, total_torque: f64, dt: f64);

        // --- IBM 移动体辅助函数 ---
        pub fn lbm_ibm_compute_mdf_moving(g: *mut LatticeGridHandle,
            ms: *mut IbmMarkerSetHandle,
            dx: f64, dt: f64, n_iter: c_int,
            target_ux: *const f64, target_uy: *const f64);
        pub fn lbm_ibm_compute_body_force_torque(
            ms: *const IbmMarkerSetHandle,
            cx: f64, cy: f64,
            out_fx: *mut f64, out_fy: *mut f64, out_torque: *mut f64);
        pub fn lbm_ibm_interpolate_only(g: *const LatticeGridHandle,
            ms: *mut IbmMarkerSetHandle, dx: f64);
        /// 在任意位置插值流体速度（TwoPoint δ 核，适用于内部拉格朗日点）。
        /// out_ux/out_uy 长度须 ≥ n。
        pub fn lbm_ibm_interpolate_at_points(g: *const LatticeGridHandle,
            x: *const f64, y: *const f64, n: c_int,
            dx: f64,
            out_ux: *mut f64, out_uy: *mut f64);
        pub fn lbm_ibm_get_marker_velocities(ms: *const IbmMarkerSetHandle,
            out_ux: *mut f64, out_uy: *mut f64);
        pub fn lbm_ibm_update_marker_positions(ms: *mut IbmMarkerSetHandle,
            x: *const f64, y: *const f64, n: c_int);

        // --- IBM 柔性体逐标记点目标速度 ---
        /// 设置所有标记点的统一目标速度（静止体：0,0）。
        pub fn lbm_ibm_set_uniform_target(ms: *mut IbmMarkerSetHandle, ux: f64, uy: f64);
        /// 设置各标记点的逐点目标速度（柔性体 / 旋转刚体）。
        pub fn lbm_ibm_set_marker_targets(ms: *mut IbmMarkerSetHandle,
            ux: *const f64, uy: *const f64, n: c_int);
        /// 读取各标记点的目标速度。
        pub fn lbm_ibm_get_marker_targets(ms: *const IbmMarkerSetHandle,
            out_ux: *mut f64, out_uy: *mut f64);
        /// 根据刚体运动状态（质心速度 + 角速度）为所有标记点设置目标速度。
        pub fn lbm_ibm_set_rigid_body_targets(ms: *mut IbmMarkerSetHandle,
            cx: f64, cy: f64, ux_cm: f64, uy_cm: f64, omega: f64);

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

    /// 将所有节点的初始密度设为 rho0。
    ///
    /// 必须在 [`LbmSolver::new`] **之前**调用，使求解器构造函数用 rho0 初始化平衡态。
    /// 若不调用，默认 rho0 = 1.0（格子单位）。
    pub fn fill_rho(&mut self, rho0: f64) {
        unsafe { ffi::lbm_grid_fill_rho(self.ptr, rho0) }
    }
    pub fn ux (&self, idx: i32) -> f64 { unsafe { ffi::lbm_grid_ux (self.ptr, idx) } }
    pub fn uy (&self, idx: i32) -> f64 { unsafe { ffi::lbm_grid_uy (self.ptr, idx) } }

    /// 将体力场清零（每个 IBM 时间步开始前调用，防止上一步残留力场被累积）。
    pub fn zero_force(&mut self) { unsafe { ffi::lbm_grid_zero_force(self.ptr) } }

    /// 在任意位置插值流体速度（TwoPoint δ 核，适用于内部拉格朗日点）。
    ///
    /// 将欧拉流体速度场 u 插值到给定的 n 个点 (x[], y[]) 处。
    /// x/y 须等长；返回 (out_ux, out_uy) 两个向量，长度与输入相同。
    ///
    /// 用途：为 Lagrangian 方案（`internal_mass_scheme = "lagrangian"`）的
    /// 内部拉格朗日点插值流体速度 u*(t)，供 `compute_internal_momentum()` 使用。
    pub fn interpolate_at_points(&self, x: &[f64], y: &[f64], dx: f64)
        -> (Vec<f64>, Vec<f64>)
    {
        let n = x.len().min(y.len());
        let mut out_ux = vec![0.0_f64; n];
        let mut out_uy = vec![0.0_f64; n];
        if n > 0 {
            unsafe {
                ffi::lbm_ibm_interpolate_at_points(
                    self.ptr as *const _,
                    x.as_ptr(), y.as_ptr(), n as i32,
                    dx,
                    out_ux.as_mut_ptr(), out_uy.as_mut_ptr())
            }
        }
        (out_ux, out_uy)
    }

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

/// 将每个进程的单个 `i32` 值 gather 到 root 进程，返回长度为 `nprocs` 的 Vec。
///
/// 非 root 进程返回空 Vec；未启用 MPI 时（nprocs=1）返回 `vec![send_val]`。
pub fn mpi_gather_int(send_val: i32, root: i32) -> Vec<i32> {
    let nprocs = mpi_size() as usize;
    let rank   = mpi_rank();
    if rank == root {
        let mut buf = vec![0i32; nprocs];
        unsafe { ffi::lbm_mpi_gather_int(send_val, buf.as_mut_ptr(), root) };
        buf
    } else {
        unsafe { ffi::lbm_mpi_gather_int(send_val, std::ptr::null_mut(), root) };
        Vec::new()
    }
}

/// MPI_Gatherv：将各进程的变长 `f64` 数组 gather 到 root 进程。
///
/// - `data`   — 本进程要发送的数据切片
/// - `root`   — 根进程编号
///
/// root 进程返回拼合后的全局 Vec（行主序）；非 root 进程返回空 Vec。
/// 未启用 MPI 时（nprocs=1）直接返回 `data` 的克隆。
///
/// 调用方需先通过 [`mpi_gather_int`] 收集各进程的 `send_count`，
/// 再调用本函数。
pub fn mpi_gatherv_f64(data: &[f64], recv_counts: &[i32], displs: &[i32], root: i32) -> Vec<f64> {
    let rank = mpi_rank();
    let total: i32 = recv_counts.iter().sum();
    if rank == root {
        let mut buf = vec![0.0f64; total as usize];
        unsafe {
            ffi::lbm_mpi_gatherv_f64(
                data.as_ptr(),       data.len() as i32,
                buf.as_mut_ptr(),    recv_counts.as_ptr(),
                displs.as_ptr(),     root,
            );
        }
        buf
    } else {
        unsafe {
            ffi::lbm_mpi_gatherv_f64(
                data.as_ptr(),        data.len() as i32,
                std::ptr::null_mut(), std::ptr::null(),
                std::ptr::null(),     root,
            );
        }
        Vec::new()
    }
}

/// MPI 屏障同步；未启用 MPI 时为空操作。
pub fn mpi_barrier() {
    unsafe { ffi::lbm_mpi_barrier() };
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

    println!("Parallel config:");
    if omp_on {
        let threads = unsafe { ffi::lbm_openmp_max_threads() };
        println!("  OpenMP : enabled   threads = {} (controlled by OMP_NUM_THREADS)", threads);
    } else {
        println!("  OpenMP : disabled");
    }
    if mpi_on {
        let rank  = mpi_rank();
        let procs = mpi_size();
        println!("  MPI    : enabled   nprocs = {}  rank = {}", procs, rank);
    } else {
        println!("  MPI    : disabled");
    }
    if cuda_on {
        println!("  GPU    : enabled (CUDA)");
    } else {
        println!("  GPU    : disabled");
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
    /// 绑定 MPI 一维（Y 方向）域分解：之后每次 `step()` 的流式迁移后自动执行幽灵行交换。
    /// 传入 `None` 可解除绑定（恢复单进程模式）。
    pub fn attach_mpi(&mut self, decomp: Option<&mut LbmMpiDecomp>) {
        let h = decomp.map_or(std::ptr::null_mut(), |d| d.as_mut_ptr());
        unsafe { ffi::lbm_solver_attach_mpi(self.ptr, h) };
    }

    /// 绑定 MPI 二维（XY 方向）块分解：之后每次 `step()` 自动执行 2D 幽灵层交换。
    /// 传入 `None` 可解除绑定。
    pub fn attach_mpi2d(&mut self, decomp: Option<&mut LbmMpiDecomp2D>) {
        let h = decomp.map_or(std::ptr::null_mut(), |d| d.as_mut_ptr());
        unsafe { ffi::lbm_solver_attach_mpi2d(self.ptr, h) };
    }

    /// 绑定 MPI 三维（XYZ 方向）块分解：之后每次 `step()` 自动执行 3D 幽灵层交换。
    /// 传入 `None` 可解除绑定。
    pub fn attach_mpi3d(&mut self, decomp: Option<&mut LbmMpiDecomp3D>) {
        let h = decomp.map_or(std::ptr::null_mut(), |d| d.as_mut_ptr());
        unsafe { ffi::lbm_solver_attach_mpi3d(self.ptr, h) };
    }
}

// ---------------------------------------------------------------------------
/// `lbm::MpiDecomp2D` 的安全封装（二维 XY 块分解）
// ---------------------------------------------------------------------------
pub struct LbmMpiDecomp2D {
    ptr: *mut ffi::MpiDecomp2DHandle,
}

unsafe impl Send for LbmMpiDecomp2D {}

impl LbmMpiDecomp2D {
    /// 创建 MPI 二维块分解（需先调用 `mpi_init()`），每侧使用 1 个幽灵层。
    ///
    /// - `global_nx`：全局 X 方向节点数
    /// - `global_ny`：全局 Y 方向节点数
    /// - `px`：X 方向进程数
    /// - `py`：Y 方向进程数
    ///
    /// `px * py` 必须等于 MPI 进程总数，否则返回 `None`。
    /// 未启用 MPI 时也返回 `None`。
    pub fn new(global_nx: i32, global_ny: i32, px: i32, py: i32) -> Option<Self> {
        let ptr = unsafe { ffi::lbm_mpi_decomp2d_new(global_nx, global_ny, px, py) };
        if ptr.is_null() { None } else { Some(LbmMpiDecomp2D { ptr }) }
    }

    /// 与 [`LbmMpiDecomp2D::new`] 相同，但额外指定每侧幽灵层数 `n_ghost`（≥1）。
    ///
    /// - `n_ghost = 1`（默认）：适用于 TwoPoint IBM 核。
    /// - `n_ghost = 2`：适用于 FourPoint IBM 核在 MPI 边界附近的标记点；
    ///   此时 `grid_ny()` 增大 `2*(n_ghost-1)` 行，`phys_y0()` 返回 `n_ghost`。
    ///
    /// 对应 TOML 配置 `[mpi] ibm_halo_width`。
    pub fn new_n(global_nx: i32, global_ny: i32, px: i32, py: i32, n_ghost: i32) -> Option<Self> {
        let ptr = unsafe { ffi::lbm_mpi_decomp2d_new_n(global_nx, global_ny, px, py, n_ghost) };
        if ptr.is_null() { None } else { Some(LbmMpiDecomp2D { ptr }) }
    }

    /// 本进程本地网格含幽灵列的 nx。
    pub fn grid_nx(&self) -> i32 {
        unsafe { ffi::lbm_mpi_decomp2d_grid_nx(self.ptr as *const _) }
    }

    /// 本进程本地网格含幽灵行的 ny。
    pub fn grid_ny(&self) -> i32 {
        unsafe { ffi::lbm_mpi_decomp2d_grid_ny(self.ptr as *const _) }
    }

    /// 本进程物理区域在全局坐标系中的 x 起始坐标。
    pub fn x_start(&self) -> i32 {
        unsafe { ffi::lbm_mpi_decomp2d_x_start(self.ptr as *const _) }
    }

    /// 本进程物理区域在全局坐标系中的 y 起始坐标。
    pub fn y_start(&self) -> i32 {
        unsafe { ffi::lbm_mpi_decomp2d_y_start(self.ptr as *const _) }
    }

    /// 本进程物理列数（不含幽灵列）。
    pub fn local_nx(&self) -> i32 {
        unsafe { ffi::lbm_mpi_decomp2d_local_nx(self.ptr as *const _) }
    }

    /// 本进程物理行数（不含幽灵行）。
    pub fn local_ny(&self) -> i32 {
        unsafe { ffi::lbm_mpi_decomp2d_local_ny(self.ptr as *const _) }
    }

    /// 物理区域在本地网格中的 X 偏移（0 = 无西幽灵列；n_ghost = 有西幽灵列）。
    pub fn phys_x0(&self) -> i32 {
        unsafe { ffi::lbm_mpi_decomp2d_phys_x0(self.ptr as *const _) }
    }

    /// 物理区域在本地网格中的 Y 偏移（0 = 无南幽灵行；n_ghost = 有南幽灵行）。
    pub fn phys_y0(&self) -> i32 {
        unsafe { ffi::lbm_mpi_decomp2d_phys_y0(self.ptr as *const _) }
    }

    /// 原始可变指针（仅供 LbmSolver::attach_mpi2d 内部使用）
    pub(crate) fn as_mut_ptr(&mut self) -> *mut ffi::MpiDecomp2DHandle { self.ptr }
}

impl Drop for LbmMpiDecomp2D {
    fn drop(&mut self) {
        unsafe { ffi::lbm_mpi_decomp2d_free(self.ptr) };
    }
}

// ---------------------------------------------------------------------------
/// 三维（XYZ 方向）MPI 块分解描述符（支持 D3Q19/D3Q27 幽灵层交换）
///
/// 当 `pz=1` 时等价于 [`LbmMpiDecomp2D`]；当 `pz=1, py=1` 时等价于 1D X 切片。
///
/// # 用途
/// - 三维 LBM 并行域分解（D3Q19/D3Q27）
/// - 绑定到求解器后，每步自动执行 6 方向幽灵层交换（Z/Y/X）
///
/// # 示例（框架，无 MPI 时 px*py*pz 须为 1）
/// ```no_run
/// let decomp3d = LbmMpiDecomp3D::new(256, 256, 128, 2, 2, 2)
///     .expect("需要 8 个 MPI 进程且已启用 MPI");
/// println!("local grid: {}×{}×{}",
///          decomp3d.grid_nx(), decomp3d.grid_ny(), decomp3d.grid_nz());
/// ```
pub struct LbmMpiDecomp3D {
    ptr: *mut ffi::MpiDecomp3DHandle,
}

impl LbmMpiDecomp3D {
    /// 创建三维 XYZ 块分解（需先调用 [`mpi_init()`]，且 `px*py*pz == nprocs`）。
    ///
    /// 若 MPI 未启用、`px*py*pz != nprocs` 或创建失败，返回 `None`。
    pub fn new(global_nx: i32, global_ny: i32, global_nz: i32,
               px: i32, py: i32, pz: i32) -> Option<Self> {
        let ptr = unsafe {
            ffi::lbm_mpi_decomp3d_new(global_nx, global_ny, global_nz, px, py, pz)
        };
        if ptr.is_null() { None } else { Some(LbmMpiDecomp3D { ptr }) }
    }

    /// 本地网格含幽灵层的 nx
    pub fn grid_nx(&self) -> i32 { unsafe { ffi::lbm_mpi_decomp3d_grid_nx(self.ptr) } }
    /// 本地网格含幽灵层的 ny
    pub fn grid_ny(&self) -> i32 { unsafe { ffi::lbm_mpi_decomp3d_grid_ny(self.ptr) } }
    /// 本地网格含幽灵层的 nz
    pub fn grid_nz(&self) -> i32 { unsafe { ffi::lbm_mpi_decomp3d_grid_nz(self.ptr) } }
    /// 本进程物理区域在全局坐标中的 X 起始坐标
    pub fn x_start(&self) -> i32 { unsafe { ffi::lbm_mpi_decomp3d_x_start(self.ptr) } }
    /// 本进程物理区域在全局坐标中的 Y 起始坐标
    pub fn y_start(&self) -> i32 { unsafe { ffi::lbm_mpi_decomp3d_y_start(self.ptr) } }
    /// 本进程物理区域在全局坐标中的 Z 起始坐标
    pub fn z_start(&self) -> i32 { unsafe { ffi::lbm_mpi_decomp3d_z_start(self.ptr) } }
    /// 本进程物理列数（不含幽灵列）
    pub fn local_nx(&self) -> i32 { unsafe { ffi::lbm_mpi_decomp3d_local_nx(self.ptr) } }
    /// 本进程物理行数（不含幽灵行）
    pub fn local_ny(&self) -> i32 { unsafe { ffi::lbm_mpi_decomp3d_local_ny(self.ptr) } }
    /// 本进程物理层数（不含幽灵层）
    pub fn local_nz(&self) -> i32 { unsafe { ffi::lbm_mpi_decomp3d_local_nz(self.ptr) } }
    /// 物理区域在本地网格中的 X 偏移（0 或 1）
    pub fn phys_x0(&self) -> i32 { unsafe { ffi::lbm_mpi_decomp3d_phys_x0(self.ptr) } }
    /// 物理区域在本地网格中的 Y 偏移（0 或 1）
    pub fn phys_y0(&self) -> i32 { unsafe { ffi::lbm_mpi_decomp3d_phys_y0(self.ptr) } }
    /// 物理区域在本地网格中的 Z 偏移（0 或 1）
    pub fn phys_z0(&self) -> i32 { unsafe { ffi::lbm_mpi_decomp3d_phys_z0(self.ptr) } }

    #[doc(hidden)]
    pub fn as_mut_ptr(&mut self) -> *mut ffi::MpiDecomp3DHandle { self.ptr }
}

impl Drop for LbmMpiDecomp3D {
    fn drop(&mut self) {
        unsafe { ffi::lbm_mpi_decomp3d_free(self.ptr) };
    }
}

unsafe impl Send for LbmMpiDecomp3D {}

// ---------------------------------------------------------------------------
/// 多重网格嵌套关系树（`lbm::MgTree` 的安全封装）
///
/// 管理 AMR（Adaptive Mesh Refinement）风格的网格层次结构：
/// 粗网格（level=0）作为根节点，细化网格块（patches）作为子节点，
/// 支持任意嵌套深度和数量，以及 2D/3D 两种维度。
///
/// # Traversal modes (implemented in C++)
/// - `traverse_coarse_to_fine`: coarse-to-fine (BFS), for per-level LBM time-stepping
/// - `traverse_fine_to_coarse`: fine-to-coarse (BFS reverse), for residual transfer
///
/// # Example (2D two-level nesting)
/// ```no_run
/// use lbm_bindings::{LbmMgTree, LbmGrid, LatticeModel};
///
/// // Create coarse grid (256x256, 2D)
/// let mut tree = LbmMgTree::new(0, 255, 0, 255, 0, 0, false)
///     .expect("MgTree creation failed");
///
/// // Nest a fine grid inside the coarse grid (centre 64x64, refine ratio 2)
/// let fine = tree.add_level_from_root(96, 159, 96, 159, 0, 0, 2)
///     .expect("add_level failed");
///
/// println!("tree depth: {}", tree.max_level());   // 1
/// println!("node count: {}", tree.node_count()); // 2
/// ```
pub struct LbmMgTree {
    ptr: *mut ffi::MgTreeHandle,
    /// 按添加顺序存储所有节点句柄：`nodes[0]` = 根节点，`nodes[1..]` = 各细化层节点。
    /// 索引直接用于 `set_grid_by_idx` / `set_solver_by_idx` / `add_child_level`。
    nodes: Vec<*mut ffi::MgNodeHandle>,
}

impl LbmMgTree {
    /// 创建多重网格树，根节点（最粗网格）的空间范围为
    /// `[x0, x1] × [y0, y1] × [z0, z1]`（全局格子坐标，含端点）。
    ///
    /// - 2D 仿真：令 `z0=z1=0`, `is_3d=false`
    /// - 3D 仿真：令 `is_3d=true`，`z0..z1` 给出 Z 方向范围
    pub fn new(x0: i32, x1: i32, y0: i32, y1: i32,
               z0: i32, z1: i32, is_3d: bool) -> Option<Self> {
        let ptr = unsafe {
            ffi::lbm_mg_tree_new(x0, x1, y0, y1, z0, z1, if is_3d { 1 } else { 0 })
        };
        if ptr.is_null() { return None; }
        let root = unsafe { ffi::lbm_mg_tree_root(ptr) };
        Some(LbmMgTree { ptr, nodes: vec![root] })
    }

    /// 在指定父节点（`parent_idx`）下添加一个细化子区域，返回新节点的索引。
    ///
    /// - `parent_idx = 0` 表示根节点；`parent_idx = k` 表示第 k 个已添加的节点。
    /// - 坐标 `x0..x1 × y0..y1` 为**父节点坐标系**中的格子范围（含端点）。
    /// - 返回新节点索引（可直接传给 [`set_grid_by_idx`] / [`set_solver_by_idx`]）；
    ///   若父索引越界或 C++ 侧返回 null 则返回 `None`。
    pub fn add_child_level(&mut self, parent_idx: usize,
                           x0: i32, x1: i32, y0: i32, y1: i32,
                           z0: i32, z1: i32,
                           refine_ratio: i32) -> Option<usize> {
        let parent = *self.nodes.get(parent_idx)?;
        let node = unsafe {
            ffi::lbm_mg_tree_add_level(self.ptr, parent, x0, x1, y0, y1, z0, z1, refine_ratio)
        };
        if node.is_null() { return None; }
        let idx = self.nodes.len();
        self.nodes.push(node);
        Some(idx)
    }

    /// 将 LatticeGrid 绑定到指定节点（0 = 根节点）。
    pub fn set_grid_by_idx(&mut self, idx: usize, grid: &mut LbmGrid) {
        if let Some(&node) = self.nodes.get(idx) {
            unsafe { ffi::lbm_mg_node_set_grid(node, grid.as_mut_ptr()) };
        }
    }

    /// 将 Solver 绑定到指定节点（0 = 根节点）。
    /// mg_step_recursive 要求所有节点均已绑定 Solver。
    pub fn set_solver_by_idx(&mut self, idx: usize, solver: &mut LbmSolver) {
        if let Some(&node) = self.nodes.get(idx) {
            unsafe { ffi::lbm_mg_node_set_solver(node, solver.ptr) };
        }
    }

    /// 返回树中最深的层级（根节点为 0）
    pub fn max_level(&self) -> i32 {
        unsafe { ffi::lbm_mg_tree_max_level(self.ptr) }
    }

    /// 返回树中所有节点数（包括根节点）
    pub fn node_count(&self) -> i32 {
        unsafe { ffi::lbm_mg_tree_node_count(self.ptr) }
    }

    /// 执行一次递归多重网格时间步推进（从根节点开始，Lagrava 2012 五步算法）。
    ///
    /// - 每层 MPI 幽灵层交换由各层 `Solver::step()` 内部完成，无需额外 MPI 调用。
    /// - 必须在所有节点上先调用 [`set_grid_by_idx`] 和 [`set_solver_by_idx`]。
    ///
    /// 返回 `0` 表示成功，`-1` 表示参数错误（节点未绑定 grid/solver）。
    pub fn mg_step_recursive(&mut self, fringe_width: i32) -> i32 {
        let root = unsafe { ffi::lbm_mg_tree_root(self.ptr) };
        unsafe { ffi::lbm_mg_step_recursive(root, fringe_width) }
    }
}

impl Drop for LbmMgTree {
    fn drop(&mut self) {
        unsafe { ffi::lbm_mg_tree_free(self.ptr) };
    }
}

unsafe impl Send for LbmMgTree {}

// ---------------------------------------------------------------------------
/// 设置 OpenMP 线程数（等价于 `OMP_NUM_THREADS` 环境变量，但在进程内即时生效）。
///
/// 若未以 `LBM_ENABLE_OPENMP=ON` 编译，此函数为空操作。
///
/// # 示例
///
/// ```no_run
/// lbm_bindings::set_omp_num_threads(8);  // 使用 8 个 OpenMP 线程
/// ```
pub fn set_omp_num_threads(n: i32) {
    unsafe { ffi::lbm_omp_set_num_threads(n) };
}

// ---------------------------------------------------------------------------
// 固体边界（BB / IBB）安全封装
// ---------------------------------------------------------------------------

/// 将圆柱（圆心 `(cx, cy)`，半径 `radius`）内部节点标记为固体，
/// 并精确计算每个流-固链接方向的 IBB 壁面距离分数 `q`。
///
/// 必须在创建 `LbmGrid` 之后、开始时间步循环之前调用。
/// 之后需调用 [`mark_solid_bc`] 设置反弹方案。
pub fn mark_solid_cylinder(grid: &mut LbmGrid, cx: f64, cy: f64, radius: f64) {
    unsafe { ffi::lbm_mark_solid_cylinder(grid.ptr, cx, cy, radius) };
}

/// 将矩形区域 `[i0,i1] × [j0,j1]`（格子坐标，含端点）标记为固体。
/// 矩形面上的 `q` 默认为 0.5（退化为标准半步长反弹）。
pub fn mark_solid_rectangle(grid: &mut LbmGrid, i0: i32, j0: i32, i1: i32, j1: i32) {
    unsafe { ffi::lbm_mark_solid_rectangle(grid.ptr, i0, j0, i1, j1) };
}

/// 从外部 CSV 网格文件加载固体边界（第三方网格接口）。
///
/// 文件格式（每行一个边界点，以逗号分隔）：
/// ```text
/// x, y [, q]
/// ```
/// - `x, y`：边界点坐标（浮点数，格子单位）
/// - `q`：IBB 壁面距离分数（可选；缺省 0.5，退化为半步长反弹）
///
/// 注释行（以 `#` 开头）和空行会被忽略。
///
/// 若文件无法打开或内容为空，函数静默返回（不标记任何节点）。
pub fn mark_solid_from_mesh_file(grid: &mut LbmGrid, filename: &str) {
    let c_str = std::ffi::CString::new(filename).expect("mark_solid_from_mesh_file: invalid filename");
    unsafe { ffi::lbm_mark_solid_from_mesh_file(grid.ptr, c_str.as_ptr()) };
}

/// 设置求解器使用的固体反弹方案。
///
/// | `bc_mode` | 方案 | 说明 |
/// |-----------|------|------|
/// | `0` | `None` | 禁用固体边界（默认，全流体模式） |
/// | `1` | `BounceBack` | 半步长反弹（Ladd 1994；q=0.5，一阶精度） |
/// | `2` | `InterpolatedBounceBack` | Bouzidi 插值反弹（2001；精确 q，二阶精度） |
pub fn mark_solid_bc(solver: &mut LbmSolver, bc_mode: i32) {
    unsafe { ffi::lbm_solver_set_solid_bc(solver.ptr, bc_mode) };
}

/// 将当前所有已标记固体节点（`solid==1` 且 `solid_bc_node==0`）的逐节点反弹方案设为 `bc_mode`。
///
/// 应在每个固体体 `mark_solid_*()` 调用后立即调用，以实现多固体混合 BC 场景下的逐体方案分配。
/// 先调用 `mark_solid_*()` 标记一个固体体，再调用本函数写入该体的方案；
/// 下一个固体体标记后再次调用，则只会填充新标记节点（`solid_bc_node==0` 的节点）。
///
/// | `bc_mode` | 方案 |
/// |-----------|------|
/// | `0` | 无操作（节点保留 0，运行时以全局 `set_solid_bc_type` 为准） |
/// | `1` | BounceBack（半步长反弹） |
/// | `2` | InterpolatedBounceBack（Bouzidi 插值反弹） |
pub fn assign_solid_bc_unmarked(grid: &mut LbmGrid, bc_mode: i32) {
    unsafe { ffi::lbm_grid_assign_solid_bc_unmarked(grid.ptr, bc_mode as std::os::raw::c_int) };
}

/// 用动量交换法（Momentum Exchange Algorithm，MEA）计算固体所受的合力。
///
/// 调用时机：`solver.step()` 之后（即 BB/IBB 已施加、`f_tmp` 保存碰后分布函数时）。
///
/// # MPI 模式
/// 在 MPI 块分解中，本函数仅统计本进程物理区域内的贡献。调用方需将
/// 所有进程的 `(fx, fy)` 通过 `MPI_Allreduce(SUM)` 求和，才能得到全局力。
/// 可通过 [`mpi_allreduce_sum_f64`] 完成此操作。
///
/// # 参数
/// - `grid`：格子网格引用（须已调用 `mark_solid_cylinder/rectangle`）
/// - `phys_i0/j0/i1/j1`：本进程物理区域范围（非 MPI 时传 `0/0/nx-1/ny-1`）
///
/// # 返回值
/// `(fx, fy)`：固体受到的 x / y 方向合力（格子单位，ρ₀=1）
pub fn compute_solid_force(grid: &LbmGrid,
                            phys_i0: i32, phys_j0: i32,
                            phys_i1: i32, phys_j1: i32) -> (f64, f64) {
    let mut fx = 0.0_f64;
    let mut fy = 0.0_f64;
    unsafe {
        ffi::lbm_compute_solid_force(
            grid.ptr as *const _,
            phys_i0, phys_j0, phys_i1, phys_j1,
            &mut fx, &mut fy,
        );
    }
    (fx, fy)
}

/// MPI 全局归约：对所有进程的一个 `f64` 值求和，结果广播到所有进程。
///
/// 未启用 MPI 时直接返回输入值（无操作）。
/// 用于将各 MPI 进程的固体受力局部值归约为全局力：
/// ```ignore
/// let (local_fx, local_fy) = compute_solid_force(&grid, i0, j0, i1, j1);
/// let fx = mpi_allreduce_sum_f64(local_fx);
/// let fy = mpi_allreduce_sum_f64(local_fy);
/// ```
pub fn mpi_allreduce_sum_f64(local_val: f64) -> f64 {
    let mut result = local_val;
    unsafe { ffi::lbm_mpi_allreduce_sum_f64(local_val, &mut result) };
    result
}

/// 清除所有固体节点标记（`solid`、`q_ibb`、`solid_bc_node`）。
///
/// 在运动固体每步重新标记前调用，防止遗留上步的固体节点。
/// 典型用法（每步循环内）：
/// ```ignore
/// clear_solid(grid);
/// mark_solid_cylinder(grid, new_cx, new_cy, radius);
/// assign_solid_bc_unmarked(grid, 2);  // IBB
/// ```
pub fn clear_solid(grid: &mut LbmGrid) {
    unsafe { ffi::lbm_clear_solid(grid.ptr) }
}

/// 运动刚体半步长反弹（Ladd 1994 移动壁面修正）。
///
/// 用法同 [`solver.step()`] 内部的 BB，但增加了壁面速度 Ladd 修正项：
///   `f_ᾱ(x_f) = f_α*(x_f) - 2*w_α*(c_α·U_wall)/cs²`
///
/// 须在 `solver.step()` 之后、下一步 `collide()` 之前调用。
/// 固体节点须在本步已由 `mark_solid_cylinder()` 等函数重新标记。
///
/// # 参数
/// - `cx, cy`：质心当前坐标（格子单位）
/// - `ux_cm, uy_cm`：质心速度（格子单位/时间步）
/// - `omega`：角速度（rad/时间步，逆时针为正）
/// - `phys_i0/j0/i1/j1`：物理区域（MPI 时从 `PartitionInfo` 获取；非 MPI 传 0/0/nx-1/ny-1）
pub fn apply_solid_bb_moving_rigid(
    grid: &mut LbmGrid,
    cx: f64, cy: f64,
    ux_cm: f64, uy_cm: f64, omega: f64,
    phys_i0: i32, phys_j0: i32,
    phys_i1: i32, phys_j1: i32,
) {
    unsafe {
        ffi::lbm_apply_solid_bb_moving_rigid(
            grid.ptr, cx, cy, ux_cm, uy_cm, omega,
            phys_i0, phys_j0, phys_i1, phys_j1,
        )
    }
}

/// 运动刚体 Bouzidi IBB（Ladd 移动壁面修正）。
///
/// 用法与 [`apply_solid_bb_moving_rigid`] 相同，但使用 IBB 插值（更精确的壁面位置）。
pub fn apply_solid_ibb_moving_rigid(
    grid: &mut LbmGrid,
    cx: f64, cy: f64,
    ux_cm: f64, uy_cm: f64, omega: f64,
    phys_i0: i32, phys_j0: i32,
    phys_i1: i32, phys_j1: i32,
) {
    unsafe {
        ffi::lbm_apply_solid_ibb_moving_rigid(
            grid.ptr, cx, cy, ux_cm, uy_cm, omega,
            phys_i0, phys_j0, phys_i1, phys_j1,
        )
    }
}

// ---------------------------------------------------------------------------
/// IBM（浸入边界法）Lagrangian 标记点集封装
// ---------------------------------------------------------------------------

/// `ibm::MarkerSet` 的安全封装（Lagrangian 标记点集）。
///
/// 封装了 IBM 三大方案的步进函数：
/// - [`LbmIbmMarkerSet::step_mdf`]：多重直接力法（MDF-IBM，Wang 2008 / Suzuki & Inamuro 2011）
/// - [`LbmIbmMarkerSet::step_penalty`]：罚函数反馈力法（Goldstein 1993）
/// - [`LbmIbmMarkerSet::step_mls`]：移动最小二乘 + 直接力（Wang 2009）
///
/// # 典型用法（MDF-IBM 圆柱绕流）
/// ```ignore
/// use lbm_bindings::{LbmGrid, LbmSolver, LbmIbmMarkerSet};
///
/// let mut ms = LbmIbmMarkerSet::new_circle(150.0, 50.0, 10.0, 64);
///
/// for _ in 0..n_steps {
///     solver.collide(&mut grid);
///     solver.stream(&mut grid);
///     ms.step_mdf(&mut grid, 1.0, 1.0, 3);   // IBM 力写入 grid.force
///     solver.apply_guo_force(&mut grid);       // 用体力更新宏观量
/// }
/// ```
pub struct LbmIbmMarkerSet {
    ptr:        *mut ffi::IbmMarkerSetHandle,
    n_markers:  usize,
    /// 用于 Penalty-IBM 的 x/y 方向速度误差积分（调用方无需直接访问）
    integral_x: Vec<f64>,
    integral_y: Vec<f64>,
    /// 用于 IVC-IBM 固定物体优化版（step_ivc_stationary）的 LU 分解缓存
    ivc_cache:  *mut c_void,
}

impl LbmIbmMarkerSet {
    /// 创建均匀分布在圆柱表面的标记点集。
    ///
    /// # 参数
    /// - `cx`, `cy`：圆心格子坐标
    /// - `radius`：半径（格子单位）
    /// - `n_markers`：标记点数量（建议 ≥ `(2π·radius).ceil() as u32`）
    pub fn new_circle(cx: f64, cy: f64, radius: f64, n_markers: u32) -> Self {
        let ptr = unsafe {
            ffi::lbm_ibm_marker_set_new_circle(cx, cy, radius, n_markers as i32)
        };
        assert!(!ptr.is_null(), "lbm_ibm_marker_set_new_circle returned null");
        let n = n_markers as usize;
        LbmIbmMarkerSet {
            ptr,
            n_markers: n,
            integral_x: vec![0.0; n],
            integral_y: vec![0.0; n],
            ivc_cache:  std::ptr::null_mut(),
        }
    }

    /// 创建沿 x 轴均匀分布的直线丝状体标记点集。
    ///
    /// # 参数
    /// - `x0`, `y0`：起点格子坐标
    /// - `length`：丝状体长度（格子单位）
    /// - `n_markers`：标记点数量
    pub fn new_filament(x0: f64, y0: f64, length: f64, n_markers: u32) -> Self {
        let ptr = unsafe {
            ffi::lbm_ibm_marker_set_new_filament(x0, y0, length, n_markers as i32)
        };
        assert!(!ptr.is_null(), "lbm_ibm_marker_set_new_filament returned null");
        let n = n_markers as usize;
        LbmIbmMarkerSet {
            ptr,
            n_markers: n,
            integral_x: vec![0.0; n],
            integral_y: vec![0.0; n],
            ivc_cache:  std::ptr::null_mut(),
        }
    }

    /// 从坐标数组创建标记点集（Python FFI / 外部网格接口）。
    ///
    /// # 参数
    /// - `x`, `y`：标记点坐标数组（格子单位，长度 `n_markers`）
    /// - `ds`：弧长/面积元素数组（长度 `n_markers`）；传空 slice 时由 C++ 端自动计算
    ///
    /// # Panics
    /// 若 `x.len() != y.len()` 或长度为 0。
    pub fn new_from_coords(x: &[f64], y: &[f64], ds: &[f64]) -> Self {
        assert_eq!(x.len(), y.len(), "x and y must have the same length");
        assert!(!x.is_empty(), "marker coordinate arrays must not be empty");
        let n = x.len();
        let ds_ptr: *const f64 = if ds.len() == n { ds.as_ptr() } else { std::ptr::null() };
        let ptr = unsafe {
            ffi::lbm_ibm_marker_set_from_coords(
                x.as_ptr(), y.as_ptr(), ds_ptr, n as i32,
            )
        };
        assert!(!ptr.is_null(), "lbm_ibm_marker_set_from_coords returned null");
        LbmIbmMarkerSet {
            ptr,
            n_markers: n,
            integral_x: vec![0.0; n],
            integral_y: vec![0.0; n],
            ivc_cache:  std::ptr::null_mut(),
        }
    }

    /// 从外部 CSV 文件加载标记点集（第三方网格接口）。
    ///
    /// 文件格式（每行一个标记点，以逗号分隔）：
    /// ```text
    /// x, y [, z [, ds]]
    /// ```
    /// - `x, y`：标记点坐标（必需）
    /// - `z`：z 坐标（可选，缺省 0.0）
    /// - `ds`：弧长/面积元素（可选；缺省值为相邻标记点间距的平均值）
    ///
    /// 注释行（以 `#` 开头）和空行会被忽略。
    ///
    /// # 错误
    /// 若文件无法打开或格式错误，返回 `Err`。
    pub fn new_from_file(filename: &str) -> Result<Self, String> {
        let c_str = std::ffi::CString::new(filename)
            .map_err(|e| format!("new_from_file: invalid filename: {e}"))?;
        let ptr = unsafe { ffi::lbm_ibm_marker_set_from_file(c_str.as_ptr()) };
        if ptr.is_null() {
            return Err(format!(
                "LbmIbmMarkerSet::new_from_file: failed to load markers from {:?}", filename
            ));
        }
        let n = unsafe { ffi::lbm_ibm_marker_set_size(ptr as *const _) } as usize;
        Ok(LbmIbmMarkerSet {
            ptr,
            n_markers: n,
            integral_x: vec![0.0; n],
            integral_y: vec![0.0; n],
            ivc_cache:  std::ptr::null_mut(),
        })
    }

    /// 返回标记点数量。
    pub fn len(&self) -> usize { self.n_markers }

    /// MDF-IBM 一步（多重直接力法，Luo 2007）。
    ///
    /// 调用时机：`collide()` + `stream()` 之后，宏观量更新之前。
    /// 计算结果写入 `grid.force`，供 Guo 体力格式使用。
    ///
    /// # MPI 跨块处理
    /// IBM 插值读取包含幽灵层的本地网格速度（halo_exchange 已填充幽灵层）。
    /// 若标记点支撑域（4 格宽）不超出幽灵层，则跨块计算自动正确。
    /// 若需要跨块力展布，须在本函数后执行力场幽灵层归并。
    ///
    /// @param grid    格子网格（读写 force 字段）
    /// @param dx      格子间距
    /// @param dt      时间步长
    /// @param n_iter  子迭代次数（推荐 2–4）
    pub fn step_mdf(&mut self, grid: &mut LbmGrid, dx: f64, dt: f64, n_iter: i32) {
        unsafe {
            ffi::lbm_ibm_compute_mdf(grid.ptr, self.ptr, dx, dt, n_iter)
        }
    }

    /// Penalty-IBM 一步（罚函数反馈力法，Goldstein 1993）。
    ///
    /// 内部维护积分向量（不需外部管理）。
    ///
    /// @param grid   格子网格
    /// @param dx     格子间距
    /// @param dt     时间步长
    /// @param alpha  比例增益（大正数，如 8.0/dt²）
    /// @param beta   积分增益（非负数，可设 0.0）
    pub fn step_penalty(&mut self, grid: &mut LbmGrid,
                        dx: f64, dt: f64, alpha: f64, beta: f64) {
        unsafe {
            ffi::lbm_ibm_compute_penalty(
                grid.ptr, self.ptr, dx, dt, alpha, beta,
                self.integral_x.as_mut_ptr(),
                self.integral_y.as_mut_ptr(),
                0.0, 0.0,   // 静止固体：u_target = (0, 0)
            )
        }
    }

    /// 隐式 MLS-IBM 一步（MLS 插值 + MLS 伴随展布 + 多步迭代修正，JCP 2025）。
    ///
    /// 相比旧版 `step_mls`，本函数使用：
    ///   - MLS 伴随展布（J^T）替代 Peskin δ，满足离散伴随一致性
    ///   - 多步迭代修正（默认 n_iter=3）逼近无滑移条件
    ///
    /// 参考：2025 JCP "An implicit moving-least-squares immersed boundary method
    ///       for high fidelity fluid-structure interaction simulations"
    ///
    /// @param grid  格子网格
    /// @param dx    格子间距
    /// @param dt    时间步长
    pub fn step_mls(&mut self, grid: &mut LbmGrid, dx: f64, dt: f64) {
        unsafe {
            ffi::lbm_ibm_compute_mls(grid.ptr, self.ptr, dx, dt)
        }
    }

    /// 隐式 MLS-IBM 一步（可指定迭代次数和目标速度）。
    ///
    /// @param grid       格子网格
    /// @param dx         格子间距
    /// @param dt         时间步长
    /// @param n_iter     迭代次数（建议 2–4）
    /// @param u_target_x 目标 x 速度（静止固体取 0.0）
    /// @param u_target_y 目标 y 速度（静止固体取 0.0）
    pub fn step_mls_implicit(&mut self, grid: &mut LbmGrid,
                              dx: f64, dt: f64,
                              n_iter: i32,
                              u_target_x: f64, u_target_y: f64) {
        unsafe {
            ffi::lbm_ibm_compute_mls_implicit(
                grid.ptr, self.ptr, dx, dt, n_iter, u_target_x, u_target_y)
        }
    }

    /// 原始 MLS-IBM 一步（MLS 插值 + MLS 形状函数展布，Algorithm 1，JCP 2025）。
    ///
    /// 单步直接力法，无迭代修正。无滑移残差大于隐式 MLS，但计算量最小。
    ///
    /// @param grid       格子网格
    /// @param dx         格子间距
    /// @param dt         时间步长
    pub fn step_mls_original(&mut self, grid: &mut LbmGrid,
                              dx: f64, dt: f64) {
        unsafe {
            ffi::lbm_ibm_compute_mls_original(
                grid.ptr, self.ptr, dx, dt)
        }
    }

    /// 显式 MLS-IBM 一步（MLS 插值 + MLS 展布 + Z 修正，Algorithm 2，JCP 2025）。
    ///
    /// 在原始 MLS 基础上加全局标量修正 Z = Σ(F·g)/Σ|g|²。
    /// 注意：Z 修正破坏守恒性，推荐使用 `step_mls`（隐式）。
    ///
    /// @param grid       格子网格
    /// @param dx         格子间距
    /// @param dt         时间步长
    pub fn step_mls_explicit(&mut self, grid: &mut LbmGrid,
                              dx: f64, dt: f64) {
        unsafe {
            ffi::lbm_ibm_compute_mls_explicit(
                grid.ptr, self.ptr, dx, dt)
        }
    }

    /// 隐式速度校正 IBM 一步（IVC-IBM，Wu & Shu 2009）。
    ///
    /// 以边界点速度修正 δu_B^l 为未知量，求解 m×m 线性方程组
    /// A·X = B 强制满足非滑移边界条件（参考论文 Eq.28–30）：
    ///   - A_{lk} = Δs_k · Σ_{i,j} D_ij^l · D_ij^k · Δx²
    ///   - B^l = U_target^l − u*(X_B^l)（速度亏量）
    ///   - 力密度：f_B^l = (2ρ/dt) · δu_B^l
    ///
    /// 每步重建矩阵 A，适用于移动物体。对静止物体请使用 `step_ivc_stationary`。
    ///
    /// 参考：J. Wu & C. Shu (2009) J. Comput. Phys. 228:1963–1979
    ///
    /// @param grid       格子网格
    /// @param dx         格子间距（通常 = 1.0）
    /// @param dt         时间步长（通常 = 1.0）
    pub fn step_ivc(&mut self, grid: &mut LbmGrid,
                     dx: f64, dt: f64) {
        unsafe {
            ffi::lbm_ibm_compute_ivc(
                grid.ptr, self.ptr, dx, dt)
        }
    }

    /// 隐式速度校正 IBM 一步（IVC-IBM，固定物体，LU 分解缓存）。
    ///
    /// 对几何固定的浸入边界，首次调用时构建矩阵 A 并完成 LU 分解（缓存于内部），
    /// 后续每步仅执行 LU 代换，节省矩阵构建开销。
    ///
    /// 内部缓存由 `LbmIbmMarkerSet` 自动管理（通过 `ivc_cache` 字段）；
    /// 销毁 `LbmIbmMarkerSet` 时自动释放缓存。
    ///
    /// @param grid       格子网格
    /// @param dx         格子间距（通常 = 1.0）
    /// @param dt         时间步长（通常 = 1.0）
    pub fn step_ivc_stationary(&mut self, grid: &mut LbmGrid,
                                dx: f64, dt: f64) {
        unsafe {
            ffi::lbm_ibm_compute_ivc_stationary(
                grid.ptr, self.ptr, dx, dt,
                &mut self.ivc_cache)
        }
    }

    /// 设置所有标记点的统一目标速度（静止体使用 0.0, 0.0）。
    ///
    /// 调用此方法后，所有 IBM 方法（MDF/Penalty/MLS/IVC）将以此速度作为无滑移目标。
    /// 对于运动刚体，建议使用 [`set_rigid_body_targets`] 逐点设置（包含旋转）。
    pub fn set_uniform_target(&mut self, ux: f64, uy: f64) {
        unsafe { ffi::lbm_ibm_set_uniform_target(self.ptr, ux, uy) }
    }

    /// 设置各标记点的逐点目标速度（柔性体 / 旋转刚体）。
    ///
    /// `ux` 和 `uy` 的长度须 ≥ `n_markers`。多余元素被忽略，不足时只更新前 N 个。
    pub fn set_marker_targets(&mut self, ux: &[f64], uy: &[f64]) {
        let n = ux.len().min(uy.len()).min(self.n_markers) as i32;
        unsafe {
            ffi::lbm_ibm_set_marker_targets(self.ptr, ux.as_ptr(), uy.as_ptr(), n)
        }
    }

    /// 读取各标记点的当前目标速度（调用前须已通过 `set_*_targets` 设置）。
    pub fn get_marker_targets(&self) -> (Vec<f64>, Vec<f64>) {
        let mut ux = vec![0.0_f64; self.n_markers];
        let mut uy = vec![0.0_f64; self.n_markers];
        unsafe {
            ffi::lbm_ibm_get_marker_targets(self.ptr as *const _, ux.as_mut_ptr(), uy.as_mut_ptr())
        }
        (ux, uy)
    }

    /// 根据刚体运动状态设置逐点目标速度（含旋转）。
    ///
    /// 每个标记点 m 的目标速度：
    ///   `Uk = (ux_cm - omega*(mk.y - cy), uy_cm + omega*(mk.x - cx))`
    ///
    /// # 参数
    /// - `cx, cy`：刚体质心坐标（格子单位，本步更新后的位置）
    /// - `ux_cm, uy_cm`：质心速度
    /// - `omega`：角速度（逆时针为正）
    pub fn set_rigid_body_targets(&mut self,
                                   cx: f64, cy: f64,
                                   ux_cm: f64, uy_cm: f64, omega: f64) {
        unsafe { ffi::lbm_ibm_set_rigid_body_targets(self.ptr, cx, cy, ux_cm, uy_cm, omega) }
    }

    /// 同步更新标记点坐标（刚体运动后用来将当前位置写回 MarkerSet）。
    ///
    /// `x` / `y` 长度须 ≥ `n_markers`。
    pub fn update_positions(&mut self, x: &[f64], y: &[f64]) {
        let n = x.len().min(y.len()).min(self.n_markers) as i32;
        unsafe { ffi::lbm_ibm_update_marker_positions(self.ptr, x.as_ptr(), y.as_ptr(), n) }
    }

    /// 读取所有标记点的插值流体速度（最近一次 IBM step 后的值）。
    pub fn get_marker_velocities(&self) -> (Vec<f64>, Vec<f64>) {
        let mut ux = vec![0.0_f64; self.n_markers];
        let mut uy = vec![0.0_f64; self.n_markers];
        unsafe {
            ffi::lbm_ibm_get_marker_velocities(
                self.ptr as *const _, ux.as_mut_ptr(), uy.as_mut_ptr())
        }
        (ux, uy)
    }

    /// 计算 IBM 总力 + 总力矩（绕质心 `(cx, cy)`）。
    ///
    /// 返回：`(Ftot_x, Ftot_y, Ttot)`，均为施加到**流体**上的力/力矩；
    /// 固体所受反作用力 = `(-Ftot_x, -Ftot_y, -Ttot)`。
    pub fn compute_body_force_and_torque(&self, cx: f64, cy: f64) -> (f64, f64, f64) {
        let mut fx = 0.0_f64;
        let mut fy = 0.0_f64;
        let mut torque = 0.0_f64;
        unsafe {
            ffi::lbm_ibm_compute_body_force_torque(
                self.ptr as *const _, cx, cy,
                &mut fx, &mut fy, &mut torque)
        }
        (fx, fy, torque)
    }

    /// 仅插值流体速度到所有标记点（不计算体力，不展布）。
    ///
    /// 用于对内部拉格朗日点做速度采样（内部质量方案 C）。
    pub fn interpolate_only(&mut self, grid: &LbmGrid, dx: f64) {
        unsafe { ffi::lbm_ibm_interpolate_only(grid.ptr as *const _, self.ptr, dx) }
    }

    /// 读取所有标记点的 Lagrangian 力 `(fx, fy)`。
    /// 在 `step_*()` 调用后调用此函数以获取当前步的力值。
    pub fn get_forces(&self) -> (Vec<f64>, Vec<f64>) {
        let mut fx = vec![0.0_f64; self.n_markers];
        let mut fy = vec![0.0_f64; self.n_markers];
        unsafe {
            ffi::lbm_ibm_get_forces(self.ptr as *const _, fx.as_mut_ptr(), fy.as_mut_ptr())
        }
        (fx, fy)
    }

    /// 读取所有标记点的当前坐标 `(x, y)`。
    pub fn get_positions(&self) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0_f64; self.n_markers];
        let mut y = vec![0.0_f64; self.n_markers];
        unsafe {
            ffi::lbm_ibm_get_positions(self.ptr as *const _, x.as_mut_ptr(), y.as_mut_ptr())
        }
        (x, y)
    }

    /// 读取单个标记点的当前坐标 `(x, y)`（便捷方法）。
    pub fn get_marker_position(&self, idx: usize) -> (f64, f64) {
        let (x, y) = self.get_positions();
        (x[idx.min(self.n_markers - 1)], y[idx.min(self.n_markers - 1)])
    }

    /// 重置 Penalty-IBM 积分（用于重启仿真）。
    pub fn reset_penalty_integrals(&mut self) {
        self.integral_x.fill(0.0);
        self.integral_y.fill(0.0);
    }

    /// 计算 IBM 固体受力合力（第三方网格接口）。
    ///
    /// 对所有 Lagrangian 标记点执行加权求和：
    ///   `F_x = Σ mk.fx * mk.ds`,  `F_y = Σ mk.fy * mk.ds`
    ///
    /// 固体所受流体合力 = `(-F_x, -F_y)`（牛顿第三定律）。
    ///
    /// 调用时机：任一 `step_*()` 方法调用之后。
    ///
    /// # MPI 说明
    /// 在 MPI 模式下，`adapt_to_partition()` 使各进程仅处理归属自身的标记点子集，
    /// 故此函数仅返回本进程所持有标记点的局部合力；若需全局合力，
    /// 调用方须额外执行 `MPI_Allreduce` 求和。
    pub fn compute_body_force(&self) -> (f64, f64) {
        let mut fx = 0.0_f64;
        let mut fy = 0.0_f64;
        unsafe {
            ffi::lbm_ibm_compute_body_force(self.ptr as *const _, &mut fx, &mut fy);
        }
        (fx, fy)
    }

    /// MPI 分区适配：将标记点坐标从全局格子坐标系转换到本进程本地格子坐标系，
    /// 并设置 IBM 函数所用的"归属"行/列范围，避免跨 MPI 块的双重计数。
    ///
    /// # 调用时机
    /// 在 MPI 初始化并获知分区信息后调用一次（通常在仿真开始前）。
    /// 非 MPI 或单进程模式下无需调用（默认 owner 范围涵盖全域）。
    ///
    /// # 参数
    /// - `x_start`：本分区物理域在全局 X 方向的起始格点索引
    /// - `y_start`：本分区物理域在全局 Y 方向的起始格点索引
    /// - `phys_x0`：本地网格物理列的起始列索引（含幽灵列时 ≥ 1）
    /// - `phys_y0`：本地网格物理行的起始行索引（含幽灵行时 ≥ 1）
    /// - `local_nx`：本分区物理列数
    /// - `local_ny`：本分区物理行数
    pub fn adapt_to_partition(&mut self,
                               x_start: i32, y_start: i32,
                               phys_x0: i32, phys_y0: i32,
                               local_nx: i32, local_ny: i32) {
        unsafe {
            ffi::lbm_ibm_marker_set_adapt_to_partition(
                self.ptr,
                x_start, y_start, phys_x0, phys_y0, local_nx, local_ny,
            );
        }
    }
}

/// IBM MPI 幽灵层辅助函数（独立函数，不附属于 marker set）
///
/// ## 使用时序（MPI 模式下每步）
/// ```text
/// ibm_halo_exchange_u_2d(grid, decomp2d);  // 步骤 1：填充幽灵行真实邻居速度
/// marker_set.step_*(grid, dx, dt);          // 步骤 2：插值 + 力计算 + 展布
/// ibm_halo_reduce_force_2d(grid, decomp2d); // 步骤 3：归并幽灵行力贡献
/// ```
///
/// 两个函数在 `nprocs==1` 时均为空操作。

/// 在 `step_ibm()` 前交换幽灵行/列的速度场。
///
/// `solver.step()` 完成后幽灵行 `u` 来自本地边界行外推（PUSH 流式迁移覆盖了
/// `halo_exchange` 写入的邻居 f，再 `compute_macroscopic` 得到），而非邻居真实速度。
/// 本函数通过 `MPI_Sendrecv` 将物理边界行/列的 `u` 正确填充到幽灵行/列，
/// 使 `interpolate_velocity()` 的支撑域计算物理正确。
pub fn ibm_halo_exchange_u_2d(grid: &mut LbmGrid, decomp: &mut LbmMpiDecomp2D) {
    unsafe { ffi::lbm_ibm_halo_exchange_u_2d(grid.ptr, decomp.ptr) }
}

/// 在 `step_ibm()` 后归并幽灵行/列的力贡献。
///
/// `spread_force()` 可能向幽灵行/列写入力，这些力属于邻居物理区域的一部分。
/// 本函数将幽灵行/列力通过 `MPI_Sendrecv` 发回各自邻居并在其物理行/列上累加，
/// 然后清零本地幽灵行/列，确保跨 MPI 边界的 IBM 力展布物理上完整。
pub fn ibm_halo_reduce_force_2d(grid: &mut LbmGrid, decomp: &mut LbmMpiDecomp2D) {
    unsafe { ffi::lbm_ibm_halo_reduce_force_2d(grid.ptr, decomp.ptr) }
}

impl Drop for LbmIbmMarkerSet {
    fn drop(&mut self) {
        if !self.ivc_cache.is_null() {
            unsafe { ffi::lbm_ibm_ivc_stationary_cache_free(&mut self.ivc_cache) };
        }
        if !self.ptr.is_null() {
            unsafe { ffi::lbm_ibm_marker_set_free(self.ptr) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

// Safety: MarkerSet 内部无线程共享状态；Send/Sync 允许在线程间传递（同一时刻只能一个线程访问）。
unsafe impl Send for LbmIbmMarkerSet {}
unsafe impl Sync for LbmIbmMarkerSet {}

// ---------------------------------------------------------------------------
// 2D 刚体求解器封装（RigidBodySolver2D / Suzuki & Inamuro 2011）
// ---------------------------------------------------------------------------

/// `fsi::RigidBodySolver2D` 的安全封装（2D 自由运动刚体求解器）。
///
/// 封装了 Suzuki & Inamuro (2011) 的刚体运动方程积分，支持 4 种内部质量修正方案（A/B-1/B-2/C），
/// 用于 IBM 中封闭结构的内部流体伪动量补偿。
///
/// # 典型用法（IBM 自由运动圆柱）
///
/// ```ignore
/// // 建立标记点集（圆形）
/// let mut ms = LbmIbmMarkerSet::new_circle(cx0, cy0, radius, n_markers);
///
/// // 建立刚体求解器
/// let ref_x: Vec<f64> = ms.iter().map(|mk| mk.x - cx0).collect();
/// let ref_y: Vec<f64> = ms.iter().map(|mk| mk.y - cy0).collect();
/// let mut rb = LbmRigidBody2D::new(mass, inertia, rho_b, rho_f,
///                                   cx0, cy0, Scheme::FengRigidBody,
///                                   true, &ref_x, &ref_y, &[], &[]);
///
/// for _ in 0..n_steps {
///     solver.step(&mut grid);
///     // 用当前刚体状态设置逐标记点目标速度
///     let (cx, cy, ux_cm, uy_cm, _, omega) = rb.state();
///     ms.set_rigid_body_targets(cx, cy, ux_cm, uy_cm, omega);
///     ms.step_mdf(&mut grid, dx, dt, n_iter);
///     // 计算合力矩并推进刚体
///     let (ftot_x, ftot_y, ttot) = ms.compute_body_force_and_torque(cx, cy);
///     rb.advance(-ftot_x, -ftot_y, -ttot, dt);   // 固体受流体反作用力（符号取反）
///     // 同步标记点位置
///     let (bx, by) = rb.boundary_positions();
///     ms.update_positions(&bx, &by);
/// }
/// ```
#[derive(Debug)]
pub struct LbmRigidBody2D {
    ptr: *mut ffi::RigidBody2DHandle,
    n_bnd: usize,
    n_int: usize,
}

/// 内部质量修正方案（对应 `fsi::InternalMassScheme`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RigidBodyScheme {
    /// 方案 A：无内部质量修正（仅适用于开放结构或外部不需要补偿的情况）。
    None = 0,
    /// 方案 B-1：Uhlmann (2005) 刚体内部质量修正。
    UhlmannRigidBody = 1,
    /// 方案 B-2：Feng & Michaelides (2004) 刚体内部质量修正。
    FengRigidBody = 2,
    /// 方案 C：内部拉格朗日点（Suzuki & Inamuro 2011 直接方法）。
    LagrangianPoints = 3,
}

impl LbmRigidBody2D {
    /// 创建 2D 刚体求解器。
    ///
    /// # 参数
    /// - `mass, inertia`：刚体质量 m 与转动惯量 Izz（格子单位）
    /// - `rho_b, rho_f`：刚体密度与流体参考密度（格子单位）
    /// - `cx0, cy0`：初始质心坐标
    /// - `scheme`：内部质量方案
    /// - `is_closed`：是否为封闭结构（控制是否应用内部质量修正）
    /// - `ref_x, ref_y`：边界标记点参考坐标（体固系，长度 n_bnd）
    /// - `int_ref_x, int_ref_y`：内部点参考坐标（仅方案 C 使用）
    pub fn new(mass: f64, inertia: f64,
               rho_b: f64, rho_f: f64,
               cx0: f64, cy0: f64,
               scheme: RigidBodyScheme,
               is_closed: bool,
               ref_x: &[f64], ref_y: &[f64],
               int_ref_x: &[f64], int_ref_y: &[f64]) -> Self {
        let n_bnd = ref_x.len();
        let n_int = int_ref_x.len();
        let ptr = unsafe {
            ffi::lbm_rigid2d_create(
                mass, inertia, rho_b, rho_f, cx0, cy0,
                scheme as std::os::raw::c_int,
                if is_closed { 1 } else { 0 },
                ref_x.as_ptr(), ref_y.as_ptr(), n_bnd as std::os::raw::c_int,
                if int_ref_x.is_empty() { std::ptr::null() } else { int_ref_x.as_ptr() },
                if int_ref_y.is_empty() { std::ptr::null() } else { int_ref_y.as_ptr() },
                n_int as std::os::raw::c_int,
            )
        };
        Self { ptr, n_bnd, n_int }
    }

    /// 获取当前运动状态：`(cx, cy, ux, uy, theta, omega)`。
    pub fn state(&self) -> (f64, f64, f64, f64, f64, f64) {
        let (mut cx, mut cy) = (0.0_f64, 0.0_f64);
        let (mut ux, mut uy) = (0.0_f64, 0.0_f64);
        let (mut theta, mut omega) = (0.0_f64, 0.0_f64);
        unsafe {
            ffi::lbm_rigid2d_get_state(self.ptr as *const _,
                &mut cx, &mut cy, &mut ux, &mut uy, &mut theta, &mut omega);
        }
        (cx, cy, ux, uy, theta, omega)
    }

    /// 设置初始速度（质心速度 + 角速度）。
    pub fn set_velocity(&mut self, ux: f64, uy: f64, omega: f64) {
        unsafe { ffi::lbm_rigid2d_set_velocity(self.ptr, ux, uy, omega) }
    }

    /// 获取边界标记点当前绝对坐标 `(x[], y[])`。
    pub fn boundary_positions(&self) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0_f64; self.n_bnd];
        let mut y = vec![0.0_f64; self.n_bnd];
        unsafe { ffi::lbm_rigid2d_get_boundary_positions(self.ptr as *const _, x.as_mut_ptr(), y.as_mut_ptr()) }
        (x, y)
    }

    /// 获取边界标记点当前速度（由刚体旋转算出）`(ux[], uy[])`。
    pub fn boundary_velocities(&self) -> (Vec<f64>, Vec<f64>) {
        let mut ux = vec![0.0_f64; self.n_bnd];
        let mut uy = vec![0.0_f64; self.n_bnd];
        unsafe { ffi::lbm_rigid2d_get_boundary_velocities(self.ptr as *const _, ux.as_mut_ptr(), uy.as_mut_ptr()) }
        (ux, uy)
    }

    /// 获取内部点当前绝对坐标（方案 C）。
    pub fn internal_positions(&self) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0_f64; self.n_int];
        let mut y = vec![0.0_f64; self.n_int];
        unsafe { ffi::lbm_rigid2d_get_internal_positions(self.ptr as *const _, x.as_mut_ptr(), y.as_mut_ptr()) }
        (x, y)
    }

    /// 设置内部点的插值流体速度（方案 C，调用前须先 `interpolate_only(grid, ms_int, dx)`）。
    pub fn set_internal_velocities(&mut self, ux: &[f64], uy: &[f64]) {
        if ux.len() < self.n_int || uy.len() < self.n_int { return; }
        unsafe { ffi::lbm_rigid2d_set_internal_velocities(self.ptr, ux.as_ptr(), uy.as_ptr()) }
    }

    /// 计算内部动量 Pin, Lin（方案 C）。须先设置内部点速度。
    pub fn compute_internal_momentum(&mut self) {
        unsafe { ffi::lbm_rigid2d_compute_internal_momentum(self.ptr) }
    }

    /// 推进刚体一个时间步。
    ///
    /// # 参数
    /// - `total_fx/fy`：IBM 施加到**固体**上的总力（= -IBM 流体力，调用方取反）
    /// - `total_torque`：IBM 施加到固体上的总力矩（同上，取反）
    /// - `dt`：时间步长
    pub fn advance(&mut self, total_fx: f64, total_fy: f64, total_torque: f64, dt: f64) {
        unsafe { ffi::lbm_rigid2d_advance(self.ptr, total_fx, total_fy, total_torque, dt) }
    }

    /// 边界标记点数量。
    pub fn n_boundary(&self) -> usize { self.n_bnd }

    /// 内部点数量（方案 C）。
    pub fn n_internal(&self) -> usize { self.n_int }
}

impl Drop for LbmRigidBody2D {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { ffi::lbm_rigid2d_free(self.ptr) }
            self.ptr = std::ptr::null_mut();
        }
    }
}

unsafe impl Send for LbmRigidBody2D {}
unsafe impl Sync for LbmRigidBody2D {}

/// `lbm::GpuSolver` 的安全封装（D2Q9 BGK on CUDA）
///
/// 支持两种主循环模式：
///
/// **同步模式**（适合小网格 / 无频繁 I/O）：
/// ```ignore
/// let mut gpu = LbmGpuSolver::new(&mut grid, omega).unwrap();
/// gpu.add_boundary_condition(...);
/// for _ in 0..nsteps {
///     gpu.step();  // 阻塞，GPU 计算完成后返回
/// }
/// gpu.download(&mut grid);
/// ```
///
/// **异步双缓冲模式**（推荐，GPU 计算与磁盘 I/O 并行）：
/// ```ignore
/// let mut gpu = LbmGpuSolver::new(&mut grid, omega).unwrap();
/// gpu.add_boundary_condition(...);
/// let mut pending = false;
/// let mut buf: i32 = 0;
/// for step in 0..nsteps {
///     if pending {
///         gpu.sync_async_download();                          // 等待拷贝完成
///         let n = gpu.n() as usize;
///         let rho = unsafe { std::slice::from_raw_parts(gpu.pinned_rho(buf), n) };
///         let u   = unsafe { std::slice::from_raw_parts(gpu.pinned_u(buf), n * 2) };
///         write_snapshot(rho, u, n, step - 1);               // 写盘（GPU 并行计算中）
///         buf ^= 1;
///     }
///     gpu.step_async();                                       // 非阻塞，立即返回
///     if step % write_interval == 0 {
///         gpu.enqueue_async_download_rho_u(buf);             // 异步拷贝，非阻塞
///         pending = true;
///     } else { pending = false; }
/// }
/// gpu.wait_compute();
/// if pending { gpu.sync_async_download(); write_snapshot(...); }
/// ```
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

    // ------------------------------------------------------------------
    // 同步数据传输
    // ------------------------------------------------------------------

    /// 将 GPU 端完整状态（f、f_tmp、ρ、u）下载到 CPU 网格
    /// （用于 CPU 端边界条件，或需要完整状态检查点的场景）。
    pub fn download(&self, grid: &mut LbmGrid) {
        unsafe { ffi::lbm_gpu_download(self.ptr, grid.as_mut_ptr()) };
    }

    /// 仅将 ρ、u 下载到 CPU 网格（6.3 MB @512²，用于输出快照）。
    /// 比 download() 少传输 7× 数据量。
    pub fn download_rho_u(&self, grid: &mut LbmGrid) {
        unsafe { ffi::lbm_gpu_download_rho_u(self.ptr, grid.as_mut_ptr()) };
    }

    /// 将 CPU 网格 f 上传到 GPU（CPU 端边界条件修正后调用）。
    pub fn upload(&mut self, grid: &mut LbmGrid) {
        unsafe { ffi::lbm_gpu_upload(self.ptr, grid.as_mut_ptr()) };
    }

    // ------------------------------------------------------------------
    // 异步输出管线（推荐用于有频繁 I/O 的大规模仿真）
    // ------------------------------------------------------------------

    /// 异步单步执行（在 compute_stream 上提交全部核函数，立即返回）。
    /// 与 step() 的区别：不阻塞 CPU，允许 CPU 同时执行磁盘写入等操作。
    pub fn step_async(&mut self) {
        unsafe { ffi::lbm_gpu_step_async(self.ptr) };
    }

    /// 等待 compute_stream 完成（cudaStreamSynchronize）。
    /// 在 step_async() 后、读取 GPU 结果之前调用。
    pub fn wait_compute(&mut self) {
        unsafe { ffi::lbm_gpu_wait_compute(self.ptr) };
    }

    /// 将 d_rho/d_u 异步拷贝到固定主机双缓冲区 buf_idx（0 或 1）的 io_stream 上。
    /// io_stream 自动等待 compute_done 事件，不阻塞 CPU。
    /// `buf_idx` 必须为 0 或 1；调用方应在连续两次输出步之间交替使用。
    pub fn enqueue_async_download_rho_u(&mut self, buf_idx: i32) {
        unsafe { ffi::lbm_gpu_enqueue_async_download_rho_u(self.ptr, buf_idx) };
    }

    /// 等待 io_stream（异步拷贝）完成（cudaStreamSynchronize）。
    /// 完成后可以安全读取 pinned_rho() / pinned_u() 中的数据。
    pub fn sync_async_download(&mut self) {
        unsafe { ffi::lbm_gpu_sync_async_download(self.ptr) };
    }

    /// 返回固定主机缓冲区中 ρ 数组的原始指针。
    ///
    /// # Safety
    /// - 必须在 [`sync_async_download()`] 返回后才能读取此指针指向的数据。
    /// - 指针的有效生命周期与 `LbmGpuSolver` 实例绑定；销毁 solver 后指针失效。
    /// - 若再次调用 `enqueue_async_download_rho_u(buf_idx)` 且缓冲区写入完成后，
    ///   该指针的内容会被新数据覆盖。双缓冲设计（0/1 交替）可避免读写竞争。
    /// - `buf_idx` 必须为 0 或 1，否则行为未定义。
    pub fn pinned_rho(&self, buf_idx: i32) -> *const f64 {
        unsafe { ffi::lbm_gpu_pinned_rho(self.ptr as *const _, buf_idx) }
    }

    /// 返回固定主机缓冲区中 u 数组的原始指针（布局：`[ux0, uy0, ux1, uy1, …]`，长度 = `n() * 2`）。
    ///
    /// # Safety
    /// 与 [`pinned_rho()`] 相同——必须在 [`sync_async_download()`] 返回后读取，
    /// 且指针有效性与 solver 生命周期绑定。
    pub fn pinned_u(&self, buf_idx: i32) -> *const f64 {
        unsafe { ffi::lbm_gpu_pinned_u(self.ptr as *const _, buf_idx) }
    }

    /// 返回总节点数 n（= nx × ny）。
    pub fn n(&self) -> i32 {
        unsafe { ffi::lbm_gpu_n(self.ptr as *const _) }
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
