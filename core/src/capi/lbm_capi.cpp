// C ABI 封装层 — 允许 Rust（及任何 C 消费者）在不依赖 C++ 名称修饰的情况下调用 C++ 核心库
#include "lbm/lattice.hpp"
#include "lbm/solver.hpp"
#include "lbm/mpi_decomp.hpp"
#include "plugins/plugin_registry.hpp"
#include <cstdlib>
#include <new>

#ifdef LBM_ENABLE_MPI
#include <mpi.h>
#endif

#ifdef LBM_ENABLE_CUDA
#include "lbm/gpu_solver.hpp"
#endif

#ifdef LBM_ENABLE_OPENMP
#include <omp.h>
#endif

// ---------------------------------------------------------------------------
// C 回调函数类型别名（必须与 plugin_registry.cpp 中的声明一致）
// Rust 对应类型: ffi::BoundaryFn / MeshFn / MotionFn / FlexibleFn — bindings/src/lib.rs
// ---------------------------------------------------------------------------
extern "C" {

typedef void (*lbm_boundary_fn) (void* grid, int step, void* userdata);
// Rust 对应: ffi::BoundaryFn
typedef void (*lbm_mesh_adapt_fn)(void* grid, int step, void* userdata);
// Rust 对应: ffi::MeshFn
typedef void (*lbm_motion_fn)   (void* grid, void* markers,
                                  double dt, int step, void* userdata);
// Rust 对应: ffi::MotionFn
typedef void (*lbm_flexible_fn) (void* markers, double dt, int step,
                                  void* userdata);
// Rust 对应: ffi::FlexibleFn

// 声明于 plugin_registry.cpp；Rust 封装: register_plugins() — bindings/src/lib.rs
void lbm_set_plugins(
    lbm_boundary_fn   boundary_fn,  void* boundary_data,
    lbm_mesh_adapt_fn mesh_fn,      void* mesh_data,
    lbm_motion_fn     motion_fn,    void* motion_data,
    lbm_flexible_fn   flexible_fn,  void* flexible_data);

} // extern "C"

extern "C" {

// ---------------------------------------------------------------------------
// LatticeGrid 函数
// ---------------------------------------------------------------------------

/// 在堆上创建 LatticeGrid 并返回指针；若分配失败返回 nullptr。
/// Rust 封装: LbmGrid::new() — bindings/src/lib.rs
lbm::LatticeGrid* lbm_grid_new(int nx, int ny, int nz, int model_id)
{
    // 验证 model_id 以避免越界枚举转换产生未定义行为
    if (model_id < 0 || model_id > 2) return nullptr;
    auto model = static_cast<lbm::LatticeModel>(model_id);
    return new (std::nothrow) lbm::LatticeGrid(nx, ny, nz, model);
}

/// 释放由 lbm_grid_new 创建的 LatticeGrid。
/// Rust 触发: <LbmGrid as Drop>::drop() (RAII 自动调用)
void lbm_grid_free(lbm::LatticeGrid* g)
{
    delete g;
}

// Rust 封装: LbmGrid::nx() / ny() / nz() — bindings/src/lib.rs
int lbm_grid_nx(const lbm::LatticeGrid* g) { return g ? g->nx : 0; }
int lbm_grid_ny(const lbm::LatticeGrid* g) { return g ? g->ny : 0; }
int lbm_grid_nz(const lbm::LatticeGrid* g) { return g ? g->nz : 0; }

/// 返回节点 idx 处的宏观密度 ρ；越界时返回 0。
/// Rust 封装: LbmGrid::rho()
double lbm_grid_rho(const lbm::LatticeGrid* g, int idx)
{
    if (!g || idx < 0 || idx >= g->size()) return 0.0;
    return g->rho[idx];
}

/// 返回节点 idx 处的 x 方向速度；越界时返回 0。
/// Rust 封装: LbmGrid::ux()
double lbm_grid_ux(const lbm::LatticeGrid* g, int idx)
{
    if (!g || idx < 0 || idx >= g->size()) return 0.0;
    return g->u[idx * g->dim() + 0];
}

/// 返回节点 idx 处的 y 方向速度；越界时返回 0。
/// Rust 封装: LbmGrid::uy()
double lbm_grid_uy(const lbm::LatticeGrid* g, int idx)
{
    if (!g || idx < 0 || idx >= g->size()) return 0.0;
    return g->u[idx * g->dim() + 1];
}

// ---------------------------------------------------------------------------
// Solver 函数
// ---------------------------------------------------------------------------

/// 创建求解器并返回指针；指针无效或参数非法时返回 nullptr。
/// Rust 封装: LbmSolver::new() — bindings/src/lib.rs
lbm::Solver* lbm_solver_new(lbm::LatticeGrid* g, double omega, int cm_id)
{
    if (!g) return nullptr;
    // 验证碰撞模型 id
    if (cm_id < 0 || cm_id > 1) return nullptr;
    auto cm = static_cast<lbm::CollisionModel>(cm_id);
    return new (std::nothrow) lbm::Solver(*g, omega, cm);
}

/// 向求解器注册一个边界条件，每步 step() 后自动施加。
///
/// @param s        求解器指针（由 lbm_solver_new 创建）
/// @param bc_type  边界类型（与 C++ lbm::BCType 枚举值对应）：
///                   0=BounceBack        1=BounceBackFullWay
///                   2=ZouHe_Velocity    3=ZouHe_Pressure
///                   4=FullyDeveloped    5=FreeOutlet
///                   6=Guo_Extrapolation 7=Periodic
/// @param face     面编号：0=West, 1=East, 2=South, 3=North, 4=Bottom, 5=Top
/// @param ux,uy,uz 规定速度分量（ZouHe_Velocity / Guo_Extrapolation 速度模式使用）
/// @param rho      规定密度（ZouHe_Pressure 使用；Guo_Extrapolation 中 rho>0 为压力模式，rho=0 为速度模式）
/// Rust 封装: LbmSolver::add_boundary_condition() — bindings/src/lib.rs
void lbm_solver_add_bc(lbm::Solver* s,
                        int bc_type, int face,
                        double ux, double uy, double uz,
                        double rho)
{
    if (!s) return;
    if (bc_type < 0 || bc_type > 7) return;   // 与 BCType 枚举成员数同步
    if (face    < 0 || face    > 5) return;

    lbm::BoundaryCondition bc;
    bc.type = static_cast<lbm::BCType>(bc_type);
    bc.face = static_cast<lbm::Face>(face);
    bc.ux   = ux;
    bc.uy   = uy;
    bc.uz   = uz;
    bc.rho  = rho;
    s->add_boundary_condition(bc);
}

/// 释放由 lbm_solver_new 创建的 Solver。
/// Rust 触发: <LbmSolver as Drop>::drop() (RAII 自动调用)
void lbm_solver_free(lbm::Solver* s)
{
    delete s;
}

/// 推进仿真一步并调用所有已注册的插件。
/// Rust 封装: LbmSolver::step() — bindings/src/lib.rs
///
/// 这是简化版接口，向所有插件传递 `step_index = 0` 和 `dt = 1.0`。
/// 若插件需要准确的步骤计数或物理时间（例如规定运动轨迹、自适应调度），
/// 请改用 `lbm_solver_step_n()`，它接受显式的 step_index 和 dt。
///
/// 单步内插件调用顺序：
///   1. `IMotionPlugin::update()`       — 更新固体/网格位置（碰撞前）
///   2. `Solver::step()`                — 标准 LBM 碰撞 + 流式迁移
///   3. `IBoundaryPlugin::apply()`      — 自定义边界条件（流式迁移后）
///   4. `IMeshPlugin::adapt()`          — 网格自适应（流式迁移后）
///   5. `IFlexibleSolverPlugin::step()` — 柔性体推进（流式迁移后）
void lbm_solver_step(lbm::Solver* s, lbm::LatticeGrid* g)
{
    if (!s || !g) return;

    auto& reg = lbm::PluginRegistry::instance();

    // 1. 运动插件：碰撞前更新固体/标记点位置
    //    （此 ABI 无 MarkerSet 句柄 — 传入 nullptr）
    reg.update_motion(*g, nullptr, /*dt=*/1.0, /*step=*/0);

    // 2. 标准 LBM 步骤（碰撞 + 流式迁移 + 宏观量更新）
    // 求解器持有构造时传入的网格引用，此处变量 g 仅用于一致性
    (void)g;
    s->step();

    // 3. 自定义边界条件插件
    reg.apply_boundary(*g, /*step=*/0);

    // 4. 网格自适应插件
    reg.adapt_mesh(*g, /*step=*/0);

    // 5. 柔性体插件（此简化 ABI 无 MarkerSet）
    reg.step_flexible(nullptr, /*dt=*/1.0, /*step=*/0);
}

/// 带显式步骤索引和时间步长的扩展求解器步骤，将准确时间信息转发给插件。
/// Rust 封装: LbmSolver::step_n() — bindings/src/lib.rs
///
/// 当插件需要准确的时间信息（运动轨迹、自适应调度等）时，
/// 优先使用此函数而非 `lbm_solver_step()`。
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

// ===========================================================================
// MPI 并行接口
// ===========================================================================
extern "C" {

/// 初始化 MPI（幂等：可多次调用）。
/// 在创建 Grid/Solver 之前调用。
/// 若未启用 LBM_ENABLE_MPI，此函数为空操作，返回 0。
int lbm_mpi_init()
{
#ifdef LBM_ENABLE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) {
        MPI_Init(nullptr, nullptr);
    }
    return 1;
#else
    return 0;
#endif
}

/// 结束 MPI。在程序退出前调用（幂等）。
void lbm_mpi_finalize()
{
#ifdef LBM_ENABLE_MPI
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) {
        MPI_Finalize();
    }
#endif
}

/// 返回当前进程编号（未启用 MPI 或未初始化时返回 0）。
int lbm_mpi_rank()
{
#ifdef LBM_ENABLE_MPI
    int rank = 0, initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
#else
    return 0;
#endif
}

/// 返回进程总数（未启用 MPI 或未初始化时返回 1）。
int lbm_mpi_size()
{
#ifdef LBM_ENABLE_MPI
    int nprocs = 1, initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    return nprocs;
#else
    return 1;
#endif
}

/// 计算本进程应持有的本地行数和全局起始行偏移。
/// @param global_ny    全局 Y 方向节点数
/// @param out_y_start  输出：本地域在全局坐标下的起始行（不含幽灵）
/// @param out_local_ny 输出：本地物理行数（不含幽灵）
/// @return  含幽灵行的本地 ny：nprocs>1 时 = out_local_ny+2；nprocs=1 时 = global_ny
int lbm_mpi_local_ny(int global_ny, int* out_y_start, int* out_local_ny)
{
#ifdef LBM_ENABLE_MPI
    int rank = lbm_mpi_rank();
    int np   = lbm_mpi_size();
    int base = global_ny / np;
    int rem  = global_ny % np;
    int lny  = base + (rank < rem ? 1 : 0);
    int yst  = rank * base + (rank < rem ? rank : rem);
    if (out_y_start)   *out_y_start  = yst;
    if (out_local_ny)  *out_local_ny = lny;
    return (np > 1) ? lny + 2 : lny;
#else
    if (out_y_start)   *out_y_start  = 0;
    if (out_local_ny)  *out_local_ny = global_ny;
    return global_ny;
#endif
}

/// 指向堆上 lbm::MpiDecomp 的不透明句柄
struct MpiDecompHandle;

/// 在堆上创建 MpiDecomp；若 MPI 未初始化或未启用，返回 nullptr。
MpiDecompHandle* lbm_mpi_decomp_new(int global_nx, int global_ny)
{
#ifdef LBM_ENABLE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) return nullptr;
    auto* d = new (std::nothrow) lbm::MpiDecomp(lbm::MpiDecomp::create(global_nx, global_ny));
    return reinterpret_cast<MpiDecompHandle*>(d);
#else
    (void)global_nx; (void)global_ny;
    return nullptr;
#endif
}

/// 释放由 lbm_mpi_decomp_new 创建的 MpiDecomp。
void lbm_mpi_decomp_free(MpiDecompHandle* h)
{
#ifdef LBM_ENABLE_MPI
    delete reinterpret_cast<lbm::MpiDecomp*>(h);
#else
    (void)h;
#endif
}

/// 将 MpiDecomp 绑定到求解器；之后每次 step() 自动执行幽灵行交换。
void lbm_solver_attach_mpi(lbm::Solver* s, MpiDecompHandle* h)
{
    if (!s) return;
#ifdef LBM_ENABLE_MPI
    s->attach_mpi(reinterpret_cast<const lbm::MpiDecomp*>(h));
#else
    (void)h;
#endif
}

} // extern "C"

// ===========================================================================
// GPU（CUDA）接口
// ===========================================================================
extern "C" {

/// 指向堆上 lbm::GpuSolver 的不透明句柄
struct GpuSolverHandle;

/// 创建 GPU 求解器（上传初始数据到 GPU）。
/// 若未启用 CUDA 或 GPU 初始化失败，返回 nullptr。
GpuSolverHandle* lbm_gpu_solver_new(lbm::LatticeGrid* g, double omega)
{
#ifdef LBM_ENABLE_CUDA
    if (!g) return nullptr;
    try {
        auto* gs = new lbm::GpuSolver(*g, omega);
        return reinterpret_cast<GpuSolverHandle*>(gs);
    } catch (...) {
        return nullptr;
    }
#else
    (void)g; (void)omega;
    return nullptr;
#endif
}

/// 释放 GPU 求解器（释放设备内存）。
void lbm_gpu_solver_free(GpuSolverHandle* h)
{
#ifdef LBM_ENABLE_CUDA
    delete reinterpret_cast<lbm::GpuSolver*>(h);
#else
    (void)h;
#endif
}

/// 在 GPU 上执行 BGK 碰撞。
void lbm_gpu_collide(GpuSolverHandle* h)
{
#ifdef LBM_ENABLE_CUDA
    if (h) reinterpret_cast<lbm::GpuSolver*>(h)->collide();
#else
    (void)h;
#endif
}

/// 在 GPU 上执行流式迁移。
void lbm_gpu_stream(GpuSolverHandle* h)
{
#ifdef LBM_ENABLE_CUDA
    if (h) reinterpret_cast<lbm::GpuSolver*>(h)->stream();
#else
    (void)h;
#endif
}

/// 在 GPU 上更新宏观量（ρ、u）。
void lbm_gpu_macroscopic(GpuSolverHandle* h)
{
#ifdef LBM_ENABLE_CUDA
    if (h) reinterpret_cast<lbm::GpuSolver*>(h)->compute_macroscopic();
#else
    (void)h;
#endif
}

/// 将 GPU 端分布函数 + 宏观量下载到 CPU 网格（执行 CPU 端边界条件前调用）。
void lbm_gpu_download(GpuSolverHandle* h, lbm::LatticeGrid* g)
{
#ifdef LBM_ENABLE_CUDA
    if (h && g) reinterpret_cast<lbm::GpuSolver*>(h)->download(*g);
#else
    (void)h; (void)g;
#endif
}

/// 将 CPU 网格 f 上传到 GPU（CPU 端边界条件修正后调用）。
void lbm_gpu_upload(GpuSolverHandle* h, lbm::LatticeGrid* g)
{
#ifdef LBM_ENABLE_CUDA
    if (h && g) reinterpret_cast<lbm::GpuSolver*>(h)->upload(*g);
#else
    (void)h; (void)g;
#endif
}

// ---------------------------------------------------------------------------
// 并行状态查询接口 — 运行时打印并行配置时使用
// ---------------------------------------------------------------------------

/// 返回 1 表示编译时启用了 OpenMP，否则返回 0。
int lbm_openmp_enabled()
{
#ifdef LBM_ENABLE_OPENMP
    return 1;
#else
    return 0;
#endif
}

/// 返回 OpenMP 最大线程数（omp_get_max_threads()）。
/// 若未启用 OpenMP，返回 1（单线程）。
int lbm_openmp_max_threads()
{
#ifdef LBM_ENABLE_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

/// 返回 1 表示编译时启用了 CUDA GPU 后端，否则返回 0。
int lbm_cuda_enabled()
{
#ifdef LBM_ENABLE_CUDA
    return 1;
#else
    return 0;
#endif
}

/// 返回 1 表示编译时启用了 MPI，否则返回 0（不触发 MPI 初始化）。
int lbm_mpi_enabled()
{
#ifdef LBM_ENABLE_MPI
    return 1;
#else
    return 0;
#endif
}

} // extern "C"
