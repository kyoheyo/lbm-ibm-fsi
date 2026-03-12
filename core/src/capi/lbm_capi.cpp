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

// ---------------------------------------------------------------------------
// 二维（XY 方向）MPI 块分解接口
// ---------------------------------------------------------------------------

/// 指向堆上 lbm::MpiDecomp2D 的不透明句柄
struct MpiDecomp2DHandle;

/// 在堆上创建 MpiDecomp2D；若 MPI 未初始化或未启用，返回 nullptr。
/// px * py 必须等于 MPI 进程总数，否则返回 nullptr。
/// Rust 封装: LbmMpiDecomp2D::new() — bindings/src/lib.rs
MpiDecomp2DHandle* lbm_mpi_decomp2d_new(int global_nx, int global_ny,
                                          int px, int py)
{
#ifdef LBM_ENABLE_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (!initialized) return nullptr;
    try {
        auto* d = new lbm::MpiDecomp2D(
            lbm::MpiDecomp2D::create(global_nx, global_ny, px, py));
        return reinterpret_cast<MpiDecomp2DHandle*>(d);
    } catch (...) {
        return nullptr;
    }
#else
    (void)global_nx; (void)global_ny; (void)px; (void)py;
    return nullptr;
#endif
}

/// 释放由 lbm_mpi_decomp2d_new 创建的 MpiDecomp2D。
/// Rust 触发: <LbmMpiDecomp2D as Drop>::drop()
void lbm_mpi_decomp2d_free(MpiDecomp2DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    delete reinterpret_cast<lbm::MpiDecomp2D*>(h);
#else
    (void)h;
#endif
}

/// 将 MpiDecomp2D 绑定到求解器；之后每次 step() 自动执行二维幽灵层交换。
/// Rust 封装: LbmSolver::attach_mpi2d() — bindings/src/lib.rs
void lbm_solver_attach_mpi2d(lbm::Solver* s, MpiDecomp2DHandle* h)
{
    if (!s) return;
#ifdef LBM_ENABLE_MPI
    s->attach_mpi2d(reinterpret_cast<const lbm::MpiDecomp2D*>(h));
#else
    (void)h;
#endif
}

/// 返回二维分解中本进程的本地 nx（含幽灵列）
int lbm_mpi_decomp2d_grid_nx(const MpiDecomp2DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp2D*>(h)->grid_nx();
#else
    (void)h; return 0;
#endif
}

/// 返回二维分解中本进程的本地 ny（含幽灵行）
int lbm_mpi_decomp2d_grid_ny(const MpiDecomp2DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp2D*>(h)->grid_ny();
#else
    (void)h; return 0;
#endif
}

/// 返回二维分解中本进程的物理 X 起始坐标（全局坐标）
int lbm_mpi_decomp2d_x_start(const MpiDecomp2DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp2D*>(h)->x_start;
#else
    (void)h; return 0;
#endif
}

/// 返回二维分解中本进程的物理 Y 起始坐标（全局坐标）
int lbm_mpi_decomp2d_y_start(const MpiDecomp2DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp2D*>(h)->y_start;
#else
    (void)h; return 0;
#endif
}

/// 返回二维分解中本进程的物理列数（不含幽灵列）
int lbm_mpi_decomp2d_local_nx(const MpiDecomp2DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp2D*>(h)->local_nx;
#else
    (void)h; return 0;
#endif
}

/// 返回二维分解中本进程的物理行数（不含幽灵行）
int lbm_mpi_decomp2d_local_ny(const MpiDecomp2DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp2D*>(h)->local_ny;
#else
    (void)h; return 0;
#endif
}

/// 返回物理区域在本地网格中的 X 偏移（0 或 1；存在西幽灵列时为 1）
int lbm_mpi_decomp2d_phys_x0(const MpiDecomp2DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp2D*>(h)->phys_x0();
#else
    (void)h; return 0;
#endif
}

/// 返回物理区域在本地网格中的 Y 偏移（0 或 1；存在南幽灵行时为 1）
int lbm_mpi_decomp2d_phys_y0(const MpiDecomp2DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp2D*>(h)->phys_y0();
#else
    (void)h; return 0;
#endif
}

/// 将每个进程的一个 int 值 gather 到 root 进程的 recv_buf 数组。
/// 非 root 进程的 recv_buf 忽略（传 nullptr 即可）。
/// 未启用 MPI 时：将 send_val 复制到 recv_buf[0]（若 recv_buf 非空）。
void lbm_mpi_gather_int(int send_val, int* recv_buf, int root)
{
#ifdef LBM_ENABLE_MPI
    MPI_Gather(&send_val, 1, MPI_INT,
               recv_buf,  1, MPI_INT,
               root, MPI_COMM_WORLD);
#else
    if (recv_buf) recv_buf[0] = send_val;
    (void)root;
#endif
}

/// 将各进程的变长 double 数组 gather 到 root 进程（MPI_Gatherv）。
///
/// @param send_buf     本进程发送缓冲区
/// @param send_count   本进程发送元素个数
/// @param recv_buf     root 进程接收缓冲区（非 root 传 nullptr）
/// @param recv_counts  root 进程：各进程元素个数数组（长度 nprocs；非 root 传 nullptr）
/// @param displs       root 进程：各进程在 recv_buf 中的偏移数组（长度 nprocs；非 root 传 nullptr）
/// @param root         根进程编号
/// 未启用 MPI 时：直接把 send_buf 的 send_count 个元素复制到 recv_buf（若非空）。
void lbm_mpi_gatherv_f64(const double* send_buf, int send_count,
                          double* recv_buf,
                          const int* recv_counts,
                          const int* displs,
                          int root)
{
#ifdef LBM_ENABLE_MPI
    MPI_Gatherv(send_buf,   send_count,  MPI_DOUBLE,
                recv_buf,   recv_counts, displs, MPI_DOUBLE,
                root, MPI_COMM_WORLD);
#else
    if (recv_buf && send_buf && send_count > 0) {
        std::copy(send_buf, send_buf + send_count, recv_buf);
    }
    (void)recv_counts; (void)displs; (void)root;
#endif
}

/// MPI 全局屏障同步。未启用 MPI 时为空操作。
void lbm_mpi_barrier()
{
#ifdef LBM_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
}

} // extern "C"

// ===========================================================================
// OpenMP 线程数设置接口
// ===========================================================================
extern "C" {

/// 设置 OpenMP 线程数（等价于 omp_set_num_threads()）。
/// 若未启用 OpenMP，此函数为空操作。
/// Rust 封装: set_omp_num_threads() — bindings/src/lib.rs
void lbm_omp_set_num_threads(int n)
{
#ifdef LBM_ENABLE_OPENMP
    if (n > 0) omp_set_num_threads(n);
#else
    (void)n;
#endif
}

} // extern "C"

// ===========================================================================
// 多重网格树（MgTree）接口
// ===========================================================================
#include "lbm/mg_tree.hpp"
extern "C" {

/// 不透明句柄类型（前向声明）
struct MgTreeHandle;
struct MgNodeHandle;

/// 创建多重网格树，根节点（最粗网格）的空间范围为 [x0,x1] × [y0,y1] × [z0,z1]。
/// 2D 仿真时令 z0=z1=0。
/// Rust 封装: LbmMgTree::new() — bindings/src/lib.rs
MgTreeHandle* lbm_mg_tree_new(int x0, int x1, int y0, int y1, int z0, int z1, int is_3d)
{
    try {
        lbm::MgExtent root_extent{x0, x1, y0, y1, z0, z1};
        lbm::MgDim dim = (is_3d != 0) ? lbm::MgDim::D3 : lbm::MgDim::D2;
        auto* tree = new lbm::MgTree(root_extent, dim);
        return reinterpret_cast<MgTreeHandle*>(tree);
    } catch (...) {
        return nullptr;
    }
}

/// 释放多重网格树（含树中所有节点）。
void lbm_mg_tree_free(MgTreeHandle* h)
{
    delete reinterpret_cast<lbm::MgTree*>(h);
}

/// 在父节点内添加一个细化子区域节点。
/// 返回新创建的子节点句柄（由树管理，调用方不得释放）。
/// 若父节点为 nullptr 或 child_extent 不在父节点范围内，返回 nullptr。
/// Rust 封装: LbmMgTree::add_level() — bindings/src/lib.rs
MgNodeHandle* lbm_mg_tree_add_level(MgTreeHandle* tree, MgNodeHandle* parent,
                                     int x0, int x1, int y0, int y1, int z0, int z1,
                                     int refine_ratio)
{
    if (!tree || !parent) return nullptr;
    try {
        lbm::MgExtent child_extent{x0, x1, y0, y1, z0, z1};
        auto* node = reinterpret_cast<lbm::MgTree*>(tree)->add_level(
            reinterpret_cast<lbm::MgNode*>(parent), child_extent, refine_ratio);
        return reinterpret_cast<MgNodeHandle*>(node);
    } catch (...) {
        return nullptr;
    }
}

/// 获取根节点（最粗网格）句柄。
MgNodeHandle* lbm_mg_tree_root(MgTreeHandle* h)
{
    if (!h) return nullptr;
    return reinterpret_cast<MgNodeHandle*>(
        reinterpret_cast<lbm::MgTree*>(h)->root());
}

/// 返回树中最深的层级（根节点为 0）。
int lbm_mg_tree_max_level(const MgTreeHandle* h)
{
    if (!h) return 0;
    return reinterpret_cast<const lbm::MgTree*>(h)->max_level();
}

/// 返回树中所有节点数（包括根节点）。
int lbm_mg_tree_node_count(const MgTreeHandle* h)
{
    if (!h) return 0;
    return static_cast<int>(reinterpret_cast<const lbm::MgTree*>(h)->node_count());
}

/// 将 LatticeGrid 绑定到多重网格节点（供该层 LBM 求解使用）。
void lbm_mg_node_set_grid(MgNodeHandle* node, lbm::LatticeGrid* grid)
{
    if (!node) return;
    reinterpret_cast<lbm::MgNode*>(node)->grid = grid;
}

/// 查询节点的空间范围
int lbm_mg_node_x_start(const MgNodeHandle* h) { return h ? reinterpret_cast<const lbm::MgNode*>(h)->extent.x_start : 0; }
int lbm_mg_node_x_end  (const MgNodeHandle* h) { return h ? reinterpret_cast<const lbm::MgNode*>(h)->extent.x_end   : 0; }
int lbm_mg_node_y_start(const MgNodeHandle* h) { return h ? reinterpret_cast<const lbm::MgNode*>(h)->extent.y_start : 0; }
int lbm_mg_node_y_end  (const MgNodeHandle* h) { return h ? reinterpret_cast<const lbm::MgNode*>(h)->extent.y_end   : 0; }
int lbm_mg_node_z_start(const MgNodeHandle* h) { return h ? reinterpret_cast<const lbm::MgNode*>(h)->extent.z_start : 0; }
int lbm_mg_node_z_end  (const MgNodeHandle* h) { return h ? reinterpret_cast<const lbm::MgNode*>(h)->extent.z_end   : 0; }
int lbm_mg_node_level  (const MgNodeHandle* h) { return h ? reinterpret_cast<const lbm::MgNode*>(h)->level          : 0; }
int lbm_mg_node_refine_ratio(const MgNodeHandle* h) { return h ? reinterpret_cast<const lbm::MgNode*>(h)->refine_ratio : 1; }
int lbm_mg_node_child_count (const MgNodeHandle* h) {
    return h ? static_cast<int>(reinterpret_cast<const lbm::MgNode*>(h)->children.size()) : 0;
}

/// 延拓算子：从粗网格双线性插值 ρ/u 到细网格。
/// 两个节点必须均已绑定 LatticeGrid（通过 lbm_mg_node_set_grid 设置）。
/// 返回 0 表示成功，-1 表示参数错误。
int lbm_mg_prolong_rho_u(const MgNodeHandle* coarse, MgNodeHandle* fine)
{
    if (!coarse || !fine) return -1;
    try {
        lbm::mg_prolong_rho_u(
            *reinterpret_cast<const lbm::MgNode*>(coarse),
            *reinterpret_cast<lbm::MgNode*>(fine));
        return 0;
    } catch (...) {
        return -1;
    }
}

/// 限制算子：从细网格体积平均 ρ/u 到粗网格。
/// 两个节点必须均已绑定 LatticeGrid（通过 lbm_mg_node_set_grid 设置）。
/// 返回 0 表示成功，-1 表示参数错误。
int lbm_mg_restrict_rho_u(const MgNodeHandle* fine, MgNodeHandle* coarse)
{
    if (!fine || !coarse) return -1;
    try {
        lbm::mg_restrict_rho_u(
            *reinterpret_cast<const lbm::MgNode*>(fine),
            *reinterpret_cast<lbm::MgNode*>(coarse));
        return 0;
    } catch (...) {
        return -1;
    }
}

} // extern "C"

// ===========================================================================
// MpiDecomp3D（三维域分解预留接口）
// ===========================================================================
extern "C" {

/// 不透明句柄（前向声明）
struct MpiDecomp3DHandle;

/// 创建三维 MPI 域分解（需先调用 MPI_Init，且 px*py*pz == nprocs）。
/// 幽灵层交换暂未实现；可用于记录三维分解信息（空间范围查询等）。
/// Rust 封装: LbmMpiDecomp3D::new() — bindings/src/lib.rs
MpiDecomp3DHandle* lbm_mpi_decomp3d_new(int gnx, int gny, int gnz,
                                          int px, int py, int pz)
{
    try {
        auto* d = new lbm::MpiDecomp3D(
            lbm::MpiDecomp3D::create(gnx, gny, gnz, px, py, pz));
        return reinterpret_cast<MpiDecomp3DHandle*>(d);
    } catch (...) {
        return nullptr;
    }
}

/// 释放三维域分解对象。
void lbm_mpi_decomp3d_free(MpiDecomp3DHandle* h)
{
    delete reinterpret_cast<lbm::MpiDecomp3D*>(h);
}

/// 查询字段
int lbm_mpi_decomp3d_grid_nx(const MpiDecomp3DHandle* h) {
    return h ? reinterpret_cast<const lbm::MpiDecomp3D*>(h)->grid_nx() : 0;
}
int lbm_mpi_decomp3d_grid_ny(const MpiDecomp3DHandle* h) {
    return h ? reinterpret_cast<const lbm::MpiDecomp3D*>(h)->grid_ny() : 0;
}
int lbm_mpi_decomp3d_grid_nz(const MpiDecomp3DHandle* h) {
    return h ? reinterpret_cast<const lbm::MpiDecomp3D*>(h)->grid_nz() : 0;
}
int lbm_mpi_decomp3d_x_start(const MpiDecomp3DHandle* h) {
    return h ? reinterpret_cast<const lbm::MpiDecomp3D*>(h)->x_start : 0;
}
int lbm_mpi_decomp3d_y_start(const MpiDecomp3DHandle* h) {
    return h ? reinterpret_cast<const lbm::MpiDecomp3D*>(h)->y_start : 0;
}
int lbm_mpi_decomp3d_z_start(const MpiDecomp3DHandle* h) {
    return h ? reinterpret_cast<const lbm::MpiDecomp3D*>(h)->z_start : 0;
}

/// 返回三维分解中本进程的物理列数（不含幽灵列）
int lbm_mpi_decomp3d_local_nx(const MpiDecomp3DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp3D*>(h)->local_nx;
#else
    (void)h; return 0;
#endif
}

/// 返回三维分解中本进程的物理行数（不含幽灵行）
int lbm_mpi_decomp3d_local_ny(const MpiDecomp3DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp3D*>(h)->local_ny;
#else
    (void)h; return 0;
#endif
}

/// 返回三维分解中本进程的物理层数（不含幽灵层）
int lbm_mpi_decomp3d_local_nz(const MpiDecomp3DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp3D*>(h)->local_nz;
#else
    (void)h; return 0;
#endif
}

/// 返回物理区域在本地网格中的 X 偏移（0 或 1；存在西幽灵列时为 1）
int lbm_mpi_decomp3d_phys_x0(const MpiDecomp3DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp3D*>(h)->phys_x0();
#else
    (void)h; return 0;
#endif
}

/// 返回物理区域在本地网格中的 Y 偏移（0 或 1；存在南幽灵行时为 1）
int lbm_mpi_decomp3d_phys_y0(const MpiDecomp3DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp3D*>(h)->phys_y0();
#else
    (void)h; return 0;
#endif
}

/// 返回物理区域在本地网格中的 Z 偏移（0 或 1；存在底幽灵层时为 1）
int lbm_mpi_decomp3d_phys_z0(const MpiDecomp3DHandle* h)
{
#ifdef LBM_ENABLE_MPI
    if (!h) return 0;
    return reinterpret_cast<const lbm::MpiDecomp3D*>(h)->phys_z0();
#else
    (void)h; return 0;
#endif
}

/// 将 MpiDecomp3D 绑定到求解器；之后每次 step() 自动执行三维幽灵层交换。
/// Rust 封装: LbmSolver::attach_mpi3d() — bindings/src/lib.rs
void lbm_solver_attach_mpi3d(lbm::Solver* s, MpiDecomp3DHandle* h)
{
    if (!s) return;
#ifdef LBM_ENABLE_MPI
    s->attach_mpi3d(reinterpret_cast<const lbm::MpiDecomp3D*>(h));
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

/// 仅下载宏观量（ρ、u）到 CPU 网格（用于输出快照，比 download() 少 7× 数据量）。
void lbm_gpu_download_rho_u(GpuSolverHandle* h, lbm::LatticeGrid* g)
{
#ifdef LBM_ENABLE_CUDA
    if (h && g) reinterpret_cast<lbm::GpuSolver*>(h)->download_rho_u(*g);
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
// 异步输出管线接口
// ---------------------------------------------------------------------------

/// 异步单步执行：在 compute_stream 上提交全部核函数（BGK + 流式 + GPU-BC + 宏观量），
/// 记录 compute_done 事件，立即返回（不阻塞 CPU）。
void lbm_gpu_step_async(GpuSolverHandle* h)
{
#ifdef LBM_ENABLE_CUDA
    if (h) reinterpret_cast<lbm::GpuSolver*>(h)->step_async();
#else
    (void)h;
#endif
}

/// 等待 compute_stream 完成（cudaStreamSynchronize）。
void lbm_gpu_wait_compute(GpuSolverHandle* h)
{
#ifdef LBM_ENABLE_CUDA
    if (h) reinterpret_cast<lbm::GpuSolver*>(h)->wait_compute();
#else
    (void)h;
#endif
}

/// 将 d_rho/d_u 异步拷贝到固定主机双缓冲区 buf_idx（0 或 1）的 io_stream 上。
/// io_stream 自动等待 compute_done 事件（确保计算完成后才开始拷贝）。
void lbm_gpu_enqueue_async_download_rho_u(GpuSolverHandle* h, int buf_idx)
{
#ifdef LBM_ENABLE_CUDA
    if (h) reinterpret_cast<lbm::GpuSolver*>(h)->enqueue_async_download_rho_u(buf_idx);
#else
    (void)h; (void)buf_idx;
#endif
}

/// 等待 io_stream（异步拷贝）完成（cudaStreamSynchronize）。
void lbm_gpu_sync_async_download(GpuSolverHandle* h)
{
#ifdef LBM_ENABLE_CUDA
    if (h) reinterpret_cast<lbm::GpuSolver*>(h)->sync_async_download();
#else
    (void)h;
#endif
}

/// 返回固定主机缓冲区中 ρ 数组的指针（仅在 lbm_gpu_sync_async_download 后调用）。
const double* lbm_gpu_pinned_rho(const GpuSolverHandle* h, int buf_idx)
{
#ifdef LBM_ENABLE_CUDA
    if (h) return reinterpret_cast<const lbm::GpuSolver*>(h)->h_rho(buf_idx);
#else
    (void)h; (void)buf_idx;
#endif
    return nullptr;
}

/// 返回固定主机缓冲区中 u 数组的指针（仅在 lbm_gpu_sync_async_download 后调用）。
const double* lbm_gpu_pinned_u(const GpuSolverHandle* h, int buf_idx)
{
#ifdef LBM_ENABLE_CUDA
    if (h) return reinterpret_cast<const lbm::GpuSolver*>(h)->h_u(buf_idx);
#else
    (void)h; (void)buf_idx;
#endif
    return nullptr;
}

/// 返回总节点数 n（= nx × ny）。
int lbm_gpu_n(const GpuSolverHandle* h)
{
#ifdef LBM_ENABLE_CUDA
    if (h) return reinterpret_cast<const lbm::GpuSolver*>(h)->n();
#else
    (void)h;
#endif
    return 0;
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
