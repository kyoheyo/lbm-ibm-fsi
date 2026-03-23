#pragma once
// lbm/mpi_decomp.hpp — D2Q9/D3Q19/D3Q27 LBM 的 MPI 域分解
//
// ============================================================
// 统一视角：一维切片是二维块分解的特例
// ============================================================
// 三种域分解模式统一在 MpiDecomp2D / MpiDecomp3D 框架下：
//   一维 Y 切片：MpiDecomp2D::create(gnx, gny, 1, nprocs)  → px=1, py=nprocs
//   一维 X 切片：MpiDecomp2D::create(gnx, gny, nprocs, 1)  → px=nprocs, py=1
//   二维 XY 块 ：MpiDecomp2D::create(gnx, gny, px, py)     → px*py==nprocs
//
// MpiDecomp（旧 1D 专用结构）保留用于向后兼容；新代码请使用 MpiDecomp2D。
//
// ============================================================
// 模式一（旧）：一维（Y 方向）域分解（MpiDecomp，向后兼容）
// ============================================================
// 将全局网格沿 Y 方向均匀切分为 nprocs 个切片，每个进程持有
//   ny_local 行物理节点 + 2 行幽灵（halo）节点：
//
//    j=0              : 南幽灵行（来自 rank-1 的顶物理行）
//    j=1 .. ny_local  : 本进程物理行（对应全局 y_start .. y_end）
//    j=ny_local+1     : 北幽灵行（来自 rank+1 的底物理行）
//
// 特殊情况（nprocs=1）：ny_local = global_ny，无需额外幽灵行。
//
// ============================================================
// 模式二（主推）：二维（XY 方向）块分解（MpiDecomp2D）
// ============================================================
// 将全局网格沿 X 和 Y 方向分别切分为 px × py 块，总进程数 = px × py。
// rank = row_rank * px + col_rank，其中：
//   col_rank = rank % px  （X 方向进程坐标，0..px-1）
//   row_rank = rank / px  （Y 方向进程坐标，0..py-1）
//
// 一维特例：px=1 → col_rank=0（等价于 1D Y 切片）；py=1 → row_rank=0（1D X 切片）。
//
// 每个进程持有 local_nx × local_ny 物理节点，
// 加上最多 4 条幽灵行/列（南/北/西/东）。
// 幽灵层由 halo_exchange_d2q9_2d() 通过 MPI_Sendrecv 交换。
//
// ============================================================
// 模式三（预留）：三维（XYZ 方向）块分解（MpiDecomp3D）
// ============================================================
// 将全局三维网格沿 X/Y/Z 方向切分为 px × py × pz 块，总进程数 = px * py * pz。
// rank = pz_rank*(px*py) + row_rank*px + col_rank，其中：
//   col_rank = rank % px
//   row_rank = (rank / px) % py
//   pz_rank  = rank / (px * py)
//
// 当 pz=1 时等价于 MpiDecomp2D（2D 网格）。
// 幽灵层交换（halo_exchange_d3q19_3d）尚未实现（留作扩展接口）。
//
// ============================================================
// 边界条件约定（与 boundary.cpp 兼容）
// ============================================================
// 二维模式：
//   col_rank==0    : 注册 Face::West BC
//   col_rank==px-1 : 注册 Face::East BC
//   row_rank==0    : 注册 Face::South BC
//   row_rank==py-1 : 注册 Face::North BC
// 三维模式（额外）：
//   pz_rank==0     : 注册 Face::Bottom BC
//   pz_rank==pz-1  : 注册 Face::Top BC
//
// ============================================================
// 参考文献
// ============================================================
// Succi S. (2001) "The Lattice Boltzmann Equation for Fluid Dynamics and Beyond", §4.
// Schulz M. et al. (2002) LBM + MPI parallelisation, EPFL technical report.

#ifdef LBM_ENABLE_MPI
#include <mpi.h>
#endif

#include "lattice.hpp"

namespace lbm {

// ---------------------------------------------------------------------------
/// 一维（Y 方向）MPI 域分解描述符（向后兼容）
///
/// 建议新代码使用 MpiDecomp2D::create(gnx, gny, 1, nprocs) 代替。
// ---------------------------------------------------------------------------
struct MpiDecomp {
    int rank      = 0;              ///< 当前进程编号
    int nprocs    = 1;              ///< 进程总数
    int global_nx = 0;              ///< 全局列数
    int global_ny = 0;              ///< 全局行数
    int y_start   = 0;              ///< 本地物理起始行（全局坐标）
    int y_end     = 0;              ///< 本地物理结束行（全局坐标，含）
    int local_ny  = 0;              ///< 本地物理行数（不含幽灵行）
    int rank_south = -1;            ///< 南邻进程 rank（-1 = 无邻居，即 MPI_PROC_NULL）
    int rank_north = -1;            ///< 北邻进程 rank（-1 = 无邻居，即 MPI_PROC_NULL）

    /// 当前进程是否持有全局南物理壁（y=0）
    bool has_south_wall() const { return y_start == 0; }
    /// 当前进程是否持有全局北物理壁（y=global_ny-1）
    bool has_north_wall() const { return y_end == global_ny - 1; }

    /// 本地网格 ny（含幽灵行）：nprocs>1 时 = local_ny+2；nprocs=1 时 = local_ny
    int grid_ny() const { return (nprocs > 1) ? local_ny + 2 : local_ny; }

    /// 由全局尺寸创建 MpiDecomp（需先调用 MPI_Init）
    static MpiDecomp create(int global_nx, int global_ny);
};

// ---------------------------------------------------------------------------
/// 二维（XY 方向）MPI 块分解描述符
///
/// 进程布局（rank = row_rank * px + col_rank）：
///
///   row_rank=py-1  ┌─────┬─────┬─────┐  ← 全局北壁（row_rank==py-1 注册 Face::North BC）
///                  │     │     │     │
///   row_rank=1     ├─────┼─────┼─────┤
///                  │     │     │     │
///   row_rank=0     └─────┴─────┴─────┘  ← 全局南壁（row_rank==0 注册 Face::South BC）
///                col=0   col=1  col=2
///                ↑                   ↑
///             西壁BC               东壁BC
///
/// 一维特例：
///   px=1（1D Y 切片）: col_rank=0，无西/东幽灵列，等价于旧 MpiDecomp
///   py=1（1D X 切片）: row_rank=0，无南/北幽灵行
///
/// 每个进程的本地网格尺寸（含幽灵层）：
///   ny_with_ghost = local_ny + (has_south_ghost ? 1 : 0) + (has_north_ghost ? 1 : 0)
///   nx_with_ghost = local_nx + (has_west_ghost  ? 1 : 0) + (has_east_ghost  ? 1 : 0)
///
/// 幽灵层偏移约定：
///   j=0            : 南幽灵层（若 row_rank > 0）
///   j=1..local_ny  : 物理行（或 j=0..local_ny-1 对最南 rank）
///   i=0            : 西幽灵列（若 col_rank > 0）
///   i=1..local_nx  : 物理列（或 i=0..local_nx-1 对最西 rank）
// ---------------------------------------------------------------------------
struct MpiDecomp2D {
    int rank      = 0;              ///< 当前进程全局编号
    int nprocs    = 1;              ///< 进程总数（= px * py）
    int px        = 1;              ///< X 方向进程数
    int py        = 1;              ///< Y 方向进程数
    int col_rank  = 0;              ///< X 方向进程坐标（0..px-1）
    int row_rank  = 0;              ///< Y 方向进程坐标（0..py-1）

    int global_nx = 0;              ///< 全局 X 节点数
    int global_ny = 0;              ///< 全局 Y 节点数
    int x_start   = 0;              ///< 本地物理起始列（全局坐标）
    int x_end     = 0;              ///< 本地物理结束列（全局坐标，含）
    int y_start   = 0;              ///< 本地物理起始行（全局坐标）
    int y_end     = 0;              ///< 本地物理结束行（全局坐标，含）
    int local_nx  = 0;              ///< 本地物理列数（不含幽灵）
    int local_ny  = 0;              ///< 本地物理行数（不含幽灵）

    int rank_south = -1;            ///< 南邻 rank（row_rank-1，无邻居时为 MPI_PROC_NULL）
    int rank_north = -1;            ///< 北邻 rank（row_rank+1，无邻居时为 MPI_PROC_NULL）
    int rank_west  = -1;            ///< 西邻 rank（col_rank-1，无邻居时为 MPI_PROC_NULL）
    int rank_east  = -1;            ///< 东邻 rank（col_rank+1，无邻居时为 MPI_PROC_NULL）

    /// 每侧幽灵层数（默认 1；设为 2 时支持 FourPoint IBM 核在 MPI 边界附近的标记点）
    /// 对应 TOML 配置 [mpi] ibm_halo_width。
    /// 影响：grid_ny/nx()、phys_y0/x0()、halo_exchange_d2q9_2d()、
    ///        halo_exchange_u_2d() 以及 ibm_halo_reduce_force_2d() 的行为。
    int n_ghost    = 1;

    /// 是否持有全局南物理壁（y=0）
    bool has_south_wall() const { return y_start == 0; }
    /// 是否持有全局北物理壁（y=global_ny-1）
    bool has_north_wall() const { return y_end   == global_ny - 1; }
    /// 是否持有全局西物理壁（x=0）
    bool has_west_wall()  const { return x_start == 0; }
    /// 是否持有全局东物理壁（x=global_nx-1）
    bool has_east_wall()  const { return x_end   == global_nx - 1; }

    /// 是否存在南幽灵层（非最南 rank）
    bool has_south_ghost() const { return row_rank > 0; }
    /// 是否存在北幽灵层（非最北 rank）
    bool has_north_ghost() const { return row_rank < py - 1; }
    /// 是否存在西幽灵列（非最西 rank）
    bool has_west_ghost()  const { return col_rank > 0; }
    /// 是否存在东幽灵列（非最东 rank）
    bool has_east_ghost()  const { return col_rank < px - 1; }

    /// 本地网格含幽灵层的 nx（每侧最多 n_ghost 列）
    int grid_nx() const {
        return local_nx
             + (has_west_ghost()  ? n_ghost : 0)
             + (has_east_ghost()  ? n_ghost : 0);
    }
    /// 本地网格含幽灵层的 ny（每侧最多 n_ghost 行）
    int grid_ny() const {
        return local_ny
             + (has_south_ghost() ? n_ghost : 0)
             + (has_north_ghost() ? n_ghost : 0);
    }
    /// 物理区域在本地网格中的 x 偏移（0 或 n_ghost）
    int phys_x0() const { return has_west_ghost()  ? n_ghost : 0; }
    /// 物理区域在本地网格中的 y 偏移（0 或 n_ghost）
    int phys_y0() const { return has_south_ghost() ? n_ghost : 0; }

    /// 由全局尺寸和块数创建 MpiDecomp2D（需先调用 MPI_Init）。
    /// px * py 必须等于 MPI 进程总数，否则抛出 std::invalid_argument。
    /// n_ghost 指定每侧幽灵层数（默认 1；ibm_halo_width=2 时传入 2）。
    static MpiDecomp2D create(int global_nx, int global_ny, int px, int py,
                               int n_ghost = 1);
};

// ---------------------------------------------------------------------------
/// 三维（XYZ 方向）MPI 块分解描述符（预留接口，幽灵交换尚未实现）
///
/// 进程布局（rank = pz_rank*(px*py) + row_rank*px + col_rank）：
///   col_rank = rank % px
///   row_rank = (rank / px) % py
///   pz_rank  = rank / (px * py)
///
/// 一维/二维特例：
///   pz=1     → 等价于 MpiDecomp2D
///   pz=1,py=1 → 等价于 MpiDecomp（1D X 切片）
///   pz=1,px=1 → 等价于 MpiDecomp（1D Y 切片）
///
/// @note 本结构用于预留三维 LBM 并行扩展接口（D3Q19/D3Q27）。
///       幽灵层交换函数 halo_exchange_d3q19_3d() 尚未实现，调用将在运行时抛出异常。
///       三维 LBM 求解器（支持 D3Q19/D3Q27）也尚未实现，待未来扩展。
// ---------------------------------------------------------------------------
struct MpiDecomp3D {
    int rank      = 0;              ///< 当前进程全局编号
    int nprocs    = 1;              ///< 进程总数（= px * py * pz）
    int px        = 1;              ///< X 方向进程数
    int py        = 1;              ///< Y 方向进程数
    int pz        = 1;              ///< Z 方向进程数

    int col_rank  = 0;              ///< X 方向进程坐标（0..px-1）
    int row_rank  = 0;              ///< Y 方向进程坐标（0..py-1）
    int pz_rank   = 0;              ///< Z 方向进程坐标（0..pz-1）

    int global_nx = 0;              ///< 全局 X 节点数
    int global_ny = 0;              ///< 全局 Y 节点数
    int global_nz = 0;              ///< 全局 Z 节点数

    int x_start = 0, x_end = 0;    ///< 本地物理 X 范围（全局坐标，含端点）
    int y_start = 0, y_end = 0;    ///< 本地物理 Y 范围
    int z_start = 0, z_end = 0;    ///< 本地物理 Z 范围

    int local_nx = 0;               ///< 本地物理 X 方向列数（不含幽灵）
    int local_ny = 0;               ///< 本地物理 Y 方向行数（不含幽灵）
    int local_nz = 0;               ///< 本地物理 Z 方向层数（不含幽灵）

    /// 六邻进程 rank（MPI_PROC_NULL=-1 表示无邻居）
    int rank_west   = -1, rank_east   = -1;
    int rank_south  = -1, rank_north  = -1;
    int rank_bottom = -1, rank_top    = -1;

    /// 物理壁面查询
    bool has_west_wall()   const { return x_start == 0; }
    bool has_east_wall()   const { return x_end   == global_nx - 1; }
    bool has_south_wall()  const { return y_start == 0; }
    bool has_north_wall()  const { return y_end   == global_ny - 1; }
    bool has_bottom_wall() const { return z_start == 0; }
    bool has_top_wall()    const { return z_end   == global_nz - 1; }

    /// 幽灵层查询
    bool has_west_ghost()   const { return col_rank > 0; }
    bool has_east_ghost()   const { return col_rank < px - 1; }
    bool has_south_ghost()  const { return row_rank > 0; }
    bool has_north_ghost()  const { return row_rank < py - 1; }
    bool has_bottom_ghost() const { return pz_rank  > 0; }
    bool has_top_ghost()    const { return pz_rank  < pz - 1; }

    /// 本地网格含幽灵层的尺寸
    int grid_nx() const { return local_nx + (has_west_ghost()?1:0) + (has_east_ghost()?1:0); }
    int grid_ny() const { return local_ny + (has_south_ghost()?1:0) + (has_north_ghost()?1:0); }
    int grid_nz() const { return local_nz + (has_bottom_ghost()?1:0) + (has_top_ghost()?1:0); }

    /// 物理区域在本地网格中的偏移（0 或 1）
    int phys_x0() const { return has_west_ghost()   ? 1 : 0; }
    int phys_y0() const { return has_south_ghost()  ? 1 : 0; }
    int phys_z0() const { return has_bottom_ghost() ? 1 : 0; }

    /// 创建三维域分解（需先调用 MPI_Init）。
    /// px * py * pz 必须等于 MPI 进程总数，否则抛出 std::invalid_argument。
    /// @note 幽灵层交换尚未实现；本接口用于预留三维并行框架。
    static MpiDecomp3D create(int global_nx, int global_ny, int global_nz,
                               int px, int py, int pz);
};

#ifdef LBM_ENABLE_MPI
// ---------------------------------------------------------------------------
/// 一维模式（Y 方向）幽灵行交换
///
/// 约定：g.ny = decomp.local_ny + 2（含幽灵行，nprocs>1）
///   j=0            : 南幽灵（接收南邻顶物理行）
///   j=1..local_ny  : 物理行
///   j=local_ny+1   : 北幽灵（接收北邻底物理行）
///
/// 由 Solver::stream() 在 stream 循环之前（碰后）自动调用。
// ---------------------------------------------------------------------------
void halo_exchange_d2q9(LatticeGrid& g, const MpiDecomp& decomp);

// ---------------------------------------------------------------------------
/// 二维模式（XY 方向）幽灵层交换
///
/// 约定：g.nx = decomp.grid_nx()，g.ny = decomp.grid_ny()（含幽灵层）
///
/// 交换顺序（避免死锁）：
///   1. 南北方向：整行交换（MPI_Sendrecv，一次发送整行 f 数据）
///   2. 东西方向：整列交换（MPI_Sendrecv，列数据需先打包到临时缓冲区）
///
/// 由 Solver::stream() 在 stream 循环之前（碰后）自动调用（若绑定了 MpiDecomp2D）。
///
/// 一维特例（px=1 或 py=1）：对应方向无幽灵列/行，自动跳过对应 Sendrecv。
// ---------------------------------------------------------------------------
void halo_exchange_d2q9_2d(LatticeGrid& g, const MpiDecomp2D& decomp);

// ---------------------------------------------------------------------------
/// 三维模式（XYZ 方向）幽灵层交换（D3Q19/D3Q27）
///
/// 约定：g.nx/ny/nz 均已包含幽灵层（各方向各加 0~2 层）：
///   k=0                   : 底幽灵层（pz_rank>0 时存在）
///   k=phys_z0..+lnz-1     : 物理层
///   k=phys_z0+lnz         : 顶幽灵层（pz_rank<pz-1 时存在）
///   j=0 / j=phys_y0+lny   : 南/北幽灵行（类似二维）
///   i=0 / i=phys_x0+lnx   : 西/东幽灵列（类似二维）
///
/// 交换顺序：先 Z（底/顶），再 Y（南/北），最后 X（西/东），每方向均使用 MPI_Sendrecv。
///
/// 由 Solver::stream() 在 stream 循环之前（碰后）自动调用（若绑定了 MpiDecomp3D）。
// ---------------------------------------------------------------------------
void halo_exchange_d3q19_3d(LatticeGrid& g, const MpiDecomp3D& decomp);

// ---------------------------------------------------------------------------
/// 宏观速度场幽灵层交换（2D 分区，D2Q9）
///
/// 在 PUSH 流式迁移之后，幽灵行/列的 u 被本地边界行 f 推入覆盖，导致错误。
/// 本函数在 compute_macroscopic() 完成后交换物理边界行/列的真实 u 到邻居幽灵行/列，
/// 使 IBM 插值等需要幽灵层速度的操作可直接使用正确数据，无需再额外调用 IBM 专用函数。
///
/// 由 Solver::stream() 在 compute_macroscopic() 之后自动调用（若绑定了 MpiDecomp2D）。
// ---------------------------------------------------------------------------
void halo_exchange_u_2d(LatticeGrid& g, const MpiDecomp2D& decomp);
#endif

} // namespace lbm

