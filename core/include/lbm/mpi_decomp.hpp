#pragma once
// lbm/mpi_decomp.hpp — D2Q9 LBM 的 MPI 域分解（一维 Y 切片 + 二维 XY 块分解）
//
// ============================================================
// 模式一：一维（Y 方向）域分解（MpiDecomp）
// ============================================================
// 将全局网格沿 Y 方向均匀切分为 nprocs 个切片，每个 MPI 进程持有
//   ny_local 行物理节点 + 2 行幽灵（halo）节点：
//
//    j=0              : 南幽灵行（来自 rank-1 的顶物理行）
//    j=1 .. ny_local  : 本进程物理行（对应全局 y_start .. y_end）
//    j=ny_local+1     : 北幽灵行（来自 rank+1 的底物理行）
//
// 特殊情况（nprocs=1）：ny_local = global_ny，无需额外幽灵行（本地 ny = global_ny）。
//
// ============================================================
// 模式二：二维（XY 方向）块分解（MpiDecomp2D）
// ============================================================
// 将全局网格沿 X 和 Y 方向分别切分为 px × py 块，总进程数 = px × py。
// rank = row_rank * px + col_rank，其中：
//   col_rank = rank % px  （X 方向进程坐标，0..px-1）
//   row_rank = rank / px  （Y 方向进程坐标，0..py-1）
//
// 每个进程持有 local_nx × local_ny 物理节点，
// 加上最多 4 条幽灵行/列（南/北/西/东）。
// 幽灵行/列由 halo_exchange_d2q9_2d() 通过 MPI_Sendrecv 交换。
//
// ============================================================
// 边界条件约定（与 boundary.cpp 兼容）
// ============================================================
// 一维模式：
//   rank 0         : 注册 Face::South BC
//   rank nprocs-1  : 注册 Face::North BC
//   所有 rank      : 注册 Face::West / Face::East BC
// 二维模式：
//   col_rank==0    : 注册 Face::West BC
//   col_rank==px-1 : 注册 Face::East BC
//   row_rank==0    : 注册 Face::South BC
//   row_rank==py-1 : 注册 Face::North BC
//
// ============================================================
// 参考文献
// ============================================================
// Succi S. (2001) "The Lattice Boltzmann Equation for Fluid Dynamics and Beyond", §4.
// Schulz M. et al. (2002) LBM + MPI parallelisation, EPFL technical report.
// Wellein G. et al. (2006) On the single processor performance of simple lattice
//   Boltzmann kernels. Comput. Fluids 35(8-9):910-919.

#ifdef LBM_ENABLE_MPI
#include <mpi.h>
#endif

#include "lattice.hpp"

namespace lbm {

// ---------------------------------------------------------------------------
/// 一维（Y 方向）MPI 域分解描述符
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

    /// 本地网格含幽灵层的 nx
    int grid_nx() const {
        return local_nx + (has_west_ghost() ? 1 : 0) + (has_east_ghost() ? 1 : 0);
    }
    /// 本地网格含幽灵层的 ny
    int grid_ny() const {
        return local_ny + (has_south_ghost() ? 1 : 0) + (has_north_ghost() ? 1 : 0);
    }
    /// 物理区域在本地网格中的 x 偏移（0 或 1）
    int phys_x0() const { return has_west_ghost()  ? 1 : 0; }
    /// 物理区域在本地网格中的 y 偏移（0 或 1）
    int phys_y0() const { return has_south_ghost() ? 1 : 0; }

    /// 由全局尺寸和块数创建 MpiDecomp2D（需先调用 MPI_Init）。
    /// px * py 必须等于 MPI 进程总数，否则抛出 std::invalid_argument。
    static MpiDecomp2D create(int global_nx, int global_ny, int px, int py);
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
/// 由 Solver::stream() 在 std::swap 之后、apply_BC() 之前自动调用。
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
/// 由 Solver::stream() 在 std::swap 之后、apply_BC() 之前自动调用（若绑定了 MpiDecomp2D）。
// ---------------------------------------------------------------------------
void halo_exchange_d2q9_2d(LatticeGrid& g, const MpiDecomp2D& decomp);
#endif

} // namespace lbm
