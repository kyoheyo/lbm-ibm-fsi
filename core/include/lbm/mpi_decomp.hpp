#pragma once
// lbm/mpi_decomp.hpp — D2Q9 LBM 的一维 MPI 域分解（Y 方向切片）
//
// 设计概述
// --------
// 将全局网格沿 Y 方向均匀切分为 nprocs 个切片，每个 MPI 进程持有
//   ny_local 行物理节点 + 2 行幽灵（halo）节点：
//
//    j=0              : 南幽灵行（来自 rank-1 的顶物理行）
//    j=1 .. ny_local  : 本进程物理行（对应全局 y_start .. y_end）
//    j=ny_local+1     : 北幽灵行（来自 rank+1 的底物理行）
//
// 特殊情况（nprocs=1）：ny_local = global_ny，无需额外幽灵行（本地 ny = global_ny）。
//
// 边界条件约定（与 boundary.cpp 兼容）
// -------
//   rank 0     : j=0（南幽灵）在 BC 层面被视为物理南壁，仅 rank 0 注册 Face::South BC。
//   rank nprocs-1: j=ny_local+1 被视为物理北壁，仅最后一个 rank 注册 Face::North BC。
//   其余 rank   : 不注册 South/North BC；幽灵行由 halo_exchange_d2q9() 维护。
//
// 使用示例
// --------
//   int y_start, local_ny;
//   lbm_mpi_init(argc, argv);
//   lbm_mpi_local_ny(global_ny, &y_start, &local_ny);
//   int grid_ny = (lbm_mpi_size() > 1) ? local_ny + 2 : local_ny;
//   LatticeGrid g(nx, grid_ny, 1, LatticeModel::D2Q9);
//   auto decomp = MpiDecomp::create(nx, global_ny);
//   solver.attach_mpi(&decomp);  // step() 自动调用 halo_exchange_d2q9()
//
// 参考文献
// --------
// Succi S. (2001) "The Lattice Boltzmann Equation for Fluid Dynamics and Beyond", §4.
// Schulz M. et al. (2002) LBM + MPI parallelisation, EPFL technical report.

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

#ifdef LBM_ENABLE_MPI
// ---------------------------------------------------------------------------
/// D2Q9 幽灵行交换（Halo Exchange）
///
/// 约定：g.ny = decomp.local_ny + 2（含幽灵行）
///   j=0            : 南幽灵（接收南邻顶物理行）
///   j=1..local_ny  : 物理行
///   j=local_ny+1   : 北幽灵（接收北邻底物理行）
///
/// 使用 MPI_Sendrecv 避免死锁：奇数 rank 先发后收，偶数 rank 先收后发
/// 由 Solver::stream() 在 std::swap 之后、apply_BC() 之前自动调用。
// ---------------------------------------------------------------------------
void halo_exchange_d2q9(LatticeGrid& g, const MpiDecomp& decomp);
#endif

} // namespace lbm
