// core/src/lbm/mpi_decomp.cpp — MPI 域分解与幽灵行交换实现
// 仅在 LBM_ENABLE_MPI 宏已定义时编译（见 core/CMakeLists.txt）

#include "lbm/mpi_decomp.hpp"

#ifdef LBM_ENABLE_MPI
#include <mpi.h>
#include <algorithm>  // std::min

namespace lbm {

// ---------------------------------------------------------------------------
// 由全局网格尺寸创建一维（Y 方向）域分解描述符
// ---------------------------------------------------------------------------
MpiDecomp MpiDecomp::create(int gnx, int gny)
{
    MpiDecomp d;
    d.global_nx = gnx;
    d.global_ny = gny;

    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.nprocs);

    // 均匀分配行：前 (gny % nprocs) 个进程各多分配一行
    const int base = gny / d.nprocs;
    const int rem  = gny % d.nprocs;

    d.local_ny = base + (d.rank < rem ? 1 : 0);
    d.y_start  = d.rank * base + std::min(d.rank, rem);
    d.y_end    = d.y_start + d.local_ny - 1;

    // MPI_PROC_NULL（=-1）对 MPI_Sendrecv 是合法的"空邻居"，操作为 no-op
    d.rank_south = (d.rank > 0)             ? d.rank - 1 : MPI_PROC_NULL;
    d.rank_north = (d.rank < d.nprocs - 1)  ? d.rank + 1 : MPI_PROC_NULL;

    return d;
}

// ---------------------------------------------------------------------------
// D2Q9 幽灵行交换
//
// 布局（g.ny = local_ny + 2，nprocs > 1）：
//   j=0             : 南幽灵行（本进程南物理边界以南 1 格）
//   j=1..local_ny   : 物理行
//   j=local_ny+1    : 北幽灵行（本进程北物理边界以北 1 格）
//
// 交换规则：
//   1. 将本进程顶物理行（j=local_ny）发送给北邻，北邻接入其南幽灵（j=0）
//   2. 将本进程底物理行（j=1）    发送给南邻，南邻接入其北幽灵（j=local_ny+1）
//
// 单进程（nprocs==1）时此函数不应被调用（Solver 已做检查）。
// ---------------------------------------------------------------------------
void halo_exchange_d2q9(LatticeGrid& g, const MpiDecomp& decomp)
{
    // nprocs==1 时无需交换（应由调用方保证不进入此函数）
    if (decomp.nprocs == 1) return;

    const int nx       = decomp.global_nx;
    const int Q        = d2q9::Q;           // = 9
    const int row_size = nx * Q;            // 每行所有方向的 f 数据量（double 个数）
    const int lny      = decomp.local_ny;

    // 顶物理行：j=lny（g.ny = lny+2，所以 j=lny 是有效物理行）
    double* top_phys    = &g.f[static_cast<std::size_t>(g.idx(0, lny)) * Q];
    // 底物理行：j=1
    double* bot_phys    = &g.f[static_cast<std::size_t>(g.idx(0, 1))  * Q];
    // 南幽灵行：j=0（接收南邻顶行）
    double* south_ghost = &g.f[static_cast<std::size_t>(g.idx(0, 0))  * Q];
    // 北幽灵行：j=lny+1（接收北邻底行）
    double* north_ghost = &g.f[static_cast<std::size_t>(g.idx(0, lny + 1)) * Q];

    MPI_Status st;

    // 交换 1：向北邻发送顶物理行，从南邻接收入南幽灵行
    //   tag=0 标识"北发/南收"对（rank_north 以 tag=0 接收，rank_south 以 tag=0 接收）
    MPI_Sendrecv(
        top_phys,    row_size, MPI_DOUBLE, decomp.rank_north, /*tag=*/0,
        south_ghost, row_size, MPI_DOUBLE, decomp.rank_south, /*tag=*/0,
        MPI_COMM_WORLD, &st
    );

    // 交换 2：向南邻发送底物理行，从北邻接收入北幽灵行
    //   tag=1 标识"南发/北收"对
    MPI_Sendrecv(
        bot_phys,    row_size, MPI_DOUBLE, decomp.rank_south, /*tag=*/1,
        north_ghost, row_size, MPI_DOUBLE, decomp.rank_north, /*tag=*/1,
        MPI_COMM_WORLD, &st
    );
}

} // namespace lbm

#else
// MPI 未启用时提供空定义，防止链接器警告
namespace lbm {
MpiDecomp MpiDecomp::create(int gnx, int gny)
{
    MpiDecomp d;
    d.global_nx = gnx;
    d.global_ny = gny;
    d.local_ny  = gny;
    d.y_start   = 0;
    d.y_end     = gny - 1;
    return d;
}
} // namespace lbm
#endif
