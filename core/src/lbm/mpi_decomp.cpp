// core/src/lbm/mpi_decomp.cpp — MPI 域分解与幽灵层交换实现
// 包含一维（Y 方向）域分解（MpiDecomp）和二维（XY 方向）块分解（MpiDecomp2D）。
// 仅在 LBM_ENABLE_MPI 宏已定义时编译（见 core/CMakeLists.txt）。

#include "lbm/mpi_decomp.hpp"

#ifdef LBM_ENABLE_MPI
#include <mpi.h>
#include <algorithm>  // std::min
#include <stdexcept>  // std::invalid_argument
#include <vector>

namespace lbm {

// ===========================================================================
// 一维（Y 方向）域分解
// ===========================================================================

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
// 一维模式：D2Q9 幽灵行交换
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
    MPI_Sendrecv(
        top_phys,    row_size, MPI_DOUBLE, decomp.rank_north, /*tag=*/0,
        south_ghost, row_size, MPI_DOUBLE, decomp.rank_south, /*tag=*/0,
        MPI_COMM_WORLD, &st
    );

    // 交换 2：向南邻发送底物理行，从北邻接收入北幽灵行
    MPI_Sendrecv(
        bot_phys,    row_size, MPI_DOUBLE, decomp.rank_south, /*tag=*/1,
        north_ghost, row_size, MPI_DOUBLE, decomp.rank_north, /*tag=*/1,
        MPI_COMM_WORLD, &st
    );
}

// ===========================================================================
// 二维（XY 方向）块分解
// ===========================================================================

// ---------------------------------------------------------------------------
// 辅助函数：计算 rank r 在均匀分布 total 个条目（nprocs 个进程）时的本地数量和起始索引
//
// 分配策略：前 (total % nprocs) 个进程各多分配一个条目（处理不整除情况）
//   local_n = total/nprocs + (r < total%nprocs ? 1 : 0)
//   start   = r * (total/nprocs) + min(r, total%nprocs)
//
// 参数：
//   total    — 需要分配的条目总数（全局节点数 nx 或 ny）
//   nprocs   — 参与分配的进程总数
//   r        — 当前进程的坐标（0..nprocs-1），即 col_rank 或 row_rank
//   local_n  — 输出：本进程分到的本地条目数
//   start    — 输出：本进程在全局坐标中的起始索引
// ---------------------------------------------------------------------------
static void uniform_partition(int total, int nprocs, int r, int& local_n, int& start)
{
    const int base = total / nprocs;
    const int rem  = total % nprocs;
    local_n = base + (r < rem ? 1 : 0);
    start   = r * base + std::min(r, rem);
}

// ---------------------------------------------------------------------------
// 由全局尺寸和块数创建二维域分解描述符
//
// 进程排布（row_rank = rank / px，col_rank = rank % px）：
//   全局网格被均匀切分为 px × py 块
//   px * py == nprocs（进程总数），否则抛出 std::invalid_argument
// ---------------------------------------------------------------------------
MpiDecomp2D MpiDecomp2D::create(int gnx, int gny, int in_px, int in_py)
{
    MpiDecomp2D d;
    d.global_nx = gnx;
    d.global_ny = gny;
    d.px        = in_px;
    d.py        = in_py;

    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.nprocs);

    if (d.px * d.py != d.nprocs) {
        throw std::invalid_argument(
            "MpiDecomp2D: px*py must equal MPI process count");
    }

    d.col_rank = d.rank % d.px;   // X 方向进程坐标
    d.row_rank = d.rank / d.px;   // Y 方向进程坐标

    // 各方向均匀分配节点数
    uniform_partition(gnx, d.px, d.col_rank, d.local_nx, d.x_start);
    uniform_partition(gny, d.py, d.row_rank, d.local_ny, d.y_start);
    d.x_end = d.x_start + d.local_nx - 1;
    d.y_end = d.y_start + d.local_ny - 1;

    // 计算四邻进程 rank（MPI_PROC_NULL 对 MPI_Sendrecv 是合法的"空邻居"）
    d.rank_south = (d.row_rank > 0)          ? (d.row_rank - 1) * d.px + d.col_rank : MPI_PROC_NULL;
    d.rank_north = (d.row_rank < d.py - 1)   ? (d.row_rank + 1) * d.px + d.col_rank : MPI_PROC_NULL;
    d.rank_west  = (d.col_rank > 0)          ? d.row_rank * d.px + d.col_rank - 1 : MPI_PROC_NULL;
    d.rank_east  = (d.col_rank < d.px - 1)   ? d.row_rank * d.px + d.col_rank + 1 : MPI_PROC_NULL;

    return d;
}

// ---------------------------------------------------------------------------
// 二维模式：D2Q9 幽灵层交换
//
// 本地网格布局（phys_x0 = 0 or 1，phys_y0 = 0 or 1）：
//   j=0              : 南幽灵行（row_rank>0 时存在）
//   j=phys_y0..+lny-1: 物理行
//   j=phys_y0+lny    : 北幽灵行（row_rank<py-1 时存在）
//   i=0              : 西幽灵列（col_rank>0 时存在）
//   i=phys_x0..+lnx-1: 物理列
//   i=phys_x0+lnx    : 东幽灵列（col_rank<px-1 时存在）
//
// 交换顺序（避免死锁）：先南北后东西，均使用 MPI_Sendrecv。
// ---------------------------------------------------------------------------
void halo_exchange_d2q9_2d(LatticeGrid& g, const MpiDecomp2D& decomp)
{
    if (decomp.nprocs == 1) return;

    const int Q    = d2q9::Q;      // = 9
    const int gnx  = g.nx;         // 含幽灵层的本地 nx
    const int lnx  = decomp.local_nx;
    const int lny  = decomp.local_ny;
    const int px0  = decomp.phys_x0();   // 物理区域在 x 方向的偏移
    const int py0  = decomp.phys_y0();   // 物理区域在 y 方向的偏移

    const int row_size = gnx * Q;   // 一整行（含幽灵列）所有节点的 f 数据量

    MPI_Status st;

    // -----------------------------------------------------------------------
    // 南北方向交换（整行，row_size 个 double）
    // -----------------------------------------------------------------------
    {
        // 顶物理行 j = py0 + lny - 1
        double* top_phys    = &g.f[static_cast<std::size_t>(g.idx(0, py0 + lny - 1)) * Q];
        // 底物理行 j = py0
        double* bot_phys    = &g.f[static_cast<std::size_t>(g.idx(0, py0))            * Q];
        // 南幽灵行 j = 0（仅 row_rank>0 时存在，否则 Sendrecv 到 MPI_PROC_NULL 为 no-op）
        double* south_ghost = &g.f[static_cast<std::size_t>(g.idx(0, 0))              * Q];
        // 北幽灵行 j = py0 + lny（仅 row_rank<py-1 时存在）
        double* north_ghost = &g.f[static_cast<std::size_t>(g.idx(0, py0 + lny))      * Q];

        // 向北邻发送顶物理行，从南邻接收南幽灵行
        MPI_Sendrecv(
            top_phys,    row_size, MPI_DOUBLE, decomp.rank_north, 10,
            south_ghost, row_size, MPI_DOUBLE, decomp.rank_south, 10,
            MPI_COMM_WORLD, &st
        );
        // 向南邻发送底物理行，从北邻接收北幽灵行
        MPI_Sendrecv(
            bot_phys,    row_size, MPI_DOUBLE, decomp.rank_south, 11,
            north_ghost, row_size, MPI_DOUBLE, decomp.rank_north, 11,
            MPI_COMM_WORLD, &st
        );
    }

    // -----------------------------------------------------------------------
    // 东西方向交换（需将列数据打包到连续缓冲区，因列在行主序中不连续）
    // -----------------------------------------------------------------------
    {
        // 缓冲区大小：一列 g.ny 行 × Q 方向
        const int gny_all = g.ny;
        const int col_size = gny_all * Q;
        std::vector<double> send_west(col_size), recv_west(col_size);
        std::vector<double> send_east(col_size), recv_east(col_size);

        // 打包最西物理列（i=px0）到 send_west
        for (int j = 0; j < gny_all; ++j) {
            const double* src = &g.f[static_cast<std::size_t>(g.idx(px0, j)) * Q];
            for (int a = 0; a < Q; ++a) {
                send_west[j * Q + a] = src[a];
            }
        }
        // 打包最东物理列（i=px0+lnx-1）到 send_east
        for (int j = 0; j < gny_all; ++j) {
            const double* src = &g.f[static_cast<std::size_t>(g.idx(px0 + lnx - 1, j)) * Q];
            for (int a = 0; a < Q; ++a) {
                send_east[j * Q + a] = src[a];
            }
        }

        // 向西邻发送最西物理列，从东邻接收东幽灵列
        MPI_Sendrecv(
            send_west.data(), col_size, MPI_DOUBLE, decomp.rank_west, 20,
            recv_east.data(), col_size, MPI_DOUBLE, decomp.rank_east, 20,
            MPI_COMM_WORLD, &st
        );
        // 向东邻发送最东物理列，从西邻接收西幽灵列
        MPI_Sendrecv(
            send_east.data(), col_size, MPI_DOUBLE, decomp.rank_east, 21,
            recv_west.data(), col_size, MPI_DOUBLE, decomp.rank_west, 21,
            MPI_COMM_WORLD, &st
        );

        // 解包西幽灵列（i=0，仅 col_rank>0 时有意义）
        if (decomp.has_west_ghost()) {
            for (int j = 0; j < gny_all; ++j) {
                double* dst = &g.f[static_cast<std::size_t>(g.idx(0, j)) * Q];
                for (int a = 0; a < Q; ++a) {
                    dst[a] = recv_west[j * Q + a];
                }
            }
        }
        // 解包东幽灵列（i=px0+lnx，仅 col_rank<px-1 时有意义）
        if (decomp.has_east_ghost()) {
            for (int j = 0; j < gny_all; ++j) {
                double* dst = &g.f[static_cast<std::size_t>(g.idx(px0 + lnx, j)) * Q];
                for (int a = 0; a < Q; ++a) {
                    dst[a] = recv_east[j * Q + a];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MpiDecomp3D（MPI 启用时的实现）
//
// 注意：幽灵层交换函数（halo_exchange_d3q19_3d 等）尚未实现。
// 本结构用于预留三维并行框架，支持 D3Q19/D3Q27 格子未来扩展。
// ---------------------------------------------------------------------------
MpiDecomp3D MpiDecomp3D::create(int gnx, int gny, int gnz,
                                  int in_px, int in_py, int in_pz)
{
    MpiDecomp3D d;
    d.global_nx = gnx;
    d.global_ny = gny;
    d.global_nz = gnz;
    d.px = in_px; d.py = in_py; d.pz = in_pz;

    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.nprocs);

    if (d.px * d.py * d.pz != d.nprocs) {
        throw std::invalid_argument(
            "MpiDecomp3D: px*py*pz must equal MPI process count");
    }

    d.col_rank = d.rank % d.px;
    d.row_rank = (d.rank / d.px) % d.py;
    d.pz_rank  = d.rank / (d.px * d.py);

    // 均匀分配三个方向的节点数
    {
        const int bx = gnx / d.px, rx = gnx % d.px;
        d.local_nx = bx + (d.col_rank < rx ? 1 : 0);
        d.x_start  = d.col_rank * bx + std::min(d.col_rank, rx);
        d.x_end    = d.x_start + d.local_nx - 1;
    }
    {
        const int by = gny / d.py, ry = gny % d.py;
        d.local_ny = by + (d.row_rank < ry ? 1 : 0);
        d.y_start  = d.row_rank * by + std::min(d.row_rank, ry);
        d.y_end    = d.y_start + d.local_ny - 1;
    }
    {
        const int bz = gnz / d.pz, rz = gnz % d.pz;
        d.local_nz = bz + (d.pz_rank < rz ? 1 : 0);
        d.z_start  = d.pz_rank * bz + std::min(d.pz_rank, rz);
        d.z_end    = d.z_start + d.local_nz - 1;
    }

    // 六邻进程 rank
    d.rank_west   = (d.col_rank > 0)        ? d.pz_rank*(d.px*d.py)+d.row_rank*d.px+d.col_rank-1 : MPI_PROC_NULL;
    d.rank_east   = (d.col_rank < d.px-1)   ? d.pz_rank*(d.px*d.py)+d.row_rank*d.px+d.col_rank+1 : MPI_PROC_NULL;
    d.rank_south  = (d.row_rank > 0)        ? d.pz_rank*(d.px*d.py)+(d.row_rank-1)*d.px+d.col_rank : MPI_PROC_NULL;
    d.rank_north  = (d.row_rank < d.py-1)   ? d.pz_rank*(d.px*d.py)+(d.row_rank+1)*d.px+d.col_rank : MPI_PROC_NULL;
    d.rank_bottom = (d.pz_rank > 0)         ? (d.pz_rank-1)*(d.px*d.py)+d.row_rank*d.px+d.col_rank : MPI_PROC_NULL;
    d.rank_top    = (d.pz_rank < d.pz-1)    ? (d.pz_rank+1)*(d.px*d.py)+d.row_rank*d.px+d.col_rank : MPI_PROC_NULL;

    return d;
}

} // namespace lbm

#else
// MPI 未启用时提供空定义，防止链接器警告
#include <stdexcept>

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

MpiDecomp2D MpiDecomp2D::create(int gnx, int gny, int in_px, int in_py)
{
    MpiDecomp2D d;
    d.global_nx = gnx;
    d.global_ny = gny;
    d.px        = in_px;
    d.py        = in_py;
    d.local_nx  = gnx;
    d.local_ny  = gny;
    d.x_start   = 0;
    d.x_end     = gnx - 1;
    d.y_start   = 0;
    d.y_end     = gny - 1;
    if (in_px * in_py != 1) {
        throw std::invalid_argument("MpiDecomp2D: MPI not enabled, px*py must be 1");
    }
    return d;
}

// ---------------------------------------------------------------------------
// MpiDecomp3D（无 MPI 构建：px*py*pz 必须为 1）
// ---------------------------------------------------------------------------
MpiDecomp3D MpiDecomp3D::create(int gnx, int gny, int gnz,
                                  int in_px, int in_py, int in_pz)
{
    MpiDecomp3D d;
    d.global_nx = gnx;
    d.global_ny = gny;
    d.global_nz = gnz;
    d.px = in_px; d.py = in_py; d.pz = in_pz;
    d.local_nx  = gnx;
    d.local_ny  = gny;
    d.local_nz  = gnz;
    d.x_start = 0; d.x_end = gnx - 1;
    d.y_start = 0; d.y_end = gny - 1;
    d.z_start = 0; d.z_end = gnz - 1;
    if (in_px * in_py * in_pz != 1) {
        throw std::invalid_argument("MpiDecomp3D: MPI not enabled, px*py*pz must be 1");
    }
    return d;
}

} // namespace lbm
#endif
