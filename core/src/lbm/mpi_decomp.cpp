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
// halo_exchange_d3q19_3d：三维（XYZ 方向）幽灵层交换
//
// 本地网格布局（phys_x0/y0/z0 均为 0 或 1）：
//   k=0                  : 底幽灵层（pz_rank>0 时存在）
//   k=phys_z0..+lnz-1   : 物理层
//   k=phys_z0+lnz        : 顶幽灵层（pz_rank<pz-1 时存在）
//   j=0 / j=phys_y0+lny  : 南/北幽灵行（同 2D）
//   i=0 / i=phys_x0+lnx  : 西/东幽灵列（同 2D）
//
// 交换顺序：Z（底/顶）→ Y（南/北）→ X（西/东），均使用 MPI_Sendrecv。
// ---------------------------------------------------------------------------
void halo_exchange_d3q19_3d(LatticeGrid& g, const MpiDecomp3D& decomp)
{
    if (decomp.nprocs == 1) return;

    const int Q    = d3q19::Q;      // = 19
    const int gnx  = g.nx;          // 含幽灵层的本地 nx
    const int gny  = g.ny;          // 含幽灵层的本地 ny
    const int gnz  = g.nz;          // 含幽灵层的本地 nz
    const int lnx  = decomp.local_nx;
    const int lny  = decomp.local_ny;
    const int lnz  = decomp.local_nz;
    const int px0  = decomp.phys_x0();
    const int py0  = decomp.phys_y0();
    const int pz0  = decomp.phys_z0();

    const int slice_size = gnx * gny;          // 一个 k-层的节点数
    const int row_size   = gnx * Q;            // 一行（沿 x 方向）的 f 数据量
    const int slab_size  = slice_size * Q;     // 一个 k-层的 f 数据量

    MPI_Status st;

    // -----------------------------------------------------------------------
    // Z 方向（底/顶）：整个 XY 平面（一整层 slab）交换
    // -----------------------------------------------------------------------
    {
        // 顶物理层：k = pz0 + lnz - 1
        double* top_phys    = &g.f[static_cast<std::size_t>(g.idx(0, 0, pz0 + lnz - 1)) * Q];
        // 底物理层：k = pz0
        double* bot_phys    = &g.f[static_cast<std::size_t>(g.idx(0, 0, pz0))             * Q];
        // 底幽灵层：k = 0
        double* bottom_ghost = &g.f[static_cast<std::size_t>(g.idx(0, 0, 0))              * Q];
        // 顶幽灵层：k = pz0 + lnz
        double* top_ghost    = &g.f[static_cast<std::size_t>(g.idx(0, 0, pz0 + lnz))      * Q];

        // 向顶邻发送顶物理层，从底邻接收底幽灵层
        MPI_Sendrecv(
            top_phys,    slab_size, MPI_DOUBLE, decomp.rank_top,    30,
            bottom_ghost, slab_size, MPI_DOUBLE, decomp.rank_bottom, 30,
            MPI_COMM_WORLD, &st
        );
        // 向底邻发送底物理层，从顶邻接收顶幽灵层
        MPI_Sendrecv(
            bot_phys,  slab_size, MPI_DOUBLE, decomp.rank_bottom, 31,
            top_ghost, slab_size, MPI_DOUBLE, decomp.rank_top,    31,
            MPI_COMM_WORLD, &st
        );
    }

    // -----------------------------------------------------------------------
    // Y 方向（南/北）：整行（沿 x 方向的每层），逐 k-层发送
    // 为避免多次 Sendrecv，一次性打包整个 XZ 平面
    // -----------------------------------------------------------------------
    {
        const int xz_size = gnx * gnz * Q;    // XZ 平面（所有 k 层）的 f 数据量
        std::vector<double> send_north(xz_size), recv_north(xz_size);
        std::vector<double> send_south(xz_size), recv_south(xz_size);

        // 打包顶物理行（j = py0 + lny - 1），遍历所有 k 层
        for (int k = 0; k < gnz; ++k) {
            const double* src = &g.f[static_cast<std::size_t>(g.idx(0, py0 + lny - 1, k)) * Q];
            std::copy(src, src + row_size,
                      send_north.data() + static_cast<std::size_t>(k) * row_size);
        }
        // 打包底物理行（j = py0），遍历所有 k 层
        for (int k = 0; k < gnz; ++k) {
            const double* src = &g.f[static_cast<std::size_t>(g.idx(0, py0, k)) * Q];
            std::copy(src, src + row_size,
                      send_south.data() + static_cast<std::size_t>(k) * row_size);
        }

        // 向北邻发送顶物理行，从南邻接收南幽灵行
        MPI_Sendrecv(
            send_north.data(), xz_size, MPI_DOUBLE, decomp.rank_north, 32,
            recv_south.data(), xz_size, MPI_DOUBLE, decomp.rank_south, 32,
            MPI_COMM_WORLD, &st
        );
        // 向南邻发送底物理行，从北邻接收北幽灵行
        MPI_Sendrecv(
            send_south.data(), xz_size, MPI_DOUBLE, decomp.rank_south, 33,
            recv_north.data(), xz_size, MPI_DOUBLE, decomp.rank_north, 33,
            MPI_COMM_WORLD, &st
        );

        // 解包南幽灵行（j=0，仅 row_rank>0 时有意义）
        if (decomp.has_south_ghost()) {
            for (int k = 0; k < gnz; ++k) {
                double* dst = &g.f[static_cast<std::size_t>(g.idx(0, 0, k)) * Q];
                std::copy(recv_south.data() + static_cast<std::size_t>(k) * row_size,
                          recv_south.data() + static_cast<std::size_t>(k + 1) * row_size,
                          dst);
            }
        }
        // 解包北幽灵行（j = py0+lny，仅 row_rank<py-1 时有意义）
        if (decomp.has_north_ghost()) {
            for (int k = 0; k < gnz; ++k) {
                double* dst = &g.f[static_cast<std::size_t>(g.idx(0, py0 + lny, k)) * Q];
                std::copy(recv_north.data() + static_cast<std::size_t>(k) * row_size,
                          recv_north.data() + static_cast<std::size_t>(k + 1) * row_size,
                          dst);
            }
        }
    }

    // -----------------------------------------------------------------------
    // X 方向（西/东）：列数据（在行主序中不连续），需打包到连续缓冲区
    // -----------------------------------------------------------------------
    {
        const int yz_size = gny * gnz * Q;    // YZ 平面一列的 f 数据量
        std::vector<double> send_west(yz_size), recv_west(yz_size);
        std::vector<double> send_east(yz_size), recv_east(yz_size);

        // 打包最西物理列（i = px0），遍历所有 j,k
        for (int k = 0; k < gnz; ++k) {
            for (int j = 0; j < gny; ++j) {
                const double* src = &g.f[static_cast<std::size_t>(g.idx(px0, j, k)) * Q];
                const std::size_t off = (static_cast<std::size_t>(k) * gny + j) * Q;
                std::copy(src, src + Q, send_west.data() + off);
            }
        }
        // 打包最东物理列（i = px0 + lnx - 1）
        for (int k = 0; k < gnz; ++k) {
            for (int j = 0; j < gny; ++j) {
                const double* src = &g.f[static_cast<std::size_t>(g.idx(px0 + lnx - 1, j, k)) * Q];
                const std::size_t off = (static_cast<std::size_t>(k) * gny + j) * Q;
                std::copy(src, src + Q, send_east.data() + off);
            }
        }

        // 向西邻发送最西物理列，从东邻接收东幽灵列
        MPI_Sendrecv(
            send_west.data(), yz_size, MPI_DOUBLE, decomp.rank_west, 34,
            recv_east.data(), yz_size, MPI_DOUBLE, decomp.rank_east, 34,
            MPI_COMM_WORLD, &st
        );
        // 向东邻发送最东物理列，从西邻接收西幽灵列
        MPI_Sendrecv(
            send_east.data(), yz_size, MPI_DOUBLE, decomp.rank_east, 35,
            recv_west.data(), yz_size, MPI_DOUBLE, decomp.rank_west, 35,
            MPI_COMM_WORLD, &st
        );

        // 解包西幽灵列（i=0，仅 col_rank>0 时有意义）
        if (decomp.has_west_ghost()) {
            for (int k = 0; k < gnz; ++k) {
                for (int j = 0; j < gny; ++j) {
                    double* dst = &g.f[static_cast<std::size_t>(g.idx(0, j, k)) * Q];
                    const std::size_t off = (static_cast<std::size_t>(k) * gny + j) * Q;
                    std::copy(recv_west.data() + off, recv_west.data() + off + Q, dst);
                }
            }
        }
        // 解包东幽灵列（i = px0+lnx，仅 col_rank<px-1 时有意义）
        if (decomp.has_east_ghost()) {
            for (int k = 0; k < gnz; ++k) {
                for (int j = 0; j < gny; ++j) {
                    double* dst = &g.f[static_cast<std::size_t>(g.idx(px0 + lnx, j, k)) * Q];
                    const std::size_t off = (static_cast<std::size_t>(k) * gny + j) * Q;
                    std::copy(recv_east.data() + off, recv_east.data() + off + Q, dst);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MpiDecomp3D::create — 三维域分解描述符构造
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
