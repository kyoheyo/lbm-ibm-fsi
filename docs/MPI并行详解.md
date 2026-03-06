# LBM-IBM-FSI MPI 并行详解

本文档详细说明项目在 MPI（及 MPI + OpenMP 混合）模式下的完整求解流程，包括：域分解原理与两种方式（**一维 Y 切片**与**二维 XY 块分解**）、**多网格独立模式**、**OpenMP 线程数便捷设置**、幽灵层交换实现、边界条件约定、Rust / C API 使用示例以及跨平台构建和运行指南。

---

## 目录

1. [整体架构与并行模式选择](#1-整体架构与并行模式选择)
2. [TOML 配置快速参考](#2-toml-配置快速参考)
   - 2.1 OpenMP 线程数设置
   - 2.2 一维 Y 切片模式
   - 2.3 二维 XY 块分解模式
   - 2.4 多网格独立模式
3. [模式一：一维 Y 方向切片（1D_Y）](#3-模式一一维-y-方向切片1d_y)
   - 3.1 域分解原理
   - 3.2 幽灵行布局与 Halo Exchange
   - 3.3 边界条件约定
   - 3.4 时序图
   - 3.5 C++ 实现：`MpiDecomp::create()`
   - 3.6 幽灵行交换：`halo_exchange_d2q9()`
4. [模式二：二维 XY 块分解（2D_XY）](#4-模式二二维-xy-块分解2d_xy)
   - 4.1 域分解原理
   - 4.2 进程布局与坐标
   - 4.3 幽灵层布局与 Halo Exchange
   - 4.4 边界条件约定
   - 4.5 C++ 实现：`MpiDecomp2D::create()`
   - 4.6 幽灵层交换：`halo_exchange_d2q9_2d()`
5. [模式三：多网格独立模式（Multi-Grid）](#5-模式三多网格独立模式multi-grid)
   - 5.1 使用场景
   - 5.2 输出目录隔离
6. [MPI + OpenMP 混合并行](#6-mpi--openmp-混合并行)
   - 6.1 线程/进程分配策略
   - 6.2 TOML 配置示例
7. [Rust / C API 使用示例](#7-rust--c-api-使用示例)
   - 7.1 一维切片示例（Rust）
   - 7.2 二维块分解示例（Rust）
   - 7.3 多网格独立模式示例（Rust）
   - 7.4 C ABI 函数速查表
8. [构建与运行](#8-构建与运行)
   - 8.1 编译开关
   - 8.2 跨平台构建命令
   - 8.3 运行命令
   - 8.4 SLURM 集群脚本示例
9. [正确性保证与常见陷阱](#9-正确性保证与常见陷阱)
   - 9.1 不要对幽灵层执行二次碰撞
   - 9.2 边界条件只在拥有该物理壁的 rank 注册
   - 9.3 px × py 必须等于 MPI 进程总数
10. [性能分析与调优建议](#10-性能分析与调优建议)

---

## 1. 整体架构与并行模式选择

本项目支持以下四种并行模式，均通过 **编译时环境变量** 和 **TOML 配置文件** 控制，无需修改算法代码：

| 模式 | 编译开关 | TOML `[mpi].mode` | 适用场景 |
|------|---------|-------------------|---------|
| 单进程 OpenMP | `LBM_ENABLE_OPENMP=ON` | — | 单节点多核工作站，共享内存并行 |
| MPI 一维 Y 切片 | `LBM_ENABLE_MPI=ON` | `"1d_y"`（默认）| 集群多节点，网格 ny 远大于 nx |
| MPI 二维 XY 块 | `LBM_ENABLE_MPI=ON` | `"2d_xy"` | 集群多节点，网格接近方形或 nx 较大 |
| MPI 多网格独立 | `LBM_ENABLE_MPI=ON` | `"multi_grid"` | 参数扫描、集成测试、无需通信的批处理 |
| MPI + OpenMP | 两者均 ON | 任意 | HPC 集群，每节点多卡或多核 |

**选择建议：**
- 网格为 256×1024 → 选 `"1d_y"`（Y 方向更长，切片更均匀）
- 网格为 1024×1024 → 选 `"2d_xy"`（接近方形，2D 切分通信面积更小）
- 需同时扫描 Re = 100/500/1000/5000 → 选 `"multi_grid"`（4 进程完全独立）

---

## 2. TOML 配置快速参考

### 2.1 OpenMP 线程数设置

```toml
[parallel]
omp_num_threads = 8   # 强制使用 8 个 OpenMP 线程（0 = 使用系统默认值）
```

`omp_num_threads` 通过 `omp_set_num_threads()` 在进程内即时生效，**优先级高于** `OMP_NUM_THREADS` 环境变量。设为 `0`（默认）则由 OpenMP 运行时自动选择（通常为全部物理核心数）。

### 2.2 一维 Y 切片模式

```toml
[mpi]
mode = "1d_y"   # 可省略，此为默认值
```

运行：`mpirun -n 4 ./lbm-orchestrator configs/my.toml`（4 个进程各分配 ny/4 行）

### 2.3 二维 XY 块分解模式

```toml
[mpi]
mode      = "2d_xy"
nx_blocks = 4     # X 方向切 4 块（每块 nx/4 列）
ny_blocks = 2     # Y 方向切 2 块（每块 ny/2 行）
# nx_blocks * ny_blocks == 8 → 必须 mpirun -n 8
```

运行：`mpirun -n 8 ./lbm-orchestrator configs/my.toml`（须满足 `nx_blocks * ny_blocks == 8`）

### 2.4 多网格独立模式

```toml
[mpi]
mode = "multi_grid"
# 每个 rank 各自运行完整的 nx × ny 网格，互不通信
# 输出写入 <output.directory>/rank_<N>/
```

运行：`mpirun -n 4 ./lbm-orchestrator configs/my.toml`（4 套独立仿真并行执行）

---

## 3. 模式一：一维 Y 方向切片（1D_Y）

### 3.1 域分解原理

将 `ny` 行均匀切分到 `nprocs` 个进程，前 `ny % nprocs` 个进程各多分配一行（处理 ny 不整除的情况）：

```
全局网格 nx × ny:

   ny-1  ┌────────────────────────┐
         │   rank nprocs-1        │  ← 持有全局北壁（注册 Face::North BC）
         │   y_start_r..ny-1      │
         ├────────────────────────┤
         │        ...             │  ← 内部 rank（不注册 South/North BC）
         ├────────────────────────┤
         │   rank 1               │
         ├────────────────────────┤
     0   │   rank 0               │  ← 持有全局南壁（注册 Face::South BC）
         └────────────────────────┘
                  nx
```

**负载均衡公式**（`core/src/lbm/mpi_decomp.cpp`，`MpiDecomp::create()`）：

```cpp
const int base = gny / nprocs;
const int rem  = gny % nprocs;
local_ny = base + (rank < rem ? 1 : 0);
y_start  = rank * base + std::min(rank, rem);
y_end    = y_start + local_ny - 1;
```

### 3.2 幽灵行布局与 Halo Exchange

每个进程分配 `grid_ny = local_ny + 2` 行的本地网格（两侧各一行幽灵行）：

```
   ny_local+1  ┌────────────────┐  ← 北幽灵行 j=local_ny+1（接收 rank+1 的 j=1 行）
               │ j=local_ny     │  物理行（对应全局 y_end）
               │ ...            │
               │ j=1            │  物理行（对应全局 y_start）
   0           └────────────────┘  ← 南幽灵行 j=0（接收 rank-1 的 j=local_ny 行）
               ←    nx      →
```

**幽灵行交换实现**（`core/src/lbm/mpi_decomp.cpp`，`halo_exchange_d2q9()`）：

```cpp
// 交换 1：向北邻发送本进程顶物理行，从南邻接收南幽灵行
MPI_Sendrecv(
    top_phys,    nx*Q, MPI_DOUBLE, rank_north, 0,
    south_ghost, nx*Q, MPI_DOUBLE, rank_south, 0,
    MPI_COMM_WORLD, &st
);

// 交换 2：向南邻发送本进程底物理行，从北邻接收北幽灵行
MPI_Sendrecv(
    bot_phys,    nx*Q, MPI_DOUBLE, rank_south, 1,
    north_ghost, nx*Q, MPI_DOUBLE, rank_north, 1,
    MPI_COMM_WORLD, &st
);
```

使用 `MPI_Sendrecv` 而非 `MPI_Send` + `MPI_Recv` 避免死锁（所有进程同时收发）。`MPI_PROC_NULL`（由 `rank_south=-1` / `rank_north=-1` 传入）在最边缘 rank 上使 Sendrecv 退化为无操作，无需额外 `if` 分支。

### 3.3 边界条件约定

| 方向 | 注册 BC 的条件 | 说明 |
|------|---------------|------|
| South | `rank == 0` | 只有 rank 0 拥有全局南壁（j=0 是物理壁而非幽灵行） |
| North | `rank == nprocs-1` | 只有最后一个 rank 拥有全局北壁 |
| West | 所有 rank | 每个 rank 的本地网格均涵盖全部 nx 列，西壁存在于所有 rank |
| East | 所有 rank | 同 West |

**重要**：内部 rank（rank ∈ [1, nprocs-2]）不注册 South/North BC；其幽灵行由 `halo_exchange_d2q9()` 维护，边界层的 f 值由相邻 rank 的物理层通过 halo 交换填入，无需额外 BC 处理。

### 3.4 时序图

每个 `Solver::step()` 内部的执行顺序：

```
collide()                ← BGK/MRT 碰撞（本地，无通信，跳过幽灵行）
  ↓
stream()                 ← 流式迁移（本地，周期取模；ghost 行被"污染"）
  ↓
halo_exchange_d2q9()     ← MPI_Sendrecv：填充正确的 ghost 行（南北方向）
  │  rank r → rank r+1: j=local_ny（顶物理行）→ 对方 j=0（南幽灵）
  │  rank r → rank r-1: j=1（底物理行）       → 对方 j=local_ny+1（北幽灵）
  ↓
apply_BC()               ← rank 0 执行 South BC，最后 rank 执行 North BC
                           所有 rank 执行 West/East BC
  ↓
compute_macroscopic()    ← 更新 ρ/u（本地，含幽灵节点）
```

### 3.5 C++ 实现：`MpiDecomp::create()`

**源文件**：`core/src/lbm/mpi_decomp.cpp`，`core/include/lbm/mpi_decomp.hpp`

```cpp
// 创建一维 Y 方向域分解描述符
MpiDecomp MpiDecomp::create(int gnx, int gny)
{
    MpiDecomp d;
    d.global_nx = gnx;
    d.global_ny = gny;

    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.nprocs);

    const int base = gny / d.nprocs;
    const int rem  = gny % d.nprocs;
    d.local_ny = base + (d.rank < rem ? 1 : 0);
    d.y_start  = d.rank * base + std::min(d.rank, rem);
    d.y_end    = d.y_start + d.local_ny - 1;

    d.rank_south = (d.rank > 0)            ? d.rank - 1 : MPI_PROC_NULL;
    d.rank_north = (d.rank < d.nprocs - 1) ? d.rank + 1 : MPI_PROC_NULL;
    return d;
}

// 本地网格含幽灵行的 ny（供 LatticeGrid 构造使用）
int grid_ny() const { return (nprocs > 1) ? local_ny + 2 : local_ny; }
```

### 3.6 幽灵行不执行二次碰撞

**源文件**：`core/src/lbm/solver.cpp`，`Solver::collide_bgk()`

幽灵行持有的是相邻进程已完成碰撞的物理行数据。若再次对其碰撞会引入 "二次碰撞" 错误（过度松弛）。通过 `n_start` / `n_end` 跳过幽灵行：

```cpp
int n_start = 0, n_end = n;
#ifdef LBM_ENABLE_MPI
if (mpi_decomp_ && mpi_decomp_->nprocs > 1) {
    if (!mpi_decomp_->has_south_wall()) n_start = grid_.nx;  // 跳过南幽灵行
    if (!mpi_decomp_->has_north_wall()) n_end   = n - grid_.nx; // 跳过北幽灵行
}
#endif

for (int i = n_start; i < n_end; ++i) {
    // BGK 碰撞
}
```

---

## 4. 模式二：二维 XY 块分解（2D_XY）

### 4.1 域分解原理

将全局网格 `global_nx × global_ny` 切分为 `px × py` 个矩形块，总进程数 `nprocs = px × py`：

```
进程布局（rank = row_rank * px + col_rank）：

   row=py-1  ┌───────┬───────┬───────┐  ← 全局北壁（row_rank==py-1 注册 North BC）
             │ r=6   │ r=7   │ r=8   │    （nx_blocks=3, ny_blocks=3 示例）
   row=1     ├───────┼───────┼───────┤
             │ r=3   │ r=4   │ r=5   │  ← 内部行（不注册 South/North BC）
   row=0     └───────┴───────┴───────┘  ← 全局南壁（row_rank==0 注册 South BC）
             col=0   col=1   col=2
             ↑                       ↑
          西壁BC                   东壁BC
```

### 4.2 进程布局与坐标

| 字段 | 含义 | 计算公式 |
|------|------|---------|
| `col_rank` | X 方向进程坐标 | `rank % px` |
| `row_rank` | Y 方向进程坐标 | `rank / px` |
| `local_nx` | X 方向物理列数 | 均匀分配：`base_x + (col_rank < rem_x ? 1 : 0)` |
| `local_ny` | Y 方向物理行数 | 均匀分配：`base_y + (row_rank < rem_y ? 1 : 0)` |
| `x_start`  | 全局 X 起始坐标 | `col_rank * base_x + min(col_rank, rem_x)` |
| `y_start`  | 全局 Y 起始坐标 | `row_rank * base_y + min(row_rank, rem_y)` |

四邻 rank（用于 MPI_Sendrecv）：

```cpp
rank_south = (row_rank > 0)        ? (row_rank-1)*px + col_rank : MPI_PROC_NULL;
rank_north = (row_rank < py-1)     ? (row_rank+1)*px + col_rank : MPI_PROC_NULL;
rank_west  = (col_rank > 0)        ? row_rank*px + col_rank - 1 : MPI_PROC_NULL;
rank_east  = (col_rank < px-1)     ? row_rank*px + col_rank + 1 : MPI_PROC_NULL;
```

### 4.3 幽灵层布局

每个进程分配 `grid_nx() × grid_ny()` 的本地网格，含最多 4 个幽灵层（各 1 格）：

```
含幽灵层的本地网格（col_rank=1, row_rank=1，即四面均有邻居）：

  ┌───────────────────────────────────┐
  │ 北幽灵列（j=phys_y0+local_ny）   │  ← 来自 rank_north 的 j=phys_y0 行
  ├──┬─────────────────────────────┬──┤
  │西│ 物理区域                    │东│
  │幽│ i=phys_x0..phys_x0+local_nx│幽│
  │灵│ j=phys_y0..phys_y0+local_ny│灵│
  │列│                             │列│
  ├──┴─────────────────────────────┴──┤
  │ 南幽灵行（j=0）                   │  ← 来自 rank_south 的 j=phys_y0+local_ny-1 行
  └───────────────────────────────────┘
     ↑                                ↑
   i=0（西幽灵）          i=phys_x0+local_nx（东幽灵）
```

其中：
- `phys_x0 = has_west_ghost()  ? 1 : 0`（西幽灵列存在时物理区域从 i=1 开始）
- `phys_y0 = has_south_ghost() ? 1 : 0`（南幽灵行存在时物理区域从 j=1 开始）

### 4.4 边界条件约定

| 方向 | 注册 BC 的条件 | 检查方法 |
|------|---------------|---------|
| South | `row_rank == 0` | `decomp.has_south_wall()` |
| North | `row_rank == py-1` | `decomp.has_north_wall()` |
| West | `col_rank == 0` | `decomp.has_west_wall()` |
| East | `col_rank == px-1` | `decomp.has_east_wall()` |

### 4.5 C++ 实现：`MpiDecomp2D::create()`

**源文件**：`core/src/lbm/mpi_decomp.cpp`，`core/include/lbm/mpi_decomp.hpp`

```cpp
MpiDecomp2D MpiDecomp2D::create(int gnx, int gny, int in_px, int in_py)
{
    MpiDecomp2D d;
    d.global_nx = gnx; d.global_ny = gny;
    d.px = in_px;      d.py = in_py;

    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.nprocs);

    if (d.px * d.py != d.nprocs)
        throw std::invalid_argument("MpiDecomp2D: px*py must equal MPI process count");

    d.col_rank = d.rank % d.px;
    d.row_rank = d.rank / d.px;

    // 均匀分配 X 和 Y 方向节点数
    uniform_partition(gnx, d.px, d.col_rank, d.local_nx, d.x_start);
    uniform_partition(gny, d.py, d.row_rank, d.local_ny, d.y_start);

    // 计算四邻 rank（MPI_PROC_NULL 对 MPI_Sendrecv 是合法的"空邻居"）
    d.rank_south = (d.row_rank > 0)        ? (d.row_rank-1)*d.px + d.col_rank : MPI_PROC_NULL;
    d.rank_north = (d.row_rank < d.py-1)   ? (d.row_rank+1)*d.px + d.col_rank : MPI_PROC_NULL;
    d.rank_west  = (d.col_rank > 0)        ? d.row_rank*d.px + d.col_rank - 1 : MPI_PROC_NULL;
    d.rank_east  = (d.col_rank < d.px-1)   ? d.row_rank*d.px + d.col_rank + 1 : MPI_PROC_NULL;
    return d;
}
```

### 4.6 幽灵层交换：`halo_exchange_d2q9_2d()`

**源文件**：`core/src/lbm/mpi_decomp.cpp`

交换分两步：先交换**南北整行**（数据连续，无需打包），再交换**东西整列**（需先打包到临时 `vector<double>`，因列在行主序中不连续）：

```cpp
void halo_exchange_d2q9_2d(LatticeGrid& g, const MpiDecomp2D& decomp)
{
    const int Q = d2q9::Q;
    const int row_size = g.nx * Q;  // 一整行（含幽灵列）的 f 数据量

    // --- 步骤 1：南北方向（行数据连续，直接 Sendrecv）---
    MPI_Sendrecv(top_phys,    row_size, MPI_DOUBLE, decomp.rank_north, 10,
                 south_ghost, row_size, MPI_DOUBLE, decomp.rank_south, 10,
                 MPI_COMM_WORLD, &st);
    MPI_Sendrecv(bot_phys,    row_size, MPI_DOUBLE, decomp.rank_south, 11,
                 north_ghost, row_size, MPI_DOUBLE, decomp.rank_north, 11,
                 MPI_COMM_WORLD, &st);

    // --- 步骤 2：东西方向（列数据不连续，需先打包）---
    // 打包最西/东物理列到 send_west / send_east
    for (int j = 0; j < g.ny; ++j) {
        for (int a = 0; a < Q; ++a) {
            send_west[j*Q+a] = g.f[g.idx(px0, j) * Q + a];
            send_east[j*Q+a] = g.f[g.idx(px0+local_nx-1, j) * Q + a];
        }
    }
    // 向西邻发最西列，从东邻收东幽灵列
    MPI_Sendrecv(send_west.data(), col_size, MPI_DOUBLE, decomp.rank_west, 20,
                 recv_east.data(), col_size, MPI_DOUBLE, decomp.rank_east, 20,
                 MPI_COMM_WORLD, &st);
    // 向东邻发最东列，从西邻收西幽灵列
    MPI_Sendrecv(send_east.data(), col_size, MPI_DOUBLE, decomp.rank_east, 21,
                 recv_west.data(), col_size, MPI_DOUBLE, decomp.rank_west, 21,
                 MPI_COMM_WORLD, &st);
    // 解包到幽灵列
    if (decomp.has_west_ghost()) { /* 解包 recv_west → i=0 列 */ }
    if (decomp.has_east_ghost()) { /* 解包 recv_east → i=px0+local_nx 列 */ }
}
```

**幽灵节点不执行二次碰撞**（`solver.cpp`，`Solver::collide_bgk()`）：

```cpp
const bool use_mpi2d = (mpi_decomp2d_ && mpi_decomp2d_->nprocs > 1);
// ...
for (int i = n_start; i < n_end; ++i) {
    if (use_mpi2d) {
        const int ix = i % gnx2d, iy = i / gnx2d;
        if ((sg2d && iy==0) || (ng2d && iy==gny2d-1)) continue;  // 跳过南/北幽灵行
        if ((wg2d && ix==0) || (eg2d && ix==gnx2d-1)) continue;  // 跳过西/东幽灵列
    }
    // BGK 碰撞
}
```

---

## 5. 模式三：多网格独立模式（Multi-Grid）

### 5.1 使用场景

每个 MPI 进程运行**完全独立**的仿真（无任何 MPI 通信），适用于：

- **参数扫描**：同时运行 Re=100/500/1000/5000 四种工况
- **集成测试**：用多个独立进程并行验证不同边界条件组合
- **重启点生成**：每进程从不同初始条件出发，快速生成多组数据

```toml
[mpi]
mode = "multi_grid"
```

此模式下：
- 每个进程分配 **完整** 的 `nx × ny` 全局网格，无需幽灵层
- **无** `attach_mpi()` / `attach_mpi2d()` 绑定，`step()` 内无 MPI 通信
- 所有进程执行**相同**的边界条件和配置（若需区分，可通过 `mpi_rank()` 动态调整参数）

### 5.2 输出目录隔离

多网格模式下，`main.rs` 自动将每进程的输出写入独立子目录：

```
output/
├── rank_0/       ← rank 0 的 fluid_*.npz 等快照
│   ├── fluid_000001.npz
│   └── monitor.csv
├── rank_1/       ← rank 1 的快照
├── rank_2/
└── rank_3/
```

实现（`orchestrator/src/main.rs`）：

```rust
let output_dir = if cfg.mpi.mode == "multi_grid" && nprocs > 1 {
    format!("{}/rank_{}", cfg.output.directory, rank)
} else {
    cfg.output.directory.clone()
};
std::fs::create_dir_all(&output_dir)?;
```

---

## 6. MPI + OpenMP 混合并行

### 6.1 线程/进程分配策略

混合并行模式下，每个 MPI 进程内部使用多个 OpenMP 线程加速局部计算：

```
节点 A (rank 0..3)                    节点 B (rank 4..7)
┌────────────────────────┐            ┌────────────────────────┐
│ rank 0 (8 OMP threads) │ ←Halo→    │ rank 4 (8 OMP threads) │
│ rank 1 (8 OMP threads) │ ←Halo→    │ rank 5 (8 OMP threads) │
│ rank 2 (8 OMP threads) │ ←Halo→    │ rank 6 (8 OMP threads) │
│ rank 3 (8 OMP threads) │            │ rank 7 (8 OMP threads) │
└────────────────────────┘            └────────────────────────┘
```

**建议**：

- MPI 进程数 × OMP 线程数 ≤ 总物理核心数
- 每节点 1 个 MPI 进程 + N 个 OMP 线程，`N` = 节点物理核心数（NUMA 感知最佳）
- 或每 NUMA 域 1 个 MPI 进程 + 域内核心数个 OMP 线程

### 6.2 TOML 配置示例

```toml
[parallel]
omp_num_threads = 8   # 每个 MPI 进程使用 8 个 OpenMP 线程

[mpi]
mode      = "2d_xy"
nx_blocks = 4
ny_blocks = 2
# 总并行度 = 8 进程 × 8 线程 = 64 核
```

运行（SLURM HPC）：

```bash
mpirun --bind-to socket --map-by socket -n 8 \
    ./target/release/lbm-orchestrator configs/hybrid.toml
```

---

## 7. Rust / C API 使用示例

### 7.1 一维切片示例（Rust）

```rust
use lbm_bindings::{
    mpi_init, mpi_finalize, mpi_rank, mpi_size, mpi_local_ny,
    LbmGrid, LbmSolver, LbmMpiDecomp,
    LatticeModel, CollisionModel, BcType, Face,
    set_omp_num_threads,
};

fn main() {
    // 1. 初始化 MPI
    mpi_init();
    let rank   = mpi_rank();
    let nprocs = mpi_size();

    // 2. 可选：设置 OpenMP 线程数
    set_omp_num_threads(4);

    // 3. 计算本进程的本地网格尺寸
    let global_nx = 256i32;
    let global_ny = 256i32;
    let (grid_ny, _y_start, _local_ny) = mpi_local_ny(global_ny);
    // grid_ny = local_ny + 2（nprocs>1 时含幽灵行）

    // 4. 创建本地网格
    let mut grid = LbmGrid::new(global_nx, grid_ny, 1, LatticeModel::D2Q9);

    // 5. 创建求解器并绑定一维 MPI 分解
    let omega = 1.0 / (3.0 * 0.01 + 0.5);
    let mut solver = LbmSolver::new(&mut grid, omega, CollisionModel::Bgk);

    let mut decomp = LbmMpiDecomp::new(global_nx, global_ny)
        .expect("MPI not initialized or MPI not enabled");
    solver.attach_mpi(Some(&mut decomp));

    // 6. 注册边界条件（各 rank 只注册自己持有的物理壁面）
    if rank == 0 {
        // rank 0 持有全局南壁
        solver.add_boundary_condition(BcType::BounceBack, Face::South, 0.0, 0.0, 0.0, 1.0);
    }
    if rank == nprocs - 1 {
        // 最后 rank 持有全局北壁（顶盖速度）
        solver.add_boundary_condition(BcType::ZouHeVelocity, Face::North, 0.1, 0.0, 0.0, 0.0);
    }
    // 所有 rank 注册东西壁
    solver.add_boundary_condition(BcType::BounceBack, Face::West, 0.0, 0.0, 0.0, 1.0);
    solver.add_boundary_condition(BcType::BounceBack, Face::East, 0.0, 0.0, 0.0, 1.0);

    // 7. 主循环（step() 内部自动执行幽灵行交换）
    for _step in 0..5000 {
        solver.step(&mut grid);
    }

    mpi_finalize();
}
```

### 7.2 二维块分解示例（Rust）

```rust
use lbm_bindings::{
    mpi_init, mpi_finalize, mpi_rank, mpi_size,
    LbmGrid, LbmSolver, LbmMpiDecomp2D,
    LatticeModel, CollisionModel, BcType, Face,
};

fn main() {
    mpi_init();
    let rank   = mpi_rank();
    let nprocs = mpi_size();

    // 2D 块分解：px=4, py=2 → 需要 mpirun -n 8
    let global_nx = 512i32;
    let global_ny = 512i32;
    let px = 4i32;
    let py = 2i32;
    assert_eq!(px * py, nprocs, "px*py must equal nprocs");

    // 创建 2D 分解描述符
    let mut decomp2d = LbmMpiDecomp2D::new(global_nx, global_ny, px, py)
        .expect("2D decomp creation failed");

    // 创建本地网格（尺寸由 decomp2d 决定，含幽灵列）
    let local_nx = decomp2d.grid_nx();
    let local_ny = decomp2d.grid_ny();
    let x_start  = decomp2d.x_start();
    let y_start  = decomp2d.y_start();

    println!("rank={rank}: local={local_nx}×{local_ny}, x_start={x_start}, y_start={y_start}");

    let mut grid = LbmGrid::new(local_nx, local_ny, 1, LatticeModel::D2Q9);
    let mut solver = LbmSolver::new(&mut grid, 1.6, CollisionModel::Bgk);

    // 绑定二维分解（之后 step() 自动执行 2D 幽灵层交换）
    solver.attach_mpi2d(Some(&mut decomp2d));

    // 注册各 rank 拥有的物理壁面
    // 以顶盖驱动方腔为例：
    if decomp2d.y_start() == 0 {
        solver.add_boundary_condition(BcType::BounceBack, Face::South, 0.0, 0.0, 0.0, 1.0);
    }
    // ... (类似地注册 North、West、East)

    for _step in 0..5000 {
        solver.step(&mut grid);
    }

    mpi_finalize();
}
```

> **注意**：`decomp2d.y_start() == 0` 等价于 `decomp2d.has_south_wall()`（即 `row_rank == 0`）。在 Rust 中若需直接访问 `col_rank` / `row_rank`，可通过 C ABI 的 `lbm_mpi_decomp2d_x_start()` / `lbm_mpi_decomp2d_y_start()` 推算。

### 7.3 多网格独立模式示例（Rust）

```rust
use lbm_bindings::{mpi_init, mpi_finalize, mpi_rank, mpi_size, LbmGrid, LbmSolver, ...};

fn main() {
    mpi_init();
    let rank = mpi_rank();

    // 每个进程持有完整网格，无幽灵层，无通信
    let nx = 256i32; let ny = 256i32;
    let mut grid = LbmGrid::new(nx, ny, 1, LatticeModel::D2Q9);
    let mut solver = LbmSolver::new(&mut grid, 1.6, CollisionModel::Bgk);
    // 注册完整边界条件（所有进程相同）
    // ...

    // 也可以根据 rank 设置不同参数（如不同 Re 数）：
    // let nu = [0.01, 0.005, 0.002, 0.001][rank as usize % 4];
    // let mut solver = LbmSolver::new(&mut grid, 1.0/(3.0*nu+0.5), CollisionModel::Bgk);

    for _step in 0..5000 {
        solver.step(&mut grid);
    }
    // 输出写入 output/rank_<N>/（由 orchestrator 自动处理）

    mpi_finalize();
}
```

### 7.4 C ABI 函数速查表

| C 函数 | 对应 Rust 函数 | 说明 |
|--------|---------------|------|
| `lbm_mpi_init()` | `mpi_init()` | 初始化 MPI（幂等） |
| `lbm_mpi_finalize()` | `mpi_finalize()` | 结束 MPI |
| `lbm_mpi_rank()` | `mpi_rank()` | 当前进程编号 |
| `lbm_mpi_size()` | `mpi_size()` | 进程总数 |
| `lbm_mpi_local_ny(gny, &yst, &lny)` | `mpi_local_ny(gny)` → `(grid_ny, y_start, local_ny)` | 一维切片本地 ny |
| `lbm_mpi_decomp_new(gnx, gny)` | `LbmMpiDecomp::new(gnx, gny)` | 创建一维分解 |
| `lbm_mpi_decomp_free(h)` | `Drop for LbmMpiDecomp` | 释放一维分解 |
| `lbm_solver_attach_mpi(s, h)` | `LbmSolver::attach_mpi(d)` | 绑定一维分解 |
| `lbm_mpi_decomp2d_new(gnx, gny, px, py)` | `LbmMpiDecomp2D::new(gnx, gny, px, py)` | 创建二维分解 |
| `lbm_mpi_decomp2d_free(h)` | `Drop for LbmMpiDecomp2D` | 释放二维分解 |
| `lbm_solver_attach_mpi2d(s, h)` | `LbmSolver::attach_mpi2d(d)` | 绑定二维分解 |
| `lbm_mpi_decomp2d_grid_nx(h)` | `LbmMpiDecomp2D::grid_nx()` | 本地含幽灵列 nx |
| `lbm_mpi_decomp2d_grid_ny(h)` | `LbmMpiDecomp2D::grid_ny()` | 本地含幽灵行 ny |
| `lbm_mpi_decomp2d_x_start(h)` | `LbmMpiDecomp2D::x_start()` | 全局 X 起始坐标 |
| `lbm_mpi_decomp2d_y_start(h)` | `LbmMpiDecomp2D::y_start()` | 全局 Y 起始坐标 |
| `lbm_omp_set_num_threads(n)` | `set_omp_num_threads(n)` | 设置 OMP 线程数 |

---

## 8. 构建与运行

### 8.1 编译开关

| 环境变量（Cargo）| CMake 参数 | 说明 |
|-----------------|-----------|------|
| `LBM_ENABLE_MPI=ON` | `-DENABLE_MPI=ON` | 启用 MPI（默认 OFF） |
| `LBM_ENABLE_OPENMP=ON` | `-DENABLE_OPENMP=ON` | 启用 OpenMP（默认 OFF） |
| `LBM_ENABLE_CUDA=ON` | `-DENABLE_CUDA=ON` | 启用 CUDA GPU 后端 |

> **Windows**：需预装 [Microsoft MPI (MS-MPI)](https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi) 或 Intel MPI；运行时使用 `mpiexec -n N` 而非 `mpirun -n N`。

### 8.2 跨平台构建命令

#### Linux / macOS（Bash / Zsh）

```bash
# 仅 OpenMP
LBM_ENABLE_OPENMP=ON cargo build --release

# 仅 MPI
LBM_ENABLE_MPI=ON cargo build --release

# MPI + OpenMP 混合
LBM_ENABLE_MPI=ON LBM_ENABLE_OPENMP=ON cargo build --release

# 持久化（当前 shell 会话）
export LBM_ENABLE_MPI=ON
export LBM_ENABLE_OPENMP=ON
cargo build --release
```

#### Windows PowerShell

```powershell
$env:LBM_ENABLE_MPI    = "ON"
$env:LBM_ENABLE_OPENMP = "ON"
cargo build --release
```

#### Windows cmd

```bat
set LBM_ENABLE_MPI=ON
set LBM_ENABLE_OPENMP=ON
cargo build --release
```

#### 纯 CMake 构建（不依赖 Cargo，适合 HPC 集成测试）

```bash
cmake -B build \
      -DENABLE_MPI=ON \
      -DENABLE_OPENMP=ON \
      -DENABLE_CUDA=OFF \
      -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel
```

### 8.3 运行命令

#### 一维 Y 切片（4 进程）

```bash
# 在 TOML 中设置 [mpi] mode = "1d_y"（或省略，默认值）
mpirun -n 4 ./target/release/lbm-orchestrator configs/lid_driven_cavity.toml
```

#### 二维 XY 块分解（4×2 = 8 进程）

```toml
# configs/mpi2d.toml
[mpi]
mode      = "2d_xy"
nx_blocks = 4
ny_blocks = 2
```

```bash
mpirun -n 8 ./target/release/lbm-orchestrator configs/mpi2d.toml
```

#### 多网格独立模式（4 套独立仿真）

```toml
# configs/multi_grid.toml
[mpi]
mode = "multi_grid"
```

```bash
mpirun -n 4 ./target/release/lbm-orchestrator configs/multi_grid.toml
# 输出：output/rank_0/ output/rank_1/ output/rank_2/ output/rank_3/
```

#### MPI + OpenMP 混合

```toml
# configs/hybrid.toml
[parallel]
omp_num_threads = 4

[mpi]
mode      = "2d_xy"
nx_blocks = 2
ny_blocks = 2
```

```bash
# 每节点 4 进程，每进程 4 OMP 线程 = 16 核
mpirun -n 4 ./target/release/lbm-orchestrator configs/hybrid.toml
```

#### Windows（MS-MPI）

```powershell
mpiexec -n 4 .\target\release\lbm-orchestrator.exe configs\lid_driven_cavity.toml
```

### 8.4 SLURM 集群脚本示例

#### 一维切片（单节点 16 进程）

```bash
#!/bin/bash
#SBATCH --job-name=lbm_mpi1d
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=1

module load openmpi/4.1

mpirun -n 16 ./target/release/lbm-orchestrator configs/lid_driven_cavity.toml
```

#### 二维块分解（2 节点，每节点 4 进程 × 4 线程）

```bash
#!/bin/bash
#SBATCH --job-name=lbm_mpi2d_hybrid
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

export OMP_PROC_BIND=close
export OMP_PLACES=cores

# 通过 TOML [parallel].omp_num_threads = 4 设置（或环境变量）
mpirun -n 8 ./target/release/lbm-orchestrator configs/hybrid.toml
```

---

## 9. 正确性保证与常见陷阱

### 9.1 不要对幽灵层执行二次碰撞

**问题**：幽灵层节点中的 f 值已由相邻进程完成了一次碰撞。若本进程再次对其执行碰撞（"二次碰撞"），分布函数被过度松弛，边界附近密度/速度出现错误。

**修复**（已实现）：
- 一维模式：`collide_bgk()` / `collide_mrt()` 用 `n_start` / `n_end` 跳过幽灵行
- 二维模式：`collide_bgk()` / `collide_mrt()` 用 `ix/iy` 检查逐节点跳过幽灵层节点

### 9.2 边界条件只在拥有该物理壁的 rank 注册

**问题**：若所有 rank 都注册了 South BC，内部 rank 的 j=0 行是幽灵行而非物理南壁，错误地施加 BC 会破坏 halo 数据。

**检查方法**（C++）：
```cpp
if (decomp.has_south_wall()) {  // rank==0（1D）或 row_rank==0（2D）
    solver.add_boundary_condition(bc_south);
}
```

**检查方法**（Rust，二维分解）：
```rust
if decomp2d.y_start() == 0 {  // row_rank == 0
    solver.add_boundary_condition(BcType::BounceBack, Face::South, ...);
}
```

### 9.3 px × py 必须等于 MPI 进程总数

`lbm_mpi_decomp2d_new(gnx, gny, px, py)` 在 `px * py != nprocs` 时返回 `nullptr`（Rust 侧返回 `None`）。`Solver::attach_mpi2d(nullptr)` 退化为解除绑定，不会崩溃，但会静默回退到无 MPI 模式。

**orchestrator 自动检测**（`main.rs`）：若检测到 `nx_blocks * ny_blocks != nprocs`，打印警告并退化为 1D Y 切片模式：

```
[warn] mpi.nx_blocks(4) * mpi.ny_blocks(2) = 8 ≠ nprocs(4). Falling back to 1d_y decomposition.
```

---

## 10. 性能分析与调优建议

### 通信量分析

| 模式 | 每步通信量（D2Q9，n=N²）| 备注 |
|------|------------------------|------|
| 一维 Y（P 进程）| 2 × nx × 9 × 8 B × P | 只有南北方向 2 次 |
| 二维 XY（px×py）| 2 × (lnx + lny) × 9 × 8 B × P | 南北+东西 4 次；东西需打包列 |

**二维优势**：当网格接近方形（nx ≈ ny）且进程数多时，每进程本地面积 O(N²/P) 而通信周长 O(N/√P)，二维分解的通信计算比更优（O(1/√P)）。一维分解通信周长为 O(N)，计算比 O(1/√P) 仅当 py≈1 时与二维持平。

### OpenMP 与 MPI 的协同

- **OpenMP 粒度**：碰撞、流式迁移和宏观量计算均用 `#pragma omp parallel for schedule(static)` 并行，各节点完全独立，效率极高
- **MPI 粒度**：每进程分配的 `local_ny × local_nx` 越大，碰撞/流式迁移的 OpenMP 效率越高；幽灵层占比（`2/local_ny` 或 `2/(local_nx+local_ny)`）越小
- **最佳比例**：经验上每进程 `local_ny ≥ 32`（或 `local_nx ≥ 32`），避免幽灵层比例过高

### 内存访问模式

行主序存储（`idx(i, j) = i + nx * j`）下：
- 流式迁移内层循环沿 `i`（X 方向）连续，缓存友好
- 幽灵行（j 方向）交换为连续内存（`row_size = nx * Q` 个 double），MPI 直接传指针
- 幽灵列（i 方向）交换需手动打包（`halo_exchange_d2q9_2d()` 中的 `for j` 循环），每次约 `gny × Q × 8` 字节额外内存带宽

---

*本文档对应代码版本：`core/include/lbm/mpi_decomp.hpp`，`core/src/lbm/mpi_decomp.cpp`，`core/src/lbm/solver.cpp`，`bindings/src/lib.rs`，`orchestrator/src/config.rs`，`orchestrator/src/main.rs`。*
