# MPI 并行实现原理

> 本文档详细介绍 `lbm-ibm-fsi` 项目中 MPI 并行加速的设计思路、网格分块原理、幽灵层（Halo）数据交换机制、边界条件的进程隔离，以及输出模式等。所有说明均结合具体代码文件及行号，便于对照阅读。

---

## 目录

1. [总体架构](#1-总体架构)
2. [域分解策略](#2-域分解策略)
   - [一维 Y 切片分解（MpiDecomp）](#21-一维-y-切片分解mpiDecomp)
   - [二维 XY 块分解（MpiDecomp2D）](#22-二维-xy-块分解mpiDecomp2D)
   - [三维 XYZ 块分解（MpiDecomp3D）](#23-三维-xyz-块分解mpiDecomp3D)
3. [幽灵层机制与进程间数据交换](#3-幽灵层机制与进程间数据交换)
   - [为什么需要幽灵层](#31-为什么需要幽灵层)
   - [一维模式幽灵行交换](#32-一维模式幽灵行交换)
   - [二维模式幽灵层交换](#33-二维模式幽灵层交换)
   - [三维模式幽灵层交换](#34-三维模式幽灵层交换)
   - [交换时机](#35-交换时机)
4. [边界条件的进程隔离](#4-边界条件的进程隔离)
   - [PhysicalBounds 结构体](#41-physicalBounds-结构体)
   - [per-rank BC 注册逻辑](#42-per-rank-bc-注册逻辑)
   - [坐标范围输出](#43-坐标范围输出)
5. [输出模式](#5-输出模式)
6. [TOML 配置参考](#6-toml-配置参考)
7. [关键代码位置索引](#7-关键代码位置索引)

---

## 1. 总体架构

本项目使用 **进程级 MPI 并行**（`MPI_COMM_WORLD`）对全局计算域进行空间分解。每个 MPI 进程独立拥有一块子网格，在每个时间步内：

```
全局时间步循环：
  1. collide()         — 本地 BGK/MRT 碰撞（无需通信）
  2. halo_exchange()   — 交换相邻进程的幽灵行/列（MPI_Sendrecv）
  3. stream()          — 本地流式迁移（使用已更新的幽灵数据）
  4. apply_BC()        — 仅在持有物理壁面的进程上施加边界条件
  5. compute_macro()   — 更新宏观量（密度、速度）
```

MPI 通信只在步骤 2 发生，每时间步通信量极少（仅边界层数据），可扩展性强。

**代码入口：**
- `core/src/lbm/solver.cpp`（第 61~125 行）：`Solver::step()` 编排上述五步
- `orchestrator/src/main.rs`（第 89~152 行）：Rust 层初始化 MPI 描述符并调用求解器

---

## 2. 域分解策略

### 2.1 一维 Y 切片分解（`MpiDecomp`）

**原理：** 将全局网格沿 **Y 方向**均匀切分为 `nprocs` 个水平条带，每个进程持有若干连续物理行。

```
全局网格（ny 行）：

  rank 3  ┌──────────────────────────────┐  y = [y3_start, ny-1]
          │    物理行 (local_ny3 行)      │
  rank 2  ├──────────────────────────────┤  y = [y2_start, y3_start-1]
          │    物理行 (local_ny2 行)      │
  rank 1  ├──────────────────────────────┤  y = [y1_start, y2_start-1]
          │    物理行 (local_ny1 行)      │
  rank 0  └──────────────────────────────┘  y = [0, y1_start-1]
           ← nx 列 →
```

**分配策略（整除不均时前若干进程各多分一行）：**

```
local_ny[r] = base + (r < ny % nprocs ? 1 : 0)
y_start[r]  = r * base + min(r, ny % nprocs)
```

**本地网格布局（含幽灵行，`nprocs > 1`）：**

```
j = 0            : 南幽灵行（来自 rank-1 的最北物理行）
j = 1            : 本进程最南物理行（全局 y_start）
    ...
j = local_ny     : 本进程最北物理行（全局 y_end）
j = local_ny+1   : 北幽灵行（来自 rank+1 的最南物理行）
```

**代码位置：**
- 结构体定义：`core/include/lbm/mpi_decomp.hpp` 第 83~104 行（`struct MpiDecomp`）
- 创建函数：`core/src/lbm/mpi_decomp.cpp` 第 22~43 行（`MpiDecomp::create()`）

---

### 2.2 二维 XY 块分解（`MpiDecomp2D`）

**原理：** 将全局网格同时沿 **X 和 Y 方向**切分为 `px × py` 个矩形块，总进程数 = `px × py`。

```
进程布局（px=3, py=2，共 6 个进程）：

  row_rank=1  ┌────────┬────────┬────────┐  ← 全局北壁（row_rank==py-1 注册 Face::North BC）
              │rank 3  │rank 4  │rank 5  │
  row_rank=0  └────────┴────────┴────────┘  ← 全局南壁（row_rank==0 注册 Face::South BC）
              col=0    col=1    col=2
              ↑                         ↑
           西壁BC                     东壁BC
```

**进程坐标计算：**

```
rank = row_rank * px + col_rank
col_rank = rank % px    （X 方向坐标）
row_rank = rank / px    （Y 方向坐标）
```

**一维特例：**
- `px=1`（1D Y 切片）：所有进程 `col_rank=0`，无西/东幽灵列，等价于旧 `MpiDecomp`
- `py=1`（1D X 切片）：所有进程 `row_rank=0`，无南/北幽灵行

**本地网格布局（phys_x0/phys_y0 为 0 或 1）：**

```
j=0                         : 南幽灵行（row_rank>0 时存在）
j=phys_y0 .. phys_y0+lny-1 : 物理行
j=phys_y0+lny               : 北幽灵行（row_rank<py-1 时存在）

i=0                         : 西幽灵列（col_rank>0 时存在）
i=phys_x0 .. phys_x0+lnx-1 : 物理列
i=phys_x0+lnx               : 东幽灵列（col_rank<px-1 时存在）
```

**邻居进程 rank 计算（`MPI_PROC_NULL=-1` 表示无邻居，对 `MPI_Sendrecv` 是合法的"空邻居"）：**

```cpp
rank_south = (row_rank > 0)        ? (row_rank-1)*px + col_rank : MPI_PROC_NULL;
rank_north = (row_rank < py-1)     ? (row_rank+1)*px + col_rank : MPI_PROC_NULL;
rank_west  = (col_rank > 0)        ? row_rank*px + col_rank-1   : MPI_PROC_NULL;
rank_east  = (col_rank < px-1)     ? row_rank*px + col_rank+1   : MPI_PROC_NULL;
```

**代码位置：**
- 结构体定义：`core/include/lbm/mpi_decomp.hpp` 第 134~190 行（`struct MpiDecomp2D`）
- 创建函数：`core/src/lbm/mpi_decomp.cpp` 第 127~165 行（`MpiDecomp2D::create()`）

---

### 2.3 三维 XYZ 块分解（`MpiDecomp3D`）

**原理：** 将三维网格沿 X/Y/Z 三个方向切分为 `px × py × pz` 块。当前用于**预留三维 LBM 并行扩展**，幽灵层交换已实现（见 `core/src/lbm/mpi_decomp.cpp` 第 350~571 行）。

**进程坐标计算：**

```
rank = pz_rank*(px*py) + row_rank*px + col_rank
col_rank = rank % px
row_rank = (rank / px) % py
pz_rank  = rank / (px * py)
```

**代码位置：**
- 结构体定义：`core/include/lbm/mpi_decomp.hpp` 第 209~268 行（`struct MpiDecomp3D`）

---

## 3. 幽灵层机制与进程间数据交换

### 3.1 为什么需要幽灵层

LBM 流式迁移步骤中，节点 `(i, j)` 的分布函数 `f_α` 会沿速度方向 `c_α` 迁移到相邻节点。当分区边界的节点需要访问**属于另一个进程的邻居节点**时，就必须提前从该进程获取那一行/列数据，暂存在"幽灵行/列"中。

**关键约束：** 幽灵层交换必须发生在 **stream 循环之前**（碰后），而不是之后。若在 stream 之后交换，物理行已包含迁移后的数据，上游数据被覆盖，导致分区接口处出现 O(1) 数量级的速度不连续（已验证）。

> 参见注释：`core/src/lbm/solver.cpp` stream() 函数注释块

---

### 3.2 一维模式幽灵行交换

**函数：** `lbm::halo_exchange_d2q9()`（`core/src/lbm/mpi_decomp.cpp` 第 47~92 行）

**交换规则（两次 `MPI_Sendrecv`，无死锁风险）：**

```
交换 1：
  本进程 顶物理行（j=local_ny）  → 北邻进程 南幽灵行（j=0）
  南邻进程 北幽灵行（j=lny+1）   ← 南邻进程 顶物理行

交换 2：
  本进程 底物理行（j=1）         → 南邻进程 北幽灵行（j=lny+1）
  本进程 北幽灵行（j=lny+1）     ← 北邻进程 底物理行（j=1）
```

**数据量：** 每次交换 `nx × Q × sizeof(double)` 字节（一整行所有速度方向的分布函数）。

---

### 3.3 二维模式幽灵层交换

**函数：** `lbm::halo_exchange_d2q9_2d()`（`core/src/lbm/mpi_decomp.cpp` 第 168~295 行）

**交换分两步进行，先南北后东西：**

**步骤 1 — 南北方向（整行数据，行主序中连续，可直接传输）：**

```
交换 1a（tag=10）：
  本进程 顶物理行（j=phys_y0+lny-1） → 北邻 南幽灵行（j=0）
  本进程 南幽灵行（j=0）             ← 南邻 顶物理行

交换 1b（tag=11）：
  本进程 底物理行（j=phys_y0）       → 南邻 北幽灵行
  本进程 北幽灵行（j=phys_y0+lny）   ← 北邻 底物理行
```

**步骤 2 — 东西方向（列数据，行主序中不连续，需先打包）：**

由于 `LatticeGrid` 使用行主序存储，同一列的相邻节点在内存中并不连续，因此需要手动打包（pack）成连续缓冲区再发送，接收后解包（unpack）写回对应幽灵列：

```
打包最西物理列（i=phys_x0）       → send_west[j*Q + a]
打包最东物理列（i=phys_x0+lnx-1）→ send_east[j*Q + a]

交换（tag=20/21）：
  本进程 send_west → 西邻（西邻接收后写入其东幽灵列）
  本进程 recv_east ← 东邻 最西物理列（写入本进程东幽灵列 i=phys_x0+lnx）
  本进程 send_east → 东邻
  本进程 recv_west ← 西邻 最东物理列（写入本进程西幽灵列 i=0）
```

**代码细节（`core/src/lbm/mpi_decomp.cpp` 第 228~295 行）：**

```cpp
// 向西邻发送最西物理列，从东邻接收东幽灵列（tag=20）
MPI_Sendrecv(send_west.data(), col_size, MPI_DOUBLE, decomp.rank_west,  20,
             recv_east.data(), col_size, MPI_DOUBLE, decomp.rank_east,  20,
             MPI_COMM_WORLD, &st);
// 向东邻发送最东物理列，从西邻接收西幽灵列（tag=21）
MPI_Sendrecv(send_east.data(), col_size, MPI_DOUBLE, decomp.rank_east, 21,
             recv_west.data(), col_size, MPI_DOUBLE, decomp.rank_west,  21,
             MPI_COMM_WORLD, &st);
```

---

### 3.4 三维模式幽灵层交换

**函数：** `lbm::halo_exchange_d3q19_3d()`（`core/src/lbm/mpi_decomp.cpp` 第 300~571 行）

**交换顺序：Z（底/顶）→ Y（南/北）→ X（西/东）**，每方向均使用 `MPI_Sendrecv`。

- Z 方向：交换整个 XY 平面（slab），数据在内存中连续，可直接传输
- Y 方向：需打包 XZ 平面（所有 k 层的一行）
- X 方向：需打包 YZ 平面（所有 j/k 层的一列），内存不连续，需 3 重循环打包

---

### 3.5 交换时机

幽灵层交换由 `Solver::stream()` 在 push-scheme 流式迁移循环**之前**自动触发：

```cpp
// core/src/lbm/solver.cpp（stream() 函数内部）
#ifdef LBM_ENABLE_MPI
if (mpi_decomp_ && mpi_decomp_->nprocs > 1)
    halo_exchange_d2q9(grid_, *mpi_decomp_);
if (mpi_decomp2d_ && mpi_decomp2d_->nprocs > 1)
    halo_exchange_d2q9_2d(grid_, *mpi_decomp2d_);
if (mpi_decomp3d_ && mpi_decomp3d_->nprocs > 1)
    halo_exchange_d3q19_3d(grid_, *mpi_decomp3d_);
#endif
// ... 随后执行 stream 循环 ...
```

---

## 4. 边界条件的进程隔离

### 4.1 `PhysicalBounds` 结构体

在 MPI 并行模式下，每个进程的本地网格包含幽灵行/列。如果直接将边界条件（BC）施加在本地网格的 `j=0` 或 `j=ny-1` 行（即幽灵行），就会污染幽灵数据，导致下一时间步的幽灵交换将错误的 BC 值传递给相邻进程，从而在分区接口处产生速度/密度阶跃。

`PhysicalBounds` 结构体（`core/include/lbm/boundary.hpp` 第 59~100 行）通过显式记录物理行/列范围来解决这一问题：

```cpp
struct PhysicalBounds {
    int  j_s = 0, j_n = ny-1;   // 物理南/北边界行索引（本地坐标）
    int  i_w = 0, i_e = nx-1;   // 物理西/东边界列索引（本地坐标）
    bool has_south_wall = true;  // 本进程是否持有全局南壁
    bool has_north_wall = true;  // 本进程是否持有全局北壁
    bool has_west_wall  = true;  // 本进程是否持有全局西壁
    bool has_east_wall  = true;  // 本进程是否持有全局东壁
};
```

`apply_boundary_conditions()` 在遍历壁面节点时以 `j_s`/`j_n`/`i_w`/`i_e` 替代硬编码的 `0`/`ny-1`/`0`/`nx-1`，并用 `has_*_wall` 标志决定是否真正施加该面的 BC。

**在 `Solver::step()` 中计算 `PhysicalBounds`（`core/src/lbm/solver.cpp` 第 78~121 行）：**

```
非 MPI：    j_s=0, j_n=ny-1, i_w=0, i_e=nx-1, 所有 has_*_wall=true
1D Y 切片： j_s=1, j_n=ny-2, i_w=0, i_e=nx-1
            has_south_wall = (rank 0 才为 true)
            has_north_wall = (rank nprocs-1 才为 true)
2D 块分解： j_s=phys_y0, j_n=phys_y0+local_ny-1
            i_w=phys_x0, i_e=phys_x0+local_nx-1
            has_*_wall 均按各自进程是否位于全局边缘而设置
```

---

### 4.2 per-rank BC 注册逻辑

Rust orchestrator 在 `orchestrator/src/main.rs`（第 397~434 行）中，为每个进程判断哪些面 BC 应注册：

```rust
let apply_bc = if effective_mode == "block" && nprocs > 1 {
    if let Some(ref d) = _decomp2d {
        match face {
            Face::South => y_start == 0,              // 仅全局南壁进程注册
            Face::North => y_start + local_ny == gny, // 仅全局北壁进程注册
            Face::West  => x_start == 0,
            Face::East  => x_start + local_nx == gnx,
            _           => true,  // Bottom/Top 等面所有进程均注册（3D 情形）
        }
    }
} else {
    true  // 非 MPI 模式：全部进程注册全部 BC
};
```

这确保了**只有持有该物理壁面的进程**才会注册并施加对应 BC，内部进程不会错误地在非壁面的行上施加边界条件。

---

### 4.3 坐标范围输出

为便于调试，BC 注册完成后会以进程顺序输出每个进程的全局坐标范围及已注册的 BC：

```
*** rank   0  全局坐标范围: x=[0, 512), y=[0, 128) ***
  rank   0, South, BounceBack  (ux=0.0000, uy=0.0000, rho=1.0000)
  rank   0, West,  BounceBack  (ux=0.0000, uy=0.0000, rho=1.0000)
  rank   0, East,  BounceBack  (ux=0.0000, uy=0.0000, rho=1.0000)
*** rank   1  全局坐标范围: x=[0, 512), y=[128, 256) ***
  rank   1, West,  BounceBack  (ux=0.0000, uy=0.0000, rho=1.0000)
  rank   1, East,  BounceBack  (ux=0.0000, uy=0.0000, rho=1.0000)
...
```

`***` 标记便于在多进程输出中快速定位各分区信息。三维分解时还会输出 `z=[…)` 范围。

**代码位置：** `orchestrator/src/main.rs` 第 357~376 行

输出顺序通过 `mpi_barrier()` 轮转保证（第 472~487 行）：

```rust
for r in 0..nprocs {
    if rank == r {
        for line in &bc_log { println!("{}", line); }
        let _ = std::io::stdout().flush();
    }
    lbm_bindings::mpi_barrier();  // 所有进程同步后轮到下一个
}
```

---

## 5. 输出模式

在 MPI block 模式下，每个进程将其本地物理区域的快照写入独立子目录，避免多进程同时写同一文件引发竞态条件：

```
output/
├── rank_0/
│   ├── fluid_000000.npz   ← rank 0 的物理区域快照（已剥离幽灵层）
│   ├── fluid_000100.npz
│   └── ...
├── rank_1/
│   └── ...
└── ...
```

每个 NPZ 文件包含 `x_start`、`y_start`、`global_nx`、`global_ny` 等元数据，供后处理脚本（`python/lbm_post/vtk_reader.py`）拼合成全局视图。

`PartitionInfo` 结构体（`orchestrator/src/output.rs` 第 34~90 行）描述每个进程的分区信息：

```rust
pub struct PartitionInfo {
    pub phys_x0:   usize,  // 物理区域在本地网格中的 x 偏移（0 或 1）
    pub phys_y0:   usize,  // 物理区域在本地网格中的 y 偏移（0 或 1）
    pub local_nx:  usize,  // 本地物理 x 节点数（不含幽灵）
    pub local_ny:  usize,  // 本地物理 y 节点数（不含幽灵）
    pub x_start:   usize,  // 本地区域在全局坐标中的 x 起始
    pub y_start:   usize,  // 本地区域在全局坐标中的 y 起始
    pub global_nx: usize,  // 全局 x 节点数
    pub global_ny: usize,  // 全局 y 节点数
}
```

**合并输出：** 设置 `[output] combine_blocks = true`，rank-0 通过 `MPI_Gatherv` 收集所有分区数据，将完整全局场写入 `output/fluid_*.npz`（`orchestrator/src/output.rs` 第 478~715 行）。

---

## 6. TOML 配置参考

```toml
[mpi]
# MPI 并行模式
mode = "block"        # "block"（域分解）| "independent" | "multigrid"

# 二维块分解：nx_blocks × ny_blocks = 进程总数（mpirun -n N）
nx_blocks = 1         # X 方向分块数（px）；1 → 1D Y 切片
ny_blocks = 4         # Y 方向分块数（py）

# 三维块分解（预留）
nz_blocks = 1         # Z 方向分块数（pz）；1 → 退化为 2D

[output]
combine_blocks = false   # true → rank-0 合并所有分区到单文件
```

**典型启动命令（4 进程 1D Y 切片）：**

```bash
mpirun -n 4 ./lbm-fsi config.toml
```

---

## 7. 关键代码位置索引

| 功能                              | 文件                                        | 行号       |
|-----------------------------------|---------------------------------------------|------------|
| `MpiDecomp` 结构体定义            | `core/include/lbm/mpi_decomp.hpp`           | 83~104     |
| `MpiDecomp2D` 结构体定义          | `core/include/lbm/mpi_decomp.hpp`           | 134~190    |
| `MpiDecomp3D` 结构体定义          | `core/include/lbm/mpi_decomp.hpp`           | 209~268    |
| 幽灵交换函数声明                  | `core/include/lbm/mpi_decomp.hpp`           | 270~313    |
| 一维域分解 `MpiDecomp::create()`  | `core/src/lbm/mpi_decomp.cpp`               | 22~43      |
| 一维幽灵行交换                    | `core/src/lbm/mpi_decomp.cpp`               | 47~92      |
| 二维域分解 `MpiDecomp2D::create()`| `core/src/lbm/mpi_decomp.cpp`               | 127~165    |
| 二维幽灵层交换                    | `core/src/lbm/mpi_decomp.cpp`               | 168~295    |
| 三维幽灵层交换                    | `core/src/lbm/mpi_decomp.cpp`               | 300~571    |
| `Solver::step()` 时间步编排       | `core/src/lbm/solver.cpp`                   | 61~125     |
| `PhysicalBounds` 结构体           | `core/include/lbm/boundary.hpp`             | 59~100     |
| `apply_boundary_conditions()`     | `core/src/lbm/boundary.cpp`                 | 765~855    |
| per-rank BC 注册                  | `orchestrator/src/main.rs`                  | 397~434    |
| 坐标范围输出（`***` 标记）        | `orchestrator/src/main.rs`                  | 357~376    |
| BC 顺序输出（barrier 轮转）       | `orchestrator/src/main.rs`                  | 472~487    |
| `PartitionInfo` 结构体            | `orchestrator/src/output.rs`                | 34~90      |
| 分区快照写入 `rank_N/` 子目录     | `orchestrator/src/main.rs`                  | 521~529    |
| `MPI_Gatherv` 合并输出            | `orchestrator/src/output.rs`                | 478~715    |
| Python 拼合工具                   | `python/examples/combine_blocks.py`         | 全文       |
| Rust MPI bindings                 | `bindings/src/lib.rs`                       | 全文       |
| CMake MPI 链接配置                | `CMakeLists.txt`                            | 31~48      |
