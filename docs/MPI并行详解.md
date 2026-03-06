# LBM-IBM-FSI MPI 并行详解

本文档详细说明项目在 MPI（及 MPI + OpenMP 混合）模式下的完整求解流程，包括：**三种并行模式**（XY 块分解、多进程独立、嵌套多重网格）、一维切片作为二维块分解的特例、三维扩展预留接口、幽灵层交换实现、OpenMP 线程数便捷设置、以及 Rust / C API 使用示例。

---

## 目录

1. [整体架构与并行模式选择](#1-整体架构与并行模式选择)
2. [TOML 配置快速参考](#2-toml-配置快速参考)
3. [模式一：XY 块分解（`"block"`）](#3-模式一xy-块分解block)
   - 3.1 统一视角：一维切片是二维块分解的特例
   - 3.2 进程布局与坐标
   - 3.3 幽灵层布局与 Halo Exchange
   - 3.4 边界条件约定
   - 3.5 时序图
   - 3.6 C++ 实现：`MpiDecomp2D::create()`
   - 3.7 幽灵层交换：`halo_exchange_d2q9_2d()`
   - 3.8 三维预留：`MpiDecomp3D`
   - 3.9 块分解模式下的输出策略
4. [模式二：多进程独立（`"independent"`）](#4-模式二多进程独立independent)
   - 4.1 使用场景
   - 4.2 输出目录隔离
5. [模式三：嵌套式多重网格（`"multigrid"`）](#5-模式三嵌套式多重网格multigrid)
   - 5.1 概述与适用场景
   - 5.2 多重网格树数据结构（`MgTree` / `MgNode`）
   - 5.3 树节点的空间范围（`MgExtent`）
   - 5.4 N 叉树结构与任意嵌套
   - 5.5 三维支持（`MgDim::D3`）
   - 5.6 与 MPI 域分解的关联
   - 5.7 遍历接口
   - 5.8 当前实现状态与扩展路线
6. [MPI + OpenMP 混合并行](#6-mpi--openmp-混合并行)
   - 6.1 OpenMP 线程数设置
   - 6.2 线程/进程分配策略
7. [Rust / C API 使用示例](#7-rust--c-api-使用示例)
   - 7.1 XY 块分解示例（1D Y 切片 ≡ nx_blocks=1）
   - 7.2 二维块分解示例
   - 7.3 多进程独立模式示例
   - 7.4 多重网格树创建示例
   - 7.5 C ABI 函数速查表
8. [构建与运行](#8-构建与运行)
   - 8.1 编译开关
   - 8.2 跨平台构建命令
   - 8.3 运行命令
   - 8.4 SLURM 集群脚本示例
9. [正确性保证与常见陷阱](#9-正确性保证与常见陷阱)
10. [性能分析与调优建议](#10-性能分析与调优建议)

---

## 1. 整体架构与并行模式选择

本项目支持三种 MPI 并行模式，全部通过编译时环境变量和 TOML 配置控制：

| 模式 | TOML `[mpi].mode` | 旧名称（向后兼容）| 适用场景 |
|------|-------------------|-----------------|---------|
| **XY 块分解** | `"block"`（默认）| `"1d_y"`、`"1d_x"`、`"2d_xy"` | 集群多节点并行求解同一网格 |
| **多进程独立** | `"independent"` | `"multi_grid"` | 参数扫描、无通信批处理 |
| **嵌套多重网格** | `"multigrid"` | — | AMR 风格细化（框架就绪，算子待实现）|

**XY 块分解的统一视角**：

```
一维 Y 切片 = block（nx_blocks=1, ny_blocks=nprocs）
一维 X 切片 = block（nx_blocks=nprocs, ny_blocks=1）
二维 XY 块  = block（nx_blocks=px, ny_blocks=py, px*py=nprocs）
```

---

## 2. TOML 配置快速参考

### 2.1 OpenMP 线程数设置

```toml
[parallel]
omp_num_threads = 8   # 强制使用 8 个 OpenMP 线程（0 = 使用系统默认值）
```

### 2.2 一维 Y 切片（nx_blocks=1，ny_blocks=nprocs）

```toml
[mpi]
mode      = "block"   # 新名称（旧 "1d_y" 仍可用但会提示弃用）
nx_blocks = 1         # X 方向 1 块（不分割 X）
ny_blocks = 0         # 0 = 自动：ny_blocks = nprocs（Y 方向均匀切片）
# mpirun -n 4 → ny_blocks 自动为 4
```

### 2.3 一维 X 切片（nx_blocks=nprocs，ny_blocks=1）

```toml
[mpi]
mode      = "block"
nx_blocks = 4   # 需 mpirun -n 4
ny_blocks = 1
```

### 2.4 二维 XY 块分解（4×2 = 8 进程）

```toml
[mpi]
mode      = "block"
nx_blocks = 4   # X 方向切 4 块
ny_blocks = 2   # Y 方向切 2 块（须满足 4×2==8 == mpirun -n 8）
```

### 2.5 多进程独立模式

```toml
[mpi]
mode = "independent"  # 旧 "multi_grid" 仍可用
```

### 2.6 嵌套多重网格模式

```toml
[mpi]
mode = "multigrid"
# 嵌套关系通过代码（LbmMgTree API）配置
# 当前版本：退化为独立模式；MgTree 框架可单独使用
```

### 2.7 三维预留（nz_blocks）

```toml
[mpi]
mode      = "block"
nx_blocks = 2
ny_blocks = 2
nz_blocks = 2   # 3D 扩展预留，需配合 3D 求解器（D3Q19/D3Q27）
```

---

## 3. 模式一：XY 块分解（`"block"`）

### 3.1 统一视角：一维切片是二维块分解的特例

```
px=1, py=nprocs → 1D Y 切片（列不分割，行均匀切分）
px=nprocs, py=1 → 1D X 切片（行不分割，列均匀切分）
px*py=nprocs     → 2D XY 块分解（两方向均切分）
```

底层统一使用 `MpiDecomp2D` 实现。`px=1` 时，无西/东幽灵列，行交换等价于旧的一维 `MpiDecomp` 模式。

### 3.2 进程布局与坐标

```
进程布局（rank = row_rank * px + col_rank）：

  row=py-1  ┌─────┬─────┬─────┐  ← 全局北壁（row_rank==py-1 注册 North BC）
            │ r=6 │ r=7 │ r=8 │    示例：px=3, py=3
  row=1     ├─────┼─────┼─────┤
            │ r=3 │ r=4 │ r=5 │
  row=0     └─────┴─────┴─────┘  ← 全局南壁（row_rank==0 注册 South BC）
            col=0  col=1  col=2
            ↑                 ↑
          西壁BC             东壁BC
```

均匀分配策略（`uniform_partition()`）：

```
local_n = total/nprocs + (r < total%nprocs ? 1 : 0)
start   = r * (total/nprocs) + min(r, total%nprocs)
```

### 3.3 幽灵层布局

含幽灵层的本地网格（col_rank=1, row_rank=1，四面均有邻居）：

```
  ┌─────────────────────────────────────────┐
  │ 北幽灵行 j=local_ny+phys_y0             │ ← 来自 rank_north 的 j=phys_y0 行
  ├──┬───────────────────────────────────┬──┤
  │西│ 物理区域                          │东│
  │幽│ i=phys_x0 .. phys_x0+local_nx-1  │幽│
  │灵│ j=phys_y0 .. phys_y0+local_ny-1  │灵│
  │列│                                   │列│
  ├──┴───────────────────────────────────┴──┤
  │ 南幽灵行 j=0                            │ ← 来自 rank_south 的 j=local_ny 行
  └─────────────────────────────────────────┘
```

- `phys_x0 = has_west_ghost() ? 1 : 0`（1D Y 切片时 phys_x0=0，无西幽灵）
- `phys_y0 = has_south_ghost() ? 1 : 0`

### 3.4 边界条件约定

| 方向 | 注册 BC 的条件 | 检查方法 |
|------|---------------|---------|
| South | `row_rank == 0` | `decomp.has_south_wall()` |
| North | `row_rank == py-1` | `decomp.has_north_wall()` |
| West | `col_rank == 0` | `decomp.has_west_wall()` |
| East | `col_rank == px-1` | `decomp.has_east_wall()` |

1D Y 切片特例（px=1）：col_rank=0 ≡ 有西壁且有东壁（所有 rank 均注册 West/East BC）。

### 3.5 时序图

每个 `Solver::step()` 内部的执行顺序：

```
collide()                ← BGK/MRT 碰撞（跳过所有幽灵节点）
  ↓
stream()                 ← 流式迁移（本地，周期取模）
  ↓
halo_exchange_d2q9_2d()  ← MPI_Sendrecv：先南北后东西
  │  px=1 时跳过东西方向 Sendrecv（无幽灵列）
  ↓
apply_BC()               ← 各 rank 只施加自己持有的物理壁面 BC
  ↓
compute_macroscopic()    ← 更新 ρ/u
```

### 3.6 C++ 实现：`MpiDecomp2D::create()`

**文件**：`core/src/lbm/mpi_decomp.cpp`

```cpp
// 二维块分解（px=1 → 1D Y；py=1 → 1D X；px*py=nprocs → 2D）
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

    uniform_partition(gnx, d.px, d.col_rank, d.local_nx, d.x_start);
    uniform_partition(gny, d.py, d.row_rank, d.local_ny, d.y_start);
    d.x_end = d.x_start + d.local_nx - 1;
    d.y_end = d.y_start + d.local_ny - 1;

    // 四邻进程（MPI_PROC_NULL 对边界 rank 合法）
    d.rank_south = (d.row_rank > 0)        ? (d.row_rank-1)*d.px + d.col_rank : MPI_PROC_NULL;
    d.rank_north = (d.row_rank < d.py-1)   ? (d.row_rank+1)*d.px + d.col_rank : MPI_PROC_NULL;
    d.rank_west  = (d.col_rank > 0)        ? d.row_rank*d.px + d.col_rank-1   : MPI_PROC_NULL;
    d.rank_east  = (d.col_rank < d.px-1)   ? d.row_rank*d.px + d.col_rank+1   : MPI_PROC_NULL;
    return d;
}
```

### 3.7 幽灵层交换：`halo_exchange_d2q9_2d()`

交换分两步：

**步骤 1（南北，行数据连续）**：
```cpp
// 向北邻发送顶物理行，从南邻接收南幽灵行
MPI_Sendrecv(top_phys, row_size, MPI_DOUBLE, decomp.rank_north, 10,
             south_ghost, row_size, MPI_DOUBLE, decomp.rank_south, 10, ...);
// 向南邻发送底物理行，从北邻接收北幽灵行
MPI_Sendrecv(bot_phys, row_size, MPI_DOUBLE, decomp.rank_south, 11,
             north_ghost, row_size, MPI_DOUBLE, decomp.rank_north, 11, ...);
```

**步骤 2（东西，列数据不连续，需打包）**：
```cpp
// 打包最西/东物理列到 send_west / send_east（for j 循环）
// 向西邻发最西列，从东邻收东幽灵列
MPI_Sendrecv(send_west, col_size, MPI_DOUBLE, decomp.rank_west, 20,
             recv_east,  col_size, MPI_DOUBLE, decomp.rank_east, 20, ...);
// 向东邻发最东列，从西邻收西幽灵列
MPI_Sendrecv(send_east, col_size, MPI_DOUBLE, decomp.rank_east, 21,
             recv_west,  col_size, MPI_DOUBLE, decomp.rank_west, 21, ...);
// 解包 recv_west/recv_east 到幽灵列
```

**1D Y 切片优化**：`px=1` 时 `has_west_ghost()=false` 且 `has_east_ghost()=false`，步骤 2 中所有打包循环为空（gny × Q × 8B 额外内存带宽为零），等价于原有 1D 实现的性能。

### 3.8 三维预留：`MpiDecomp3D`

`MpiDecomp3D` 是为未来三维 LBM（D3Q19/D3Q27）预留的域分解框架：

```cpp
// 进程布局：rank = pz_rank*(px*py) + row_rank*px + col_rank
// 六邻进程：rank_west/east/south/north/bottom/top
// 六方向幽灵层：has_west/east/south/north/bottom/top_ghost()
// 退化关系：
//   pz=1       → 等价于 MpiDecomp2D
//   pz=1,py=1  → 等价于 1D X 切片
//   pz=1,px=1  → 等价于 1D Y 切片
static MpiDecomp3D create(int gnx, int gny, int gnz, int px, int py, int pz);
```

**当前状态**：创建/空间范围查询 API 就绪；幽灵层交换（6 方向）和对应的三维 LBM 求解器待未来实现。

### 3.9 块分解模式下的输出策略

#### 分区快照输出（默认）

每个 rank 将本地物理分区数据写入各自的子目录，互不干扰：

```
output/
├── rank_0/
│   ├── fluid_000500.npz    ← 含 x_start/y_start/global_nx/global_ny 元数据
│   └── monitor.csv
├── rank_1/
│   ├── fluid_000500.npz
│   └── monitor.csv
├── rank_2/
│   └── fluid_000500.npz
└── rank_3/
    └── fluid_000500.npz
```

NPZ 分区文件中额外存储了位置元数据，供后处理工具拼合全局场：

| 附加键名 | 类型 | 含义 |
|---------|------|------|
| `x_start` | int64 | 本分区在全局坐标系中的 X 起始位置 |
| `y_start` | int64 | 本分区在全局坐标系中的 Y 起始位置 |
| `global_nx` | int64 | 全局域 X 总节点数 |
| `global_ny` | int64 | 全局域 Y 总节点数 |

#### 计算中全局合并输出（`combine_blocks = true`）

在 TOML 中设置 `combine_blocks = true` 后，rank-0 在每次写快照时自动通过
`MPI_Gatherv` 收集全部分区数据，拼合为全局场后写出到 `<directory>/fluid_<NNNNNN>.<ext>`：

```toml
[output]
write_interval = 500
directory      = "output/my_run"
format         = "npz"           # 或 "tecplot_asc" / "tecplot_bin"
combine_blocks = true            # 默认 false；仅 block 模式有效
```

开启后输出目录结构变为：

```
output/my_run/
├── fluid_000500.npz    ← rank-0 写出的全局合并快照（含全域完整流场）
├── fluid_001000.npz
├── rank_0/
│   ├── fluid_000500.npz
│   └── fluid_001000.npz
├── rank_1/
│   ├── fluid_000500.npz
│   └── fluid_001000.npz
└── ...
```

若同时启用 `plot_interval`（需 `python-ffi` 特性），绘图使用全局合并场，
生成完整流域的速度幅值云图、涡量云图和流线图。

#### 计算完成后离线合并（`combine_block_snapshots`）

若计算时未开启 `combine_blocks = true`（或需要在事后转换格式），可使用
`python/examples/combine_blocks.py` 脚本或 Python API 进行离线合并（详见
docs/运行时实现详解.md §13.4）：

```bash
# 将 NPZ 分区文件合并为全局 .dat
python3 python/examples/combine_blocks.py \
    --dir output/my_run --fmt dat --out output/my_run/global
```

---

## 4. 模式二：多进程独立（`"independent"`）

### 4.1 使用场景

每个 MPI 进程运行**完全独立**的仿真（无任何 MPI 通信），适用于：

- **参数扫描**：同时运行 Re = 100/500/1000/5000 四种工况
- **集成测试**：并行验证不同边界条件组合
- **重启点生成**：各进程从不同初始条件出发，快速生成多组数据

```toml
[mpi]
mode = "independent"   # 旧名 "multi_grid" 仍可用
```

此模式下：
- 每个进程分配**完整**的 `nx × ny` 全局网格，无幽灵层
- `step()` 内无 MPI 通信
- 所有进程执行相同配置（若需区分，可通过 `mpi_rank()` 动态调整参数）

### 4.2 输出目录隔离

多进程独立模式下，每进程的输出写入独立子目录：

```
output/
├── rank_0/       ← rank 0 的快照
│   ├── fluid_000001.npz
│   └── monitor.csv
├── rank_1/
├── rank_2/
└── rank_3/
```

---

## 5. 模式三：嵌套式多重网格（`"multigrid"`）

### 5.1 概述与适用场景

"多重网格"（multigrid/AMR 风格）指细网格**嵌套**在粗网格上，可实现任意嵌套层数：

```
全局粗网格（level=0，256×256）
  └── 细化区域 A（level=1，中心 64×64，加密比 2）
        └── 更细区域 A1（level=2，中心 16×16，加密比 2）
  └── 细化区域 B（level=1，右下角 32×32，加密比 2）
```

**适用场景**：
- 在流场中的高梯度区域（如边界层、激波、涡）使用细网格
- 其他区域使用粗网格，降低全局计算量
- AMR（自适应网格细化）前处理/框架搭建

### 5.2 多重网格树数据结构（`MgTree` / `MgNode`）

**文件**：`core/include/lbm/mg_tree.hpp`，`core/src/lbm/mg_tree.cpp`

```
MgTree（N 叉树，管理所有节点的所有权）
│
├── root（MgNode，level=0，粗网格）
│   ├── extent: {0,255, 0,255, 0,0}  ← 全局范围
│   ├── grid:   *LatticeGrid          ← 绑定粗网格实例（可空）
│   └── children:
│       ├── MgNode（level=1，细化区域 A）
│       │   ├── extent: {96,159, 96,159, 0,0}
│       │   ├── refine_ratio: 2
│       │   └── children:
│       │       └── MgNode（level=2）
│       └── MgNode（level=1，细化区域 B）
│           ├── extent: {192,255, 0,63, 0,0}
│           └── refine_ratio: 2
```

### 5.3 树节点的空间范围（`MgExtent`）

```cpp
struct MgExtent {
    int x_start, x_end;   // 全局 X 格子坐标（0-based，含端点）
    int y_start, y_end;   // 全局 Y 格子坐标
    int z_start, z_end;   // 全局 Z 格子坐标（2D 时设 0，nz()=1）

    int nx() const { return x_end - x_start + 1; }
    int ny() const { return y_end - y_start + 1; }
    int nz() const { return (z_end > z_start) ? z_end - z_start + 1 : 1; }
    bool is_3d() const { return z_end > z_start; }
    bool contains(const MgExtent& other) const;  // 包含关系（验证子节点合法性）
    bool overlaps(const MgExtent& other) const;  // 重叠检测
};
```

`add_level()` 会在 `child_extent` 不在 `parent->extent` 内时抛出 `std::invalid_argument`，防止越界嵌套。

### 5.4 N 叉树结构与任意嵌套

**树的增长**（`add_level()`）：

```cpp
MgTree tree({0,255, 0,255, 0,0});           // 根节点（256×256 粗网格）

// 第一层：嵌套两个细化区域（中心+角落）
auto* c1a = tree.add_level(root, {96,159, 96,159, 0,0}, 2);   // 加密比 2
auto* c1b = tree.add_level(root, {192,255, 0,63, 0,0},  2);

// 第二层：在 c1a 内再嵌套
auto* c2  = tree.add_level(c1a, {112,143, 112,143, 0,0}, 2);

// 深度任意：
auto* c3  = tree.add_level(c2, {120,135, 120,135, 0,0}, 4);  // 加密比 4
```

**等价数据结构**：
- 2D 各向同性加密（refine_ratio=2）→ **四叉树**（quadtree）
- 3D 各向同性加密（refine_ratio=2）→ **八叉树**（octree）
- 一维方向加密 → **二叉树**（binary tree）
- 非均匀加密（不同方向不同 ratio）→ **N 叉树**

### 5.5 三维支持（`MgDim::D3`）

```cpp
MgTree tree_3d({0,63, 0,63, 0,63}, MgDim::D3);  // 64×64×64 粗网格

// 在粗网格中嵌套细化区域（z 方向有非零范围）
auto* fine_3d = tree_3d.add_level(root,
    {16,47, 16,47, 16,47},  // 三维嵌套区域
    2                        // 加密比（3D: 每格→2×2×2=8 个细格）
);

// volume_ratio() = refine_ratio^3 = 8（D3 模式）
```

### 5.6 与 MPI 域分解的关联

每个 `MgNode` 持有可选的 `decomp` 指针，支持**并行多重网格**：

```cpp
MgNode {
    // ...
    void* decomp = nullptr;  // 可绑定 MpiDecomp2D* 或 MpiDecomp3D*（按层独立分解）
};
```

**块分解与多重网格树的统一**：

```
全局粗网格（level=0）
  │ decomp = MpiDecomp2D::create(256, 256, 2, 2)  ← 粗层用 4 进程 2×2 分解
  └── 细化区域（level=1）
        │ decomp = MpiDecomp2D::create(64, 64, 2, 2)  ← 细层也用 4 进程（独立分解）
        └── 更细区域（level=2）
              │ decomp = nullptr  ← 最细层单进程运行
```

### 5.7 遍历接口

```cpp
// 粗→细（BFS）：先推进粗网格，再推进细网格
tree.traverse_coarse_to_fine([](MgNode* node) {
    if (node->has_grid()) {
        // 在该层执行 LBM 步骤
        // node->grid->solver->step();  // 待 LBM-MG 算子实现后启用
    }
});

// 细→粗（BFS 逆序）：残差传递（restriction）
tree.traverse_fine_to_coarse([](MgNode* node) {
    // 将细层残差传递到粗层（待实现）
});

// 查询
int max_lv  = tree.max_level();    // 最大层级（根=0）
int n_nodes = tree.node_count();   // 总节点数
auto lvl1   = tree.nodes_at_level(1);  // 所有 level=1 节点
```

### 5.8 当前实现状态与扩展路线

| 功能 | 状态 | 说明 |
|------|------|------|
| `MgTree` / `MgNode` 数据结构 | ✅ 就绪 | N 叉树，支持 2D/3D，任意嵌套 |
| `add_level()` / `traverse_*()` | ✅ 就绪 | 含边界检查和 BFS 遍历 |
| `LbmMgTree` Rust 封装 | ✅ 就绪 | 完整 FFI + RAII |
| C ABI（`lbm_mg_tree_*`） | ✅ 就绪 | 创建/释放/节点查询 |
| `decomp` 字段绑定 | ✅ 接口就绪 | 调用方负责传入 `MpiDecomp2D*` |
| LBM 层间插值算子（prolongation） | 🔲 待实现 | 细→粗速度/分布函数插值 |
| LBM 层间限制算子（restriction） | 🔲 待实现 | 粗→细残差传递 |
| 时间步同步（粗/细层时间步之比） | 🔲 待实现 | 加密比 r 时细层时间步 = 1/r |
| 自动 AMR 细化判断（误差估计）| 🔲 待实现 | 基于局部梯度/涡量的自适应 |

---

## 6. MPI + OpenMP 混合并行

### 6.1 OpenMP 线程数设置

```toml
[parallel]
omp_num_threads = 8   # 每个 MPI 进程使用 8 个 OpenMP 线程
```

等价于 `omp_set_num_threads(8)`，优先级高于 `OMP_NUM_THREADS` 环境变量。

Rust API：`lbm_bindings::set_omp_num_threads(8)`
C API：`lbm_omp_set_num_threads(8)`

### 6.2 线程/进程分配策略

```
节点 A（rank 0..3）                节点 B（rank 4..7）
┌──────────────────────────┐       ┌──────────────────────────┐
│ rank 0 (8 OMP threads)  │←Halo→ │ rank 4 (8 OMP threads)  │
│ rank 1 (8 OMP threads)  │←Halo→ │ rank 5 (8 OMP threads)  │
│ rank 2 (8 OMP threads)  │←Halo→ │ rank 6 (8 OMP threads)  │
│ rank 3 (8 OMP threads)  │       │ rank 7 (8 OMP threads)  │
└──────────────────────────┘       └──────────────────────────┘
总并行度 = 8 进程 × 8 线程 = 64 核
```

**建议**：MPI 进程数 × OMP 线程数 ≤ 总物理核心数；每 NUMA 域分配 1 个 MPI 进程 + 域内核心数个 OMP 线程以最大化 NUMA 亲和性。

---

## 7. Rust / C API 使用示例

### 7.1 XY 块分解示例（1D Y 切片 ≡ nx_blocks=1）

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

    let global_nx = 256i32;
    let global_ny = 256i32;

    // 1D Y 切片 = 2D 块分解（px=1, py=nprocs）
    let mut decomp2d = LbmMpiDecomp2D::new(global_nx, global_ny, 1, nprocs)
        .expect("分解创建失败");

    let local_nx = decomp2d.grid_nx();  // = global_nx（px=1 时无西/东幽灵）
    let local_ny = decomp2d.grid_ny();  // = local_ny + 南/北幽灵
    let y_start  = decomp2d.y_start();

    println!("rank={rank}: y_start={y_start}, local grid={local_nx}×{local_ny}");

    let mut grid   = LbmGrid::new(local_nx, local_ny, 1, LatticeModel::D2Q9);
    let mut solver = LbmSolver::new(&mut grid, 1.6, CollisionModel::Bgk);

    // 绑定 2D 分解（px=1 时 halo_exchange_d2q9_2d 自动跳过东西方向）
    solver.attach_mpi2d(Some(&mut decomp2d));

    // 注册边界条件（只注册本进程持有的物理壁面）
    if decomp2d.y_start() == 0 {
        solver.add_boundary_condition(BcType::BounceBack, Face::South, 0.0, 0.0, 0.0, 1.0);
    }
    if decomp2d.y_start() + local_ny == global_ny {
        solver.add_boundary_condition(BcType::ZouHeVelocity, Face::North, 0.1, 0.0, 0.0, 0.0);
    }
    // px=1：所有 rank 拥有西/东壁
    solver.add_boundary_condition(BcType::BounceBack, Face::West, 0.0, 0.0, 0.0, 1.0);
    solver.add_boundary_condition(BcType::BounceBack, Face::East, 0.0, 0.0, 0.0, 1.0);

    for _ in 0..5000 { solver.step(&mut grid); }
    mpi_finalize();
}
```

### 7.2 二维块分解示例

```rust
// 2D 块分解：px=4, py=2 → 需要 mpirun -n 8
let (global_nx, global_ny, px, py) = (512i32, 512i32, 4i32, 2i32);
assert_eq!(px * py, nprocs, "px*py must equal nprocs");

let mut decomp2d = LbmMpiDecomp2D::new(global_nx, global_ny, px, py)
    .expect("分解创建失败");

let local_nx = decomp2d.grid_nx();
let local_ny = decomp2d.grid_ny();
let x_start  = decomp2d.x_start();
let y_start  = decomp2d.y_start();

// 注册各 rank 拥有的物理壁面
if y_start == 0 {
    solver.add_boundary_condition(BcType::BounceBack, Face::South, ...);
}
if x_start == 0 {
    solver.add_boundary_condition(BcType::BounceBack, Face::West, ...);
}
// ... East, North 类似
```

### 7.3 多进程独立模式示例

```rust
mpi_init();
let rank = mpi_rank();

// 每个进程持有完整网格，无幽灵层，无通信
let mut grid   = LbmGrid::new(256, 256, 1, LatticeModel::D2Q9);
let mut solver = LbmSolver::new(&mut grid, 1.6, CollisionModel::Bgk);
// 也可通过 rank 设置不同 omega（不同 Re）：
// let omega = [1.0/(3.0*0.01+0.5), 1.0/(3.0*0.005+0.5), ...][rank as usize];
for _ in 0..5000 { solver.step(&mut grid); }
// 输出写入 output/rank_<rank>/（由 orchestrator 自动处理）
mpi_finalize();
```

### 7.4 多重网格树创建示例

```rust
use lbm_bindings::LbmMgTree;

// 创建 2D 双层嵌套（256×256 粗网格 + 中心 64×64 细网格）
let mut tree = LbmMgTree::new(0, 255, 0, 255, 0, 0, false)
    .expect("MgTree creation failed");

// 在粗网格中嵌套细网格（中心区域，加密比 2）
let fine_node = tree.add_level_from_root(96, 159, 96, 159, 0, 0, 2)
    .expect("add_level failed");

println!("最大层级: {}", tree.max_level());   // 1
println!("节点总数: {}", tree.node_count());  // 2

// 再嵌套第二层（加密比 4）
let finer_node = tree.add_level(fine_node, 112, 143, 112, 143, 0, 0, 4)
    .expect("add_level failed");
println!("最大层级: {}", tree.max_level());   // 2
println!("节点总数: {}", tree.node_count());  // 3

// 3D 示例：
let mut tree_3d = LbmMgTree::new(0, 63, 0, 63, 0, 63, true)
    .expect("3D MgTree creation failed");
let fine_3d = tree_3d.add_level_from_root(16, 47, 16, 47, 16, 47, 2)
    .expect("add_level failed");
```

### 7.5 C ABI 函数速查表

**MPI 块分解接口**

| C 函数 | 对应 Rust 函数 | 说明 |
|--------|---------------|------|
| `lbm_mpi_init()` | `mpi_init()` | 初始化 MPI |
| `lbm_mpi_finalize()` | `mpi_finalize()` | 结束 MPI |
| `lbm_mpi_rank()` | `mpi_rank()` | 当前进程编号 |
| `lbm_mpi_size()` | `mpi_size()` | 进程总数 |
| `lbm_mpi_decomp2d_new(gnx,gny,px,py)` | `LbmMpiDecomp2D::new(...)` | 创建 2D 块分解（1D 用 px=1 或 py=1）|
| `lbm_mpi_decomp2d_free(h)` | `Drop for LbmMpiDecomp2D` | 释放 |
| `lbm_solver_attach_mpi2d(s,h)` | `solver.attach_mpi2d(d)` | 绑定 2D 分解 |
| `lbm_mpi_decomp2d_grid_nx(h)` | `d.grid_nx()` | 含幽灵层 nx |
| `lbm_mpi_decomp2d_grid_ny(h)` | `d.grid_ny()` | 含幽灵层 ny |
| `lbm_mpi_decomp2d_x_start(h)` | `d.x_start()` | 全局 X 起始 |
| `lbm_mpi_decomp2d_y_start(h)` | `d.y_start()` | 全局 Y 起始 |
| `lbm_mpi_decomp3d_new(gnx,gny,gnz,px,py,pz)` | `LbmMpiDecomp3D::new(...)` | 创建 3D 块分解（预留）|
| `lbm_mpi_decomp3d_grid_nz(h)` | `d.grid_nz()` | 含幽灵层 nz（3D）|
| `lbm_mpi_decomp3d_z_start(h)` | `d.z_start()` | 全局 Z 起始（3D）|

**多重网格树接口**

| C 函数 | 对应 Rust 函数 | 说明 |
|--------|---------------|------|
| `lbm_mg_tree_new(x0,x1,y0,y1,z0,z1,is_3d)` | `LbmMgTree::new(...)` | 创建树（2D/3D）|
| `lbm_mg_tree_free(h)` | `Drop for LbmMgTree` | 释放树及所有节点 |
| `lbm_mg_tree_add_level(tree,parent,x0,x1,y0,y1,z0,z1,r)` | `tree.add_level(parent,...)` | 添加细化子区域 |
| `lbm_mg_tree_root(h)` | `tree.root()` | 获取根节点 |
| `lbm_mg_tree_max_level(h)` | `tree.max_level()` | 最大层级 |
| `lbm_mg_tree_node_count(h)` | `tree.node_count()` | 节点总数 |
| `lbm_mg_node_set_grid(node,grid)` | — | 绑定 LatticeGrid |
| `lbm_mg_node_level(node)` | — | 节点层级 |
| `lbm_mg_node_refine_ratio(node)` | — | 加密比 |
| `lbm_mg_node_child_count(node)` | — | 子节点数 |

**OpenMP 接口**

| C 函数 | 对应 Rust 函数 | 说明 |
|--------|---------------|------|
| `lbm_omp_set_num_threads(n)` | `set_omp_num_threads(n)` | 设置 OMP 线程数 |

---

## 8. 构建与运行

### 8.1 编译开关

| 环境变量（Cargo）| CMake 参数 | 说明 |
|-----------------|-----------|------|
| `LBM_ENABLE_MPI=ON` | `-DENABLE_MPI=ON` | 启用 MPI（默认 OFF） |
| `LBM_ENABLE_OPENMP=ON` | `-DENABLE_OPENMP=ON` | 启用 OpenMP（默认 OFF） |
| `LBM_ENABLE_CUDA=ON` | `-DENABLE_CUDA=ON` | 启用 CUDA GPU 后端 |

### 8.2 跨平台构建命令

**Linux / macOS**：
```bash
# MPI + OpenMP 混合
LBM_ENABLE_MPI=ON LBM_ENABLE_OPENMP=ON cargo build --release
```

**Windows PowerShell**：
```powershell
$env:LBM_ENABLE_MPI = "ON"; $env:LBM_ENABLE_OPENMP = "ON"
cargo build --release
```

### 8.3 运行命令

**一维 Y 切片（4 进程，nx_blocks=1 自动）**：
```bash
# configs/my.toml: [mpi] mode="block" nx_blocks=1 ny_blocks=0
mpirun -n 4 ./target/release/lbm-orchestrator configs/my.toml
```

**二维 XY 块（4×2 = 8 进程）**：
```bash
# [mpi] mode="block" nx_blocks=4 ny_blocks=2
mpirun -n 8 ./target/release/lbm-orchestrator configs/my.toml
```

**多进程独立（4 套独立仿真）**：
```bash
# [mpi] mode="independent"
mpirun -n 4 ./target/release/lbm-orchestrator configs/my.toml
# 输出: output/rank_0/ output/rank_1/ output/rank_2/ output/rank_3/
```

**Windows（MS-MPI）**：
```powershell
mpiexec -n 4 .\target\release\lbm-orchestrator.exe configs\my.toml
```

### 8.4 SLURM 集群脚本示例

**2D 块分解 + OpenMP 混合（2 节点，每节点 4 进程 × 4 线程）**：
```bash
#!/bin/bash
#SBATCH --nodes=2 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=4
export OMP_PROC_BIND=close; export OMP_PLACES=cores
# TOML: [mpi] mode="block" nx_blocks=4 ny_blocks=2
#        [parallel] omp_num_threads=4
mpirun -n 8 ./target/release/lbm-orchestrator configs/hybrid.toml
```

---

## 9. 正确性保证与常见陷阱

### 9.1 幽灵节点不执行二次碰撞

幽灵节点持有相邻进程已完成碰撞的数据；若再次碰撞会引入"二次松弛"误差。`collide_bgk()` / `collide_mrt()` 通过 ix/iy 检查跳过所有幽灵节点。

### 9.2 边界条件只在拥有物理壁的 rank 注册

```rust
if decomp2d.y_start() == 0 {              // row_rank==0 → 南壁
    solver.add_boundary_condition(BcType::BounceBack, Face::South, ...);
}
if decomp2d.x_start() == 0 {              // col_rank==0 → 西壁
    solver.add_boundary_condition(BcType::BounceBack, Face::West, ...);
}
// 类似地检查 North、East
```

### 9.3 nx_blocks × ny_blocks 必须等于进程数

若检测到不匹配，orchestrator 会打印警告并自动退化为 1D Y 切片（nx_blocks=1, ny_blocks=nprocs）：
```
[warn] mpi.nx_blocks(4) * mpi.ny_blocks(2) = 8 ≠ nprocs(4).
       Falling back to 1D Y slice (nx_blocks=1, ny_blocks=nprocs).
```

### 9.4 MgTree 子节点范围必须在父节点内

`add_level()` 在 `child_extent` 不是 `parent->extent` 子集时抛出 `std::invalid_argument`。确保坐标正确，2D 时 `z_start=z_end=0`。

---

## 10. 性能分析与调优建议

### 通信量对比

| 模式 | 每步通信量（D2Q9，N² 节点，P 进程）| 说明 |
|------|-----------------------------------|------|
| 1D Y（px=1, py=P）| 2 × nx × 9 × 8 B × P | 仅南北 2 次 Sendrecv |
| 1D X（px=P, py=1）| 2 × ny × 9 × 8 B × P | 仅东西 2 次（含打包开销）|
| 2D XY（px×py=P）| 2 × (lnx+lny) × 9 × 8 B × P | 4 次 Sendrecv，东西需打包 |

**2D 优势**：方形网格（nx≈ny）下，每进程本地面积 O(N²/P)，通信周长 O(N/√P)，通信计算比 O(1/√P) 远优于 1D 的 O(1/P^{1/2})。

### 选择建议

- **nx << ny**：选 1D Y（py=P, px=1）
- **ny << nx**：选 1D X（px=P, py=1）
- **nx ≈ ny**：选 2D XY（px≈py≈√P）
- **无需通信，参数扫描**：选 independent
- **局部高精度需求**：使用 MgTree 搭建多层嵌套框架

---

*对应代码版本：`core/include/lbm/mpi_decomp.hpp`，`core/include/lbm/mg_tree.hpp`，`core/src/lbm/mpi_decomp.cpp`，`core/src/lbm/mg_tree.cpp`，`core/src/capi/lbm_capi.cpp`，`bindings/src/lib.rs`，`orchestrator/src/config.rs`，`orchestrator/src/main.rs`。*
