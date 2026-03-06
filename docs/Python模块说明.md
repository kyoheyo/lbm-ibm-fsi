# Python 模块说明文档

## 目录

1. [概述](#一概述)
2. [安装与依赖](#二安装与依赖)
3. [包目录结构](#三包目录结构)
4. [lbm_pre — 预处理包](#四lbm_pre--预处理包)
   - 4.1 lbm_pre.config_gen — TOML 配置生成
   - 4.2 lbm_pre.geometry — IBM 标记点生成
   - 4.3 lbm_pre.mesh — Gmsh 网格生成
   - 4.4 lbm_pre.bridge — Rust FFI 接口
5. [lbm_post — 后处理包](#五lbm_post--后处理包)
   - 5.1 lbm_post.vtk_reader — 快照文件读取
   - 5.2 lbm_post.plot — 可视化绘图
   - 5.3 lbm_post.analysis — 定量后处理
   - 5.4 lbm_post.bridge — Rust FFI 接口
6. [典型工作流](#六典型工作流)
7. [与 Rust 主控的集成方式](#七与-rust-主控的集成方式)
8. [数据布局约定](#八数据布局约定)

---

## 一、概述

Python 层分为两个子包：

| 子包 | 职责 |
|------|------|
| `lbm_pre` | 前处理：生成 TOML 配置文件、Lagrangian IBM 标记点、Gmsh 网格；提供 Rust FFI 入口 |
| `lbm_post` | 后处理：读取求解器输出文件（NPZ / Tecplot）、流场可视化、定量分析；提供 Rust FFI 入口 |

两个子包可以**独立于 Rust 主控**单独在纯 Python 脚本中使用，也可以通过 Rust `pyo3` 绑定在仿真过程中被主控程序实时调用。

---

## 二、安装与依赖

### 2.1 Python 版本要求

Python ≥ 3.11（使用了 `tomllib` 标准库和 `X | Y` 类型联合语法）。

### 2.2 依赖包

| 包 | 最低版本 | 用途 |
|----|----------|------|
| `numpy` | ≥ 1.24 | 所有数组运算 |
| `matplotlib` | ≥ 3.7 | 流场可视化 |
| `scipy` | ≥ 1.10 | 高级数值处理（保留接口） |
| `gmsh` | ≥ 4.11 | 网格生成（`lbm_pre.mesh` 使用，可选） |
| `pytest` | ≥ 7.4 | 单元测试（开发依赖） |

> `gmsh` 为**可选依赖**——若未安装，`lbm_pre.mesh.MeshBuilder.build()` 会抛出 `ImportError`，其他模块不受影响。

### 2.3 安装

```bash
# 以可编辑模式安装（开发时推荐）
pip install -e "python/[test]"

# 或只安装运行时依赖
pip install -e python/
```

---

## 三、包目录结构

```
python/
├── pyproject.toml          # 打包配置（setuptools）
│
├── lbm_pre/                # 预处理子包
│   ├── __init__.py
│   ├── config_gen.py       # TOML 配置生成器
│   ├── geometry.py         # IBM 标记点生成
│   ├── mesh.py             # Gmsh 背景网格生成
│   └── bridge.py           # Rust pyo3 FFI 接口
│
├── lbm_post/               # 后处理子包
│   ├── __init__.py
│   ├── vtk_reader.py       # 快照文件读取器（NPZ / Tecplot）
│   ├── plot.py             # Matplotlib 可视化
│   ├── analysis.py         # 定量后处理（误差、积分量等）
│   └── bridge.py           # Rust pyo3 FFI 接口
│
├── examples/
│   ├── lid_driven_cavity.py    # 顶盖驱动方腔端到端示例
│   └── plot_solver_output.py   # 通用后处理命令行脚本
│
└── tests/                  # pytest 测试套件
    ├── conftest.py
    ├── test_config_gen.py
    ├── test_geometry.py
    ├── test_mesh.py
    ├── test_vtk_reader.py
    ├── test_plot.py
    ├── test_analysis.py
    └── test_bridge.py
```

---

## 四、lbm_pre — 预处理包

### 4.1 `lbm_pre.config_gen` — TOML 配置生成

**模块职责**：以 Python 数据类（dataclass）的形式构造求解器配置，并序列化为 Rust 主控可读取的 TOML 文件。主要用途包括：参数扫描自动化、从脚本批量生成工况、以及在 CI/CD 流水线中生成测试用例。

#### 4.1.1 辅助函数

```python
lbm_pre.config_gen.reynolds_to_nu(Re, U, L) -> float
```

将雷诺数转换为格子运动粘度：

$$\nu = \frac{U \cdot L}{Re}$$

LBM 稳定性要求 $\tau \in (0.5, 2)$，等价于 $\nu \in (0.05, 0.5)$。若 $\nu$ 超出此范围，函数会向 stderr 打印警告，但不会抛出异常（允许用户在极端参数下测试）。

| 参数 | 类型 | 说明 |
|------|------|------|
| `Re` | float | 雷诺数 |
| `U` | float | 参考速度（格子单位） |
| `L` | float | 参考长度（格点数） |

```python
lbm_pre.config_gen.load_config(path) -> SolverConfig
```

从已有 TOML 文件反序列化为 `SolverConfig` 对象（使用标准库 `tomllib`）。

#### 4.1.2 数据类 `BoundaryConditionConfig`

```python
@dataclass
class BoundaryConditionConfig:
    bc_type : str        # 边界条件类型（见下表）
    face    : str        # 所在面："west"|"east"|"south"|"north"|"bottom"|"top"
    ux      : float = 0.0
    uy      : float = 0.0
    uz      : float = 0.0
    rho     : float = 1.0
```

`bc_type` 取值：

| 取值 | 类型 | 说明 |
|------|------|------|
| `"bounce_back"` | **固壁 BC** | 半步长反弹，无滑移，2 阶精度（推荐） |
| `"bounce_back_full_way"` | **固壁 BC** | 全步长反弹，无滑移，1 阶精度 |
| `"zou_he_velocity"` | 流体进/出口 BC | Zou-He 速度 BC，规定 ux/uy |
| `"zou_he_pressure"` | 流体进/出口 BC | Zou-He 压力 BC，规定 rho |
| `"fully_developed"` | 流体进/出口 BC | 充分发展出口（零法向梯度） |
| `"free_outlet"` | 流体进/出口 BC | 自由出口（等价于 `fully_developed`） |
| `"guo_extrapolation"` | 流体进/出口 BC | 郭照立非平衡外推格式（Guo 2002） |

#### 4.1.3 数据类 `SimConfig`

```python
@dataclass
class SimConfig:
    n_steps        : int   = 10000
    dt             : float = 1.0
    lattice_model  : str   = "D2Q9"    # "D2Q9" | "D3Q19" | "D3Q27"
    collision_model: str   = "MRT"     # "BGK" | "MRT"
```

#### 4.1.4 数据类 `FluidConfig`

```python
@dataclass
class FluidConfig:
    nx   : int   = 100
    ny   : int   = 100
    nz   : int   = 1
    nu   : float = 0.01    # 格子运动粘度（建议 0.05–0.5）
    rho0 : float = 1.0     # 初始均匀密度
    boundary_conditions: list[BoundaryConditionConfig] = []
```

#### 4.1.5 数据类 `StructureConfig`

```python
@dataclass
class StructureConfig:
    young_modulus : float = 1.0e3     # 杨氏模量 E（格子单位）
    second_moment : float = 1.0e-6   # 截面二次矩 I
    density       : float = 1.5      # 结构密度（相对于流体）
    area          : float = 0.01     # 横截面积
    length        : float = 0.3      # 弹性梁长度（格点数）
    n_elements    : int   = 10       # 有限元单元数
```

#### 4.1.6 数据类 `IbmConfig`

```python
@dataclass
class IbmConfig:
    geometry    : str   = "filament"  # "circle" | "filament"
    x0          : float = 50.0        # 圆心 x（circle）或起点 x（filament）
    y0          : float = 50.0        # 圆心 y 或起点 y
    size        : float = 10.0        # 半径（circle）或长度（filament）
    n_markers   : int   = 64          # Lagrangian 标记点数量
    delta_kernel: str   = "four_point"# "two_point" | "four_point"
```

#### 4.1.7 数据类 `OutputConfig`

```python
@dataclass
class OutputConfig:
    write_interval: int = 500
    directory     : str = "output"
    format        : str = "npz"   # "npz" | "tecplot_asc" | "tecplot_bin"
```

#### 4.1.8 顶层数据类 `SolverConfig`

```python
@dataclass
class SolverConfig:
    simulation: SimConfig
    fluid     : FluidConfig
    structure : Optional[StructureConfig] = None   # FSI 时设置
    ibm       : Optional[IbmConfig]      = None   # IBM 时设置
    output    : OutputConfig = OutputConfig()
```

**方法 `SolverConfig.write(path, *, comment="")`**

将配置序列化为 TOML 文件并写入磁盘。

```python
cfg = SolverConfig.lid_driven_cavity(nx=128, ny=128, Re=400)
path = cfg.write("configs/lid400.toml",
                 comment="Re=400 顶盖驱动方腔，用于 Ghia 基准对比")
```

**预置工厂方法**：

| 方法 | 说明 | 关键参数 |
|------|------|----------|
| `lid_driven_cavity(nx, ny, Re, U_lid)` | 顶盖驱动方腔流 | Re=100，U_lid=0.1 |
| `flow_around_cylinder(nx, ny, Re, U_inlet)` | 绕圆柱流（含 IBM 浸入边界）| Re=100，U_inlet=0.05 |
| `pressure_driven_channel(nx, ny, nu, rho_in, rho_out)` | 压力驱动通道流（Poiseuille 流）| rho_in=1.005，rho_out=1.0 |
| `velocity_inlet_channel(nx, ny, Re, U_inlet)` | 速度入口 + 充分发展出口通道流 | Re=30，U_inlet=0.05 |
| `velocity_inlet_free_outlet(nx, ny, Re, U_inlet)` | 速度入口 + 压力出口 + 上下壁自由出口 | Re=80，U_inlet=0.05 |
| `fsi_filament(nx, ny, Re, U_inlet)` | 速度入口 + 浸入柔性丝状体 FSI | Re=200，U_inlet=0.05 |

**示例：**

```python
from lbm_pre.config_gen import SolverConfig, reynolds_to_nu

# 方法一：工厂方法快速生成
cfg = SolverConfig.lid_driven_cavity(nx=100, ny=100, Re=100.0)
cfg.write("output/lid100.toml")

# 方法二：手动构造精细控制
from lbm_pre.config_gen import (
    SimConfig, FluidConfig, BoundaryConditionConfig, OutputConfig
)
cfg = SolverConfig(
    simulation=SimConfig(n_steps=50000, collision_model="MRT"),
    fluid=FluidConfig(
        nx=200, ny=80, nu=reynolds_to_nu(200, 0.05, 80),
        boundary_conditions=[
            BoundaryConditionConfig("zou_he_velocity", "west", ux=0.05),
            BoundaryConditionConfig("zou_he_pressure", "east", rho=1.0),
            BoundaryConditionConfig("bounce_back", "south"),
            BoundaryConditionConfig("bounce_back", "north"),
        ],
    ),
    output=OutputConfig(write_interval=200, directory="output/channel"),
)
cfg.write("configs/channel.toml")

# 方法三：从已有 TOML 文件读取后修改
from lbm_pre.config_gen import load_config
cfg = load_config("configs/channel.toml")
cfg.simulation.n_steps = 100000
cfg.write("configs/channel_long.toml")
```

---

### 4.2 `lbm_pre.geometry` — IBM 标记点生成

**模块职责**：在格子空间中生成 Lagrangian IBM 标记点（位置坐标 + 弧长权重），支持圆形、椭圆、直线丝状体、三次 Bézier 曲线等几何形状。生成的标记点可写入 CSV 文件供求解器读取，也可直接通过 FFI 传递给 Rust。

所有坐标使用**格子单位**（Δx = 1）。

#### 4.2.1 数据类 `MarkerArray`

```python
@dataclass
class MarkerArray:
    x  : np.ndarray   # 标记点 x 坐标，形状 (N,)
    y  : np.ndarray   # 标记点 y 坐标，形状 (N,)
    ds : np.ndarray   # 弧长权重，形状 (N,)（IBM 力展布积分元）
    tag: str = ""     # 可选标签
```

**方法**：

| 方法 | 返回值 | 说明 |
|------|--------|------|
| `len(m)` | int | 标记点数量 N |
| `m.to_xy()` | `(N, 2)` ndarray | 合并 x/y 为二维数组 |
| `m.centroid()` | `(float, float)` | 平均位置 (x̄, ȳ) |

#### 4.2.2 形状生成函数

**`circle_markers(cx, cy, radius, n=64, *, tag="circle")`**

在圆周上均匀生成 n 个标记点，弧长元 ds = 2πr / n。

```python
from lbm_pre.geometry import circle_markers

m = circle_markers(cx=50.0, cy=40.0, radius=8.0, n=100)
print(m.x.shape)        # (100,)
print(m.ds[0])          # 2π×8/100 ≈ 0.503
```

**`ellipse_markers(cx, cy, a, b, n=64, *, tag="ellipse")`**

在椭圆周上按弧长均匀分布 n 个标记点（自适应角度采样，使 ds 近似相等）。

```python
m = ellipse_markers(cx=50, cy=50, a=15, b=8, n=80)
```

**`filament_markers(x0, y0, x1, y1, n=32, *, tag="filament")`**

在直线段 (x0, y0) → (x1, y1) 上均匀分布 n 个标记点。

```python
m = filament_markers(x0=25.0, y0=20.0, x1=25.0, y1=60.0, n=40)
```

**`bezier_markers(control_points, n=64, *, tag="bezier")`**

按弧长均匀分布在三次 Bézier 曲线上；`control_points` 必须恰好包含 4 个 (x, y) 控制点。

```python
m = bezier_markers([(10, 20), (20, 60), (60, 60), (70, 20)], n=60)
```

#### 4.2.3 I/O 函数

**`write_markers_csv(markers, path)`**

将标记点写入 CSV 文件（可供 Rust 主控的 `[ibm] csv_path` 字段读取）：

```
# tag: circle
x,y,ds
41.995,50.000,0.503
42.482,50.498,0.503
...
```

**`read_markers_csv(path)`**

从上述格式的 CSV 文件读回 `MarkerArray`。

**示例：**

```python
from lbm_pre.geometry import circle_markers, write_markers_csv

m = circle_markers(50, 40, 8, n=128, tag="cylinder")
write_markers_csv(m, "configs/cylinder_markers.csv")
```

---

### 4.3 `lbm_pre.mesh` — Gmsh 网格生成

**模块职责**：提供基于 Gmsh 的背景网格生成器，用于预可视化计算域轮廓、为 IBM 标记点播种表面节点、以及与体拟合网格（body-fitted mesh）的混合耦合场景。LBM 求解器本身使用笛卡尔欧拉网格，网格文件不直接输入求解器，而用于可视化和前处理辅助。

> **注意**：`lbm_pre.mesh` 仅在安装了 `gmsh` 包时才能使用。若未安装，模块可以正常导入，但调用 `MeshBuilder.build()` 时会抛出 `ImportError`。

#### 4.3.1 数据类 `MeshInfo`

```python
@dataclass
class MeshInfo:
    n_nodes        : int
    n_elements     : int
    lc_min         : float            # 最小特征网格尺寸
    lc_max         : float            # 最大特征网格尺寸
    dim            : int              # 网格维度（2 或 3）
    physical_groups: dict[str, int]   # 物理组名称 → Gmsh tag
```

#### 4.3.2 类 `MeshBuilder`

链式调用构建器，用于生成带有可选障碍物（圆形或矩形孔洞）的矩形 LBM 计算域背景网格。

**构造函数**

```python
MeshBuilder(nx, ny, nz=1, *, lx=None, ly=None, lz=None)
```

| 参数 | 说明 |
|------|------|
| `nx, ny, nz` | 格点分辨率 |
| `lx, ly, lz` | 物理域尺寸（默认等于格点数，即 Δx = 1） |

**链式方法**

| 方法 | 说明 |
|------|------|
| `set_mesh_size(lc)` | 设置全局特征网格尺寸 |
| `add_circle_obstacle(cx, cy, radius, mesh_size, tag)` | 添加圆形障碍物（孔洞） |
| `add_rectangle_obstacle(x0, y0, width, height, mesh_size, tag)` | 添加矩形障碍物（孔洞） |
| `build()` | 构建 Gmsh 模型（不写盘） |
| `write(path)` | 将网格写入 `.msh` 文件 |
| `info()` | 返回 `MeshInfo`（需先调用 `build()`） |
| `export_surface_nodes(tag)` | 提取物理组边界节点坐标（返回 (N, 3) ndarray） |

`MeshBuilder` 支持作为上下文管理器使用（`with` 语句），在离开时自动调用 `gmsh.finalize()`：

```python
with MeshBuilder(200, 80) \
        .set_mesh_size(2.0) \
        .add_circle_obstacle(cx=50, cy=40, radius=8, mesh_size=1.0) as mb:
    mb.build()
    mb.write("domain.msh")
    info = mb.info()
    print(f"节点数：{info.n_nodes}，单元数：{info.n_elements}")
    # 从圆形障碍物边界提取表面节点作为 IBM 标记点种子
    nodes = mb.export_surface_nodes("circle")
```

**辅助函数 `write_mesh_info(info, path=None)`**

将 `MeshInfo` 输出到控制台或写入 JSON 文件（当 `path` 不为 None 时）。

---

### 4.4 `lbm_pre.bridge` — Rust FFI 接口

**模块职责**：提供被 Rust `pyo3` 调用的函数入口点，参数类型均为 Python 基础类型（int/float/list），无需在 Rust 侧处理高级数据类。

#### `geometry_markers_raw(geometry, x0, y0, size, n_markers)`

生成 IBM Lagrangian 标记点，返回三个 Python list。

| 参数 | 类型 | 说明 |
|------|------|------|
| `geometry` | str | `"circle"` 或 `"filament"` |
| `x0, y0` | float | 圆心（circle）或起点（filament），格子单位 |
| `size` | float | 半径（circle）或长度（filament） |
| `n_markers` | int | 标记点数量 |

返回：`(x: list[float], y: list[float], ds: list[float])`

```python
# 纯 Python 调用示例
from lbm_pre.bridge import geometry_markers_raw

x, y, ds = geometry_markers_raw("circle", 50.0, 40.0, 8.0, 128)
print(len(x))   # 128
```

Rust 侧调用方式（`pyo3`）：
```rust
let (x, y, ds): (Vec<f64>, Vec<f64>, Vec<f64>) =
    py_module.call1("geometry_markers_raw",
                    ("circle", 50.0, 40.0, 8.0, 128_i64))?
             .extract()?;
```

---

## 五、lbm_post — 后处理包

### 5.1 `lbm_post.vtk_reader` — 快照文件读取

**模块职责**：将求解器输出的三种格式（NPZ / ASCII Tecplot / 二进制 Tecplot）统一读入标准化的 Python 数据对象，供绘图和分析函数使用。

#### 5.1.1 数据类 `FieldSnapshot`

封装单时间步的欧拉场数据：

```python
@dataclass
class FieldSnapshot:
    step   : int              # 时间步索引
    time   : float            # 物理时间（= step × dt）
    nx, ny : int              # 网格分辨率
    rho    : np.ndarray       # 密度场，形状 (ny, nx)
    ux     : np.ndarray       # x 方向速度，形状 (ny, nx)
    uy     : np.ndarray       # y 方向速度，形状 (ny, nx)
    force_x: Optional[ndarray]  # IBM 体力 x（可为 None）
    force_y: Optional[ndarray]  # IBM 体力 y（可为 None）
```

**派生量方法**：

| 方法 | 返回形状 | 说明 |
|------|----------|------|
| `velocity_magnitude()` | `(ny, nx)` | $|u| = \sqrt{u_x^2 + u_y^2}$ |
| `vorticity()` | `(ny, nx)` | $\omega_z = \partial u_y/\partial x - \partial u_x/\partial y$（二阶中心差分） |
| `pressure(cs2=1/3)` | `(ny, nx)` | $p = c_s^2 \cdot \rho$ |
| `stream_function()` | `(ny, nx)` | 沿 y 方向积分 $u_x$ 近似得到流函数 $\psi$ |

#### 5.1.2 数据类 `MarkerSnapshot`

封装单时间步的 Lagrangian IBM 标记点数据：

```python
@dataclass
class MarkerSnapshot:
    step  : int
    x, y  : np.ndarray          # 标记点位置，形状 (N,)
    fx, fy: Optional[ndarray]   # IBM 力密度（可为 None）
    ux, uy: Optional[ndarray]   # 插值流体速度（可为 None）
```

#### 5.1.3 `NpzReader`（默认格式，推荐）

读取 Rust 求解器写出的 `.npz` 归档文件。

```python
class NpzReader:
    def __init__(self, directory: str | Path)
    def __len__(self) -> int
    def __iter__(self) -> Iterator[FieldSnapshot]
    def steps(self) -> list[int]
    def read(self, path_or_step: str | Path | int) -> FieldSnapshot
    def last(self) -> FieldSnapshot
```

`.npz` 文件内数组键：

| 键 | 形状 | 类型 | 说明 |
|----|------|------|------|
| `rho` | `(ny, nx)` | float64 | 密度场 |
| `ux` | `(ny, nx)` | float64 | x 速度 |
| `uy` | `(ny, nx)` | float64 | y 速度 |
| `step` | 标量 | int64 | 时间步 |
| `time` | 标量 | float64 | 物理时间 |
| `force_x`, `force_y` | `(ny, nx)` | float64 | IBM 体力（可选） |

```python
from lbm_post.vtk_reader import NpzReader

reader = NpzReader("output/lid_cavity")
print(f"共 {len(reader)} 个快照，时间步：{reader.steps()}")

last = reader.last()
print(last.rho.shape)        # (100, 100)

for snap in reader:          # 遍历所有快照
    print(snap.step, snap.velocity_magnitude().max())
```

#### 5.1.4 `TecplotAscReader`

读取 ASCII Tecplot POINT 格式 `.dat` 文件（`format = "tecplot_asc"`）。

```python
class TecplotAscReader:
    def __init__(self, directory: str | Path)
    def steps(self) -> list[int]
    def read(self, path_or_step) -> FieldSnapshot
    def last(self) -> FieldSnapshot
```

#### 5.1.5 `TecplotBinReader`

读取二进制 Tecplot TDV112 格式 `.plt` 文件（`format = "tecplot_bin"`）。

```python
class TecplotBinReader:
    def __init__(self, directory: str | Path)
    def steps(self) -> list[int]
    def read(self, path_or_step) -> FieldSnapshot
    def last(self) -> FieldSnapshot
```

#### 5.1.6 自动格式检测 `load_snapshot`

```python
load_snapshot(path: str | Path) -> FieldSnapshot
```

根据文件后缀（`.npz` / `.dat` / `.plt`）自动选择读取器，返回 `FieldSnapshot`。

```python
from lbm_post.vtk_reader import load_snapshot

snap = load_snapshot("output/fluid_001000.plt")  # 自动识别为二进制 Tecplot
```

#### 5.1.7 测试辅助函数

模块还提供两个仅供测试和示例使用的辅助函数（不属于正式 API）：

- `make_synthetic_lid_cavity(nx, ny, U_lid, step)` — 生成顶盖驱动方腔的合成流场快照
- `save_snapshot_npz(snap, path)` — 将 `FieldSnapshot` 保存为 `.npz` 文件

---

### 5.2 `lbm_post.plot` — 可视化绘图

**模块职责**：基于 Matplotlib 提供标准化的流场可视化函数，所有函数返回 `(fig, ax)` 元组，供调用方进一步自定义或保存。内部使用非交互后端（`Agg`），可在无显示器的 HPC 环境中运行。

所有绘图函数的第一个参数均为 `FieldSnapshot`（或 Sequence），关键字参数均有合理默认值。

#### 5.2.1 流场云图

**`plot_velocity_magnitude(snap, *, n_levels=64, cmap="viridis", add_colorbar=True, figsize=(8,6))`**

速度幅值 $|u|$ 填充等值线图。

**`plot_pressure(snap, *, n_levels=64, cmap="coolwarm", add_colorbar=True, figsize=(8,6))`**

压力 $p = c_s^2 \cdot \rho$ 填充等值线图，使用对称双斜率归一化。

**`plot_vorticity(snap, *, n_levels=64, cmap="RdBu_r", symmetric=True, add_colorbar=True, figsize=(8,6))`**

z 方向涡量 $\omega_z$ 填充等值线图，`symmetric=True` 时以 0 为中心对称着色。

**`plot_rho(snap, *, n_levels=64, cmap="plasma", add_colorbar=True, figsize=(8,6))`**

密度场 $\rho$ 填充等值线图。

**`plot_velocity_vectors(snap, *, every=5, scale=None, cmap="viridis", background="magnitude", figsize=(8,6))`**

速度矢量箭头图，叠加背景云图。

| `background` 取值 | 背景场 |
|-------------------|--------|
| `"magnitude"` | 速度幅值（默认） |
| `"pressure"` | 压力 |
| `"vorticity"` | 涡量 |
| `"none"` | 无背景 |

`every` 控制采样间隔（减少箭头密度），`scale=None` 时自动缩放。

**`plot_streamlines(snap, *, density=1.5, cmap="viridis", linewidth_scale=1.0, figsize=(8,6))`**

积分流线图，流线颜色和宽度均按速度幅值着色/缩放。

#### 5.2.2 IBM 与结构绘图

**`plot_markers(snaps_fluid, snaps_markers, *, background="vorticity", marker_color="red", marker_size=3.0, figsize=(8,6))`**

在流场背景上叠加 Lagrangian IBM 标记点散点图。

**`plot_beam_deformation(x, y, y_ref=None, *, step=None, figsize=(8,4))`**

绘制柔性梁变形曲线，可选同时绘制未变形参考构型（灰色虚线）。

#### 5.2.3 时序与收敛图

**`plot_convergence(steps, values, *, label="residual", xlabel="Time step", ylabel=None, log_scale=True, figsize=(8,4))`**

绘制标量量（残差、阻力、升力……）随时间步的变化曲线，`log_scale=True` 时使用对数 y 轴。

**`plot_velocity_profile(snap, *, x_slices=None, y_slices=None, component="ux", figsize=(8,5))`**

在指定格线位置绘制一维速度剖面，可同时叠加多条剖面。

#### 5.2.4 保存辅助 `save_figure`

```python
save_figure(fig, path, *, dpi=150, tight=True) -> Path
```

以合理默认值将 `fig` 保存到 `path`，格式由后缀推断（`.png` / `.pdf` / `.svg` 等），若目录不存在则自动创建。保存后自动调用 `plt.close(fig)` 释放内存。

**典型绘图示例：**

```python
from lbm_post.vtk_reader import NpzReader
from lbm_post.plot import (
    plot_velocity_magnitude, plot_streamlines,
    plot_vorticity, plot_velocity_profile, save_figure
)

snap = NpzReader("output/lid_cavity").last()

# 速度幅值云图
fig, ax = plot_velocity_magnitude(snap, cmap="hot", n_levels=128)
ax.set_title("Re=400 顶盖驱动方腔")
save_figure(fig, "figures/velocity.png", dpi=200)

# 流线图
fig, _ = plot_streamlines(snap, density=2.0)
save_figure(fig, "figures/streamlines.png")

# 中心线速度剖面（x = nx/2 处）
fig, _ = plot_velocity_profile(snap, x_slices=[snap.nx // 2], component="ux")
save_figure(fig, "figures/ux_profile.png")
```

---

### 5.3 `lbm_post.analysis` — 定量后处理

**模块职责**：提供定量分析函数，包括 IBM 阻力/升力系数计算、误差范数、收敛阶估计、单点时序监测和体积平均流动统计量等。

#### 5.3.1 数据容器

**`ForceSummary`**

```python
@dataclass
class ForceSummary:
    step    : int
    fx_total: float    # 体上合力 x 分量
    fy_total: float    # 体上合力 y 分量
    cd      : Optional[float]   # 阻力系数 Cd
    cl      : Optional[float]   # 升力系数 Cl
```

**`BulkQuantities`**

```python
@dataclass
class BulkQuantities:
    step      : int
    ke        : float   # 体积平均动能  E = 0.5 · mean(|u|²)
    enstrophy : float   # 体积平均拟能  Z = 0.5 · mean(ωz²)
    rho_mean  : float   # 体积平均密度
    rho_std   : float   # 密度标准差（近不可压时应接近 0）
    div_max   : float   # max|∇·u|（不可压残差）
```

#### 5.3.2 力与系数

**`drag_lift_coefficients(markers, *, rho_ref=1.0, U_ref=0.1, D_ref=1.0)`**

从 IBM 标记点力密度计算阻力和升力系数：

$$C_d = \frac{F_x}{0.5\,\rho_{ref}\,U_{ref}^2\,D_{ref}}, \quad C_l = \frac{F_y}{0.5\,\rho_{ref}\,U_{ref}^2\,D_{ref}}$$

若 `markers.fx is None`，返回的 `fx_total`、`fy_total`、`cd`、`cl` 均为 `nan`。

```python
from lbm_post.analysis import drag_lift_coefficients
from lbm_post.vtk_reader import NpzReader

# 假设求解器已输出标记点快照（在同一目录下以 markers_NNNNNN.npz 存储）
markers = ...   # 加载 MarkerSnapshot
fs = drag_lift_coefficients(markers, rho_ref=1.0, U_ref=0.05, D_ref=16)
print(f"Cd = {fs.cd:.4f},  Cl = {fs.cl:.4f}")
```

#### 5.3.3 微分量

**`compute_divergence(snap)`**

计算速度散度 $\nabla \cdot \mathbf{u} = \partial u_x/\partial x + \partial u_y/\partial y$，返回形状 `(ny, nx)` 的数组。对于不可压 LBM，散度应接近 0（量级为 Ma²）。

**`compute_q_criterion(snap)`**

计算二维 Q 准则（涡核识别）：

$$Q = \tfrac{1}{2}(|\mathbf{\Omega}|^2 - |\mathbf{S}|^2)$$

其中 $\mathbf{\Omega}$ 为反对称旋转率张量，$\mathbf{S}$ 为对称应变率张量。$Q > 0$ 的区域为旋转主导区域（涡核）。

#### 5.3.4 单点监测

**`monitor_point(snapshots, xi, yj, field="ux")`**

从快照序列中提取单个格点 `(xi, yj)` 处的时序数据。

| `field` 取值 | 说明 |
|--------------|------|
| `"ux"` | x 速度（默认） |
| `"uy"` | y 速度 |
| `"rho"` | 密度 |
| `"magnitude"` | 速度幅值 |
| `"vorticity"` | 涡量 |

返回 `(steps: ndarray[int], values: ndarray[float])`。

```python
from lbm_post.analysis import monitor_point
from lbm_post.plot import plot_convergence, save_figure

steps, ux_vals = monitor_point(snapshots, xi=50, yj=50, field="ux")
fig, _ = plot_convergence(steps, ux_vals, label="ux at (50,50)", log_scale=False)
save_figure(fig, "figures/monitor.png")
```

#### 5.3.5 误差范数

**`l2_error(snap, ref, field="ux")`**

相对 L2 误差：$\|f - f_{ref}\|_2 / \|f_{ref}\|_2$

**`linf_error(snap, ref, field="ux")`**

相对 L∞ 误差：$\|f - f_{ref}\|_\infty / \|f_{ref}\|_\infty$

`field` 可为 `"ux"`, `"uy"`, `"rho"`, `"magnitude"`。

```python
from lbm_post.analysis import l2_error

coarse = NpzReader("output/coarse").last()
fine   = NpzReader("output/fine").last()
print(f"L2 error(ux) = {l2_error(coarse, fine, 'ux'):.4e}")
```

#### 5.3.6 收敛阶估计

**`convergence_rate(grid_spacings, errors)`**

对多个网格分辨率的误差做对数最小二乘拟合，估计空间收敛阶 $p$（$\epsilon \propto \Delta x^p$）。

```python
from lbm_post.analysis import convergence_rate

dx_list   = [1/16, 1/32, 1/64]
err_list  = [1.2e-3, 3.1e-4, 7.8e-5]
p = convergence_rate(dx_list, err_list)
print(f"收敛阶 p ≈ {p:.2f}")   # 应接近 2（半步长反弹为 2 阶精度）
```

#### 5.3.7 体积平均量

**`compute_bulk_quantities(snap)`**

一次调用计算全部体积平均统计量，返回 `BulkQuantities`。

```python
from lbm_post.analysis import compute_bulk_quantities

bq = compute_bulk_quantities(snap)
print(f"KE = {bq.ke:.6e},  Enstrophy = {bq.enstrophy:.6e}")
print(f"ρ̄ = {bq.rho_mean:.6f},  σ_ρ = {bq.rho_std:.2e}")
print(f"max|∇·u| = {bq.div_max:.2e}")
```

---

### 5.4 `lbm_post.bridge` — Rust FFI 接口

**模块职责**：提供被 Rust `pyo3` 调用的实时可视化入口，在不写 NPZ 文件的情况下直接从内存数组渲染 PNG 云图。

#### `plot_field_raw(rho, ux, uy, nx, ny, step, time, out_dir, field="velocity_magnitude")`

从行主序的平坦浮点数列表渲染流场云图并保存为 PNG。

| 参数 | 类型 | 说明 |
|------|------|------|
| `rho, ux, uy` | `Sequence[float]` | 长度 `nx × ny` 的行主序一维数组 |
| `nx, ny` | int | 网格分辨率 |
| `step` | int | 时间步（用于文件名和标题） |
| `time` | float | 物理时间 |
| `out_dir` | str | 输出目录（不存在时自动创建） |
| `field` | str | `"velocity_magnitude"` \| `"vorticity"` \| `"pressure"` \| `"streamlines"` |

输出文件：`<out_dir>/<field>_<NNNNNN>.png`

```python
# 纯 Python 调用示例
from lbm_post.bridge import plot_field_raw
import numpy as np

nx, ny = 100, 100
rho_flat = np.ones(nx * ny).tolist()
ux_flat  = (np.random.randn(nx * ny) * 0.01).tolist()
uy_flat  = (np.random.randn(nx * ny) * 0.01).tolist()

plot_field_raw(rho_flat, ux_flat, uy_flat, nx, ny,
               step=1000, time=1000.0,
               out_dir="output/live_plots",
               field="vorticity")
```

Rust 侧调用（通过 `pyo3`）：
```rust
let out: () = py_module
    .call1("plot_field_raw", (rho_vec, ux_vec, uy_vec,
                               nx, ny, step, time,
                               out_dir, "velocity_magnitude"))?
    .extract()?;
```

---

## 六、典型工作流

### 6.1 纯 Python 端到端示例（无 Rust）

```
configs/channel.toml ←── [lbm_pre.config_gen]
                              ↓
             Rust 求解器（或仿真器）运行
                              ↓
output/snapshots/*.npz ←── 求解器写出
                              ↓
      [lbm_post.vtk_reader] 读取快照
                              ↓
      [lbm_post.plot]     绘图 → figures/*.png
      [lbm_post.analysis] 计算 Cd, L2 error, 体积平均量
```

完整示例见 `python/examples/lid_driven_cavity.py`。

### 6.2 参数扫描

```python
from lbm_pre.config_gen import SolverConfig

for Re in [100, 200, 400, 1000]:
    cfg = SolverConfig.lid_driven_cavity(Re=Re)
    cfg.write(f"configs/lid_Re{Re}.toml")
```

### 6.3 通用后处理命令行脚本

```bash
cd python/
# 对 output/lid_cavity/ 目录下的所有快照绘图
python3 examples/plot_solver_output.py \
    --dir output/lid_cavity \
    --fmt npz \
    --field velocity_magnitude \
    --out figures/
```

支持的 `--fmt`：`npz`、`tecplot_asc`、`tecplot_bin`；支持的 `--field`：`velocity_magnitude`、`vorticity`、`pressure`、`streamlines`、`rho`。

---

## 七、与 Rust 主控的集成方式

Python 子包通过两种方式被 Rust 调用：

### 7.1 子进程模式（`[python]` TOML 节）

Rust 主控在仿真前/后通过 `std::process::Command` 启动 Python 解释器，执行配置文件中指定的脚本：

```toml
[python]
interpreter  = "python3"
pre_script   = "scripts/pre.py"
post_script  = "scripts/post.py"
pythonpath   = "python"
```

数据通过文件系统（TOML / NPZ）传递，模块间完全解耦。

### 7.2 FFI 直接调用（`pyo3`，`--features python-ffi`）

Rust 主控通过 `pyo3` 在同一进程内调用 Python 函数，零文件 I/O，适用于实时云图输出：

| Rust 调用 | Python 函数 |
|-----------|-------------|
| `lbm_pre.bridge.geometry_markers_raw(...)` | 生成 IBM 标记点，返回三个 `Vec<f64>` |
| `lbm_post.bridge.plot_field_raw(...)` | 实时渲染流场 PNG |

启用方式：

```bash
cargo build --features python-ffi
```

详细的 Rust ↔ Python FFI 调用链见 `docs/说明文档.md` §5.4。

---

## 八、数据布局约定

- 所有二维场数组均采用**行主序（row-major）**，形状为 `(ny, nx)`。
- 索引 `arr[j, i]` 对应格点 `(i, j)`，即 `i` 为 x 方向（列），`j` 为 y 方向（行）。
- `np.gradient(field, axis=1)` 计算 x 方向梯度，`axis=0` 计算 y 方向梯度。
- 所有坐标和速度均使用**格子单位**（Δx = Δt = 1）。

```
         j (y 方向)
         ↑
    ny-1 │  arr[ny-1, 0]  ···  arr[ny-1, nx-1]
         │
       0 │  arr[0, 0]     ···  arr[0, nx-1]
         └─────────────────────────────────→ i (x 方向)
              0                    nx-1
```

---

> **相关文档**
> - 项目整体架构：`docs/说明文档.md`
> - C++ 核心与 Rust 绑定：`docs/rust_cpp封装详解.md`
> - 构建系统详解：`docs/构建系统详解.md`
> - 运行时实现详解：`docs/运行时实现详解.md`
> - 示例代码：`python/examples/lid_driven_cavity.py`
