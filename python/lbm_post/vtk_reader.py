"""
lbm_post.vtk_reader
===================
读取 LBM+IBM+FSI 求解器生成的快照文件。

求解器按时间步写出欧拉流场（密度、速度）和拉格朗日 IBM 标记点位置。
本模块将这些文件读入 NumPy 数组，供 :mod:`lbm_post.plot` 和
:mod:`lbm_post.analysis` 使用。

支持三种格式
-----------
* ``.npz``        — NumPy 压缩归档（默认格式，Rust 求解器原生输出）
* ``.dat``        — ASCII Tecplot POINT 格式
* ``.plt``        — 二进制 Tecplot TDV112 格式

数据布局约定
-----------
所有二维场数组采用**行主序**，形状为 ``(ny, nx)``——
即索引 ``[j, i]`` 对应格点 ``(i, j)``（x 为第一坐标，y 为第二坐标）。

公开 API
-------
FieldSnapshot       — 封装单时间步欧拉场数据的数据类
MarkerSnapshot      — 封装单时间步拉格朗日数据的数据类
NpzReader           — 读取 Rust 求解器写出的 .npz 文件
TecplotAscReader    — 读取 ASCII Tecplot .dat 文件
TecplotBinReader    — 读取二进制 Tecplot .plt（TDV112）文件
load_snapshot()     — 自动检测格式并加载（支持 .npz / .dat / .plt）
"""

from __future__ import annotations

import re
import struct
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

__all__ = [
    "FieldSnapshot",
    "MarkerSnapshot",
    "NpzReader",
    "TecplotAscReader",
    "TecplotBinReader",
    "load_snapshot",
]


# ---------------------------------------------------------------------------
# 数据容器
# ---------------------------------------------------------------------------

@dataclass
class FieldSnapshot:
    """
    单时间步的欧拉流场数据。

    属性
    ----
    step    : 时间步索引
    time    : 物理时间
    nx, ny  : 网格分辨率
    rho     : 密度场，形状 (ny, nx)
    ux      : x 方向速度，形状 (ny, nx)
    uy      : y 方向速度，形状 (ny, nx)
    force_x : IBM 体力 x 分量，形状 (ny, nx)，可为 None
    force_y : IBM 体力 y 分量，形状 (ny, nx)，可为 None
    """
    step: int
    time: float
    nx: int
    ny: int
    rho: np.ndarray
    ux: np.ndarray
    uy: np.ndarray
    force_x: Optional[np.ndarray] = None
    force_y: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # 派生量
    # ------------------------------------------------------------------

    def velocity_magnitude(self) -> np.ndarray:
        """返回速度幅值 |u| = sqrt(ux² + uy²)，形状 (ny, nx)。"""
        return np.hypot(self.ux, self.uy)

    def vorticity(self) -> np.ndarray:
        """
        返回涡量 z 分量  ωz = ∂uy/∂x − ∂ux/∂y，
        采用二阶中心差分计算，形状 (ny, nx)。
        """
        duy_dx = np.gradient(self.uy, axis=1)   # ∂uy/∂x（x 为第 1 轴）
        dux_dy = np.gradient(self.ux, axis=0)   # ∂ux/∂y（y 为第 0 轴）
        return duy_dx - dux_dy

    def pressure(self, cs2: float = 1.0 / 3.0) -> np.ndarray:
        """
        返回 LBM 压力  p = cs² · ρ，形状 (ny, nx)。
        """
        return cs2 * self.rho

    def stream_function(self) -> np.ndarray:
        """
        通过沿 y 方向积分 ux 近似计算流函数 ψ。
        适用于封闭腔体流的流线绘制，形状 (ny, nx)。
        """
        psi = np.zeros_like(self.ux)
        # 沿 y 方向积分：ψ(i, j) = ψ(i, j-1) + ux(i, j) Δy
        for j in range(1, self.ny):
            psi[j, :] = psi[j - 1, :] + self.ux[j, :]
        return psi


@dataclass
class MarkerSnapshot:
    """
    单时间步的拉格朗日 IBM 标记点数据。

    属性
    ----
    step    : 时间步索引
    x, y    : 标记点位置（长度为 N 的一维数组）
    fx, fy  : 各标记点处的 IBM 力密度
    ux, uy  : 各标记点处插值得到的流体速度
    """
    step: int
    x: np.ndarray
    y: np.ndarray
    fx: Optional[np.ndarray] = None
    fy: Optional[np.ndarray] = None
    ux: Optional[np.ndarray] = None
    uy: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# NPZ 读取器（主要输出格式）
# ---------------------------------------------------------------------------

class NpzReader:
    """
    读取求解器写出的 ``.npz`` 归档文件。

    每个 ``.npz`` 文件内应包含以下数组：

    ``rho``   : (ny, nx) float64  密度场
    ``ux``    : (ny, nx) float64  x 方向速度
    ``uy``    : (ny, nx) float64  y 方向速度
    ``step``  : 标量 int           时间步索引
    ``time``  : 标量 float         物理时间
    可选：
    ``force_x``, ``force_y``  : (ny, nx) float64  IBM 体力

    文件命名规范：``<目录>/fluid_<NNNNNN>.npz``。
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._files: list[Path] = sorted(
            self.directory.glob("fluid_*.npz"),
            key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
        )

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self):
        for p in self._files:
            yield self.read(p)

    def steps(self) -> list[int]:
        """返回已有快照的时间步索引（升序）。"""
        return [
            int(re.search(r"(\d+)", p.stem).group(1))
            for p in self._files
        ]

    def read(self, path_or_step: "str | Path | int") -> FieldSnapshot:
        """
        加载一个快照。

        参数
        ----
        path_or_step : .npz 文件路径，或整数时间步索引
        """
        if isinstance(path_or_step, int):
            p = self.directory / f"fluid_{path_or_step:06d}.npz"
        else:
            p = Path(path_or_step)

        data = np.load(p)
        rho = data["rho"]
        ux  = data["ux"]
        uy  = data["uy"]
        ny, nx = rho.shape
        return FieldSnapshot(
            step=int(data.get("step", 0)),
            time=float(data.get("time", 0.0)),
            nx=nx,
            ny=ny,
            rho=rho,
            ux=ux,
            uy=uy,
            force_x=data["force_x"] if "force_x" in data else None,
            force_y=data["force_y"] if "force_y" in data else None,
        )

    def last(self) -> FieldSnapshot:
        """加载最后一个（时间步最大）快照。"""
        if not self._files:
            raise FileNotFoundError(f"目录 {self.directory} 中没有 fluid_*.npz 文件")
        return self.read(self._files[-1])


# ---------------------------------------------------------------------------
# ASCII Tecplot 读取器（.dat 文件）
# ---------------------------------------------------------------------------

class TecplotAscReader:
    """
    读取 Rust 求解器写出的 ASCII Tecplot POINT 格式文件（``.dat``）。

    期望的文件格式：

    .. code-block:: text

        TITLE = "LBM Flow Field step=000100 time=100.000"
        VARIABLES = "X" "Y" "RHO" "UX" "UY"
        ZONE T="fluid", I=100, J=100, K=1, DATAPACKING=POINT, SOLUTIONTIME=100.0
        0.5 0.5 1.00012345e+00 1.23456789e-03 4.56789012e-04
        1.5 0.5 ...
        ...

    文件命名规范：``<目录>/fluid_<NNNNNN>.dat``。
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._files: list[Path] = sorted(
            self.directory.glob("fluid_*.dat"),
            key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
        )

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self):
        for p in self._files:
            yield self.read(p)

    def steps(self) -> list[int]:
        """返回已有快照的时间步索引（升序）。"""
        return [
            int(re.search(r"(\d+)", p.stem).group(1))
            for p in self._files
        ]

    def read(self, path_or_step: "str | Path | int") -> FieldSnapshot:
        """
        加载一个 ASCII Tecplot 快照。

        参数
        ----
        path_or_step : .dat 文件路径，或整数时间步索引
        """
        if isinstance(path_or_step, int):
            p = self.directory / f"fluid_{path_or_step:06d}.dat"
        else:
            p = Path(path_or_step)

        with open(p, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # 解析标题行中的 step 和 time
        step_val = 0
        time_val = 0.0
        for line in lines:
            m = re.search(r"step=(\d+)", line)
            if m:
                step_val = int(m.group(1))
            m = re.search(r"time=([\d.]+)", line)
            if m:
                time_val = float(m.group(1))

        # 解析 Zone 行中的 I（nx）和 J（ny）
        nx, ny = 0, 0
        for line in lines:
            m = re.search(r"I=(\d+)", line)
            if m:
                nx = int(m.group(1))
            m = re.search(r"J=(\d+)", line)
            if m:
                ny = int(m.group(1))
        if nx == 0 or ny == 0:
            raise ValueError(f"无法从文件 {p} 中解析网格尺寸（I/J）")

        # 跳过头部行（TITLE + VARIABLES + ZONE），读取数据行
        # 头部行通常有 3 行，但也可能更多——以第一个纯数字行为准
        data_lines: list[str] = []
        for line in lines:
            # 数据行以数字（或负号）开头
            stripped = line.strip()
            if stripped and (stripped[0].isdigit() or stripped[0] == '-'):
                data_lines.append(stripped)

        if len(data_lines) != nx * ny:
            raise ValueError(
                f"文件 {p} 中数据行数 {len(data_lines)} 与网格大小 {nx}×{ny} 不符"
            )

        # 解析数字并重新排列为 (ny, nx) 数组
        # 每行格式：X Y RHO UX UY
        rho_flat = np.empty(nx * ny, dtype=np.float64)
        ux_flat  = np.empty(nx * ny, dtype=np.float64)
        uy_flat  = np.empty(nx * ny, dtype=np.float64)
        for idx, dl in enumerate(data_lines):
            vals = dl.split()
            # 列顺序：0=X, 1=Y, 2=RHO, 3=UX, 4=UY
            rho_flat[idx] = float(vals[2])
            ux_flat[idx]  = float(vals[3])
            uy_flat[idx]  = float(vals[4])

        return FieldSnapshot(
            step=step_val,
            time=time_val,
            nx=nx,
            ny=ny,
            rho=rho_flat.reshape(ny, nx),
            ux=ux_flat.reshape(ny, nx),
            uy=uy_flat.reshape(ny, nx),
        )

    def last(self) -> FieldSnapshot:
        """加载最后一个（时间步最大）快照。"""
        if not self._files:
            raise FileNotFoundError(f"目录 {self.directory} 中没有 fluid_*.dat 文件")
        return self.read(self._files[-1])


# ---------------------------------------------------------------------------
# 二进制 Tecplot 读取器（.plt，TDV112 格式）
# ---------------------------------------------------------------------------

class TecplotBinReader:
    """
    读取 Rust 求解器写出的二进制 Tecplot PLT 文件（TDV112 格式）。

    TDV112 二进制格式结构（本实现仅支持本求解器产生的有序网格格式）：

    1. 魔数 ``#!TDV112``（8 字节）+ 字节序标志 i32=1
    2. 文件类型 i32（0 = 全场）
    3. 数据集标题（空字符终止 i32 序列）
    4. 变量数量 i32
    5. 变量名（各自空字符终止 i32 序列）
    6. Zone 头（标记 f32=299.0 + 元信息）
    7. EOH 标记（f32=357.0）
    8. Zone 数据区域（数据标记 f32=299.0 + 各变量 float64 数组）

    文件命名规范：``<目录>/fluid_<NNNNNN>.plt``。
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self._files: list[Path] = sorted(
            self.directory.glob("fluid_*.plt"),
            key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)),
        )

    def __len__(self) -> int:
        return len(self._files)

    def __iter__(self):
        for p in self._files:
            yield self.read(p)

    def steps(self) -> list[int]:
        """返回已有快照的时间步索引（升序）。"""
        return [
            int(re.search(r"(\d+)", p.stem).group(1))
            for p in self._files
        ]

    def read(self, path_or_step: "str | Path | int") -> FieldSnapshot:
        """
        加载一个二进制 Tecplot PLT 快照。

        参数
        ----
        path_or_step : .plt 文件路径，或整数时间步索引
        """
        if isinstance(path_or_step, int):
            p = self.directory / f"fluid_{path_or_step:06d}.plt"
        else:
            p = Path(path_or_step)

        with open(p, "rb") as f:
            raw = f.read()

        pos = 0

        def read_i32() -> int:
            nonlocal pos
            val = struct.unpack_from("<i", raw, pos)[0]
            pos += 4
            return val

        def read_f32() -> float:
            nonlocal pos
            val = struct.unpack_from("<f", raw, pos)[0]
            pos += 4
            return val

        def read_f64() -> float:
            nonlocal pos
            val = struct.unpack_from("<d", raw, pos)[0]
            pos += 8
            return val

        def read_tec_string() -> str:
            """读取空字符终止的 i32 序列字符串。"""
            chars = []
            while True:
                code = read_i32()
                if code == 0:
                    break
                chars.append(chr(code))
            return "".join(chars)

        # -----------------------------------------------------------------------
        # 1. 验证魔数
        # -----------------------------------------------------------------------
        magic = raw[pos:pos + 8]
        pos += 8
        if magic != b"#!TDV112":
            raise ValueError(f"文件 {p} 不是有效的 TDV112 二进制 Tecplot 文件（魔数错误）")

        # 字节序标志（应为 1）
        byte_order = read_i32()
        if byte_order != 1:
            raise ValueError(f"文件 {p} 使用大端字节序，本读取器仅支持小端（Intel）格式")

        # -----------------------------------------------------------------------
        # 2. 文件类型（0 = 全场，忽略）
        # -----------------------------------------------------------------------
        _file_type = read_i32()

        # -----------------------------------------------------------------------
        # 3. 数据集标题
        # -----------------------------------------------------------------------
        title = read_tec_string()
        # 从标题中提取 step 和 time
        step_val = 0
        time_val = 0.0
        m = re.search(r"step=(\d+)", title)
        if m:
            step_val = int(m.group(1))
        m = re.search(r"time=([\d.]+)", title)
        if m:
            time_val = float(m.group(1))

        # -----------------------------------------------------------------------
        # 4. 变量数量
        # -----------------------------------------------------------------------
        n_vars = read_i32()  # 本实现固定为 5：X, Y, RHO, UX, UY

        # -----------------------------------------------------------------------
        # 5. 变量名（逐一读取，忽略内容）
        # -----------------------------------------------------------------------
        var_names = [read_tec_string() for _ in range(n_vars)]

        # -----------------------------------------------------------------------
        # 6. Zone 头（标记 299.0 + 元信息）
        # -----------------------------------------------------------------------
        zone_marker = read_f32()
        if abs(zone_marker - 299.0) > 0.01:
            raise ValueError(f"文件 {p} 中 Zone 标记值错误（期望 299.0，实际 {zone_marker}）")

        _zone_name   = read_tec_string()
        _parent_zone = read_i32()
        _strand_id   = read_i32()
        time_val     = read_f64()  # 使用 Zone 中记录的时间（更精确）
        _zone_color  = read_i32()
        _zone_type   = read_i32()  # 0 = Ordered
        _var_location = read_i32()
        _raw_local   = read_i32()
        _face_neigh  = read_i32()
        nx = read_i32()
        ny = read_i32()
        _nz = read_i32()  # 二维时为 1
        _n_aux_data  = read_i32()

        # -----------------------------------------------------------------------
        # 7. EOH 标记（357.0）
        # -----------------------------------------------------------------------
        eoh = read_f32()
        if abs(eoh - 357.0) > 0.01:
            raise ValueError(f"文件 {p} 中 EOH 标记值错误（期望 357.0，实际 {eoh}）")

        # -----------------------------------------------------------------------
        # 8. 数据区域
        # -----------------------------------------------------------------------
        # 区域数据标记（299.0）
        data_marker = read_f32()
        if abs(data_marker - 299.0) > 0.01:
            raise ValueError(f"文件 {p} 中数据区域标记值错误（期望 299.0，实际 {data_marker}）")

        # 各变量格式（2 = float64）
        var_formats = [read_i32() for _ in range(n_vars)]

        _has_passive = read_i32()
        _has_sharing = read_i32()
        _share_zone  = read_i32()

        # 按变量顺序读取数据（X Y RHO UX UY）
        n = nx * ny
        vars_data: list[np.ndarray] = []
        for fmt in var_formats:
            if fmt == 2:  # float64
                arr = np.frombuffer(raw, dtype="<f8", count=n, offset=pos)
                pos += n * 8
            elif fmt == 1:  # float32
                arr = np.frombuffer(raw, dtype="<f4", count=n, offset=pos).astype(np.float64)
                pos += n * 4
            else:
                raise ValueError(f"文件 {p} 中变量格式 {fmt} 未知（支持 1=float32，2=float64）")
            vars_data.append(arr)

        # 按变量名确定 rho/ux/uy 的索引（X=0, Y=1, RHO=2, UX=3, UY=4）
        name_map = {name.upper(): i for i, name in enumerate(var_names)}
        rho_idx = name_map.get("RHO", 2)
        ux_idx  = name_map.get("UX",  3)
        uy_idx  = name_map.get("UY",  4)

        return FieldSnapshot(
            step=step_val,
            time=time_val,
            nx=nx,
            ny=ny,
            rho=vars_data[rho_idx].reshape(ny, nx),
            ux=vars_data[ux_idx].reshape(ny, nx),
            uy=vars_data[uy_idx].reshape(ny, nx),
        )

    def last(self) -> FieldSnapshot:
        """加载最后一个（时间步最大）快照。"""
        if not self._files:
            raise FileNotFoundError(f"目录 {self.directory} 中没有 fluid_*.plt 文件")
        return self.read(self._files[-1])


# ---------------------------------------------------------------------------
# VTU 读取器（XML 非结构化 VTK，保留向后兼容）
# ---------------------------------------------------------------------------

class VtkReader:
    """
    读取求解器生成的 VTK 矩形网格文件（.vtk）或 XML VTU 文件（.vtu）。

    需要 ``vtk`` Python 包或本模块内置的手写 XML 解析器
    （后者用于避免引入重型 VTK 依赖）。

    仅支持 ASCII VTK / inline-base64 VTU；二进制文件请安装 ``vtk`` 或 ``meshio``。
    """

    @staticmethod
    def read_vtu(path: str | Path) -> FieldSnapshot:
        """
        解析 VTU（XML VTK 非结构化网格）文件。

        支持名为 ``rho``、``ux``、``uy`` 的 ASCII 点数据数组。
        """
        p = Path(path)
        tree = ET.parse(p)
        root = tree.getroot()

        # 提取 Piece 节点范围
        piece = root.find(".//Piece")
        if piece is None:
            raise ValueError(f"文件 {p} 中没有 <Piece> 元素")

        nx_str = piece.get("NumberOfPoints", "0")
        n_total = int(nx_str)

        arrays: dict[str, np.ndarray] = {}
        for da in root.findall(".//PointData/DataArray"):
            name = da.get("Name", "")
            fmt  = da.get("format", "ascii").lower()
            n_comp = int(da.get("NumberOfComponents", "1"))
            if fmt == "ascii":
                vals = np.fromstring(da.text or "", sep=" ", dtype=np.float64)
            else:
                warnings.warn(f"VTK 二进制格式暂不完全支持，跳过变量 {name}")
                continue
            if n_comp > 1:
                vals = vals.reshape(-1, n_comp)
            arrays[name] = vals

        # 从 "rho"（1 分量）重建网格形状
        rho_flat = arrays.get("rho", np.ones(n_total))
        # 猜测接近正方形的网格
        nx = int(np.round(np.sqrt(n_total)))
        ny = n_total // nx

        def _reshape(arr: np.ndarray) -> np.ndarray:
            return arr[:ny * nx].reshape(ny, nx)

        ux_data = arrays.get("ux", np.zeros(n_total))
        uy_data = arrays.get("uy", np.zeros(n_total))

        step_match = re.search(r"_(\d+)\.vtu$", p.name)
        step = int(step_match.group(1)) if step_match else 0

        return FieldSnapshot(
            step=step, time=float(step), nx=nx, ny=ny,
            rho=_reshape(rho_flat),
            ux=_reshape(ux_data),
            uy=_reshape(uy_data),
        )


# ---------------------------------------------------------------------------
# 自动格式检测加载器
# ---------------------------------------------------------------------------

def load_snapshot(path: str | Path) -> FieldSnapshot:
    """
    从单个快照文件自动加载，支持 ``.npz``、``.dat`` 和 ``.plt`` 格式。

    参数
    ----
    path : 快照文件路径

    返回
    ----
    FieldSnapshot

    异常
    ----
    ValueError : 文件格式不受支持
    """
    p = Path(path)
    if p.suffix == ".npz":
        return NpzReader(p.parent).read(p)
    elif p.suffix == ".dat":
        return TecplotAscReader(p.parent).read(p)
    elif p.suffix == ".plt":
        return TecplotBinReader(p.parent).read(p)
    elif p.suffix in {".vtu", ".vtk"}:
        return VtkReader.read_vtu(p)
    else:
        raise ValueError(f"不支持的快照格式：{p.suffix}（支持 .npz / .dat / .plt / .vtu）")


# ---------------------------------------------------------------------------
# 合成数据生成器（用于测试，无需运行真实求解器）
# ---------------------------------------------------------------------------

def make_synthetic_lid_cavity(nx: int = 64, ny: int = 64,
                               U_lid: float = 0.1,
                               step: int = 10000) -> FieldSnapshot:
    """
    使用简单解析近似（线性剖面 + 抛物线修正）生成合成的顶盖驱动方腔快照，
    适用于在不运行求解器的情况下对后处理流水线进行单元测试。

    参数
    ----
    nx, ny  : 网格分辨率
    U_lid   : 盖板速度
    step    : 时间步索引
    """
    y_vals = np.linspace(0, 1, ny)
    x_vals = np.linspace(0, 1, nx)
    X, Y = np.meshgrid(x_vals, y_vals)

    # 简单解析近似：Stokes 流型抛物线剖面
    ux = U_lid * Y * (1 - Y) * 4        # y 方向抛物线分布
    uy = U_lid * X * (1 - X) * 4 * 0.1  # 小量横向速度

    rho = np.ones((ny, nx))

    return FieldSnapshot(
        step=step, time=float(step), nx=nx, ny=ny,
        rho=rho, ux=ux, uy=uy,
    )


def save_snapshot_npz(snap: FieldSnapshot, path: str | Path) -> Path:
    """
    将 :class:`FieldSnapshot` 保存为 ``.npz`` 文件
    （测试用途及 Rust 求解器推荐输出格式）。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "rho": snap.rho,
        "ux":  snap.ux,
        "uy":  snap.uy,
        "step": np.array(snap.step),
        "time": np.array(snap.time),
    }
    if snap.force_x is not None:
        arrays["force_x"] = snap.force_x
    if snap.force_y is not None:
        arrays["force_y"] = snap.force_y
    np.savez(p, **arrays)
    return p
