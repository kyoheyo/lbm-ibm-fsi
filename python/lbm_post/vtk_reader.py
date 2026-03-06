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
    "combine_block_snapshots",
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
# MPI 块分解结果合并工具（计算后离线拼合）
# ---------------------------------------------------------------------------

def combine_block_snapshots(
    output_dir: "str | Path",
    fmt: str = "npz",
    out_dir: "str | Path | None" = None,
) -> "list[Path]":
    """
    将 MPI 块分解模式下离散输出的各 rank 分区快照拼合为全局完整流场文件。

    求解器以 ``mpi.mode = "block"`` 运行时，每个 MPI rank 将本地物理分区数据
    写入 ``<output_dir>/rank_<N>/fluid_<NNNNNN>.<ext>``。本函数扫描所有
    ``rank_*`` 子目录，按时间步号对齐后，依据每个分区文件中记录的
    ``x_start / y_start / global_nx / global_ny`` 元数据将各分区拼合到同一
    全局数组，并将结果写出到 ``<out_dir>/fluid_<NNNNNN>.<ext>``。

    参数
    ----
    output_dir : 包含 ``rank_0/``、``rank_1/`` … 子目录的基础输出目录
    fmt        : 输出格式 — ``"npz"``（默认）、``"dat"``（ASCII Tecplot）、
                 ``"plt"``（二进制 Tecplot TDV112）
    out_dir    : 合并文件的写出目录；缺省时写到 ``<output_dir>/combined/``

    返回
    ----
    已写出的合并文件路径列表（按时间步升序排列）。

    异常
    ----
    FileNotFoundError : ``output_dir`` 不存在或其中没有任何 ``rank_*`` 子目录
    ValueError        : 分区文件缺少 ``x_start / y_start / global_nx / global_ny``
                        元数据（须以 NPZ 格式输出；.dat/.plt 分区文件暂不含元数据）

    示例
    ----
    >>> from lbm_post.vtk_reader import combine_block_snapshots
    >>> written = combine_block_snapshots("output/my_run", fmt="npz")
    >>> print(written[0])
    output/my_run/combined/fluid_001000.npz

    注意
    ----
    当前版本仅支持以 NPZ 格式输出的分区文件（因为 .npz 文件内嵌了分区元数据）。
    若使用 ``tecplot_asc`` / ``tecplot_bin`` 格式输出，请改用 ``combine_blocks=true``
    配置项在计算过程中由求解器直接写出全局合并文件。
    """
    base = Path(output_dir)
    if not base.exists():
        raise FileNotFoundError(f"输出目录不存在：{base}")

    if fmt not in {"npz", "dat", "plt"}:
        raise ValueError(f"不支持的输出格式 {fmt!r}，可选：'npz'、'dat'、'plt'")
    if fmt in {"dat", "plt"}:
        raise ValueError(
            f"格式 {fmt!r} 的分区文件不包含位置元数据（x_start/y_start），"
            "无法自动拼合。请在 TOML 中设置 combine_blocks=true 以在计算时"
            "由求解器直接写出全局合并文件。"
        )

    # 扫描所有 rank_* 子目录
    rank_dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.startswith("rank_")],
        key=lambda d: int(d.name.split("_", 1)[1]),
    )
    if not rank_dirs:
        raise FileNotFoundError(
            f"目录 {base} 中没有找到任何 rank_* 子目录。"
            f"请确认求解器使用了 mpi.mode=\"block\" 并成功运行。"
        )

    dst_dir = Path(out_dir) if out_dir is not None else base / "combined"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # 收集所有 rank 的文件列表，按步号索引
    from collections import defaultdict
    step_to_files: "dict[int, list[Path]]" = defaultdict(list)
    for rd in rank_dirs:
        for p in sorted(rd.glob(f"fluid_*.{fmt}"),
                        key=lambda x: int(re.search(r"(\d+)", x.stem).group(1))):
            step_no = int(re.search(r"(\d+)", p.stem).group(1))
            step_to_files[step_no].append(p)

    if not step_to_files:
        raise FileNotFoundError(
            f"在 {base}/rank_*/ 目录中未找到任何 fluid_*.{fmt} 文件。"
        )

    written: list[Path] = []

    for step_no in sorted(step_to_files.keys()):
        files = step_to_files[step_no]
        if not files:
            continue

        # 读第一个分区以获取全局尺寸和时间
        first_data = np.load(files[0])
        if "global_nx" not in first_data or "global_ny" not in first_data:
            raise ValueError(
                f"文件 {files[0]} 中缺少 global_nx/global_ny 元数据。"
                "请确认求解器以 NPZ 格式输出分区数据，且每个 .npz 文件包含"
                "x_start / y_start / global_nx / global_ny 字段。"
            )
        gnx  = int(first_data["global_nx"])
        gny  = int(first_data["global_ny"])
        step_val = int(first_data.get("step", step_no))
        time_val = float(first_data.get("time", 0.0))

        # 初始化全局数组（覆盖顺序不影响结果，因为各分区不重叠）
        g_rho = np.zeros((gny, gnx), dtype=np.float64)
        g_ux  = np.zeros((gny, gnx), dtype=np.float64)
        g_uy  = np.zeros((gny, gnx), dtype=np.float64)

        for fpath in files:
            d = np.load(fpath)
            if "x_start" not in d or "y_start" not in d:
                raise ValueError(
                    f"文件 {fpath} 中缺少 x_start/y_start 元数据，"
                    "无法确定该分区在全局坐标系中的位置。"
                )
            xs  = int(d["x_start"])
            ys  = int(d["y_start"])
            rho = d["rho"]   # shape (local_ny, local_nx)
            ux  = d["ux"]
            uy  = d["uy"]
            local_ny, local_nx = rho.shape
            # 将分区放置到全局数组中
            g_rho[ys:ys + local_ny, xs:xs + local_nx] = rho
            g_ux [ys:ys + local_ny, xs:xs + local_nx] = ux
            g_uy [ys:ys + local_ny, xs:xs + local_nx] = uy

        # 写出合并文件
        out_path = dst_dir / f"fluid_{step_no:06d}.npz"
        np.savez_compressed(
            out_path,
            rho=g_rho, ux=g_ux, uy=g_uy,
            step=np.array(step_val),
            time=np.array(time_val),
        )
        written.append(out_path)

    return written


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
