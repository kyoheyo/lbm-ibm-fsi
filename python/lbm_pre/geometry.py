"""
lbm_pre.geometry
================
为放置 IBM Lagrangian 标记点及定义浸没边界形状提供几何辅助工具，
生成的形状数据将传入求解器。

所有坐标均以**格子单位**（Δx = 1，除非另有说明）表示。

Public API
----------
circle_markers()    — 在圆形上均匀分布标记点
filament_markers()  — 在直线丝状体上均匀分布标记点
bezier_markers()    — 在三次 Bézier 曲线上均匀分布标记点
write_markers_csv() — 将标记点坐标写入 CSV 文件供求解器读取
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "MarkerArray",
    "circle_markers",
    "ellipse_markers",
    "filament_markers",
    "bezier_markers",
    "write_markers_csv",
    "read_markers_csv",
]

# ---------------------------------------------------------------------------
# MarkerArray
# ---------------------------------------------------------------------------

@dataclass
class MarkerArray:
    """
    IBM Lagrangian 标记点位置及弧长元素的集合。

    Attributes
    ----------
    x, y    : 标记点坐标的一维数组
    ds      : 每个标记点对应的弧长（或面积）元素的一维数组
    tag     : 可选标签（如 'circle'、'filament'）
    """
    x: np.ndarray
    y: np.ndarray
    ds: np.ndarray
    tag: str = ""

    def __len__(self) -> int:
        return len(self.x)

    def to_xy(self) -> np.ndarray:
        """返回形状为 (N, 2) 的 (x, y) 位置数组。"""
        return np.column_stack([self.x, self.y])

    def centroid(self) -> tuple[float, float]:
        """返回平均 (x, y) 位置（质心）。"""
        return float(self.x.mean()), float(self.y.mean())


# ---------------------------------------------------------------------------
# 标记点构造函数
# ---------------------------------------------------------------------------

def circle_markers(cx: float, cy: float, radius: float,
                   n: int = 64, *, tag: str = "circle") -> MarkerArray:
    """
    在圆形上生成 *n* 个均匀分布的标记点。

    Parameters
    ----------
    cx, cy  : 圆心坐标（格子单位）
    radius  : 圆形半径
    n       : 标记点数量
    tag     : 标签

    Returns
    -------
    MarkerArray
    """
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x = cx + radius * np.cos(theta)
    y = cy + radius * np.sin(theta)
    ds = np.full(n, 2.0 * np.pi * radius / n)
    return MarkerArray(x=x, y=y, ds=ds, tag=tag)


def ellipse_markers(cx: float, cy: float,
                    a: float, b: float,
                    n: int = 64, *, tag: str = "ellipse") -> MarkerArray:
    """
    在椭圆上生成 *n* 个弧长均匀分布的标记点。

    采用自适应角度间距，使各弧长元素近似相等，
    满足 IBM delta 函数的数值要求。

    Parameters
    ----------
    cx, cy  : 圆心
    a, b    : 半轴长（a = 水平方向，b = 垂直方向）
    n       : 标记点数量
    """
    # 对参数曲线过采样，再按等弧长重新采样
    N_fine = max(n * 100, 10000)
    t_fine = np.linspace(0.0, 2.0 * np.pi, N_fine, endpoint=False)
    xf = cx + a * np.cos(t_fine)
    yf = cy + b * np.sin(t_fine)

    dx = np.diff(xf, append=xf[0] - xf[-1])
    dy = np.diff(yf, append=yf[0] - yf[-1])

    # 累积弧长
    seg_len = np.sqrt(dx**2 + dy**2)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = arc[-1]
    ds_val = total_len / n

    # 按等弧长间隔采样
    s_targets = np.linspace(0.0, total_len, n, endpoint=False)
    x = np.interp(s_targets, arc[:-1], xf)
    y = np.interp(s_targets, arc[:-1], yf)
    ds = np.full(n, ds_val)
    return MarkerArray(x=x, y=y, ds=ds, tag=tag)


def filament_markers(x0: float, y0: float,
                     x1: float, y1: float,
                     n: int = 32, *, tag: str = "filament") -> MarkerArray:
    """
    在直线丝状体上生成 *n* 个均匀间距的标记点。

    Parameters
    ----------
    (x0, y0) : 起点
    (x1, y1) : 终点
    n        : 标记点数量（包含两个端点）
    """
    t = np.linspace(0.0, 1.0, n)
    x = x0 + t * (x1 - x0)
    y = y0 + t * (y1 - y0)
    length = np.hypot(x1 - x0, y1 - y0)
    ds = np.full(n, length / n)
    return MarkerArray(x=x, y=y, ds=ds, tag=tag)


def bezier_markers(control_points: Sequence[Tuple[float, float]],
                   n: int = 64, *, tag: str = "bezier") -> MarkerArray:
    """
    在由 *control_points*（4 个控制点的三次 Bézier 曲线）定义的曲线上
    生成 *n* 个弧长均匀分布的标记点。

    Parameters
    ----------
    control_points : (x, y) 元组列表 — 必须恰好包含 4 个点
    n              : 输出标记点数量
    """
    pts = np.asarray(control_points, dtype=float)
    if len(pts) != 4:
        raise ValueError("bezier_markers requires exactly 4 control points (cubic Bézier)")

    N_fine = max(n * 100, 10000)
    t_fine = np.linspace(0.0, 1.0, N_fine)
    # De Casteljau / Bernstein 计算
    xf = (
        (1 - t_fine)**3 * pts[0, 0]
        + 3 * (1 - t_fine)**2 * t_fine * pts[1, 0]
        + 3 * (1 - t_fine) * t_fine**2 * pts[2, 0]
        + t_fine**3 * pts[3, 0]
    )
    yf = (
        (1 - t_fine)**3 * pts[0, 1]
        + 3 * (1 - t_fine)**2 * t_fine * pts[1, 1]
        + 3 * (1 - t_fine) * t_fine**2 * pts[2, 1]
        + t_fine**3 * pts[3, 1]
    )

    seg_len = np.sqrt(np.diff(xf)**2 + np.diff(yf)**2)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = arc[-1]
    ds_val = total_len / n

    s_targets = np.linspace(0.0, total_len, n, endpoint=False)
    x = np.interp(s_targets, arc[:-1], xf[:-1])
    y = np.interp(s_targets, arc[:-1], yf[:-1])
    return MarkerArray(x=x, y=y, ds=np.full(n, ds_val), tag=tag)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_markers_csv(markers: MarkerArray, path: str | Path) -> Path:
    """
    将标记点位置写入可被求解器读取的 CSV 文件。

    格式::

        # tag: <tag>
        x,y,ds
        <x0>,<y0>,<ds0>
        ...

    Parameters
    ----------
    markers : MarkerArray
    path    : 输出文件路径

    Returns
    -------
    写出文件的 Path。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="") as f:
        f.write(f"# tag: {markers.tag}\n")
        writer = csv.writer(f)
        writer.writerow(["x", "y", "ds"])
        for xi, yi, dsi in zip(markers.x, markers.y, markers.ds):
            writer.writerow([f"{xi:.10g}", f"{yi:.10g}", f"{dsi:.10g}"])
    return p


def read_markers_csv(path: str | Path) -> MarkerArray:
    """
    读取由 :func:`write_markers_csv` 写出的标记点 CSV 文件。

    Returns
    -------
    MarkerArray
    """
    p = Path(path)
    tag = ""
    rows: list[list[float]] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line.startswith("# tag:"):
                tag = line.split(":", 1)[1].strip()
            elif line.startswith("#") or line.startswith("x"):
                continue
            else:
                parts = line.split(",")
                rows.append([float(v) for v in parts])

    data = np.array(rows)
    return MarkerArray(x=data[:, 0], y=data[:, 1], ds=data[:, 2], tag=tag)
