"""
lbm_post.bridge
===============
供 Rust 编排器通过 pyo3 调用的 FFI 接口入口。

所有函数接受普通 Python 标量/列表而非高层数据类，
以便无需复杂对象编组即可从 Rust 调用。
这些函数同样可直接从纯 Python 代码中使用。

Public API
----------
plot_field_raw()  — 将原始平坦数组重塑为 FieldSnapshot 并保存为 PNG
"""

from __future__ import annotations

import pathlib
from typing import Sequence

import numpy as np

from .vtk_reader import FieldSnapshot
from .plot import (
    plot_velocity_magnitude,
    plot_vorticity,
    plot_pressure,
    plot_streamlines,
    save_figure,
)

__all__ = ["plot_field_raw"]

# 字段名 → 绘图函数的映射
_PLOTTERS = {
    "velocity_magnitude": plot_velocity_magnitude,
    "vorticity":          plot_vorticity,
    "pressure":           plot_pressure,
    "streamlines":        plot_streamlines,
}


def plot_field_raw(
    rho: Sequence[float],
    ux: Sequence[float],
    uy: Sequence[float],
    nx: int,
    ny: int,
    step: int,
    time: float,
    out_dir: str,
    field: str = "velocity_magnitude",
) -> None:
    """
    从原始平坦场数组渲染等值线 PNG 并保存到磁盘。

    这是 Rust pyo3 FFI 桥接的主要入口；也可直接从 Python 调用。

    不会创建中间 ``.npz`` 文件——数组在进程内直接重塑为
    NumPy 二维数组。

    Parameters
    ----------
    rho, ux, uy : 长度为 ``nx * ny`` 的行主序平坦序列
    nx, ny      : 网格维度
    step        : 时间步索引（用于文件名和标题）
    time        : 物理时间（用于坐标轴标签）
    out_dir     : 输出目录；不存在时自动创建
    field       : 要绘制的字段：
                  ``"velocity_magnitude"`` | ``"vorticity"`` |
                  ``"pressure"``           | ``"streamlines"``

    Output
    ------
    保存至 ``<out_dir>/<field>_<NNNNNN>.png``。
    """
    rho_arr = np.asarray(rho, dtype=np.float64).reshape(ny, nx)
    ux_arr  = np.asarray(ux,  dtype=np.float64).reshape(ny, nx)
    uy_arr  = np.asarray(uy,  dtype=np.float64).reshape(ny, nx)

    snap = FieldSnapshot(
        step=step, time=time, nx=nx, ny=ny,
        rho=rho_arr, ux=ux_arr, uy=uy_arr,
    )

    plotter = _PLOTTERS.get(field, plot_velocity_magnitude)
    fig, _ = plotter(snap)

    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_figure(fig, out / f"{field}_{step:06d}.png")
