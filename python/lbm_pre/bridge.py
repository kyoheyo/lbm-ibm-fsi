"""
lbm_pre.bridge
==============
供 Rust 编排器通过 pyo3 调用的 FFI 接口入口。

所有函数接受普通 Python 标量/列表而非高层数据类，
以便无需复杂对象编组即可从 Rust 调用。
这些函数同样可直接从纯 Python 代码中使用。

Public API
----------
geometry_markers_raw()  — 生成 IBM 拉格朗日标记，返回 (x, y, ds) 列表
"""

from __future__ import annotations

from typing import Tuple

from .geometry import (
    circle_markers,
    filament_markers,
    MarkerArray,
)

__all__ = ["geometry_markers_raw"]


def geometry_markers_raw(
    geometry: str,
    x0: float,
    y0: float,
    size: float,
    n_markers: int,
) -> Tuple[list, list, list]:
    """
    在不产生任何文件的情况下生成 IBM 拉格朗日标记点位置。

    这是 Rust pyo3 FFI 桥接的主要入口；也可直接从 Python 调用。

    Parameters
    ----------
    geometry  : ``"circle"`` 或 ``"filament"``
    x0, y0    : 圆心（circle）或起始点（filament），格子单位
    size      : 半径（circle）或长度（filament），格子单位
    n_markers : 拉格朗日标记点数量

    Returns
    -------
    ``(x, y, ds)`` — 长度为 *n_markers* 的三个普通 Python 列表，
    pyo3 可直接提取（Rust 侧无需 NumPy 依赖）。

    Raises
    ------
    ValueError
        若 *geometry* 不是 ``"circle"`` 或 ``"filament"``。
    """
    if geometry == "circle":
        m: MarkerArray = circle_markers(x0, y0, size, n_markers)
    elif geometry == "filament":
        # 沿 y 轴从 (x0, y0) 到 (x0, y0 + size) 的直线细丝
        m = filament_markers(x0, y0, x0, y0 + size, n_markers)
    else:
        raise ValueError(
            f"Unknown geometry {geometry!r}. Supported: 'circle', 'filament'."
        )
    return m.x.tolist(), m.y.tolist(), m.ds.tolist()
