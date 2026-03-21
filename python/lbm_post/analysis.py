"""
lbm_post.analysis
=================
用于 LBM+IBM+FSI 仿真数据的定量后处理例程。

Public API
----------
drag_lift_coefficients()  — 计算沉浸体上的阻力系数 Cd 和升力系数 Cl
compute_vorticity()       — 完整涡量场（委托给 FieldSnapshot）
compute_divergence()      — ∇·u（不可压缩流应 ≈ 0）
compute_q_criterion()     — 用于涡旋识别的 Q 准则（二维）
monitor_point()           — 在单个格点提取时间序列
l2_error()                — 两个场快照之间的 L2 误差
linf_error()              — 两个场快照之间的 L∞ 误差
convergence_rate()        — 估算空间收敛阶次
compute_bulk_quantities() — 体积平均动能、拟能等统计量
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from .vtk_reader import FieldSnapshot, MarkerSnapshot

__all__ = [
    "ForceSummary",
    "drag_lift_coefficients",
    "compute_divergence",
    "compute_q_criterion",
    "monitor_point",
    "l2_error",
    "linf_error",
    "convergence_rate",
    "compute_bulk_quantities",
    "BulkQuantities",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ForceSummary:
    """沉浸体上的 IBM 合力汇总。"""
    step: int
    fx_total: float   #: x 方向总力
    fy_total: float   #: y 方向总力
    cd: Optional[float] = None   #: 阻力系数  Cd = Fx / (0.5 ρ U² D)
    cl: Optional[float] = None   #: 升力系数  Cl = Fy / (0.5 ρ U² D)


@dataclass
class BulkQuantities:
    """单快照的体积平均流场统计量。"""
    step: int
    ke: float        #: 平均动能  E = 0.5 * mean(|u|²)
    enstrophy: float #: 平均拟能  Z = 0.5 * mean(ωz²)
    rho_mean: float  #: 体积平均密度
    rho_std: float   #: 密度标准差（不可压缩时 ≈ 0）
    div_max: float   #: 最大 |∇·u|（不可压缩性残差）


# ---------------------------------------------------------------------------
# Force / drag / lift
# ---------------------------------------------------------------------------

def drag_lift_coefficients(
    markers: MarkerSnapshot,
    *,
    rho_ref: float = 1.0,
    U_ref: float = 0.1,
    D_ref: float = 1.0,
) -> ForceSummary:
    """
    计算 IBM 标记力的阻力系数和升力系数。

    IBM 对每个标记处流体施加的力为 ``(fx, fy) * ds``。
    物体所受反力为其负值：
        Fx = -∑ fx·ds,   Fy = -∑ fy·ds

    Parameters
    ----------
    markers : 含 fx、fy 的拉格朗日标记快照
    rho_ref : 参考密度
    U_ref   : 参考速度（如进口速度）
    D_ref   : 参考长度（如圆柱直径）

    Returns
    -------
    ForceSummary
    """
    if markers.fx is None or markers.fy is None:
        return ForceSummary(step=markers.step,
                            fx_total=float("nan"),
                            fy_total=float("nan"))

    # 物体所受反力是流体所受力的负值
    fx_total = -float(np.sum(markers.fx))
    fy_total = -float(np.sum(markers.fy))

    dyn_pressure = 0.5 * rho_ref * U_ref**2 * D_ref
    cd = fx_total / dyn_pressure if dyn_pressure != 0.0 else float("nan")
    cl = fy_total / dyn_pressure if dyn_pressure != 0.0 else float("nan")

    return ForceSummary(step=markers.step,
                        fx_total=fx_total,
                        fy_total=fy_total,
                        cd=cd, cl=cl)


# ---------------------------------------------------------------------------
# Differential quantities
# ---------------------------------------------------------------------------

def compute_divergence(snap: FieldSnapshot) -> np.ndarray:
    """
    计算速度散度 ∇·u = ∂ux/∂x + ∂uy/∂y。

    对于完全不可压缩求解器，此值应恒为零；
    对于 LBM，其量级正比于马赫数的平方。

    Returns
    -------
    div_u : 形状为 (ny, nx) 的数组
    """
    dux_dx = np.gradient(snap.ux, axis=1)
    duy_dy = np.gradient(snap.uy, axis=0)
    return dux_dx + duy_dy


def compute_q_criterion(snap: FieldSnapshot) -> np.ndarray:
    """
    计算用于涡旋识别的二维 Q 准则。

    二维 Q 准则定义为：
        Q = -0.5 * (∂ux/∂x · ∂uy/∂y − ∂ux/∂y · ∂uy/∂x)
          = 0.5 * (|Ω|² − |S|²)
    其中 Ω 为 ∇u 的反对称部分，S 为对称部分。
    Q > 0 的区域以旋转为主（涡核）。

    Returns
    -------
    Q : 形状为 (ny, nx) 的数组
    """
    dux_dx = np.gradient(snap.ux, axis=1)
    dux_dy = np.gradient(snap.ux, axis=0)
    duy_dx = np.gradient(snap.uy, axis=1)
    duy_dy = np.gradient(snap.uy, axis=0)

    # 对称应变率张量 S
    Sxx = dux_dx
    Sxy = 0.5 * (dux_dy + duy_dx)
    Syy = duy_dy
    S2 = Sxx**2 + 2 * Sxy**2 + Syy**2

    # 反对称旋转率张量 Ω
    Oxy = 0.5 * (duy_dx - dux_dy)
    O2 = 2 * Oxy**2

    return 0.5 * (O2 - S2)


# ---------------------------------------------------------------------------
# Point monitoring
# ---------------------------------------------------------------------------

def monitor_point(
    snapshots: Sequence[FieldSnapshot],
    xi: int,
    yj: int,
    field: str = "ux",
) -> tuple[np.ndarray, np.ndarray]:
    """
    提取单个格点 ``(xi, yj)`` 处 ``field`` 的时间序列。

    Parameters
    ----------
    snapshots : FieldSnapshot 的可迭代序列（按时间步排序）
    xi, yj    : 格点索引（x 索引，y 索引）
    field     : 'ux'、'uy'、'rho'、'magnitude' 或 'vorticity' 之一

    Returns
    -------
    steps  : 时间步索引的一维 int 数组
    values : 场值的一维 float 数组
    """
    steps_list: list[int] = []
    values_list: list[float] = []
    for snap in snapshots:
        steps_list.append(snap.step)
        if field == "ux":
            val = snap.ux[yj, xi]
        elif field == "uy":
            val = snap.uy[yj, xi]
        elif field == "rho":
            val = snap.rho[yj, xi]
        elif field == "magnitude":
            val = float(snap.velocity_magnitude()[yj, xi])
        elif field == "vorticity":
            val = float(snap.vorticity()[yj, xi])
        else:
            raise ValueError(f"Unknown field: {field!r}")
        values_list.append(float(val))

    return np.array(steps_list), np.array(values_list)


# ---------------------------------------------------------------------------
# Error norms
# ---------------------------------------------------------------------------

def l2_error(snap: FieldSnapshot, ref: FieldSnapshot,
             field: str = "ux") -> float:
    """
    计算 *snap* 与参考解之间的相对 L2 误差。

    ``err = ‖snap.field − ref.field‖₂ / ‖ref.field‖₂``

    Parameters
    ----------
    snap  : 计算快照
    ref   : 参考（解析或细网格）快照
    field : 'ux'、'uy'、'rho' 或 'magnitude'
    """
    def _get(s: FieldSnapshot) -> np.ndarray:
        if field == "magnitude":
            return s.velocity_magnitude()
        return getattr(s, field)

    f = _get(snap)
    r = _get(ref)
    denom = float(np.linalg.norm(r))
    if denom == 0.0:
        return float(np.linalg.norm(f - r))
    return float(np.linalg.norm(f - r)) / denom


def linf_error(snap: FieldSnapshot, ref: FieldSnapshot,
               field: str = "ux") -> float:
    """
    计算 *snap* 与参考解之间的相对 L∞ 误差。
    """
    def _get(s: FieldSnapshot) -> np.ndarray:
        if field == "magnitude":
            return s.velocity_magnitude()
        return getattr(s, field)

    f = _get(snap)
    r = _get(ref)
    denom = float(np.abs(r).max())
    if denom == 0.0:
        return float(np.abs(f - r).max())
    return float(np.abs(f - r).max()) / denom


def convergence_rate(
    grid_spacings: Sequence[float],
    errors: Sequence[float],
) -> float:
    """
    通过对数-对数最小二乘拟合估算空间收敛阶次。

    Parameters
    ----------
    grid_spacings : Δx 值（递减顺序）
    errors        : 对应的误差范数

    Returns
    -------
    p : 估算的收敛阶次（error ∝ Δxᵖ）
    """
    log_dx = np.log(np.array(grid_spacings, dtype=float))
    log_e  = np.log(np.array(errors, dtype=float))
    # 最小二乘斜率
    A = np.column_stack([log_dx, np.ones_like(log_dx)])
    p, _ = np.linalg.lstsq(A, log_e, rcond=None)[:2]
    return float(p[0])


# ---------------------------------------------------------------------------
# Bulk quantities
# ---------------------------------------------------------------------------

def compute_bulk_quantities(snap: FieldSnapshot) -> BulkQuantities:
    """
    计算单个快照的体积平均流场统计量。

    Returns
    -------
    BulkQuantities
    """
    ke = 0.5 * float(np.mean(snap.ux**2 + snap.uy**2))
    omega = snap.vorticity()
    enstrophy = 0.5 * float(np.mean(omega**2))
    rho_mean = float(np.mean(snap.rho))
    rho_std  = float(np.std(snap.rho))
    div = compute_divergence(snap)
    div_max = float(np.abs(div).max())

    return BulkQuantities(
        step=snap.step,
        ke=ke,
        enstrophy=enstrophy,
        rho_mean=rho_mean,
        rho_std=rho_std,
        div_max=div_max,
    )
