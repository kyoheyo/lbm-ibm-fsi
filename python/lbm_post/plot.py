"""
lbm_post.plot
=============
基于 Matplotlib 的 LBM+IBM+FSI 仿真结果可视化模块。

所有绘图函数接受 :class:`~lbm_post.vtk_reader.FieldSnapshot`
并返回 ``(fig, ax)`` 元组，调用者可进一步定制或保存。

Public API
----------
plot_velocity_magnitude()   — |u| 的填充等值线图
plot_velocity_vectors()     — quiver 或流线叠加图
plot_pressure()             — 压力 p = cs²·ρ 的填充等值线图
plot_vorticity()            — ωz 的填充等值线图
plot_streamlines()          — 积分流线图
plot_rho()                  — 密度场
plot_markers()              — 拉格朗日 IBM 标记点位置
plot_beam_deformation()     — 柔性梁挠曲曲线
plot_convergence()          — 残差/误差随时间步的变化曲线
save_figure()               — 辅助函数：以合理默认参数保存图形
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")   # 非交互式后端（在无显示器环境中安全使用）
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.font_manager as _fm
import numpy as np

# ---------------------------------------------------------------------------
# CJK 字体检测：若系统中存在支持 CJK 的字体，
# 则将其添加到 sans-serif 字体列表最前，以确保标题和标签中的
# 中/日/韩文字正确渲染，而非产生"Glyph missing"警告
# （或在无 CJK 字体的 Windows 上导致 tight_layout 崩溃）。
# ---------------------------------------------------------------------------
_CJK_CANDIDATES = [
    "Microsoft YaHei",      # Windows
    "SimHei",               # Windows fallback
    "PingFang SC",          # macOS
    "Noto Sans CJK SC",     # Linux (Google Noto)
    "WenQuanYi Micro Hei",  # Linux (Wen Quan Yi)
    "Arial Unicode MS",     # cross-platform (if installed)
]
_installed_fonts = {f.name for f in _fm.fontManager.ttflist}
_cjk_font = next((f for f in _CJK_CANDIDATES if f in _installed_fonts), None)
if _cjk_font:
    matplotlib.rcParams["font.sans-serif"] = (
        [_cjk_font] + matplotlib.rcParams.get("font.sans-serif", [])
    )
    matplotlib.rcParams["axes.unicode_minus"] = False

from .vtk_reader import FieldSnapshot, MarkerSnapshot

__all__ = [
    "plot_velocity_magnitude",
    "plot_velocity_vectors",
    "plot_pressure",
    "plot_vorticity",
    "plot_streamlines",
    "plot_rho",
    "plot_markers",
    "plot_beam_deformation",
    "plot_convergence",
    "plot_velocity_profile",
    "save_figure",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _grid_axes(snap: FieldSnapshot) -> tuple[np.ndarray, np.ndarray]:
    """从 FieldSnapshot 返回 (x, y) 一维坐标数组。"""
    x = np.arange(snap.nx, dtype=float)
    y = np.arange(snap.ny, dtype=float)
    return x, y


def _make_fig(title: str, snap: Optional[FieldSnapshot] = None,
              figsize: tuple[float, float] = (8, 6)) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=figsize)
    if snap is not None:
        ax.set_aspect("equal")
        ax.set_xlabel("x (lattice units)")
        ax.set_ylabel("y (lattice units)")
        ax.set_title(f"{title}  [step {snap.step}]")
    else:
        ax.set_title(title)
    return fig, ax


# ---------------------------------------------------------------------------
# Field plots
# ---------------------------------------------------------------------------

def plot_velocity_magnitude(
    snap: FieldSnapshot,
    *,
    n_levels: int = 64,
    cmap: str = "viridis",
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (8, 6),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    |u| 速度幅值的填充等值线图。

    Parameters
    ----------
    snap        : 流体场快照
    n_levels    : 等值线层数
    cmap        : Matplotlib 颜色映射名称
    add_colorbar: 是否绘制颜色条
    figsize     : 图形尺寸（英寸）
    vmin, vmax  : 显式颜色范围边界（格子单位/步）。
                  当两者均为 *None*（默认）且速度场几乎均匀
                  （峰峰变化 < 均值的 1%）时，范围自动钳位到
                  ``[0, 1.5 × 均值]``，以防机器精度噪声（≤ 10⁻¹⁴）
                  被放大为虚假的视觉伪影（如假速度反转或树状条纹）。

    Returns
    -------
    (fig, ax)

    Notes
    -----
    自动钳位仅在 ``vmin`` 和 ``vmax`` 均为 ``None`` 时生效。
    若需观察亚百分比变化，请传入显式数值，如 ``vmin=0.049, vmax=0.051``。
    仅传入其中一个参数时，另一个保持默认值。
    """
    fig, ax = _make_fig("Velocity magnitude |u|", snap, figsize)
    x, y = _grid_axes(snap)
    mag = snap.velocity_magnitude()

    # ------------------------------------------------------------------
    # 智能归一化：防止 Matplotlib 在速度场几乎均匀时对机器精度噪声自动缩放。
    #
    # 背景说明：在自由出口/充分发展出口通道中，稳态为近似均匀流
    # （所有 |u| ≈ u_inlet）。经过数千步后，峰峰变化可降至 10⁻¹⁵，
    # 远低于相对均值的双精度 epsilon。若无显式边界，contourf 会将颜色
    # 范围自动缩放到这个极小区间，使不可见的噪声表现为剧烈的速度反转
    # 或条纹图案，视觉上看起来物理上错误，实则并非如此。
    #
    # 启发式规则：若相对变化 < 均值的 1%，则钳位到 [0, 1.5·均值]。
    # ------------------------------------------------------------------
    if vmin is None and vmax is None:
        mean_mag = float(np.mean(mag))
        if mean_mag > 1e-30:
            rel_var = float(mag.max() - mag.min()) / mean_mag
            if rel_var < 1e-2:
                vmin = 0.0
                vmax = mean_mag * 1.5

    if vmin is not None or vmax is not None:
        v0 = vmin if vmin is not None else float(mag.min())
        v1 = vmax if vmax is not None else float(mag.max())
        levels = np.linspace(v0, v1, n_levels)
        cf = ax.contourf(x, y, mag, levels=levels, cmap=cmap, extend="both")
    else:
        cf = ax.contourf(x, y, mag, levels=n_levels, cmap=cmap)

    if add_colorbar:
        fig.colorbar(cf, ax=ax, label="|u| (lattice units/step)")
    return fig, ax


def plot_velocity_vectors(
    snap: FieldSnapshot,
    *,
    every: int = 5,
    scale: Optional[float] = None,
    cmap: str = "viridis",
    background: str = "magnitude",
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    速度场的 quiver（箭头）图，可选填充等值线背景。

    Parameters
    ----------
    snap        : 流体场快照
    every       : 每隔 N 个点采样一次（减少视觉混乱）
    scale       : quiver 缩放比例（None = 自动）
    cmap        : 背景字段的颜色映射
    background  : 'magnitude'、'pressure'、'vorticity' 或 'none'
    figsize     : 图形尺寸
    """
    fig, ax = _make_fig("Velocity vectors", snap, figsize)
    x, y = _grid_axes(snap)
    X, Y = np.meshgrid(x, y)

    # 背景场
    if background == "magnitude":
        bg = snap.velocity_magnitude()
        label = "|u|"
    elif background == "pressure":
        bg = snap.pressure()
        label = "p"
    elif background == "vorticity":
        bg = snap.vorticity()
        label = "ωz"
        cmap = "RdBu_r"
    else:
        bg = None
        label = ""

    if bg is not None:
        cf = ax.contourf(x, y, bg, levels=64, cmap=cmap)
        fig.colorbar(cf, ax=ax, label=label)

    # 为 quiver 降采样
    sl = slice(None, None, every)
    ax.quiver(X[sl, sl], Y[sl, sl],
              snap.ux[sl, sl], snap.uy[sl, sl],
              scale=scale, color="white", alpha=0.7, linewidth=0.5)
    return fig, ax


def plot_pressure(
    snap: FieldSnapshot,
    *,
    n_levels: int = 64,
    cmap: str = "coolwarm",
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    LBM 压力 p = cs² · ρ 的填充等值线图。
    """
    fig, ax = _make_fig("Pressure (p = cs² · ρ)", snap, figsize)
    x, y = _grid_axes(snap)
    p = snap.pressure()
    vmin, vmax = p.min(), p.max()
    if vmin == vmax:
        norm = None
    else:
        vcenter = (vmin + vmax) / 2
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    cf = ax.contourf(x, y, p, levels=n_levels, cmap=cmap, norm=norm)
    if add_colorbar:
        fig.colorbar(cf, ax=ax, label="p (lattice units)")
    return fig, ax


def plot_vorticity(
    snap: FieldSnapshot,
    *,
    n_levels: int = 64,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    z 方向涡量 ωz = ∂uy/∂x − ∂ux/∂y 的填充等值线图。
    """
    fig, ax = _make_fig("Vorticity ωz", snap, figsize)
    x, y = _grid_axes(snap)
    omega = snap.vorticity()

    if symmetric:
        v = np.abs(omega).max()
        # 防止全零（或仅含机器精度）涡量时出现问题，
        # 如均匀通道流。使用 v ≤ 1e-10 的 TwoSlopeNorm 会将
        # 10⁻¹⁵ 噪声放大为视觉上明显的虚假涡旋。
        norm = mcolors.TwoSlopeNorm(vmin=-v, vcenter=0.0, vmax=v) if v > 1e-10 else None
    else:
        norm = None

    cf = ax.contourf(x, y, omega, levels=n_levels, cmap=cmap, norm=norm)
    if add_colorbar:
        fig.colorbar(cf, ax=ax, label="ωz (1/step)")
    return fig, ax


def plot_streamlines(
    snap: FieldSnapshot,
    *,
    density: float = 1.5,
    cmap: str = "viridis",
    linewidth_scale: float = 1.0,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    以速度幅值着色的积分流线图。
    """
    fig, ax = _make_fig("Streamlines", snap, figsize)
    x, y = _grid_axes(snap)
    mag = snap.velocity_magnitude()
    lw = linewidth_scale * (0.5 + 2.0 * mag / (mag.max() + 1e-30))
    strm = ax.streamplot(
        x, y, snap.ux, snap.uy,
        color=mag, cmap=cmap, linewidth=lw,
        density=density, arrowsize=1.0,
    )
    fig.colorbar(strm.lines, ax=ax, label="|u| (lattice units/step)")
    return fig, ax


def plot_rho(
    snap: FieldSnapshot,
    *,
    n_levels: int = 64,
    cmap: str = "plasma",
    add_colorbar: bool = True,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    密度场 ρ 的填充等值线图。
    """
    fig, ax = _make_fig("Density ρ", snap, figsize)
    x, y = _grid_axes(snap)
    cf = ax.contourf(x, y, snap.rho, levels=n_levels, cmap=cmap)
    if add_colorbar:
        fig.colorbar(cf, ax=ax, label="ρ (lattice units)")
    return fig, ax


# ---------------------------------------------------------------------------
# IBM / structure plots
# ---------------------------------------------------------------------------

def plot_markers(
    snaps_fluid: FieldSnapshot,
    snaps_markers: MarkerSnapshot,
    *,
    background: str = "vorticity",
    marker_color: str = "red",
    marker_size: float = 3.0,
    figsize: tuple[float, float] = (8, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    在流体场背景上叠加拉格朗日 IBM 标记点位置。
    """
    if background == "vorticity":
        fig, ax = plot_vorticity(snaps_fluid, figsize=figsize)
    elif background == "magnitude":
        fig, ax = plot_velocity_magnitude(snaps_fluid, figsize=figsize)
    else:
        fig, ax = _make_fig("Markers", snaps_fluid, figsize)

    ax.scatter(snaps_markers.x, snaps_markers.y,
               c=marker_color, s=marker_size, zorder=5,
               label="IBM markers")
    ax.legend(loc="upper right", fontsize=8)
    return fig, ax


def plot_beam_deformation(
    x: np.ndarray,
    y: np.ndarray,
    y_ref: Optional[np.ndarray] = None,
    *,
    step: Optional[int] = None,
    figsize: tuple[float, float] = (8, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """
    绘制柔性梁的变形形状。

    Parameters
    ----------
    x       : 梁节点 x 坐标（一维）
    y       : 梁节点 y 坐标（变形后，一维）
    y_ref   : 未变形 y 坐标（一维，可选——以灰色绘制）
    step    : 时间步标签（用于标题）
    figsize : 图形尺寸
    """
    title = "Beam deformation" + (f"  [step {step}]" if step is not None else "")
    fig, ax = plt.subplots(figsize=figsize)
    if y_ref is not None:
        ax.plot(x, y_ref, "--", color="grey", linewidth=1, label="reference")
    ax.plot(x, y, "-o", color="steelblue", linewidth=2, markersize=4,
            label="deformed")
    ax.set_xlabel("x (lattice units)")
    ax.set_ylabel("y (lattice units)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig, ax


# ---------------------------------------------------------------------------
# Convergence / time-series plots
# ---------------------------------------------------------------------------

def plot_convergence(
    steps: Sequence[int],
    values: Sequence[float],
    *,
    label: str = "residual",
    xlabel: str = "Time step",
    ylabel: Optional[str] = None,
    log_scale: bool = True,
    figsize: tuple[float, float] = (8, 4),
) -> tuple[plt.Figure, plt.Axes]:
    """
    绘制标量量（残差、阻力、升力……）随时间步的变化曲线。

    Parameters
    ----------
    steps   : 时间步索引的一维序列
    values  : 标量值的一维序列
    label   : 曲线标签
    xlabel  : x 轴标签
    ylabel  : y 轴标签（默认为 *label*）
    log_scale: 是否使用对数 y 轴（适用于残差）
    figsize : 图形尺寸
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(steps, values, linewidth=1.5, label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or label)
    ax.set_title(f"{label} vs. {xlabel}")
    if log_scale and all(v > 0 for v in values):
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    return fig, ax


def plot_velocity_profile(
    snap: FieldSnapshot,
    *,
    x_slices: Optional[Sequence[int]] = None,
    y_slices: Optional[Sequence[int]] = None,
    component: str = "ux",
    figsize: tuple[float, float] = (8, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    在指定网格线处绘制一维速度剖面。

    Parameters
    ----------
    snap       : 流体场快照
    x_slices   : 竖直剖面的 x 索引列表（常数 x）
    y_slices   : 水平剖面的 y 索引列表（常数 y）
    component  : 'ux' 或 'uy'
    figsize    : 图形尺寸
    """
    field = getattr(snap, component)
    fig, ax = plt.subplots(figsize=figsize)

    if x_slices:
        for xi in x_slices:
            y_arr = np.arange(snap.ny)
            ax.plot(field[:, xi], y_arr, label=f"x={xi}")
        ax.set_xlabel(component)
        ax.set_ylabel("y")

    if y_slices:
        for yj in y_slices:
            x_arr = np.arange(snap.nx)
            ax.plot(x_arr, field[yj, :], label=f"y={yj}")
        ax.set_xlabel("x")
        ax.set_ylabel(component)

    ax.set_title(f"{component} profiles  [step {snap.step}]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return fig, ax


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: str | Path, *,
                dpi: int = 150,
                tight: bool = True) -> Path:
    """
    将 *fig* 保存到 *path*。

    Parameters
    ----------
    fig     : Matplotlib 图形对象
    path    : 输出路径（格式由后缀推断：.png、.pdf、.svg……）
    dpi     : 每英寸点数（适用于光栅格式）
    tight   : 保存前调用 ``tight_layout()``

    Returns
    -------
    已保存文件的 Path。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p
