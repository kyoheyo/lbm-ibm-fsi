"""
lbm_post.plot
=============
Matplotlib-based visualisation for LBM+IBM+FSI simulation results.

All plotting functions accept a :class:`~lbm_post.vtk_reader.FieldSnapshot`
and return a ``(fig, ax)`` tuple so the caller can further customise or save.

Public API
----------
plot_velocity_magnitude()   — filled-contour of |u|
plot_velocity_vectors()     — quiver or streamplot overlay
plot_pressure()             — filled-contour of pressure p = cs²·ρ
plot_vorticity()            — filled-contour of ωz
plot_streamlines()          — integrated streamlines
plot_rho()                  — density field
plot_markers()              — Lagrangian IBM marker positions
plot_beam_deformation()     — flexible beam deflection curve
plot_convergence()          — residual / error vs. time-step
save_figure()               — helper: save with sensible defaults
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe in headless environments)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

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
    "save_figure",
]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _grid_axes(snap: FieldSnapshot) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) 1-D coordinate arrays from a FieldSnapshot."""
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
    Filled-contour plot of velocity magnitude |u|.

    Parameters
    ----------
    snap        : fluid field snapshot
    n_levels    : number of contour levels
    cmap        : Matplotlib colormap name
    add_colorbar: whether to draw a colourbar
    figsize     : figure size in inches
    vmin, vmax  : explicit colour-range bounds (lattice units/step).
                  When both are *None* (default) and the velocity field is
                  nearly uniform (peak-to-peak variation < 1 % of the mean),
                  the range is automatically clamped to ``[0, 1.5 × mean]``
                  so that machine-precision noise (≤ 10⁻¹⁴) is not amplified
                  into spurious visual artifacts such as fake velocity
                  reversals or tree-like stripe patterns.

    Returns
    -------
    (fig, ax)

    Notes
    -----
    The auto-clamp is only applied when *both* ``vmin`` and ``vmax`` are
    ``None``.  To override it and zoom into sub-percent variations, pass
    explicit numeric values, e.g. ``vmin=0.049, vmax=0.051``.
    Passing ``None`` for either parameter keeps that bound at its default.
    """
    fig, ax = _make_fig("Velocity magnitude |u|", snap, figsize)
    x, y = _grid_axes(snap)
    mag = snap.velocity_magnitude()

    # ------------------------------------------------------------------
    # Smart normalisation: prevent matplotlib from auto-scaling to
    # machine-precision noise when the velocity field is nearly uniform.
    #
    # Background: in a free-outlet / fully-developed-outlet channel, the
    # steady state is essentially uniform flow (all |u| ≈ u_inlet).  After
    # many thousand steps the peak-to-peak variation can drop to 10⁻¹⁵,
    # well below double-precision epsilon relative to the mean.  Without
    # explicit bounds, contourf auto-scales its colour range to this tiny
    # interval, making invisible noise appear as dramatic velocity reversals
    # or striped patterns that look physically wrong but are NOT.
    #
    # Heuristic: if rel. variation < 1 % of the mean, clamp to [0, 1.5·mean].
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
    Quiver (arrow) plot of the velocity field, optionally over a
    filled-contour background.

    Parameters
    ----------
    snap        : fluid field snapshot
    every       : sub-sample every Nth point (reduces clutter)
    scale       : quiver scale (None = auto)
    cmap        : colormap for the background field
    background  : 'magnitude', 'pressure', 'vorticity', or 'none'
    figsize     : figure size
    """
    fig, ax = _make_fig("Velocity vectors", snap, figsize)
    x, y = _grid_axes(snap)
    X, Y = np.meshgrid(x, y)

    # Background
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

    # Sub-sample for quiver
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
    Filled-contour plot of the LBM pressure  p = cs² · ρ.
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
    Filled-contour plot of the z-vorticity  ωz = ∂uy/∂x − ∂ux/∂y.
    """
    fig, ax = _make_fig("Vorticity ωz", snap, figsize)
    x, y = _grid_axes(snap)
    omega = snap.vorticity()

    if symmetric:
        v = np.abs(omega).max()
        # Guard against all-zero (or machine-precision-only) vorticity,
        # e.g. a uniform channel flow.  Using TwoSlopeNorm with v ≤ 1e-10
        # would amplify 10⁻¹⁵ noise into visually prominent fake vortices.
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
    Integrated streamline plot coloured by velocity magnitude.
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
    Filled-contour plot of the density field ρ.
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
    Overlay Lagrangian IBM marker positions on a fluid-field background.
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
    Plot the deformed shape of a flexible beam.

    Parameters
    ----------
    x       : beam node x-coordinates (1-D)
    y       : beam node y-coordinates (deformed, 1-D)
    y_ref   : undeformed y-coordinates (1-D, optional — drawn in grey)
    step    : time-step label for the title
    figsize : figure size
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
    Plot a scalar quantity (residual, drag, lift, …) versus time step.

    Parameters
    ----------
    steps   : 1-D sequence of time-step indices
    values  : 1-D sequence of scalar values
    label   : curve label
    xlabel  : x-axis label
    ylabel  : y-axis label (defaults to *label*)
    log_scale: use log-y scale (useful for residuals)
    figsize : figure size
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
    Plot 1-D velocity profiles at specified grid lines.

    Parameters
    ----------
    snap       : fluid field snapshot
    x_slices   : list of x-indices for vertical profiles (constant x)
    y_slices   : list of y-indices for horizontal profiles (constant y)
    component  : 'ux' or 'uy'
    figsize    : figure size
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
    Save *fig* to *path*.

    Parameters
    ----------
    fig     : Matplotlib figure
    path    : output path (format inferred from suffix: .png, .pdf, .svg, …)
    dpi     : dots per inch (relevant for raster formats)
    tight   : call ``tight_layout()`` before saving

    Returns
    -------
    Path of the saved file.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if tight:
        fig.tight_layout()
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p
