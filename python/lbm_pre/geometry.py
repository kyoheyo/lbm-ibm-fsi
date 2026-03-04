"""
lbm_pre.geometry
================
Geometry helpers for placing IBM Lagrangian markers and defining
immersed-boundary shapes that are passed to the solver.

All coordinates are in **lattice units** (Δx = 1 unless stated otherwise).

Public API
----------
circle_markers()    — uniform markers around a circle
filament_markers()  — uniform markers along a straight filament
bezier_markers()    — markers along a cubic Bézier curve
write_markers_csv() — write marker coordinates to CSV for the solver
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
    A collection of Lagrangian IBM marker positions and arc-length elements.

    Attributes
    ----------
    x, y    : 1-D arrays of marker coordinates
    ds      : 1-D array of arc-length (or area) element per marker
    tag     : optional label (e.g. 'circle', 'filament')
    """
    x: np.ndarray
    y: np.ndarray
    ds: np.ndarray
    tag: str = ""

    def __len__(self) -> int:
        return len(self.x)

    def to_xy(self) -> np.ndarray:
        """Return (N, 2) array of (x, y) positions."""
        return np.column_stack([self.x, self.y])

    def centroid(self) -> tuple[float, float]:
        """Return the mean (x, y) position."""
        return float(self.x.mean()), float(self.y.mean())


# ---------------------------------------------------------------------------
# Marker constructors
# ---------------------------------------------------------------------------

def circle_markers(cx: float, cy: float, radius: float,
                   n: int = 64, *, tag: str = "circle") -> MarkerArray:
    """
    Generate *n* uniformly-spaced markers around a circle.

    Parameters
    ----------
    cx, cy  : centre coordinates (lattice units)
    radius  : circle radius
    n       : number of markers
    tag     : label

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
    Generate *n* markers uniformly distributed (in arc length) around an ellipse.

    Uses adaptive angular spacing so that the arc-length elements are nearly
    equal, matching the numerical requirement of the IBM delta function.

    Parameters
    ----------
    cx, cy  : centre
    a, b    : semi-axes (a = horizontal, b = vertical)
    n       : number of markers
    """
    # Over-sample the parametric curve then re-sample at equal arc-length
    N_fine = max(n * 100, 10000)
    t_fine = np.linspace(0.0, 2.0 * np.pi, N_fine, endpoint=False)
    xf = cx + a * np.cos(t_fine)
    yf = cy + b * np.sin(t_fine)

    dx = np.diff(xf, append=xf[0] - xf[-1])
    dy = np.diff(yf, append=yf[0] - yf[-1])

    # Cumulative arc length
    seg_len = np.sqrt(dx**2 + dy**2)
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = arc[-1]
    ds_val = total_len / n

    # Sample at equal arc-length intervals
    s_targets = np.linspace(0.0, total_len, n, endpoint=False)
    x = np.interp(s_targets, arc[:-1], xf)
    y = np.interp(s_targets, arc[:-1], yf)
    ds = np.full(n, ds_val)
    return MarkerArray(x=x, y=y, ds=ds, tag=tag)


def filament_markers(x0: float, y0: float,
                     x1: float, y1: float,
                     n: int = 32, *, tag: str = "filament") -> MarkerArray:
    """
    Generate *n* uniformly-spaced markers along a straight filament.

    Parameters
    ----------
    (x0, y0) : start point
    (x1, y1) : end point
    n        : number of markers (includes both endpoints)
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
    Generate *n* markers uniformly distributed in arc-length along a cubic
    Bézier curve defined by *control_points* (4 points for cubic).

    Parameters
    ----------
    control_points : list of (x, y) tuples — must have exactly 4 entries
    n              : number of output markers
    """
    pts = np.asarray(control_points, dtype=float)
    if len(pts) != 4:
        raise ValueError("bezier_markers requires exactly 4 control points (cubic Bézier)")

    N_fine = max(n * 100, 10000)
    t_fine = np.linspace(0.0, 1.0, N_fine)
    # De Casteljau / Bernstein evaluation
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
    Write marker positions to a CSV file consumable by the solver.

    Format::

        # tag: <tag>
        x,y,ds
        <x0>,<y0>,<ds0>
        ...

    Parameters
    ----------
    markers : MarkerArray
    path    : output file path

    Returns
    -------
    Path of the written file.
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
    Read a marker CSV file written by :func:`write_markers_csv`.

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
