"""
lbm_pre.bridge
==============
FFI-friendly entry points called by the Rust orchestrator via pyo3.

All functions accept plain Python scalars / lists rather than high-level
dataclasses so that they remain easy to call from Rust without marshalling
complex objects.  The functions remain independently usable from pure
Python code as well.

Public API
----------
geometry_markers_raw()  — generate IBM Lagrangian markers, return (x, y, ds) lists
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
    Generate IBM Lagrangian marker positions without producing any files.

    This function is the primary entry point for the Rust pyo3 FFI bridge;
    it can also be called directly from Python.

    Parameters
    ----------
    geometry  : ``"circle"`` or ``"filament"``
    x0, y0    : centre (circle) or start point (filament) in lattice units
    size      : radius (circle) or length (filament) in lattice units
    n_markers : number of Lagrangian markers

    Returns
    -------
    ``(x, y, ds)`` — three plain Python lists of length *n_markers* suitable
    for direct extraction by pyo3 (no NumPy dependency on the Rust side).

    Raises
    ------
    ValueError
        If *geometry* is not ``"circle"`` or ``"filament"``.
    """
    if geometry == "circle":
        m: MarkerArray = circle_markers(x0, y0, size, n_markers)
    elif geometry == "filament":
        # Straight filament from (x0, y0) to (x0, y0 + size) along y-axis
        m = filament_markers(x0, y0, x0, y0 + size, n_markers)
    else:
        raise ValueError(
            f"Unknown geometry {geometry!r}. Supported: 'circle', 'filament'."
        )
    return m.x.tolist(), m.y.tolist(), m.ds.tolist()
