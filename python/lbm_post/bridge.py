"""
lbm_post.bridge
===============
FFI-friendly entry points called by the Rust orchestrator via pyo3.

All functions accept plain Python scalars / lists rather than high-level
dataclasses so that they remain easy to call from Rust without marshalling
complex objects.  The functions remain independently usable from pure
Python code as well.

Public API
----------
plot_field_raw()  — reshape raw flat arrays into a FieldSnapshot and save a PNG
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

# Map field name → plot function
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
    Render a contour PNG from raw flat field arrays and save it to disk.

    This is the primary entry point for the Rust pyo3 FFI bridge; it can
    also be called directly from Python.

    No intermediate ``.npz`` file is created — the arrays are reshaped into
    NumPy 2-D arrays in-process.

    Parameters
    ----------
    rho, ux, uy : flat row-major sequences of length ``nx * ny``
    nx, ny      : grid dimensions
    step        : time-step index (used in file name and title)
    time        : physical time (used as an axis label)
    out_dir     : output directory; created automatically if absent
    field       : which field to plot:
                  ``"velocity_magnitude"`` | ``"vorticity"`` |
                  ``"pressure"``           | ``"streamlines"``

    Output
    ------
    Saves ``<out_dir>/<field>_<NNNNNN>.png``.
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
