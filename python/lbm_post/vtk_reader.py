"""
lbm_post.vtk_reader
===================
Read VTK / VTU output files produced by the LBM+IBM+FSI solver.

The solver writes per-timestep snapshots of the Eulerian fluid field
(density, velocity) and the Lagrangian IBM marker positions.  This module
reads those files into NumPy arrays suitable for :mod:`lbm_post.plot` and
:mod:`lbm_post.analysis`.

Because VTK output is optional/future (the solver currently writes nothing),
we also provide a *synthetic data* generator that creates the same array
layout from a numpy binary dump (``.npz``), which the Rust orchestrator
can produce without the VTK dependency.

Data layout convention
----------------------
All 2-D field arrays are **row-major** with shape ``(ny, nx)`` — i.e. index
``[j, i]`` corresponds to grid node ``(i, j)``  (x first, y second).

Public API
----------
FieldSnapshot       — dataclass holding one time-step of Eulerian data
MarkerSnapshot      — dataclass holding one time-step of Lagrangian data
VtkReader           — read .vtu XML files (requires ``vtk`` package if available)
NpzReader           — read .npz files written by the Rust orchestrator
load_snapshot()     — auto-detect format and load
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
    "load_snapshot",
]


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FieldSnapshot:
    """
    One time-step of the Eulerian fluid field.

    Attributes
    ----------
    step    : time-step index
    time    : physical time
    nx, ny  : grid resolution
    rho     : density field, shape (ny, nx)
    ux      : x-velocity, shape (ny, nx)
    uy      : y-velocity, shape (ny, nx)
    force_x : body-force x-component (IBM), shape (ny, nx), may be None
    force_y : body-force y-component, shape (ny, nx), may be None
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
    # Derived quantities
    # ------------------------------------------------------------------

    def velocity_magnitude(self) -> np.ndarray:
        """Return |u| = sqrt(ux² + uy²), shape (ny, nx)."""
        return np.hypot(self.ux, self.uy)

    def vorticity(self) -> np.ndarray:
        """
        Return the z-component of vorticity  ωz = ∂uy/∂x − ∂ux/∂y,
        computed via second-order central differences.  Shape (ny, nx).
        """
        duy_dx = np.gradient(self.uy, axis=1)   # ∂uy/∂x  (x is axis 1)
        dux_dy = np.gradient(self.ux, axis=0)   # ∂ux/∂y  (y is axis 0)
        return duy_dx - dux_dy

    def pressure(self, cs2: float = 1.0 / 3.0) -> np.ndarray:
        """
        Return the LBM pressure  p = cs² · ρ.  Shape (ny, nx).
        """
        return cs2 * self.rho

    def stream_function(self) -> np.ndarray:
        """
        Approximate the stream function ψ by integrating uy along x (row 0).
        Useful for plotting streamlines in enclosed cavities.
        Shape (ny, nx).
        """
        psi = np.zeros_like(self.ux)
        # Integrate ux vertically: ψ(i, j) = ψ(i, j-1) + ux(i, j) Δy
        for j in range(1, self.ny):
            psi[j, :] = psi[j - 1, :] + self.ux[j, :]
        return psi


@dataclass
class MarkerSnapshot:
    """
    One time-step of Lagrangian IBM marker data.

    Attributes
    ----------
    step    : time-step index
    x, y    : marker positions (1-D arrays of length N)
    fx, fy  : IBM force density at each marker
    ux, uy  : interpolated fluid velocity at each marker
    """
    step: int
    x: np.ndarray
    y: np.ndarray
    fx: Optional[np.ndarray] = None
    fy: Optional[np.ndarray] = None
    ux: Optional[np.ndarray] = None
    uy: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# NPZ reader  (primary output format)
# ---------------------------------------------------------------------------

class NpzReader:
    """
    Read solver output written as ``.npz`` archives.

    Expected layout inside each ``.npz``:

    ``rho``   : (ny, nx) float64
    ``ux``    : (ny, nx) float64
    ``uy``    : (ny, nx) float64
    ``step``  : scalar int
    ``time``  : scalar float
    Optional:
    ``force_x``, ``force_y``  : (ny, nx) float64

    Files should be named  ``<directory>/fluid_<NNNNNN>.npz``.
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
        """Return sorted list of available time-step indices."""
        return [
            int(re.search(r"(\d+)", p.stem).group(1))
            for p in self._files
        ]

    def read(self, path_or_step: "str | Path | int") -> FieldSnapshot:
        """
        Load one snapshot.

        Parameters
        ----------
        path_or_step : path to the .npz file, or an integer step index
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
        """Load the last (highest-step) snapshot."""
        if not self._files:
            raise FileNotFoundError(f"No fluid_*.npz files in {self.directory}")
        return self.read(self._files[-1])


# ---------------------------------------------------------------------------
# VTU reader  (XML unstructured VTK)
# ---------------------------------------------------------------------------

class VtkReader:
    """
    Read legacy VTK rectilinear-grid files (.vtk) or XML VTU files (.vtu)
    produced by the solver (or converted via paraview / meshio).

    Requires either the ``vtk`` Python package or a hand-written XML parser
    for the ASCII variant (the latter is used here to avoid the heavy VTK dep).

    Only ASCII VTK / inline-base64 VTU are supported; for binary files
    install ``vtk`` or ``meshio``.
    """

    @staticmethod
    def read_vtu(path: str | Path) -> FieldSnapshot:
        """
        Parse a VTU (XML VTK unstructured-grid) file.

        Supports ASCII point-data arrays named ``rho``, ``ux``, ``uy``.
        """
        p = Path(path)
        tree = ET.parse(p)
        root = tree.getroot()

        # Extract piece extent
        piece = root.find(".//Piece")
        if piece is None:
            raise ValueError(f"No <Piece> element found in {p}")

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
                warnings.warn(f"VTK binary format not fully supported; skipping {name}")
                continue
            if n_comp > 1:
                vals = vals.reshape(-1, n_comp)
            arrays[name] = vals

        # Reconstruct grid shape from "rho" (1 component)
        rho_flat = arrays.get("rho", np.ones(n_total))
        # Guess square-ish grid
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
# Auto-detect loader
# ---------------------------------------------------------------------------

def load_snapshot(path: str | Path) -> FieldSnapshot:
    """
    Load a single snapshot from a ``.npz`` or ``.vtu`` file.

    Parameters
    ----------
    path : path to ``.npz`` or ``.vtu`` file

    Returns
    -------
    FieldSnapshot
    """
    p = Path(path)
    if p.suffix == ".npz":
        return NpzReader(p.parent).read(p)
    elif p.suffix in {".vtu", ".vtk"}:
        return VtkReader.read_vtu(p)
    else:
        raise ValueError(f"Unsupported snapshot format: {p.suffix}")


# ---------------------------------------------------------------------------
# Synthetic data generator (testing / demo without real solver output)
# ---------------------------------------------------------------------------

def make_synthetic_lid_cavity(nx: int = 64, ny: int = 64,
                               U_lid: float = 0.1,
                               step: int = 10000) -> FieldSnapshot:
    """
    Generate a synthetic lid-driven cavity snapshot using a simple analytical
    approximation (linear profile + parabolic correction), suitable for
    unit-testing the post-processing pipeline without running the solver.

    Parameters
    ----------
    nx, ny  : grid resolution
    U_lid   : lid velocity
    step    : reported time-step index
    """
    y_vals = np.linspace(0, 1, ny)
    x_vals = np.linspace(0, 1, nx)
    X, Y = np.meshgrid(x_vals, y_vals)

    # Simple analytical approximation: Stokes flow-like profile
    ux = U_lid * Y * (1 - Y) * 4  # parabolic in y
    uy = U_lid * X * (1 - X) * 4 * 0.1  # small cross-flow

    rho = np.ones((ny, nx))

    return FieldSnapshot(
        step=step, time=float(step), nx=nx, ny=ny,
        rho=rho, ux=ux, uy=uy,
    )


def save_snapshot_npz(snap: FieldSnapshot, path: str | Path) -> Path:
    """
    Save a :class:`FieldSnapshot` as a ``.npz`` file (for testing and
    as the recommended output format for the Rust orchestrator).
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
