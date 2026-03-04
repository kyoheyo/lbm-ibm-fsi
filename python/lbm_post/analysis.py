"""
lbm_post.analysis
=================
Quantitative post-processing routines for LBM+IBM+FSI simulation data.

Public API
----------
drag_lift_coefficients()  — compute Cd and Cl on an immersed body
compute_vorticity()       — full vorticity field (delegates to FieldSnapshot)
compute_divergence()      — ∇·u (should be ≈ 0 for incompressible flow)
compute_q_criterion()     — Q-criterion for vortex identification (2-D)
monitor_point()           — extract time-series at a single grid point
l2_error()                — L2 error between two field snapshots
linf_error()              — L∞ error between two field snapshots
convergence_rate()        — estimate spatial order of convergence
compute_bulk_quantities() — volume-averaged kinetic energy, enstrophy
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
    """IBM force summary on the immersed body."""
    step: int
    fx_total: float   #: total force in x-direction
    fy_total: float   #: total force in y-direction
    cd: Optional[float] = None   #: drag coefficient  Cd = Fx / (0.5 ρ U² D)
    cl: Optional[float] = None   #: lift coefficient  Cl = Fy / (0.5 ρ U² D)


@dataclass
class BulkQuantities:
    """Volume-averaged quantities for a single snapshot."""
    step: int
    ke: float        #: mean kinetic energy  E = 0.5 * mean(|u|²)
    enstrophy: float #: mean enstrophy  Z = 0.5 * mean(ωz²)
    rho_mean: float  #: volume-averaged density
    rho_std: float   #: density standard deviation (≈ 0 for incompressible)
    div_max: float   #: maximum |∇·u| (incompressibility residual)


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
    Compute drag and lift coefficients from IBM marker forces.

    The IBM force on the fluid at each marker is ``(fx, fy) * ds``.
    The reaction force on the body is the negative:
        Fx = -∑ fx·ds,   Fy = -∑ fy·ds

    Parameters
    ----------
    markers : Lagrangian marker snapshot with fx, fy populated
    rho_ref : reference density
    U_ref   : reference velocity (e.g. inlet velocity)
    D_ref   : reference length (e.g. cylinder diameter)

    Returns
    -------
    ForceSummary
    """
    if markers.fx is None or markers.fy is None:
        return ForceSummary(step=markers.step,
                            fx_total=float("nan"),
                            fy_total=float("nan"))

    # The reaction force on the body is the negative of the force on the fluid
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
    Compute the velocity divergence  ∇·u = ∂ux/∂x + ∂uy/∂y.

    For a perfectly incompressible solver this should be identically zero;
    for LBM it is proportional to the Mach number squared.

    Returns
    -------
    div_u : ndarray of shape (ny, nx)
    """
    dux_dx = np.gradient(snap.ux, axis=1)
    duy_dy = np.gradient(snap.uy, axis=0)
    return dux_dx + duy_dy


def compute_q_criterion(snap: FieldSnapshot) -> np.ndarray:
    """
    Compute the 2-D Q-criterion for vortex identification.

    In 2-D, Q = -0.5 * (∂ux/∂x · ∂uy/∂y − ∂ux/∂y · ∂uy/∂x)
             = 0.5 * (|Ω|² − |S|²)
    where Ω is the antisymmetric part and S is the symmetric part of ∇u.
    Regions with Q > 0 are dominated by rotation (vortex cores).

    Returns
    -------
    Q : ndarray of shape (ny, nx)
    """
    dux_dx = np.gradient(snap.ux, axis=1)
    dux_dy = np.gradient(snap.ux, axis=0)
    duy_dx = np.gradient(snap.uy, axis=1)
    duy_dy = np.gradient(snap.uy, axis=0)

    # Symmetric rate-of-strain  S
    Sxx = dux_dx
    Sxy = 0.5 * (dux_dy + duy_dx)
    Syy = duy_dy
    S2 = Sxx**2 + 2 * Sxy**2 + Syy**2

    # Antisymmetric rotation rate  Ω
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
    Extract a time-series of ``field`` at the single grid point ``(xi, yj)``.

    Parameters
    ----------
    snapshots : iterable of FieldSnapshot (sorted by step)
    xi, yj    : grid indices (x-index, y-index)
    field     : one of 'ux', 'uy', 'rho', 'magnitude', 'vorticity'

    Returns
    -------
    steps  : 1-D int array of time-step indices
    values : 1-D float array of field values
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
    Compute the relative L2 error between *snap* and a reference solution.

    ``err = ‖snap.field − ref.field‖₂ / ‖ref.field‖₂``

    Parameters
    ----------
    snap  : computed snapshot
    ref   : reference (analytic or fine-grid) snapshot
    field : 'ux', 'uy', 'rho', or 'magnitude'
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
    Compute the relative L∞ error between *snap* and a reference solution.
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
    Estimate the spatial order of convergence via a log-log least-squares fit.

    Parameters
    ----------
    grid_spacings : Δx values (decreasing)
    errors        : corresponding error norms

    Returns
    -------
    p : estimated convergence order  (error ∝ Δxᵖ)
    """
    log_dx = np.log(np.array(grid_spacings, dtype=float))
    log_e  = np.log(np.array(errors, dtype=float))
    # Least-squares slope
    A = np.column_stack([log_dx, np.ones_like(log_dx)])
    p, _ = np.linalg.lstsq(A, log_e, rcond=None)[:2]
    return float(p[0])


# ---------------------------------------------------------------------------
# Bulk quantities
# ---------------------------------------------------------------------------

def compute_bulk_quantities(snap: FieldSnapshot) -> BulkQuantities:
    """
    Compute volume-averaged flow statistics for a snapshot.

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
