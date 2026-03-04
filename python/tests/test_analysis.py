"""
Tests for lbm_post.analysis — quantitative post-processing.
"""

import numpy as np
import pytest

from lbm_post.vtk_reader import make_synthetic_lid_cavity, FieldSnapshot, MarkerSnapshot
from lbm_post.analysis import (
    drag_lift_coefficients,
    compute_divergence,
    compute_q_criterion,
    monitor_point,
    l2_error,
    linf_error,
    convergence_rate,
    compute_bulk_quantities,
    ForceSummary,
    BulkQuantities,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_uniform_snap(nx=16, ny=16, ux_val=0.05, uy_val=0.0, rho_val=1.0,
                       step=0) -> FieldSnapshot:
    """Uniform flow field for analytical verification."""
    return FieldSnapshot(
        step=step, time=float(step), nx=nx, ny=ny,
        rho=np.full((ny, nx), rho_val),
        ux=np.full((ny, nx), ux_val),
        uy=np.full((ny, nx), uy_val),
    )


def _make_solid_body_rotation(nx=32, ny=32, omega=1.0, step=0) -> FieldSnapshot:
    """Pure solid-body rotation: ux = -ω·y,  uy = ω·x"""
    x = np.arange(nx, dtype=float) - nx / 2
    y = np.arange(ny, dtype=float) - ny / 2
    X, Y = np.meshgrid(x, y)
    return FieldSnapshot(
        step=step, time=float(step), nx=nx, ny=ny,
        rho=np.ones((ny, nx)),
        ux=-omega * Y,
        uy= omega * X,
    )


# ---------------------------------------------------------------------------
# drag_lift_coefficients
# ---------------------------------------------------------------------------

class TestDragLift:
    def test_returns_force_summary(self):
        markers = MarkerSnapshot(
            step=100,
            x=np.array([0.0, 1.0]),
            y=np.array([0.0, 0.0]),
            fx=np.array([0.5, 0.5]),
            fy=np.array([0.1, 0.1]),
        )
        result = drag_lift_coefficients(markers, rho_ref=1.0, U_ref=0.1, D_ref=1.0)
        assert isinstance(result, ForceSummary)
        # Reaction force on body = -sum(f)
        assert abs(result.fx_total - (-1.0)) < 1e-12
        assert abs(result.fy_total - (-0.2)) < 1e-12

    def test_nan_when_no_forces(self):
        markers = MarkerSnapshot(step=0, x=np.array([0.0]), y=np.array([0.0]))
        result = drag_lift_coefficients(markers)
        assert np.isnan(result.fx_total)

    def test_cd_formula(self):
        markers = MarkerSnapshot(
            step=0,
            x=np.zeros(4),
            y=np.zeros(4),
            fx=np.full(4, 1.0),   # each marker has fx=1
            fy=np.zeros(4),
        )
        # dyn = 0.5 * 1.0 * 1.0^2 * 1.0 = 0.5
        # Reaction Fx = -4.0,  Cd = -4.0 / 0.5 = -8.0
        result = drag_lift_coefficients(markers, rho_ref=1.0, U_ref=1.0, D_ref=1.0)
        assert abs(result.cd - (-8.0)) < 1e-10


# ---------------------------------------------------------------------------
# compute_divergence
# ---------------------------------------------------------------------------

class TestDivergence:
    def test_uniform_flow_zero_divergence(self):
        snap = _make_uniform_snap()
        div = compute_divergence(snap)
        np.testing.assert_allclose(div, 0.0, atol=1e-12)

    def test_divergence_shape(self):
        snap = _make_uniform_snap(nx=24, ny=16)
        div = compute_divergence(snap)
        assert div.shape == (16, 24)

    def test_solid_body_rotation_near_zero(self):
        snap = _make_solid_body_rotation()
        div = compute_divergence(snap)
        # ∇·u = ∂(-ω·y)/∂x + ∂(ω·x)/∂y = 0
        np.testing.assert_allclose(div, 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_q_criterion
# ---------------------------------------------------------------------------

class TestQCriterion:
    def test_shape(self):
        snap = _make_uniform_snap()
        Q = compute_q_criterion(snap)
        assert Q.shape == (snap.ny, snap.nx)

    def test_uniform_flow_zero(self):
        snap = _make_uniform_snap()
        Q = compute_q_criterion(snap)
        np.testing.assert_allclose(Q, 0.0, atol=1e-10)

    def test_solid_rotation_positive(self):
        snap = _make_solid_body_rotation(nx=32, ny=32, omega=1.0)
        Q = compute_q_criterion(snap)
        # Interior points should be positive (dominated by rotation)
        interior = Q[4:-4, 4:-4]
        assert (interior > -1e-8).all()


# ---------------------------------------------------------------------------
# monitor_point
# ---------------------------------------------------------------------------

class TestMonitorPoint:
    def test_returns_correct_length(self):
        snaps = [make_synthetic_lid_cavity(step=s) for s in [0, 100, 200]]
        steps, values = monitor_point(snaps, xi=10, yj=10, field="ux")
        assert len(steps) == 3
        assert len(values) == 3

    def test_steps_are_correct(self):
        snaps = [make_synthetic_lid_cavity(step=s) for s in [50, 150]]
        steps, _ = monitor_point(snaps, xi=5, yj=5)
        np.testing.assert_array_equal(steps, [50, 150])

    def test_unknown_field_raises(self):
        snaps = [make_synthetic_lid_cavity(step=0)]
        with pytest.raises(ValueError):
            monitor_point(snaps, xi=0, yj=0, field="nonexistent")

    def test_magnitude_field(self):
        snap = _make_uniform_snap(ux_val=0.03, uy_val=0.04)
        _, vals = monitor_point([snap], xi=5, yj=5, field="magnitude")
        assert abs(vals[0] - 0.05) < 1e-10


# ---------------------------------------------------------------------------
# l2_error / linf_error
# ---------------------------------------------------------------------------

class TestErrors:
    def test_identical_snapshots_zero_error(self):
        snap = make_synthetic_lid_cavity()
        assert l2_error(snap, snap) == pytest.approx(0.0, abs=1e-14)
        assert linf_error(snap, snap) == pytest.approx(0.0, abs=1e-14)

    def test_l2_error_known(self):
        ref  = _make_uniform_snap(ux_val=1.0)
        snap = _make_uniform_snap(ux_val=2.0)
        # rel L2 = |2-1|/|1| = 1.0
        assert l2_error(snap, ref, field="ux") == pytest.approx(1.0, rel=1e-10)

    def test_linf_error_known(self):
        ref  = _make_uniform_snap(ux_val=1.0)
        snap = _make_uniform_snap(ux_val=1.5)
        # rel L∞ = 0.5 / 1.0 = 0.5
        assert linf_error(snap, ref, field="ux") == pytest.approx(0.5, rel=1e-10)

    def test_magnitude_field(self):
        ref  = _make_uniform_snap(ux_val=0.3, uy_val=0.4)  # |u|=0.5
        snap = _make_uniform_snap(ux_val=0.0, uy_val=0.0)
        # rel L2 error on magnitude = 1.0
        assert l2_error(snap, ref, field="magnitude") == pytest.approx(1.0, rel=1e-10)


# ---------------------------------------------------------------------------
# convergence_rate
# ---------------------------------------------------------------------------

class TestConvergenceRate:
    def test_second_order(self):
        dx = [1.0, 0.5, 0.25]
        errors = [0.04, 0.01, 0.0025]   # ∝ dx^2
        p = convergence_rate(dx, errors)
        assert abs(p - 2.0) < 0.01

    def test_first_order(self):
        dx = [1.0, 0.5, 0.25]
        errors = [0.1, 0.05, 0.025]   # ∝ dx^1
        p = convergence_rate(dx, errors)
        assert abs(p - 1.0) < 0.01


# ---------------------------------------------------------------------------
# compute_bulk_quantities
# ---------------------------------------------------------------------------

class TestBulkQuantities:
    def test_returns_bulk_quantities(self):
        snap = make_synthetic_lid_cavity()
        bq = compute_bulk_quantities(snap)
        assert isinstance(bq, BulkQuantities)
        assert bq.ke >= 0.0
        assert bq.enstrophy >= 0.0
        assert abs(bq.rho_mean - 1.0) < 1e-10
        assert bq.rho_std == pytest.approx(0.0, abs=1e-12)

    def test_zero_flow_zero_ke(self):
        snap = _make_uniform_snap(ux_val=0.0, uy_val=0.0)
        bq = compute_bulk_quantities(snap)
        assert bq.ke == pytest.approx(0.0, abs=1e-14)
        assert bq.enstrophy == pytest.approx(0.0, abs=1e-12)

    def test_step_field(self):
        snap = make_synthetic_lid_cavity(step=42)
        bq = compute_bulk_quantities(snap)
        assert bq.step == 42
