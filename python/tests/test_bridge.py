"""
Tests for lbm_pre.bridge and lbm_post.bridge — FFI-friendly entry points.

These tests exercise the bridge functions as plain Python calls, which
validates both standalone Python use and the correctness of the logic that
the Rust pyo3 bridge will invoke.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np
import pytest

from lbm_pre.bridge import geometry_markers_raw
from lbm_post.bridge import plot_field_raw


# ---------------------------------------------------------------------------
# lbm_pre.bridge
# ---------------------------------------------------------------------------

class TestGeometryMarkersRaw:
    def test_circle_count(self):
        x, y, ds = geometry_markers_raw("circle", 50.0, 40.0, 10.0, 64)
        assert len(x) == 64
        assert len(y) == 64
        assert len(ds) == 64

    def test_circle_radius(self):
        cx, cy, r, n = 50.0, 40.0, 10.0, 128
        x, y, ds = geometry_markers_raw("circle", cx, cy, r, n)
        for xi, yi in zip(x, y):
            dist = math.hypot(xi - cx, yi - cy)
            assert abs(dist - r) < 1e-6, f"radius mismatch: {dist} != {r}"

    def test_circle_arc_length(self):
        r, n = 10.0, 64
        x, y, ds = geometry_markers_raw("circle", 0.0, 0.0, r, n)
        expected_ds = 2.0 * math.pi * r / n
        for dsi in ds:
            assert abs(dsi - expected_ds) < 1e-9

    def test_filament_count(self):
        x, y, ds = geometry_markers_raw("filament", 50.0, 20.0, 30.0, 32)
        assert len(x) == 32
        assert len(y) == 32
        assert len(ds) == 32

    def test_filament_y_range(self):
        x, y, ds = geometry_markers_raw("filament", 50.0, 20.0, 30.0, 31)
        # y should span [20, 50]
        assert abs(min(y) - 20.0) < 1e-9
        assert abs(max(y) - 50.0) < 1e-9

    def test_filament_constant_x(self):
        x, y, ds = geometry_markers_raw("filament", 25.0, 0.0, 10.0, 10)
        for xi in x:
            assert abs(xi - 25.0) < 1e-9

    def test_returns_plain_lists(self):
        x, y, ds = geometry_markers_raw("circle", 0.0, 0.0, 5.0, 16)
        assert isinstance(x, list)
        assert isinstance(y, list)
        assert isinstance(ds, list)

    def test_unknown_geometry_raises(self):
        with pytest.raises(ValueError, match="Unknown geometry"):
            geometry_markers_raw("triangle", 0.0, 0.0, 5.0, 16)


# ---------------------------------------------------------------------------
# lbm_post.bridge
# ---------------------------------------------------------------------------

class TestPlotFieldRaw:
    """Smoke-tests: check that plot_field_raw runs without error and writes a file."""

    @pytest.fixture()
    def flat_field(self):
        nx, ny = 16, 16
        n = nx * ny
        rho = [1.0] * n
        ux  = [0.05 * (i % nx) / nx for i in range(n)]
        uy  = [0.0] * n
        return rho, ux, uy, nx, ny

    def test_velocity_magnitude(self, flat_field, tmp_path):
        rho, ux, uy, nx, ny = flat_field
        plot_field_raw(rho, ux, uy, nx, ny,
                       step=100, time=100.0,
                       out_dir=str(tmp_path),
                       field="velocity_magnitude")
        assert (tmp_path / "velocity_magnitude_000100.png").exists()

    def test_vorticity(self, flat_field, tmp_path):
        rho, ux, uy, nx, ny = flat_field
        plot_field_raw(rho, ux, uy, nx, ny,
                       step=200, time=200.0,
                       out_dir=str(tmp_path),
                       field="vorticity")
        assert (tmp_path / "vorticity_000200.png").exists()

    def test_pressure(self, flat_field, tmp_path):
        rho, ux, uy, nx, ny = flat_field
        plot_field_raw(rho, ux, uy, nx, ny,
                       step=300, time=300.0,
                       out_dir=str(tmp_path),
                       field="pressure")
        assert (tmp_path / "pressure_000300.png").exists()

    def test_unknown_field_defaults_to_velocity(self, flat_field, tmp_path):
        """An unrecognized field name should fall back to velocity_magnitude."""
        rho, ux, uy, nx, ny = flat_field
        plot_field_raw(rho, ux, uy, nx, ny,
                       step=1, time=1.0,
                       out_dir=str(tmp_path),
                       field="nonexistent_field")
        # Fall-back produces velocity_magnitude file
        assert (tmp_path / "nonexistent_field_000001.png").exists()

    def test_creates_output_dir(self, flat_field, tmp_path):
        rho, ux, uy, nx, ny = flat_field
        new_dir = tmp_path / "deep" / "nested"
        plot_field_raw(rho, ux, uy, nx, ny,
                       step=1, time=1.0,
                       out_dir=str(new_dir),
                       field="vorticity")
        assert new_dir.is_dir()
        assert (new_dir / "vorticity_000001.png").exists()

    def test_accepts_numpy_arrays(self, tmp_path):
        """plot_field_raw should also accept numpy arrays (not just lists)."""
        nx, ny = 8, 8
        n = nx * ny
        rho = np.ones(n)
        ux  = np.linspace(0, 0.1, n)
        uy  = np.zeros(n)
        plot_field_raw(rho, ux, uy, nx, ny,
                       step=5, time=5.0,
                       out_dir=str(tmp_path),
                       field="velocity_magnitude")
        assert (tmp_path / "velocity_magnitude_000005.png").exists()
