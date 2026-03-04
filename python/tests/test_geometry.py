"""
Tests for lbm_pre.geometry — IBM marker construction utilities.
"""

import math
import numpy as np
import pytest

from lbm_pre.geometry import (
    MarkerArray,
    circle_markers,
    ellipse_markers,
    filament_markers,
    bezier_markers,
    write_markers_csv,
    read_markers_csv,
)


# ---------------------------------------------------------------------------
# circle_markers
# ---------------------------------------------------------------------------

class TestCircleMarkers:
    def test_count(self):
        m = circle_markers(50, 50, 10, n=64)
        assert len(m) == 64

    def test_positions_on_circle(self):
        cx, cy, r = 50.0, 50.0, 10.0
        m = circle_markers(cx, cy, r, n=128)
        dist = np.sqrt((m.x - cx)**2 + (m.y - cy)**2)
        np.testing.assert_allclose(dist, r, rtol=1e-12)

    def test_arc_length_uniform(self):
        m = circle_markers(0, 0, 5.0, n=32)
        expected_ds = 2 * math.pi * 5.0 / 32
        np.testing.assert_allclose(m.ds, expected_ds, rtol=1e-12)

    def test_total_arc_length(self):
        r, n = 7.5, 100
        m = circle_markers(0, 0, r, n=n)
        total = float(m.ds.sum())
        expected = 2 * math.pi * r
        assert abs(total - expected) / expected < 1e-10

    def test_tag(self):
        m = circle_markers(0, 0, 1, n=8, tag="my_circle")
        assert m.tag == "my_circle"

    def test_to_xy_shape(self):
        m = circle_markers(0, 0, 1, n=16)
        xy = m.to_xy()
        assert xy.shape == (16, 2)


# ---------------------------------------------------------------------------
# ellipse_markers
# ---------------------------------------------------------------------------

class TestEllipseMarkers:
    def test_count(self):
        m = ellipse_markers(0, 0, 5, 3, n=64)
        assert len(m) == 64

    def test_points_on_ellipse(self):
        a, b = 6.0, 4.0
        m = ellipse_markers(0, 0, a, b, n=200)
        # Check each point satisfies (x/a)^2 + (y/b)^2 ≈ 1
        val = (m.x / a)**2 + (m.y / b)**2
        np.testing.assert_allclose(val, 1.0, atol=5e-4)

    def test_arc_lengths_approx_uniform(self):
        m = ellipse_markers(0, 0, 5, 3, n=64)
        # Standard deviation of ds should be small relative to mean
        assert m.ds.std() / m.ds.mean() < 0.05


# ---------------------------------------------------------------------------
# filament_markers
# ---------------------------------------------------------------------------

class TestFilamentMarkers:
    def test_count(self):
        m = filament_markers(0, 0, 10, 0, n=11)
        assert len(m) == 11

    def test_start_end_positions(self):
        m = filament_markers(1.0, 2.0, 9.0, 2.0, n=5)
        np.testing.assert_allclose(m.x[0], 1.0)
        np.testing.assert_allclose(m.x[-1], 9.0)
        np.testing.assert_allclose(m.y, 2.0)

    def test_uniform_spacing(self):
        m = filament_markers(0, 0, 10, 0, n=11)
        dx = np.diff(m.x)
        np.testing.assert_allclose(dx, dx[0], rtol=1e-12)

    def test_arc_length_total(self):
        length = 8.0
        m = filament_markers(0, 0, length, 0, n=9)
        total = float(m.ds.sum())
        assert abs(total - length) / length < 1e-10

    def test_diagonal_filament(self):
        m = filament_markers(0, 0, 3, 4, n=6)
        # total arc length should equal hypotenuse = 5.0
        total = float(m.ds.sum())
        assert abs(total - 5.0) / 5.0 < 1e-10


# ---------------------------------------------------------------------------
# bezier_markers
# ---------------------------------------------------------------------------

class TestBezierMarkers:
    def test_count(self):
        ctrl = [(0, 0), (1, 2), (3, 2), (4, 0)]
        m = bezier_markers(ctrl, n=50)
        assert len(m) == 50

    def test_start_point(self):
        ctrl = [(0.0, 1.0), (1.0, 3.0), (2.0, 3.0), (3.0, 1.0)]
        m = bezier_markers(ctrl, n=64)
        # First marker should be close to the first control point
        np.testing.assert_allclose(m.x[0], ctrl[0][0], atol=0.05)
        np.testing.assert_allclose(m.y[0], ctrl[0][1], atol=0.05)

    def test_wrong_control_point_count(self):
        with pytest.raises(ValueError):
            bezier_markers([(0, 0), (1, 1)], n=10)


# ---------------------------------------------------------------------------
# I/O round-trip
# ---------------------------------------------------------------------------

class TestMarkersIO:
    def test_roundtrip_csv(self, tmp_path):
        original = circle_markers(10.0, 20.0, 5.0, n=32, tag="test_circle")
        path = tmp_path / "markers.csv"
        write_markers_csv(original, path)
        loaded = read_markers_csv(path)

        assert loaded.tag == "test_circle"
        np.testing.assert_allclose(loaded.x, original.x, rtol=1e-9)
        np.testing.assert_allclose(loaded.y, original.y, rtol=1e-9)
        np.testing.assert_allclose(loaded.ds, original.ds, rtol=1e-9)

    def test_csv_creates_parent_dirs(self, tmp_path):
        m = filament_markers(0, 0, 1, 0, n=5)
        path = tmp_path / "subdir" / "sub2" / "out.csv"
        write_markers_csv(m, path)
        assert path.exists()
