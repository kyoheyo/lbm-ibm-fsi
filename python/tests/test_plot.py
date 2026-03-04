"""
Tests for lbm_post.plot — visualisation helpers.

All tests run without a display (Agg backend) and only verify that:
  1. Functions return (fig, ax) tuples with the correct types.
  2. Figures can be saved without error.
  3. Key decorations (title, labels) are set.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lbm_post.vtk_reader import (
    make_synthetic_lid_cavity,
    MarkerSnapshot,
)
from lbm_post.plot import (
    plot_velocity_magnitude,
    plot_velocity_vectors,
    plot_pressure,
    plot_vorticity,
    plot_streamlines,
    plot_rho,
    plot_markers,
    plot_beam_deformation,
    plot_convergence,
    plot_velocity_profile,
    save_figure,
)


@pytest.fixture
def snap():
    return make_synthetic_lid_cavity(nx=32, ny=32, step=1000)


@pytest.fixture
def markers():
    n = 16
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return MarkerSnapshot(
        step=1000,
        x=50.0 + 10.0 * np.cos(theta),
        y=50.0 + 10.0 * np.sin(theta),
        fx=np.zeros(n),
        fy=np.zeros(n),
    )


# ---------------------------------------------------------------------------
# Return-type checks
# ---------------------------------------------------------------------------

class TestReturnTypes:
    def test_velocity_magnitude(self, snap):
        fig, ax = plot_velocity_magnitude(snap)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_velocity_vectors(self, snap):
        fig, ax = plot_velocity_vectors(snap)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_pressure(self, snap):
        fig, ax = plot_pressure(snap)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_vorticity(self, snap):
        fig, ax = plot_vorticity(snap)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_streamlines(self, snap):
        fig, ax = plot_streamlines(snap)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_rho(self, snap):
        fig, ax = plot_rho(snap)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_markers(self, snap, markers):
        fig, ax = plot_markers(snap, markers)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_beam_deformation(self):
        x = np.linspace(0, 1, 10)
        y = np.zeros(10)
        y_def = np.sin(np.pi * x) * 0.05
        fig, ax = plot_beam_deformation(x, y_def, y_ref=y, step=500)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_convergence(self):
        steps = list(range(0, 1000, 100))
        values = [1e-2 / (i + 1) for i in range(len(steps))]
        fig, ax = plot_convergence(steps, values, label="residual")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_velocity_profile(self, snap):
        fig, ax = plot_velocity_profile(snap, x_slices=[16], component="ux")
        assert isinstance(fig, plt.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Title / label checks
# ---------------------------------------------------------------------------

class TestTitlesLabels:
    def test_velocity_magnitude_title_contains_step(self, snap):
        fig, ax = plot_velocity_magnitude(snap)
        assert "1000" in ax.get_title()
        plt.close(fig)

    def test_vorticity_title(self, snap):
        fig, ax = plot_vorticity(snap)
        title = ax.get_title().lower()
        assert "vorticity" in title or "ωz" in title or "omega" in title
        plt.close(fig)

    def test_convergence_no_log_scale(self):
        steps = [0, 100, 200]
        values = [-0.5, 0.0, 0.5]
        fig, ax = plot_convergence(steps, values, log_scale=True)
        # values contain negatives — should NOT be log scale
        assert ax.get_yscale() == "linear"
        plt.close(fig)


# ---------------------------------------------------------------------------
# save_figure
# ---------------------------------------------------------------------------

class TestSaveFigure:
    def test_saves_png(self, snap, tmp_path):
        fig, _ = plot_velocity_magnitude(snap)
        out = save_figure(fig, tmp_path / "vel.png")
        assert out.exists()
        assert out.suffix == ".png"

    def test_saves_pdf(self, snap, tmp_path):
        fig, _ = plot_pressure(snap)
        out = save_figure(fig, tmp_path / "pressure.pdf")
        assert out.exists()

    def test_creates_parent_dirs(self, snap, tmp_path):
        fig, _ = plot_rho(snap)
        out = save_figure(fig, tmp_path / "sub" / "sub2" / "rho.png")
        assert out.exists()

    def test_returns_path(self, snap, tmp_path):
        fig, _ = plot_vorticity(snap)
        result = save_figure(fig, tmp_path / "omega.png")
        assert result.is_absolute()


# ---------------------------------------------------------------------------
# Background variants for velocity_vectors
# ---------------------------------------------------------------------------

class TestVelocityVectorsVariants:
    @pytest.mark.parametrize("bg", ["magnitude", "pressure", "vorticity", "none"])
    def test_all_backgrounds(self, snap, bg):
        fig, ax = plot_velocity_vectors(snap, background=bg)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
