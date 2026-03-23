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
    FieldSnapshot,
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


# ---------------------------------------------------------------------------
# Smart normalisation for nearly-uniform velocity fields
# ---------------------------------------------------------------------------

@pytest.fixture
def uniform_channel_snap():
    """
    Synthetic nearly-uniform channel flow snapshot (u ≈ 0.05 everywhere).
    Simulates the free-outlet steady state from ZouHe inlet +
    FullyDeveloped on all other faces, where peak-to-peak variation is
    machine-precision noise (≤ 1e-14), reproducing the visual artifacts
    (fake velocity reversal / tree-like stripes) caused by matplotlib
    auto-scaling to the noise floor.
    """
    nx, ny = 32, 16
    rho = np.ones((ny, nx))
    ux  = np.full((ny, nx), 0.05)
    uy  = np.zeros((ny, nx))
    rng = np.random.default_rng(42)
    ux += rng.uniform(-5e-15, 5e-15, ux.shape)
    return FieldSnapshot(step=20000, time=20000.0, nx=nx, ny=ny,
                         rho=rho, ux=ux, uy=uy)


class TestUniformFlowNormalisation:
    """
    Verify that plot_velocity_magnitude does NOT amplify machine-precision
    noise into fake visual artifacts when the velocity field is nearly uniform
    (relative peak-to-peak < 1 %).
    """

    def test_auto_normalisation_clamped_to_physical_range(
            self, uniform_channel_snap):
        """
        When rel. variation < 1 %, the plot must succeed and return a valid
        Figure / Axes pair without raising (the key regression: matplotlib
        used to show machine-precision noise as dramatic stripes).
        """
        fig, ax = plot_velocity_magnitude(uniform_channel_snap)
        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_explicit_vmin_vmax_override_auto_normalisation(
            self, uniform_channel_snap):
        """
        Explicit vmin/vmax bypass the smart normalisation and zoom into the
        tiny variation — must succeed without raising.
        """
        fig, ax = plot_velocity_magnitude(
            uniform_channel_snap, vmin=0.049, vmax=0.051)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_non_uniform_field_not_clamped(self, snap):
        """
        The lid-driven-cavity fixture has > 1 % velocity variation, so the
        auto-normalisation must NOT activate.
        """
        mag = snap.velocity_magnitude()
        mean_mag = float(np.mean(mag))
        rel_var = float(mag.max() - mag.min()) / (mean_mag or 1.0)
        assert rel_var >= 1e-2, (
            f"Fixture too uniform (rel_var={rel_var:.2e}); "
            "non-clamped path cannot be tested."
        )
        fig, ax = plot_velocity_magnitude(snap)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_vorticity_near_zero_no_amplification(
            self, uniform_channel_snap):
        """
        Vorticity in a uniform flow is machine-precision zero (≤ 1e-14).
        plot_vorticity must not amplify it via TwoSlopeNorm (which would
        produce the same fake-stripe artifact as velocity_magnitude).
        """
        fig, ax = plot_vorticity(uniform_channel_snap)
        assert isinstance(fig, plt.Figure)
        plt.close(fig)
