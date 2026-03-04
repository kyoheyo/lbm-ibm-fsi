"""
Tests for lbm_post.vtk_reader — snapshot data structures and I/O.
"""

import numpy as np
import pytest

from lbm_post.vtk_reader import (
    FieldSnapshot,
    MarkerSnapshot,
    NpzReader,
    load_snapshot,
    make_synthetic_lid_cavity,
    save_snapshot_npz,
)


# ---------------------------------------------------------------------------
# FieldSnapshot derived quantities
# ---------------------------------------------------------------------------

class TestFieldSnapshotDerivatives:
    @pytest.fixture
    def snap(self):
        return make_synthetic_lid_cavity(nx=32, ny=32)

    def test_velocity_magnitude_shape(self, snap):
        mag = snap.velocity_magnitude()
        assert mag.shape == (snap.ny, snap.nx)

    def test_velocity_magnitude_non_negative(self, snap):
        assert (snap.velocity_magnitude() >= 0).all()

    def test_vorticity_shape(self, snap):
        omega = snap.vorticity()
        assert omega.shape == (snap.ny, snap.nx)

    def test_pressure_shape(self, snap):
        p = snap.pressure()
        assert p.shape == (snap.ny, snap.nx)

    def test_pressure_proportional_to_rho(self, snap):
        cs2 = 1.0 / 3.0
        p = snap.pressure(cs2=cs2)
        np.testing.assert_allclose(p, cs2 * snap.rho)

    def test_stream_function_shape(self, snap):
        psi = snap.stream_function()
        assert psi.shape == (snap.ny, snap.nx)

    def test_stream_function_zero_bottom(self, snap):
        psi = snap.stream_function()
        np.testing.assert_allclose(psi[0, :], 0.0)


# ---------------------------------------------------------------------------
# NpzReader round-trip
# ---------------------------------------------------------------------------

class TestNpzReader:
    def test_save_and_load(self, tmp_path):
        snap = make_synthetic_lid_cavity(nx=16, ny=16, step=500)
        path = save_snapshot_npz(snap, tmp_path / "fluid_000500.npz")
        assert path.exists()

        reader = NpzReader(tmp_path)
        assert len(reader) == 1

        loaded = reader.read(path)
        assert loaded.step == 500
        assert loaded.nx == 16
        assert loaded.ny == 16
        np.testing.assert_allclose(loaded.rho, snap.rho)
        np.testing.assert_allclose(loaded.ux,  snap.ux)
        np.testing.assert_allclose(loaded.uy,  snap.uy)

    def test_last(self, tmp_path):
        for step in [100, 500, 1000]:
            snap = make_synthetic_lid_cavity(step=step)
            save_snapshot_npz(snap, tmp_path / f"fluid_{step:06d}.npz")

        reader = NpzReader(tmp_path)
        last = reader.last()
        assert last.step == 1000

    def test_steps_list(self, tmp_path):
        for step in [200, 400]:
            snap = make_synthetic_lid_cavity(step=step)
            save_snapshot_npz(snap, tmp_path / f"fluid_{step:06d}.npz")

        reader = NpzReader(tmp_path)
        assert reader.steps() == [200, 400]

    def test_empty_directory_raises(self, tmp_path):
        reader = NpzReader(tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.last()

    def test_iteration(self, tmp_path):
        steps_written = [100, 200, 300]
        for s in steps_written:
            save_snapshot_npz(make_synthetic_lid_cavity(step=s),
                              tmp_path / f"fluid_{s:06d}.npz")
        steps_read = [snap.step for snap in NpzReader(tmp_path)]
        assert steps_read == steps_written

    def test_with_forces(self, tmp_path):
        snap = make_synthetic_lid_cavity(nx=8, ny=8, step=0)
        snap.force_x = np.ones((8, 8)) * 0.001
        snap.force_y = np.ones((8, 8)) * 0.002
        path = save_snapshot_npz(snap, tmp_path / "fluid_000000.npz")
        loaded = NpzReader(tmp_path).read(path)
        assert loaded.force_x is not None
        np.testing.assert_allclose(loaded.force_x, snap.force_x)


# ---------------------------------------------------------------------------
# load_snapshot auto-detect
# ---------------------------------------------------------------------------

class TestLoadSnapshot:
    def test_load_npz(self, tmp_path):
        snap = make_synthetic_lid_cavity(step=42)
        path = save_snapshot_npz(snap, tmp_path / "fluid_000042.npz")
        loaded = load_snapshot(path)
        assert loaded.step == 42

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "data.xyz"
        p.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported snapshot format"):
            load_snapshot(p)


# ---------------------------------------------------------------------------
# make_synthetic_lid_cavity
# ---------------------------------------------------------------------------

class TestSyntheticData:
    def test_shape(self):
        snap = make_synthetic_lid_cavity(nx=48, ny=32)
        assert snap.rho.shape == (32, 48)
        assert snap.ux.shape == (32, 48)

    def test_density_uniform(self):
        snap = make_synthetic_lid_cavity()
        np.testing.assert_allclose(snap.rho, 1.0)

    def test_step_field(self):
        snap = make_synthetic_lid_cavity(step=9999)
        assert snap.step == 9999
