"""
Tests for lbm_pre.config_gen — TOML configuration generation.
"""

import tomllib
from pathlib import Path

import pytest

from lbm_pre.config_gen import (
    SolverConfig,
    FluidConfig,
    SimConfig,
    BoundaryConditionConfig,
    OutputConfig,
    StructureConfig,
    IbmConfig,
    reynolds_to_nu,
    load_config,
)


# ---------------------------------------------------------------------------
# reynolds_to_nu
# ---------------------------------------------------------------------------

class TestReynoldsToNu:
    def test_known_values(self):
        # Re = U*L/nu  =>  nu = U*L/Re
        nu = reynolds_to_nu(Re=100.0, U=0.1, L=100.0)
        assert abs(nu - 0.1) < 1e-12

    def test_warning_high_nu(self, capsys):
        # nu > 0.5 should warn
        reynolds_to_nu(Re=1.0, U=1.0, L=1.0)
        captured = capsys.readouterr()
        assert "WARNING" in captured.err

    def test_warning_low_nu(self, capsys):
        # nu < 0.05 should warn
        reynolds_to_nu(Re=10000.0, U=0.1, L=100.0)
        captured = capsys.readouterr()
        assert "WARNING" in captured.err

    def test_stable_range_no_warning(self, capsys):
        # nu = 0.1 is in [0.05, 0.5]
        reynolds_to_nu(Re=100.0, U=0.1, L=100.0)  # nu = 0.1
        captured = capsys.readouterr()
        assert "WARNING" not in captured.err


# ---------------------------------------------------------------------------
# SolverConfig.write / load_config round-trip
# ---------------------------------------------------------------------------

class TestConfigWriteLoad:
    def test_lid_cavity_roundtrip(self, tmp_path):
        cfg = SolverConfig.lid_driven_cavity(nx=64, ny=64, Re=100.0)
        path = cfg.write(tmp_path / "lid.toml")
        assert path.exists()
        data = load_config(path)
        assert data["simulation"]["n_steps"] == 20000
        assert data["fluid"]["nx"] == 64
        assert data["fluid"]["ny"] == 64

    def test_cylinder_roundtrip(self, tmp_path):
        cfg = SolverConfig.flow_around_cylinder(nx=200, ny=80, Re=100.0)
        path = cfg.write(tmp_path / "cyl.toml")
        data = load_config(path)
        assert data["ibm"]["geometry"] == "circle"
        assert data["fluid"]["nx"] == 200

    def test_fsi_filament_roundtrip(self, tmp_path):
        cfg = SolverConfig.fsi_filament(nx=150, ny=60, Re=200.0)
        path = cfg.write(tmp_path / "fsi.toml")
        data = load_config(path)
        assert "structure" in data
        assert data["ibm"]["geometry"] == "filament"

    def test_boundary_conditions_written(self, tmp_path):
        cfg = SolverConfig.lid_driven_cavity(nx=32, ny=32)
        path = cfg.write(tmp_path / "bc.toml")
        data = load_config(path)
        bcs = data["fluid"]["boundary_conditions"]
        assert isinstance(bcs, list)
        assert len(bcs) >= 1
        bc_types = [bc["bc_type"] for bc in bcs]
        assert "zou_he_velocity" in bc_types
        assert "bounce_back" in bc_types

    def test_comment_written(self, tmp_path):
        cfg = SolverConfig.lid_driven_cavity(nx=32, ny=32)
        path = cfg.write(tmp_path / "commented.toml",
                         comment="Test comment\nSecond line")
        text = path.read_text()
        assert "# Test comment" in text

    def test_creates_parent_dirs(self, tmp_path):
        cfg = SolverConfig.lid_driven_cavity()
        path = cfg.write(tmp_path / "deep" / "nested" / "cfg.toml")
        assert path.exists()

    def test_custom_config(self, tmp_path):
        cfg = SolverConfig(
            simulation=SimConfig(n_steps=500, lattice_model="D2Q9",
                                 collision_model="BGK"),
            fluid=FluidConfig(nx=32, ny=32, nu=0.1, rho0=1.0,
                              boundary_conditions=[
                                  BoundaryConditionConfig("bounce_back", "south"),
                              ]),
            output=OutputConfig(write_interval=50, directory="out/test"),
        )
        path = cfg.write(tmp_path / "custom.toml")
        data = load_config(path)
        assert data["simulation"]["collision_model"] == "BGK"
        assert data["simulation"]["n_steps"] == 500
        assert data["fluid"]["nu"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_load_example_config(self):
        """The existing lid_driven_cavity.toml should be loadable."""
        repo_root = Path(__file__).parents[2]
        example = repo_root / "configs" / "lid_driven_cavity.toml"
        if not example.exists():
            pytest.skip("Example config not present")
        data = load_config(example)
        assert "simulation" in data
        assert "fluid" in data
        assert data["fluid"]["nx"] == 100
