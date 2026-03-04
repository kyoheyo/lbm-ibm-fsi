"""
Tests for lbm_pre.mesh — gmsh-based domain mesh generation.

These tests build minimal meshes to verify the API contract without
incurring the overhead of meshing large domains.  Each test rebuilds
the gmsh model from scratch because gmsh is a global singleton.
"""

import math
from pathlib import Path

import numpy as np
import pytest

gmsh = pytest.importorskip("gmsh", reason="gmsh is not installed")

from lbm_pre.mesh import MeshBuilder, MeshInfo, export_surface_nodes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_builder(*, circles=None, rects=None,
                   nx=20, ny=20) -> MeshBuilder:
    """Return a MeshBuilder for a 20×20 domain with coarse mesh."""
    mb = MeshBuilder(nx, ny).set_mesh_size(4.0)
    if circles:
        for c in circles:
            mb.add_circle_obstacle(*c)
    if rects:
        for r in rects:
            mb.add_rectangle_obstacle(*r)
    return mb


# ---------------------------------------------------------------------------
# MeshInfo after build
# ---------------------------------------------------------------------------

class TestMeshBuild:
    def test_plain_domain_has_nodes(self):
        with _small_builder() as mb:
            info = mb.build().info()
        assert info.n_nodes > 0

    def test_plain_domain_has_elements(self):
        with _small_builder() as mb:
            info = mb.build().info()
        assert info.n_elements > 0

    def test_dim_is_2(self):
        with _small_builder() as mb:
            info = mb.build().info()
        assert info.dim == 2

    def test_physical_groups_contain_fluid(self):
        with _small_builder() as mb:
            info = mb.build().info()
        assert "fluid" in info.physical_groups

    def test_physical_groups_contain_boundary_faces(self):
        with _small_builder() as mb:
            info = mb.build().info()
        # At least some of the domain edges should be tagged
        pg = info.physical_groups
        # west OR east OR south OR north must be present
        boundary_names = {"west", "east", "south", "north"}
        assert len(boundary_names & set(pg.keys())) >= 1

    def test_circle_obstacle_adds_immersed_group(self):
        with _small_builder(circles=[(10, 10, 3.0)]) as mb:
            info = mb.build().info()
        assert "immersed_boundary" in info.physical_groups


# ---------------------------------------------------------------------------
# Write / file output
# ---------------------------------------------------------------------------

class TestMeshWrite:
    def test_write_msh(self, tmp_path):
        with _small_builder() as mb:
            p = mb.build().write(tmp_path / "domain.msh")
        assert p.exists()
        assert p.suffix == ".msh"
        assert p.stat().st_size > 0

    def test_write_creates_parent_dirs(self, tmp_path):
        with _small_builder() as mb:
            p = mb.build().write(tmp_path / "a" / "b" / "d.msh")
        assert p.exists()

    def test_auto_build_on_write(self, tmp_path):
        """write() should call build() automatically."""
        mb = MeshBuilder(20, 20).set_mesh_size(4.0)
        try:
            p = mb.write(tmp_path / "auto.msh")
            assert p.exists()
        finally:
            mb.finalize()


# ---------------------------------------------------------------------------
# export_surface_nodes
# ---------------------------------------------------------------------------

class TestExportSurfaceNodes:
    def test_returns_ndarray(self, tmp_path):
        with _small_builder(circles=[(10, 10, 3.0)]) as mb:
            msh = mb.build().write(tmp_path / "d.msh")
        coords = export_surface_nodes(msh, "immersed_boundary")
        assert isinstance(coords, np.ndarray)

    def test_shape_is_n_by_2(self, tmp_path):
        with _small_builder(circles=[(10, 10, 3.0)]) as mb:
            msh = mb.build().write(tmp_path / "d.msh")
        coords = export_surface_nodes(msh, "immersed_boundary")
        assert coords.ndim == 2
        assert coords.shape[1] == 2

    def test_missing_group_returns_empty(self, tmp_path):
        with _small_builder() as mb:
            msh = mb.build().write(tmp_path / "plain.msh")
        coords = export_surface_nodes(msh, "nonexistent_group")
        assert coords.shape == (0, 2)

    def test_nodes_near_circle(self, tmp_path):
        """Surface nodes should lie approximately on the circle."""
        cx, cy, r = 10.0, 10.0, 3.0
        with _small_builder(circles=[(cx, cy, r, 0.5)]) as mb:
            msh = mb.build().write(tmp_path / "c.msh")
        coords = export_surface_nodes(msh, "immersed_boundary")
        if len(coords) == 0:
            pytest.skip("No immersed boundary nodes found (coarse mesh)")
        dist = np.sqrt((coords[:, 0] - cx)**2 + (coords[:, 1] - cy)**2)
        # Nodes should be within one mesh-size of the circle
        assert (np.abs(dist - r) < 1.0).all()


# ---------------------------------------------------------------------------
# Fluent interface
# ---------------------------------------------------------------------------

class TestFluentInterface:
    def test_chaining_returns_builder(self):
        mb = MeshBuilder(20, 20)
        try:
            result = mb.set_mesh_size(3.0)
            assert result is mb
            result2 = mb.add_circle_obstacle(10, 10, 3.0)
            assert result2 is mb
        finally:
            mb.finalize()
