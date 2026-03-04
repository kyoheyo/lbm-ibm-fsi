"""
lbm_pre.mesh
============
Gmsh-based mesh generation helpers for LBM domains.

The LBM solver uses a Cartesian Eulerian grid, so this module's primary job
is to generate *background meshes* that describe the geometry boundary to the
solver (e.g. for pre-visualisation, for IBM marker placement, or for hybrid
body-fitted / LBM coupling).  It also exports mesh statistics and converts
gmsh outputs to formats understood by the solver.

Public API
----------
MeshBuilder           — fluent interface for building rectangular 2-D/3-D domains
write_mesh_info()     — print or return a dict of mesh statistics
export_surface_nodes()— extract boundary node coordinates for IBM marker seeding
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence, Tuple

import gmsh
import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MeshInfo:
    """Summary statistics returned by :meth:`MeshBuilder.build`."""
    n_nodes: int
    n_elements: int
    lc_min: float
    lc_max: float
    dim: int
    physical_groups: dict[str, int] = field(default_factory=dict)


@dataclass
class CircleObstacle:
    """A circular hole / immersed boundary inside the domain."""
    cx: float          #: centre x  (lattice units)
    cy: float          #: centre y  (lattice units)
    radius: float      #: radius    (lattice units)
    mesh_size: float   #: local characteristic length near boundary
    tag: str = "circle"


@dataclass
class RectangleObstacle:
    """A rectangular hole / immersed boundary inside the domain."""
    x0: float
    y0: float
    width: float
    height: float
    mesh_size: float
    tag: str = "rectangle"


# ---------------------------------------------------------------------------
# MeshBuilder
# ---------------------------------------------------------------------------

class MeshBuilder:
    """
    Fluent builder for 2-D rectangular LBM domains, optionally with internal
    obstacles (circular or rectangular holes) for IBM.

    Example
    -------
    >>> mb = (MeshBuilder(nx=100, ny=100)
    ...       .set_mesh_size(2.0)
    ...       .add_circle_obstacle(cx=50, cy=50, radius=10, mesh_size=1.0)
    ...       .build())
    >>> mb.write("domain.msh")
    """

    def __init__(self, nx: int, ny: int, nz: int = 1, *,
                 lx: Optional[float] = None,
                 ly: Optional[float] = None,
                 lz: Optional[float] = None) -> None:
        """
        Parameters
        ----------
        nx, ny, nz  : grid resolution (lattice units)
        lx, ly, lz  : physical domain length; defaults to nx, ny, nz (Δx = 1)
        """
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.lx = float(lx if lx is not None else nx)
        self.ly = float(ly if ly is not None else ny)
        self.lz = float(lz if lz is not None else nz)
        self._mesh_size: float = min(self.lx, self.ly) / max(nx, ny)
        self._circles: list[CircleObstacle] = []
        self._rectangles: list[RectangleObstacle] = []
        self._built = False
        self._mesh_info: Optional[MeshInfo] = None
        self._output_path: Optional[Path] = None

    # ------------------------------------------------------------------
    # Fluent setters
    # ------------------------------------------------------------------

    def set_mesh_size(self, lc: float) -> "MeshBuilder":
        """Set the global characteristic mesh size."""
        self._mesh_size = float(lc)
        return self

    def add_circle_obstacle(self, cx: float, cy: float, radius: float,
                             mesh_size: Optional[float] = None,
                             tag: str = "circle") -> "MeshBuilder":
        """Add a circular obstacle (hole) inside the domain."""
        lc = mesh_size if mesh_size is not None else self._mesh_size * 0.5
        self._circles.append(CircleObstacle(cx, cy, radius, lc, tag))
        return self

    def add_rectangle_obstacle(self, x0: float, y0: float,
                                width: float, height: float,
                                mesh_size: Optional[float] = None,
                                tag: str = "rectangle") -> "MeshBuilder":
        """Add a rectangular obstacle (hole) inside the domain."""
        lc = mesh_size if mesh_size is not None else self._mesh_size * 0.5
        self._rectangles.append(RectangleObstacle(x0, y0, width, height, lc, tag))
        return self

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> "MeshBuilder":
        """Construct the gmsh model (does not write to disk yet)."""
        gmsh.initialize(sys.argv, False)
        gmsh.option.setNumber("General.Verbosity", 1)
        gmsh.model.add("lbm_domain")

        if self.nz <= 1:
            self._build_2d()
        else:
            self._build_3d()

        # Global mesh size field
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self._mesh_size * 0.1)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self._mesh_size * 4.0)

        gmsh.model.mesh.generate(2 if self.nz <= 1 else 3)
        gmsh.model.mesh.optimize("Netgen")

        n_nodes = len(gmsh.model.mesh.get_nodes()[0])
        elem_types, elem_tags, _ = gmsh.model.mesh.get_elements()
        n_elements = sum(len(tags) for tags in elem_tags)

        groups = {}
        for dim, tag in gmsh.model.get_physical_groups():
            name = gmsh.model.get_physical_name(dim, tag)
            groups[name] = tag

        self._mesh_info = MeshInfo(
            n_nodes=n_nodes,
            n_elements=n_elements,
            lc_min=self._mesh_size * 0.1,
            lc_max=self._mesh_size * 4.0,
            dim=2 if self.nz <= 1 else 3,
            physical_groups=groups,
        )
        self._built = True
        return self

    def _build_2d(self) -> None:
        """Build 2-D rectangular domain with optional circle/rectangle holes."""
        occ = gmsh.model.occ

        # Outer rectangle
        domain = occ.add_rectangle(0.0, 0.0, 0.0, self.lx, self.ly)

        cut_tools: list[tuple[int, int]] = []

        # Circular holes
        for i, c in enumerate(self._circles):
            disk = occ.add_disk(c.cx, c.cy, 0.0, c.radius, c.radius)
            cut_tools.append((2, disk))

        # Rectangular holes
        for i, r in enumerate(self._rectangles):
            rect = occ.add_rectangle(r.x0, r.y0, 0.0, r.width, r.height)
            cut_tools.append((2, rect))

        if cut_tools:
            occ.cut([(2, domain)], cut_tools, removeObject=True, removeTool=True)

        occ.synchronize()

        # Physical groups for boundary condition tagging
        surfaces = gmsh.model.get_entities(2)
        gmsh.model.add_physical_group(2, [s[1] for s in surfaces], name="fluid")

        curves = gmsh.model.get_entities(1)
        # Tag curves by their bounding-box position
        west, east, south, north, immersed = [], [], [], [], []
        tol = min(self.lx, self.ly) * 1e-4
        for _, ctag in curves:
            xmin, ymin, _, xmax, ymax, _ = gmsh.model.get_bounding_box(1, ctag)
            cx_mid = (xmin + xmax) / 2
            cy_mid = (ymin + ymax) / 2
            if abs(cx_mid) < tol:
                west.append(ctag)
            elif abs(cx_mid - self.lx) < tol:
                east.append(ctag)
            elif abs(cy_mid) < tol:
                south.append(ctag)
            elif abs(cy_mid - self.ly) < tol:
                north.append(ctag)
            else:
                immersed.append(ctag)

        if west:
            gmsh.model.add_physical_group(1, west, name="west")
        if east:
            gmsh.model.add_physical_group(1, east, name="east")
        if south:
            gmsh.model.add_physical_group(1, south, name="south")
        if north:
            gmsh.model.add_physical_group(1, north, name="north")
        if immersed:
            gmsh.model.add_physical_group(1, immersed, name="immersed_boundary")

        # Mesh size fields
        field_list = []
        for i, c in enumerate(self._circles):
            f = gmsh.model.mesh.field.add("Ball")
            gmsh.model.mesh.field.setNumber(f, "Radius", c.radius * 2)
            gmsh.model.mesh.field.setNumber(f, "VIn", c.mesh_size)
            gmsh.model.mesh.field.setNumber(f, "VOut", self._mesh_size)
            gmsh.model.mesh.field.setNumber(f, "XCenter", c.cx)
            gmsh.model.mesh.field.setNumber(f, "YCenter", c.cy)
            gmsh.model.mesh.field.setNumber(f, "ZCenter", 0.0)
            field_list.append(f)

        if field_list:
            fmin = gmsh.model.mesh.field.add("Min")
            gmsh.model.mesh.field.setNumbers(fmin, "FieldsList", field_list)
            gmsh.model.mesh.field.setAsBackgroundMesh(fmin)
        else:
            gmsh.model.mesh.field.setAsBackgroundMesh(0)

    def _build_3d(self) -> None:
        """Build 3-D rectangular domain (no holes for now)."""
        occ = gmsh.model.occ
        box = occ.add_box(0.0, 0.0, 0.0, self.lx, self.ly, self.lz)
        occ.synchronize()
        vols = gmsh.model.get_entities(3)
        gmsh.model.add_physical_group(3, [v[1] for v in vols], name="fluid")

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def write(self, path: str | Path) -> Path:
        """Write the mesh to *path* (format inferred from suffix)."""
        if not self._built:
            self.build()
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        gmsh.write(str(p))
        self._output_path = p
        return p

    def info(self) -> MeshInfo:
        """Return mesh statistics (builds first if needed)."""
        if not self._built:
            self.build()
        assert self._mesh_info is not None
        return self._mesh_info

    def finalize(self) -> None:
        """Release gmsh resources."""
        if gmsh.is_initialized():
            gmsh.finalize()

    def __enter__(self) -> "MeshBuilder":
        return self

    def __exit__(self, *_: object) -> None:
        self.finalize()


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def export_surface_nodes(msh_path: str | Path,
                          group_name: str = "immersed_boundary") -> np.ndarray:
    """
    Extract the (x, y) coordinates of all nodes in a named physical group from
    a .msh file.  Useful for seeding IBM Lagrangian markers from a gmsh mesh.

    Parameters
    ----------
    msh_path   : path to a .msh file previously written by :meth:`MeshBuilder.write`
    group_name : physical group name to extract

    Returns
    -------
    coords : ndarray of shape (N, 2) — sorted counter-clockwise
    """
    gmsh.initialize(sys.argv, False)
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.open(str(msh_path))

    coords_list: list[np.ndarray] = []
    for dim, tag in gmsh.model.get_physical_groups():
        if gmsh.model.get_physical_name(dim, tag) == group_name:
            node_tags, node_coords = gmsh.model.mesh.get_nodes_for_physical_group(dim, tag)
            xy = node_coords.reshape(-1, 3)[:, :2]
            coords_list.append(xy)

    gmsh.finalize()

    if not coords_list:
        return np.empty((0, 2))

    coords = np.vstack(coords_list)
    # Sort by angle around centroid (counter-clockwise)
    centroid = coords.mean(axis=0)
    angles = np.arctan2(coords[:, 1] - centroid[1], coords[:, 0] - centroid[0])
    coords = coords[np.argsort(angles)]
    return coords


def make_channel_mesh(nx: int, ny: int, *,
                      circle_cx: Optional[float] = None,
                      circle_cy: Optional[float] = None,
                      circle_r: Optional[float] = None,
                      mesh_size: Optional[float] = None,
                      output: str | Path = "domain.msh") -> MeshInfo:
    """
    Convenience wrapper: rectangular channel with optional circular cylinder.

    Parameters
    ----------
    nx, ny          : grid resolution in lattice units
    circle_cx/cy/r  : if given, add a circular obstacle at this location
    mesh_size       : global characteristic length (default: ny / 20)
    output          : output file path

    Returns
    -------
    MeshInfo
    """
    lc = mesh_size or max(nx, ny) / 20.0
    with MeshBuilder(nx, ny).set_mesh_size(lc) as mb:
        if circle_cx is not None:
            mb.add_circle_obstacle(circle_cx, circle_cy, circle_r,
                                   mesh_size=lc * 0.3)
        mb.build()
        mb.write(output)
        return mb.info()
