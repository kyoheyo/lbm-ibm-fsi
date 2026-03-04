"""
lbm_pre.config_gen
==================
Generate solver TOML configuration files from Python, avoiding manual
editing and enabling parameter sweeps.

Public API
----------
SimConfig      — dataclass mirroring the TOML [simulation] section
FluidConfig    — dataclass mirroring the TOML [fluid] section
StructureConfig— dataclass mirroring the TOML [structure] section
IbmConfig      — dataclass mirroring the TOML [ibm] section
OutputConfig   — dataclass mirroring the TOML [output] section
SolverConfig   — top-level container; write() → TOML file
reynolds_to_nu()— compute lattice viscosity from Re and velocity
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Literal, Optional

__all__ = [
    "BoundaryConditionConfig",
    "SimConfig",
    "FluidConfig",
    "StructureConfig",
    "IbmConfig",
    "OutputConfig",
    "SolverConfig",
    "reynolds_to_nu",
    "load_config",
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def reynolds_to_nu(Re: float, U: float, L: float) -> float:
    """
    Return the kinematic viscosity ν = U·L / Re.

    For LBM stability we need  0.5 < τ < 2  ⟹  0.05 < ν < 0.5.
    A warning is printed when ν is outside the stable range.
    """
    nu = U * L / Re
    if not (0.05 <= nu <= 0.5):
        print(
            f"[config_gen] WARNING: ν={nu:.4f} is outside the stable LBM range "
            f"[0.05, 0.5] for Re={Re}, U={U}, L={L}.",
            file=sys.stderr,
        )
    return nu


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class BoundaryConditionConfig:
    bc_type: Literal["bounce_back", "zou_he_velocity", "zou_he_pressure"] = "bounce_back"
    face: Literal["west", "east", "south", "north", "bottom", "top"] = "south"
    ux: float = 0.0
    uy: float = 0.0
    uz: float = 0.0
    rho: float = 1.0


@dataclass
class SimConfig:
    n_steps: int = 10000
    dt: float = 1.0
    lattice_model: Literal["D2Q9", "D3Q19", "D3Q27"] = "D2Q9"
    collision_model: Literal["BGK", "MRT"] = "MRT"


@dataclass
class FluidConfig:
    nx: int = 100
    ny: int = 100
    nz: int = 1
    nu: float = 0.01
    rho0: float = 1.0
    boundary_conditions: list[BoundaryConditionConfig] = field(default_factory=list)


@dataclass
class StructureConfig:
    young_modulus: float = 1.0e3
    second_moment: float = 1.0e-6
    density: float = 1.5
    area: float = 0.01
    length: float = 0.3
    n_elements: int = 10


@dataclass
class IbmConfig:
    geometry: Literal["circle", "filament"] = "filament"
    x0: float = 50.0
    y0: float = 50.0
    size: float = 10.0
    n_markers: int = 64
    delta_kernel: Literal["two_point", "four_point"] = "four_point"


@dataclass
class OutputConfig:
    write_interval: int = 500
    directory: str = "output"


@dataclass
class SolverConfig:
    simulation: SimConfig = field(default_factory=SimConfig)
    fluid: FluidConfig = field(default_factory=FluidConfig)
    structure: Optional[StructureConfig] = None
    ibm: Optional[IbmConfig] = None
    output: OutputConfig = field(default_factory=OutputConfig)

    # ------------------------------------------------------------------
    def write(self, path: str | Path, *, comment: str = "") -> Path:
        """
        Serialise to a TOML file understood by the Rust orchestrator.

        Parameters
        ----------
        path    : output file path
        comment : optional comment line added at the top of the file

        Returns
        -------
        Path of the written file.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w") as f:
            if comment:
                for line in comment.splitlines():
                    f.write(f"# {line}\n")
                f.write("\n")
            _write_section(f, "simulation", self.simulation)
            _write_section(f, "fluid", self.fluid,
                           skip_keys={"boundary_conditions"})
            # Write boundary conditions as array of tables
            for bc in self.fluid.boundary_conditions:
                f.write("\n[[fluid.boundary_conditions]]\n")
                for k, v in asdict(bc).items():
                    f.write(f"{k} = {_toml_value(v)}\n")
            if self.structure is not None:
                _write_section(f, "structure", self.structure)
            if self.ibm is not None:
                _write_section(f, "ibm", self.ibm)
            _write_section(f, "output", self.output)
        return p

    @classmethod
    def lid_driven_cavity(cls, nx: int = 100, ny: int = 100,
                           Re: float = 100.0,
                           U_lid: float = 0.1) -> "SolverConfig":
        """
        Create a lid-driven cavity configuration.

        Parameters
        ----------
        nx, ny  : grid resolution
        Re      : Reynolds number  (Re = U·L / ν, L = ny)
        U_lid   : lid velocity in lattice units
        """
        nu = reynolds_to_nu(Re, U_lid, float(ny))
        cfg = cls(
            simulation=SimConfig(n_steps=20000, collision_model="MRT"),
            fluid=FluidConfig(
                nx=nx, ny=ny, nu=nu, rho0=1.0,
                boundary_conditions=[
                    BoundaryConditionConfig("zou_he_velocity", "north",
                                            ux=U_lid, uy=0.0),
                    BoundaryConditionConfig("bounce_back", "south"),
                    BoundaryConditionConfig("bounce_back", "west"),
                    BoundaryConditionConfig("bounce_back", "east"),
                ],
            ),
            output=OutputConfig(write_interval=500,
                                directory=f"output/lid_cavity_Re{int(Re)}"),
        )
        return cfg

    @classmethod
    def flow_around_cylinder(cls, nx: int = 200, ny: int = 80,
                              Re: float = 100.0,
                              U_inlet: float = 0.05) -> "SolverConfig":
        """
        Create a flow-around-cylinder (von Kármán vortex street) configuration.

        The cylinder is centred at (nx/4, ny/2) with radius ny/8.
        """
        nu = reynolds_to_nu(Re, U_inlet, float(ny // 4))  # D = ny/4
        r = ny // 8
        cfg = cls(
            simulation=SimConfig(n_steps=50000, collision_model="MRT"),
            fluid=FluidConfig(
                nx=nx, ny=ny, nu=nu, rho0=1.0,
                boundary_conditions=[
                    BoundaryConditionConfig("zou_he_velocity", "west",
                                            ux=U_inlet, uy=0.0),
                    BoundaryConditionConfig("zou_he_pressure", "east",
                                            rho=1.0),
                    BoundaryConditionConfig("bounce_back", "south"),
                    BoundaryConditionConfig("bounce_back", "north"),
                ],
            ),
            ibm=IbmConfig(
                geometry="circle",
                x0=float(nx // 4),
                y0=float(ny // 2),
                size=float(r),
                n_markers=int(2 * 3.14159 * r),
                delta_kernel="four_point",
            ),
            output=OutputConfig(write_interval=200,
                                directory=f"output/cylinder_Re{int(Re)}"),
        )
        return cfg

    @classmethod
    def fsi_filament(cls, nx: int = 150, ny: int = 60,
                     Re: float = 200.0,
                     U_inlet: float = 0.05) -> "SolverConfig":
        """
        Create a flexible-filament FSI configuration.
        """
        nu = reynolds_to_nu(Re, U_inlet, float(ny))
        cfg = cls(
            simulation=SimConfig(n_steps=80000, collision_model="MRT"),
            fluid=FluidConfig(
                nx=nx, ny=ny, nu=nu, rho0=1.0,
                boundary_conditions=[
                    BoundaryConditionConfig("zou_he_velocity", "west",
                                            ux=U_inlet, uy=0.0),
                    BoundaryConditionConfig("zou_he_pressure", "east",
                                            rho=1.0),
                    BoundaryConditionConfig("bounce_back", "south"),
                    BoundaryConditionConfig("bounce_back", "north"),
                ],
            ),
            structure=StructureConfig(
                young_modulus=1.0e3, second_moment=1.0e-6,
                density=1.5, area=0.01,
                length=float(ny // 3), n_elements=15,
            ),
            ibm=IbmConfig(
                geometry="filament",
                x0=float(nx // 4),
                y0=float(ny // 2) - float(ny // 6),
                size=float(ny // 3),
                n_markers=30,
                delta_kernel="four_point",
            ),
            output=OutputConfig(write_interval=400,
                                directory=f"output/fsi_filament_Re{int(Re)}"),
        )
        return cfg


# ---------------------------------------------------------------------------
# TOML reader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> dict:
    """
    Load a solver TOML config file and return it as a plain dict.

    Uses the stdlib ``tomllib`` module (Python ≥ 3.11).

    Parameters
    ----------
    path : path to the TOML file

    Returns
    -------
    dict
    """
    with open(path, "rb") as f:
        return tomllib.load(f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _toml_value(v: object) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, float):
        # Use exponential notation for very large/small values
        return f"{v:.10g}"
    return str(v)


def _write_section(f, section: str, obj: object,
                   skip_keys: set[str] | None = None) -> None:
    f.write(f"\n[{section}]\n")
    for k, v in asdict(obj).items():
        if skip_keys and k in skip_keys:
            continue
        if v is None:
            continue
        if isinstance(v, list):
            continue  # handled separately (array of tables)
        f.write(f"{k} = {_toml_value(v)}\n")
