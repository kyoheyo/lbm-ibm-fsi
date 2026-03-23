"""
lbm_pre.config_gen
==================
通过 Python 生成求解器 TOML 配置文件，避免手动编辑，
并支持参数扫描。

Public API
----------
SimConfig      — 与 TOML [simulation] 段对应的数据类
FluidConfig    — 与 TOML [fluid] 段对应的数据类
StructureConfig— 与 TOML [structure] 段对应的数据类
IbmConfig      — 与 TOML [ibm] 段对应的数据类
OutputConfig   — 与 TOML [output] 段对应的数据类
SolverConfig   — 顶层容器；write() → TOML 文件
reynolds_to_nu()— 由 Re 和速度计算格子粘度
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
    返回运动粘度 ν = U·L / Re。

    LBM 稳定性要求  0.5 < τ < 2  ⟹  0.05 < ν < 0.5。
    当 ν 超出稳定范围时打印警告。
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
    bc_type: Literal[
        "bounce_back",           # 半步长反弹（halfway bounce-back），无滑移固壁，2 阶精度
        "bounce_back_full_way",  # 全步长反弹（on-node），无滑移固壁，1 阶精度
        "zou_he_velocity",       # Zou-He 速度边界条件（进/出口规定速度）
        "zou_he_pressure",       # Zou-He 压力边界条件（进/出口规定密度/压力）
        "fully_developed",       # 充分发展出口（零法向梯度，拷贝上游一层 f）
        "free_outlet",           # 自由出口（与 fully_developed 等价）
        "guo_extrapolation",     # 郭照立非平衡外推格式（Guo et al., 2002）
    ] = "bounce_back"
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
        序列化为 Rust 编排器可读的 TOML 文件。

        Parameters
        ----------
        path    : 输出文件路径
        comment : 写入文件顶部的可选注释行

        Returns
        -------
        已写入文件的 Path。
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
        创建顶盖驱动方腔配置。

        Parameters
        ----------
        nx, ny  : 网格分辨率
        Re      : Reynolds 数（Re = U·L / ν，L = ny）
        U_lid   : 格子单位的顶盖速度
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
        创建绕圆柱流（冯·卡门涡街）配置。

        圆柱圆心位于 (nx/4, ny/2)，半径为 ny/8。
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
    def pressure_driven_channel(cls, nx: int = 200, ny: int = 60,
                                 nu: float = 0.1,
                                 rho_in: float = 1.005,
                                 rho_out: float = 1.0) -> "SolverConfig":
        """
        创建压力驱动通道（Poiseuille）流配置。

        西（进口）面与东（出口）面之间的压力差驱动流动。
        稳态时速度剖面收敛至抛物线 Poiseuille 解 u(y) = ΔP·y·(H−y) / (2ν·ρ)。

        Parameters
        ----------
        nx, ny   : 网格分辨率（ny 为通道高度 H）
        nu       : 格子单位的运动粘度
        rho_in   : 进口密度（高压侧）
        rho_out  : 出口密度（参考压力侧）
        """
        cfg = cls(
            simulation=SimConfig(n_steps=20000, collision_model="MRT"),
            fluid=FluidConfig(
                nx=nx, ny=ny, nu=nu, rho0=1.0,
                boundary_conditions=[
                    BoundaryConditionConfig("zou_he_pressure", "west",
                                            rho=rho_in),
                    BoundaryConditionConfig("zou_he_pressure", "east",
                                            rho=rho_out),
                    BoundaryConditionConfig("bounce_back", "south"),
                    BoundaryConditionConfig("bounce_back", "north"),
                ],
            ),
            output=OutputConfig(write_interval=1000,
                                directory="output/pressure_channel"),
        )
        return cfg

    @classmethod
    def velocity_inlet_channel(cls, nx: int = 300, ny: int = 60,
                                Re: float = 30.0,
                                U_inlet: float = 0.05) -> "SolverConfig":
        """
        创建速度进口通道流配置。

        西面（进口）规定均匀速度，东面采用充分发展（零梯度）出流边界条件，
        南北面为无滑移固壁。流动从均匀进口剖面向下游发展为 Poiseuille 抛物线剖面。

        Parameters
        ----------
        nx, ny    : 网格分辨率（ny 为通道高度 H）
        Re        : Reynolds 数（Re = U·H / ν）
        U_inlet   : 格子单位的进口速度
        """
        nu = reynolds_to_nu(Re, U_inlet, float(ny))
        cfg = cls(
            simulation=SimConfig(n_steps=30000, collision_model="MRT"),
            fluid=FluidConfig(
                nx=nx, ny=ny, nu=nu, rho0=1.0,
                boundary_conditions=[
                    BoundaryConditionConfig("zou_he_velocity", "west",
                                            ux=U_inlet, uy=0.0),
                    BoundaryConditionConfig("fully_developed", "east"),
                    BoundaryConditionConfig("bounce_back", "south"),
                    BoundaryConditionConfig("bounce_back", "north"),
                ],
            ),
            output=OutputConfig(write_interval=1000,
                                directory=f"output/velocity_inlet_channel_Re{int(Re)}"),
        )
        return cfg

    @classmethod
    def velocity_inlet_free_outlet(cls, nx: int = 200, ny: int = 80,
                                    Re: float = 80.0,
                                    U_inlet: float = 0.05) -> "SolverConfig":
        """
        创建速度进口/压力出口配置，南北面采用自由出口边界条件。

        适用于流体可从顶部和底部边界流出的开放域或射流类问题。
        由于南北面使用 ``free_outlet``（非固壁 BC），相邻两面均为
        非固壁 BC 的角节点（SW/SE/NW/NE）将自动跳过角落反弹修正，
        避免在开放角落施加非物理的 u = 0 约束。

        Parameters
        ----------
        nx, ny    : 网格分辨率
        Re        : Reynolds 数（Re = U·H / ν，H = ny）
        U_inlet   : 格子单位的进口速度
        """
        nu = reynolds_to_nu(Re, U_inlet, float(ny))
        cfg = cls(
            simulation=SimConfig(n_steps=25000, collision_model="MRT"),
            fluid=FluidConfig(
                nx=nx, ny=ny, nu=nu, rho0=1.0,
                boundary_conditions=[
                    BoundaryConditionConfig("zou_he_velocity", "west",
                                            ux=U_inlet, uy=0.0),
                    BoundaryConditionConfig("zou_he_pressure", "east",
                                            rho=1.0),
                    BoundaryConditionConfig("free_outlet", "south"),
                    BoundaryConditionConfig("free_outlet", "north"),
                ],
            ),
            output=OutputConfig(write_interval=1000,
                                directory=f"output/velocity_inlet_free_outlet_Re{int(Re)}"),
        )
        return cfg

    @classmethod
    def fsi_filament(cls, nx: int = 150, ny: int = 60,
                     Re: float = 200.0,
                     U_inlet: float = 0.05) -> "SolverConfig":
        """
        创建柔性细丝 FSI 配置。
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
    加载求解器 TOML 配置文件并以普通 dict 返回。

    使用标准库 ``tomllib`` 模块（Python ≥ 3.11）。

    Parameters
    ----------
    path : TOML 文件路径

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
