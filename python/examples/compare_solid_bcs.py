#!/usr/bin/env python3
"""
examples/compare_solid_bcs.py
==============================
固体边界条件比较后处理脚本。

本脚本配合 ``configs/solid_bc_comparison.toml`` 仿真输出使用，读取以下数据：

- 流场快照（ASCII Tecplot ``fluid_NNNNNN.dat``）：最后一帧
- 各圆柱受力 CSV（``cyl_ibb_force.csv``、``cyl_mdf_force.csv``、
  ``cyl_mls_force.csv``、``cyl_pen_force.csv``）

生成以下对比图像（保存至输出目录）：

1. ``velocity_magnitude.png`` — 速度幅值云图（含 4 个圆柱）
2. ``pressure.png``           — 压力云图（p = ρ·cs² = ρ/3，格子单位）
3. ``vorticity.png``          — 涡量云图（ωz = ∂uy/∂x − ∂ux/∂y）
4. ``force_fx_comparison.png`` — 阻力 Fx(t) 四方法对比曲线
5. ``force_fy_comparison.png`` — 升力 Fy(t) 四方法对比曲线

用法
----
独立运行（从仓库根目录）::

    python3 python/examples/compare_solid_bcs.py

指定输出目录::

    python3 python/examples/compare_solid_bcs.py --input output/solid_bc_comparison

在 TOML 配置中自动调用（仿真结束后）::

    [python]
    post_script = "python/examples/compare_solid_bcs.py"
    pythonpath  = "python"

此时求解器将输出目录作为第一个位置参数传入。

方法说明
--------
+---------------------------+-----------------------------------+---------------------------+
| 标签                      | 方法                              | 参考文献                  |
+===========================+===================================+===========================+
| cyl_ibb  (IBB)            | 插值反弹格式（二阶精度）          | Bouzidi 2001              |
+---------------------------+-----------------------------------+---------------------------+
| cyl_mdf  (IBM-MDF)        | 多重直接力法                      | Luo et al. 2007           |
+---------------------------+-----------------------------------+---------------------------+
| cyl_mls  (IBM-MLS)        | 移动最小二乘插值 + 直接力         | Wang et al. 2009          |
+---------------------------+-----------------------------------+---------------------------+
| cyl_pen  (IBM-Penalty)    | 罚函数反馈力法                    | Goldstein et al. 1993     |
+---------------------------+-----------------------------------+---------------------------+
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # 无显示器环境下使用非交互后端
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# sys.path 引导：确保无论工作目录如何，lbm_post 均可导入。
# __file__ → .../python/examples/compare_solid_bcs.py
# .parent  → .../python/examples/
# .parent  → .../python/          ← 包根目录
# ---------------------------------------------------------------------------
_PYTHON_DIR = Path(__file__).resolve().parent.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

# ---------------------------------------------------------------------------
# 方法元数据（标签 → 显示名称 + 曲线样式）
# ---------------------------------------------------------------------------

# 每个条目：(csv_stem, display_label, color, linestyle)
_METHODS: list[tuple[str, str, str, str]] = [
    ("cyl_ibb_force",  "IBB（插值反弹，Bouzidi 2001）",         "#1f77b4", "-"),
    ("cyl_mdf_force",  "IBM-MDF（多重直接力，Luo 2007）",        "#ff7f0e", "--"),
    ("cyl_mls_force",  "IBM-MLS（移动最小二乘，Wang 2009）",     "#2ca02c", "-."),
    ("cyl_pen_force",  "IBM-Penalty（罚函数反馈，Goldstein 1993）", "#d62728", ":"),
]


# ---------------------------------------------------------------------------
# 命令行解析
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "固体边界条件比较后处理：读取 solid_bc_comparison 仿真输出，"
            "生成压力/速度/涡量云图及各方法受力对比曲线。"
        )
    )
    p.add_argument(
        "positional_input",
        nargs="?",
        metavar="INPUT_DIR",
        help=(
            "包含快照文件和受力 CSV 的目录"
            "（作为 post_script 调用时由求解器自动传入）。"
        ),
    )
    p.add_argument(
        "--input", "-i",
        metavar="INPUT_DIR",
        default=None,
        help="包含仿真输出的目录（默认：output/solid_bc_comparison）。",
    )
    p.add_argument(
        "--output", "-o",
        metavar="OUTPUT_DIR",
        default=None,
        help="图像文件的保存目录（默认：与 INPUT_DIR 相同）。",
    )
    p.add_argument(
        "--fmt",
        metavar="FORMAT",
        default=None,
        choices=["tecplot_asc", "npz", "tecplot_bin"],
        help="快照格式（默认自动检测）。",
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        metavar="N",
        help="指定绘制某个时间步（默认：最后一帧）。",
    )
    args = p.parse_args()
    if args.positional_input is not None:
        args.input = args.positional_input
    return args


# ---------------------------------------------------------------------------
# CSV 受力数据读取
# ---------------------------------------------------------------------------

def _read_force_csv(path: Path) -> tuple[list[int], list[float], list[float]]:
    """读取受力 CSV（列：step, time, fx, fy），返回 (steps, fx_list, fy_list)。"""
    steps, fx_list, fy_list = [], [], []
    if not path.exists():
        return steps, fx_list, fy_list
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                steps.append(int(row["step"]))
                fx_list.append(float(row["fx"]))
                fy_list.append(float(row["fy"]))
            except (KeyError, ValueError):
                continue
    return steps, fx_list, fy_list


# ---------------------------------------------------------------------------
# 流场云图绘制（压力 / 速度 / 涡量）
# ---------------------------------------------------------------------------

def _plot_flow_fields(snap, output_dir: Path) -> None:
    """绘制压力云图、速度云图、涡量云图并保存到 output_dir。"""
    from lbm_post.plot import (
        plot_velocity_magnitude,
        plot_pressure,
        plot_vorticity,
        save_figure,
    )

    step_tag = f"  step={snap.step}"

    # 1. 速度幅值云图
    fig, ax = plot_velocity_magnitude(snap, figsize=(14, 10))
    ax.set_title(f"速度幅值云图 — 固体边界条件比较（Re=100）{step_tag}")
    _annotate_cylinders(ax)
    save_figure(fig, output_dir / "velocity_magnitude.png")
    plt.close(fig)
    print("  [1/5] velocity_magnitude.png")

    # 2. 压力云图
    fig, ax = plot_pressure(snap, figsize=(14, 10))
    ax.set_title(f"压力云图（p = ρ/3）— 固体边界条件比较{step_tag}")
    _annotate_cylinders(ax)
    save_figure(fig, output_dir / "pressure.png")
    plt.close(fig)
    print("  [2/5] pressure.png")

    # 3. 涡量云图
    fig, ax = plot_vorticity(snap, figsize=(14, 10))
    ax.set_title(f"涡量云图（ωz）— 固体边界条件比较{step_tag}")
    _annotate_cylinders(ax)
    save_figure(fig, output_dir / "vorticity.png")
    plt.close(fig)
    print("  [3/5] vorticity.png")


def _annotate_cylinders(ax) -> None:
    """在流场云图上标注各圆柱位置和方法名称。"""
    labels = [
        (200, 100,  "① IBB"),
        (200, 250,  "② IBM-MDF"),
        (200, 400,  "③ IBM-MLS"),
        (200, 550,  "④ IBM-Penalty"),
    ]
    for x, y, lbl in labels:
        circle = plt.Circle((x, y), 10, fill=False, edgecolor="white",
                             linewidth=1.0, linestyle="--")
        ax.add_patch(circle)
        ax.text(x + 15, y, lbl, color="white", fontsize=7,
                verticalalignment="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.5))


# ---------------------------------------------------------------------------
# 受力对比曲线
# ---------------------------------------------------------------------------

def _plot_force_comparison(input_dir: Path, output_dir: Path) -> None:
    """绘制 Fx(t) 和 Fy(t) 对比曲线，每个文件一条线。"""
    fig_fx, ax_fx = plt.subplots(figsize=(12, 5))
    fig_fy, ax_fy = plt.subplots(figsize=(12, 5))

    any_data = False
    for csv_stem, disp_label, color, ls in _METHODS:
        csv_path = input_dir / f"{csv_stem}.csv"
        steps, fx_list, fy_list = _read_force_csv(csv_path)
        if not steps:
            print(f"  [警告] 未找到受力 CSV：{csv_path}，跳过 {csv_stem}")
            continue
        any_data = True
        ax_fx.plot(steps, fx_list, label=disp_label,
                   color=color, linestyle=ls, linewidth=1.2)
        ax_fy.plot(steps, fy_list, label=disp_label,
                   color=color, linestyle=ls, linewidth=1.2)

    if not any_data:
        print("  [警告] 所有受力 CSV 均为空，跳过受力对比图")
        plt.close(fig_fx)
        plt.close(fig_fy)
        return

    for ax, comp, fname in [
        (ax_fx, "Fx（阻力）", "force_fx_comparison.png"),
        (ax_fy, "Fy（升力）", "force_fy_comparison.png"),
    ]:
        ax.set_xlabel("时间步", fontsize=12)
        ax.set_ylabel(f"{comp}（格子单位）", fontsize=12)
        ax.set_title(f"{comp} 随时间变化 — 固体边界条件对比（Re=100）", fontsize=13)
        ax.legend(fontsize=9, loc="upper right")
        ax.grid(True, alpha=0.3)
        fig = ax.get_figure()
        fig.tight_layout()
        out_path = output_dir / fname
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [4-5/5] {fname}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    # ──────────────────────────────────────────────────────────────────────
    # 输入 / 输出目录解析
    # ──────────────────────────────────────────────────────────────────────
    if args.input is not None:
        input_dir = Path(args.input)
    else:
        repo_root = _PYTHON_DIR.parent
        input_dir = repo_root / "output" / "solid_bc_comparison"

    if not input_dir.exists():
        print(
            f"[compare_solid_bcs] 错误：输入目录不存在：{input_dir}\n"
            "  请先运行仿真：\n"
            "    ./lbm-fsi configs/solid_bc_comparison.toml",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output) if args.output else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[compare_solid_bcs] 输入目录 : {input_dir}")
    print(f"[compare_solid_bcs] 输出目录 : {output_dir}")

    # ──────────────────────────────────────────────────────────────────────
    # 加载流场快照（自动检测格式）
    # ──────────────────────────────────────────────────────────────────────
    from lbm_post.vtk_reader import NpzReader, TecplotAscReader, TecplotBinReader

    fmt = args.fmt
    if fmt is None:
        if list(input_dir.glob("fluid_*.dat")):
            fmt = "tecplot_asc"
        elif list(input_dir.glob("fluid_*.npz")):
            fmt = "npz"
        elif list(input_dir.glob("fluid_*.plt")):
            fmt = "tecplot_bin"
        else:
            print(
                f"[compare_solid_bcs] 未在 {input_dir} 中找到流场快照文件。\n"
                "  跳过流场云图，仅生成受力对比曲线（若 CSV 已存在）。"
            )

    snap = None
    if fmt is not None:
        reader_map = {
            "tecplot_asc": TecplotAscReader,
            "npz":         NpzReader,
            "tecplot_bin": TecplotBinReader,
        }
        reader = reader_map[fmt](input_dir)
        if len(reader) > 0:
            snap = reader.read(args.step) if args.step is not None else reader.last()
            print(
                f"[compare_solid_bcs] 流场格式: {fmt}  快照数: {len(reader)}"
                f"  使用: step={snap.step}  time={snap.time:.2f}"
            )
        else:
            print(f"[compare_solid_bcs] {input_dir} 中无 {fmt} 格式快照，跳过流场云图。")

    # ──────────────────────────────────────────────────────────────────────
    # 1–3. 流场云图
    # ──────────────────────────────────────────────────────────────────────
    if snap is not None:
        _plot_flow_fields(snap, output_dir)
    else:
        print("  [1-3/5] 跳过流场云图（无快照数据）")

    # ──────────────────────────────────────────────────────────────────────
    # 4–5. 受力对比曲线
    # ──────────────────────────────────────────────────────────────────────
    _plot_force_comparison(input_dir, output_dir)

    print(f"\n[compare_solid_bcs] 图像已保存 → {output_dir}")
    print("生成文件:")
    for fname in [
        "velocity_magnitude.png",
        "pressure.png",
        "vorticity.png",
        "force_fx_comparison.png",
        "force_fy_comparison.png",
    ]:
        p = output_dir / fname
        if p.exists():
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname}（未生成）")


if __name__ == "__main__":
    main()
