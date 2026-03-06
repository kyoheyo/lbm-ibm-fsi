#!/usr/bin/env python3
"""
examples/plot_solver_output.py
==============================
LBM 求解器输出后处理脚本，支持 NPZ / ASCII Tecplot / 二进制 Tecplot 三种格式。

运行求解器后::

    ./target/release/lbm-ibm-fsi --config configs/lid_driven_cavity.toml

输出目录中会产生快照文件（格式取决于 TOML 中的 ``output.format``）：

* ``fluid_NNNNNN.npz`` — NumPy 压缩归档（默认，``format = "npz"``）
* ``fluid_NNNNNN.dat`` — ASCII Tecplot POINT 格式（``format = "tecplot_asc"``）
* ``fluid_NNNNNN.plt`` — 二进制 Tecplot TDV112（``format = "tecplot_bin"``）

本脚本读取上述快照并保存出版级流场图。

用法
----
独立运行（图像保存到快照同目录）::

    python3 python/examples/plot_solver_output.py

指定输入目录::

    python3 python/examples/plot_solver_output.py --input output/lid_cavity

自定义输出目录::

    python3 python/examples/plot_solver_output.py \\
        --input output/lid_cavity --output my_plots/

指定输入格式（默认自动检测）::

    python3 python/examples/plot_solver_output.py --fmt tecplot_asc

在 TOML 配置中自动调用（仿真结束后）::

    [python]
    post_script = "python/examples/plot_solver_output.py"
    pythonpath  = "python"

调用时求解器将输出目录作为第一个位置参数传入，因此无需手动指定 ``--input``。

输出文件
--------
保存到 ``<output_dir>/``：

* ``velocity_magnitude.png``   — 速度幅值云图
* ``streamlines.png``          — 流线图
* ``vorticity.png``            — 涡量云图
* ``ux_profile.png``           — 方腔中心截面 ux 剖面（x = nx//2）
* ``uy_profile.png``           — 方腔中心截面 uy 剖面（y = ny//2）
* ``monitor_velocity.png``     — 中心监测点 |u| 时间序列
                                 （仅在快照数 > 1 时生成）
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# sys.path 引导：确保无论工作目录如何，lbm_pre / lbm_post 均可导入。
# __file__ → .../python/examples/plot_solver_output.py
# .parent  → .../python/examples/
# .parent  → .../python/          ← 包的根目录
# ---------------------------------------------------------------------------
_PYTHON_DIR = Path(__file__).resolve().parent.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

# ---------------------------------------------------------------------------
# 命令行参数解析
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "可视化 LBM 求解器输出（支持 .npz / .dat / .plt 快照格式）。"
            "生成速度幅值、流线、涡量及剖面图，保存到指定输出目录。"
        )
    )
    p.add_argument(
        "positional_input",
        nargs="?",
        metavar="INPUT_DIR",
        help=(
            "包含快照文件的目录"
            "（作为 post_script 调用时由求解器自动传入）。"
        ),
    )
    p.add_argument(
        "--input", "-i",
        metavar="INPUT_DIR",
        default=None,
        help=(
            "包含快照文件的目录。"
            "默认：仓库根目录下的 'output/lid_cavity'。"
        ),
    )
    p.add_argument(
        "--output", "-o",
        metavar="OUTPUT_DIR",
        default=None,
        help=(
            "图像文件的保存目录。"
            "默认：与 INPUT_DIR 相同。"
        ),
    )
    p.add_argument(
        "--fmt",
        metavar="FORMAT",
        default=None,
        choices=["npz", "tecplot_asc", "tecplot_bin"],
        help=(
            "输入文件格式：npz（默认）、tecplot_asc（.dat）或 tecplot_bin（.plt）。"
            "若不指定则自动检测（先找 .npz，再找 .dat，最后找 .plt）。"
        ),
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        metavar="N",
        help=(
            "指定绘制某个时间步（而不是最后一帧）。"
            "对应文件名 fluid_NNNNNN.{npz|dat|plt}。"
        ),
    )
    p.add_argument(
        "--centre-x",
        type=int,
        default=None,
        metavar="I",
        help="竖向速度剖面的 x 格点索引（默认：nx//2）。",
    )
    p.add_argument(
        "--centre-y",
        type=int,
        default=None,
        metavar="J",
        help="横向速度剖面的 y 格点索引（默认：ny//2）。",
    )
    args = p.parse_args()

    # 位置参数优先于 --input（与 Rust post_script 调用约定兼容）
    if args.positional_input is not None:
        args.input = args.positional_input

    return args


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    from lbm_post.vtk_reader import NpzReader, TecplotAscReader, TecplotBinReader
    from lbm_post.plot import (
        plot_velocity_magnitude,
        plot_streamlines,
        plot_vorticity,
        plot_velocity_profile,
        save_figure,
    )
    from lbm_post.analysis import compute_bulk_quantities, monitor_point

    # ------------------------------------------------------------------
    # 解析输入 / 输出目录
    # ------------------------------------------------------------------
    if args.input is not None:
        input_dir = Path(args.input)
    else:
        # 默认：<仓库根目录>/output/lid_cavity
        repo_root = _PYTHON_DIR.parent
        input_dir = repo_root / "output" / "lid_cavity"

    if not input_dir.exists():
        print(
            f"[plot_solver_output] 错误：输入目录不存在：{input_dir}\n"
            "  请先运行求解器：\n"
            "    ./target/release/lbm-ibm-fsi --config configs/lid_driven_cavity.toml",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output) if args.output is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 自动选择读取器
    # ------------------------------------------------------------------
    fmt = args.fmt
    if fmt is None:
        # 自动检测：优先 .npz，其次 .dat，最后 .plt
        if list(input_dir.glob("fluid_*.npz")):
            fmt = "npz"
        elif list(input_dir.glob("fluid_*.dat")):
            fmt = "tecplot_asc"
        elif list(input_dir.glob("fluid_*.plt")):
            fmt = "tecplot_bin"
        else:
            print(
                f"[plot_solver_output] 错误：目录 {input_dir} 中没有快照文件\n"
                "  支持格式：fluid_*.npz（NPZ）/ fluid_*.dat（ASCII Tecplot）/ "
                "fluid_*.plt（二进制 Tecplot）\n"
                "  请先运行求解器：\n"
                "    ./target/release/lbm-ibm-fsi --config configs/lid_driven_cavity.toml",
                file=sys.stderr,
            )
            sys.exit(1)

    if fmt == "npz":
        reader = NpzReader(input_dir)
        fmt_label = "NPZ"
    elif fmt == "tecplot_asc":
        reader = TecplotAscReader(input_dir)
        fmt_label = "ASCII Tecplot (.dat)"
    else:
        reader = TecplotBinReader(input_dir)
        fmt_label = "二进制 Tecplot (.plt)"

    if len(reader) == 0:
        print(
            f"[plot_solver_output] 错误：目录 {input_dir} 中没有 {fmt_label} 快照文件\n"
            "  请先运行求解器：\n"
            "    ./target/release/lbm-ibm-fsi --config configs/lid_driven_cavity.toml",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[plot_solver_output] 在 {input_dir} 中找到 {len(reader)} 个快照（格式：{fmt_label}）")

    if args.step is not None:
        snap = reader.read(args.step)
        print(f"  使用第 {snap.step} 步快照")
    else:
        snap = reader.last()
        print(f"  使用最后一帧  step={snap.step}  time={snap.time:.2f}")

    cx = args.centre_x if args.centre_x is not None else snap.nx // 2
    cy = args.centre_y if args.centre_y is not None else snap.ny // 2

    # ------------------------------------------------------------------
    # 1. 速度幅值云图
    # ------------------------------------------------------------------
    fig, _ = plot_velocity_magnitude(snap)
    save_figure(fig, output_dir / "velocity_magnitude.png")
    print("  [1/6] velocity_magnitude.png")

    # ------------------------------------------------------------------
    # 2. 流线图
    # ------------------------------------------------------------------
    fig, _ = plot_streamlines(snap)
    save_figure(fig, output_dir / "streamlines.png")
    print("  [2/6] streamlines.png")

    # ------------------------------------------------------------------
    # 3. 涡量云图
    # ------------------------------------------------------------------
    fig, _ = plot_vorticity(snap)
    save_figure(fig, output_dir / "vorticity.png")
    print("  [3/6] vorticity.png")

    # ------------------------------------------------------------------
    # 4. ux 速度剖面
    # ------------------------------------------------------------------
    fig, _ = plot_velocity_profile(snap, x_slices=[cx], component="ux")
    save_figure(fig, output_dir / "ux_profile.png")
    print(f"  [4/6] ux_profile.png  (x={cx})")

    # ------------------------------------------------------------------
    # 5. uy 速度剖面
    # ------------------------------------------------------------------
    fig, _ = plot_velocity_profile(snap, y_slices=[cy], component="uy")
    save_figure(fig, output_dir / "uy_profile.png")
    print(f"  [5/6] uy_profile.png  (y={cy})")

    # ------------------------------------------------------------------
    # 6. 中心监测点速度时间序列
    #    （使用全部已加载快照；仅在快照数 > 1 时生成）
    # ------------------------------------------------------------------
    if len(reader) > 1:
        from lbm_post.plot import plot_convergence
        all_snaps = list(reader)
        steps_arr, ke_arr = monitor_point(all_snaps, xi=cx, yj=cy, field="magnitude")
        fig, _ = plot_convergence(
            steps_arr, ke_arr,
            label=f"|u| at ({cx},{cy})",
            log_scale=False,
        )
        save_figure(fig, output_dir / "monitor_velocity.png")
        print(f"  [6/6] monitor_velocity.png  （{len(all_snaps)} 个时间步）")
    else:
        print("  [6/6] monitor_velocity.png  跳过（只有 1 个快照）")

    # ------------------------------------------------------------------
    # 体积统计量
    # ------------------------------------------------------------------
    bq = compute_bulk_quantities(snap)
    print(f"\n第 {bq.step} 步体积统计量：")
    print(f"  动能      KE        = {bq.ke:.6f}")
    print(f"  涡量能    Enstrophy = {bq.enstrophy:.6f}")
    print(f"  密度均值  ρ mean    = {bq.rho_mean:.6f}  std = {bq.rho_std:.2e}")
    print(f"  最大散度  max|∇·u|  = {bq.div_max:.2e}")
    print(f"\n图像已保存 → {output_dir}")


if __name__ == "__main__":
    main()
