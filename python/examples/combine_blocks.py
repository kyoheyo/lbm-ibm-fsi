#!/usr/bin/env python3
"""
combine_blocks.py
=================
将 MPI 块分解计算后离散输出的各分区快照合并为全局完整流场文件。

当求解器以 ``mpi.mode = "block"`` 运行时，每个 MPI rank 将本地物理分区数据
写入 ``<output_dir>/rank_<N>/fluid_<NNNNNN>.npz``。本脚本扫描所有
``rank_*`` 子目录，依据每个 NPZ 分区文件中记录的位置元数据
（``x_start / y_start / global_nx / global_ny``）将各分区拼合为全局场，
并写出到 ``<out_dir>/fluid_<NNNNNN>.<ext>``。

用法
----
::

    # 合并为 NPZ（默认），输出到 output/my_run/combined/
    python3 python/examples/combine_blocks.py --dir output/my_run

    # 合并为 ASCII Tecplot .dat
    python3 python/examples/combine_blocks.py --dir output/my_run --fmt dat

    # 合并为二进制 Tecplot .plt，写到自定义目录
    python3 python/examples/combine_blocks.py --dir output/my_run --fmt plt \\
        --out output/my_run/global

注意
----
* 输入文件必须为 NPZ 格式（只有 NPZ 分区文件内嵌了位置元数据）。
  若求解器使用了 ``format = "tecplot_asc"`` 或 ``format = "tecplot_bin"``，
  请同时在 TOML 中设置 ``combine_blocks = true``，让求解器在计算过程中
  直接写出全局合并文件，无需本脚本。
* 本脚本需要将仓库 ``python/`` 目录加入 ``PYTHONPATH``，或在仓库根目录下
  通过 ``pip install -e python/`` 安装 ``lbm_post`` 包。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _add_python_path() -> None:
    """将仓库根目录下的 python/ 子目录加入 sys.path（若尚未加入）。"""
    repo_root = Path(__file__).resolve().parent.parent.parent
    python_dir = repo_root / "python"
    if python_dir.is_dir() and str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))


def main() -> int:
    _add_python_path()

    parser = argparse.ArgumentParser(
        description="将 MPI 块分解计算后各分区的 NPZ 快照合并为全局完整流场文件。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dir", "-d",
        required=True,
        metavar="OUTPUT_DIR",
        help="包含 rank_0/、rank_1/ … 子目录的求解器输出目录",
    )
    parser.add_argument(
        "--fmt", "-f",
        default="npz",
        choices=["npz", "dat", "plt"],
        help="输出文件格式：npz（默认）/ dat（ASCII Tecplot）/ plt（二进制 Tecplot TDV112）",
    )
    parser.add_argument(
        "--out", "-o",
        default=None,
        metavar="OUT_DIR",
        help="合并文件写出目录（默认：<dir>/combined/）",
    )
    args = parser.parse_args()

    try:
        from lbm_post.vtk_reader import combine_block_snapshots
    except ImportError as e:
        print(
            f"错误：无法导入 lbm_post（{e}）。\n"
            "请先安装：  pip install -e python/\n"
            "或将 python/ 目录加入 PYTHONPATH：  export PYTHONPATH=python",
            file=sys.stderr,
        )
        return 1

    out_dir = args.out if args.out else None
    ext = args.fmt

    print(f"输入目录 : {args.dir}")
    print(f"输出格式 : {args.fmt}  →  fluid_NNNNNN.{ext}")
    dst = out_dir if out_dir else str(Path(args.dir) / "combined")
    print(f"输出目录 : {dst}")
    print()

    try:
        written = combine_block_snapshots(args.dir, fmt=args.fmt, out_dir=out_dir)
    except FileNotFoundError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"错误：{e}", file=sys.stderr)
        return 1

    if not written:
        print("未找到可合并的分区快照文件。")
        return 0

    print(f"已合并 {len(written)} 个时间步快照：")
    for p in written:
        print(f"  {p}")
    print(f"\n合并完成。输出目录：{written[0].parent}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
