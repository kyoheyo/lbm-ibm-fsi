#!/usr/bin/env python3
"""
examples/plot_solver_output.py
==============================
Post-processing script for visualising the output of the Rust LBM solver.

After running the solver::

    ./target/release/lbm-ibm-fsi --config configs/lid_driven_cavity.toml

a directory of ``fluid_NNNNNN.npz`` snapshots is produced.  This script reads
those files and saves publication-ready flow-field plots.

Usage
-----
Run standalone (plots go next to the snapshots)::

    python3 python/examples/plot_solver_output.py

Point at a specific output directory::

    python3 python/examples/plot_solver_output.py --input output/lid_cavity

Choose a custom plot destination::

    python3 python/examples/plot_solver_output.py \\
        --input output/lid_cavity --output my_plots/

The solver can also invoke this script automatically by adding the following
section to the TOML config::

    [python]
    post_script = "python/examples/plot_solver_output.py"
    pythonpath  = "python"   # optional: makes lbm_post importable without pip

When called as a post-script the solver passes the output directory as the
first positional argument, so ``--input`` need not be specified.

Outputs
-------
Written to ``<output_dir>/``:

* ``velocity_magnitude.png``
* ``streamlines.png``
* ``vorticity.png``
* ``ux_profile.png``      — ux profile at cavity centre (x = nx//2)
* ``uy_profile.png``      — uy profile at cavity centre (y = ny//2)
* ``monitor_velocity.png`` — |u| time series at the centre monitor point
                             (only written when more than one snapshot is available)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Path bootstrap: ensure lbm_pre / lbm_post are importable regardless of the
# working directory or how this script is invoked.
# __file__ → .../python/examples/plot_solver_output.py
# .parent  → .../python/examples/
# .parent  → .../python/          ← the package root
# ---------------------------------------------------------------------------
_PYTHON_DIR = Path(__file__).resolve().parent.parent
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Visualise LBM solver output (fluid_NNNNNN.npz snapshots). "
            "Produces velocity-magnitude, streamline, vorticity and profile "
            "plots in the specified output directory."
        )
    )
    p.add_argument(
        "positional_input",
        nargs="?",
        metavar="INPUT_DIR",
        help=(
            "Directory containing fluid_*.npz snapshots "
            "(passed automatically when used as post_script)."
        ),
    )
    p.add_argument(
        "--input", "-i",
        metavar="INPUT_DIR",
        default=None,
        help=(
            "Directory containing fluid_*.npz snapshots. "
            "Defaults to 'output/lid_cavity' relative to the repo root."
        ),
    )
    p.add_argument(
        "--output", "-o",
        metavar="OUTPUT_DIR",
        default=None,
        help=(
            "Directory for plot files. "
            "Defaults to the same directory as INPUT_DIR."
        ),
    )
    p.add_argument(
        "--step",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Plot a specific time step instead of the last snapshot. "
            "Must match one of the fluid_NNNNNN.npz filenames."
        ),
    )
    p.add_argument(
        "--centre-x",
        type=int,
        default=None,
        metavar="I",
        help="Grid x-index for the vertical velocity profile (default: nx//2).",
    )
    p.add_argument(
        "--centre-y",
        type=int,
        default=None,
        metavar="J",
        help="Grid y-index for the horizontal velocity profile (default: ny//2).",
    )
    args = p.parse_args()

    # Positional arg overrides --input (compatibility with Rust post_script calling
    # convention: the solver passes the output directory as the first positional arg).
    if args.positional_input is not None:
        args.input = args.positional_input

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    from lbm_post.vtk_reader import NpzReader
    from lbm_post.plot import (
        plot_velocity_magnitude,
        plot_streamlines,
        plot_vorticity,
        plot_velocity_profile,
        save_figure,
    )
    from lbm_post.analysis import compute_bulk_quantities, monitor_point

    # ------------------------------------------------------------------
    # Resolve input / output directories
    # ------------------------------------------------------------------
    if args.input is not None:
        input_dir = Path(args.input)
    else:
        # Default: <repo_root>/output/lid_cavity
        repo_root = _PYTHON_DIR.parent
        input_dir = repo_root / "output" / "lid_cavity"

    if not input_dir.exists():
        print(
            f"[plot_solver_output] ERROR: input directory not found: {input_dir}\n"
            "  Run the solver first:\n"
            "    ./target/release/lbm-ibm-fsi --config configs/lid_driven_cavity.toml",
            file=sys.stderr,
        )
        sys.exit(1)

    output_dir = Path(args.output) if args.output is not None else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load snapshots
    # ------------------------------------------------------------------
    reader = NpzReader(input_dir)
    if len(reader) == 0:
        print(
            f"[plot_solver_output] ERROR: no fluid_*.npz files found in {input_dir}\n"
            "  Run the solver first:\n"
            "    ./target/release/lbm-ibm-fsi --config configs/lid_driven_cavity.toml",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"[plot_solver_output] Found {len(reader)} snapshot(s) in {input_dir}")

    if args.step is not None:
        snap = reader.read(args.step)
        print(f"  Using step {snap.step}")
    else:
        snap = reader.last()
        print(f"  Using last snapshot  step={snap.step}  time={snap.time:.2f}")

    cx = args.centre_x if args.centre_x is not None else snap.nx // 2
    cy = args.centre_y if args.centre_y is not None else snap.ny // 2

    # ------------------------------------------------------------------
    # 1. Velocity magnitude
    # ------------------------------------------------------------------
    fig, _ = plot_velocity_magnitude(snap)
    save_figure(fig, output_dir / "velocity_magnitude.png")
    print(f"  [1/6] velocity_magnitude.png")

    # ------------------------------------------------------------------
    # 2. Streamlines
    # ------------------------------------------------------------------
    fig, _ = plot_streamlines(snap)
    save_figure(fig, output_dir / "streamlines.png")
    print(f"  [2/6] streamlines.png")

    # ------------------------------------------------------------------
    # 3. Vorticity
    # ------------------------------------------------------------------
    fig, _ = plot_vorticity(snap)
    save_figure(fig, output_dir / "vorticity.png")
    print(f"  [3/6] vorticity.png")

    # ------------------------------------------------------------------
    # 4. Velocity profiles
    # ------------------------------------------------------------------
    fig, _ = plot_velocity_profile(snap, x_slices=[cx], component="ux")
    save_figure(fig, output_dir / "ux_profile.png")
    print(f"  [4/6] ux_profile.png  (x={cx})")

    fig, _ = plot_velocity_profile(snap, y_slices=[cy], component="uy")
    save_figure(fig, output_dir / "uy_profile.png")
    print(f"  [5/6] uy_profile.png  (y={cy})")

    # ------------------------------------------------------------------
    # 6. Velocity time series at the centre monitor point
    # (uses all loaded snapshots; skipped when only one snapshot exists)
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
        print(f"  [6/6] monitor_velocity.png  ({len(all_snaps)} time steps)")
    else:
        print("  [6/6] monitor_velocity.png  skipped (only 1 snapshot)")

    # ------------------------------------------------------------------
    # Bulk statistics
    # ------------------------------------------------------------------
    bq = compute_bulk_quantities(snap)
    print(f"\nBulk statistics at step {bq.step}:")
    print(f"  KE        = {bq.ke:.6f}")
    print(f"  Enstrophy = {bq.enstrophy:.6f}")
    print(f"  ρ mean    = {bq.rho_mean:.6f}  std = {bq.rho_std:.2e}")
    print(f"  max |∇·u| = {bq.div_max:.2e}")
    print(f"\nPlots saved → {output_dir}")


if __name__ == "__main__":
    main()
