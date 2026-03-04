mod config;
mod output;
mod python_bridge;

use clap::Parser;
use std::path::PathBuf;
use anyhow::{Context, Result};

use config::Config;
use lbm_bindings::{LatticeModel, CollisionModel, LbmGrid, LbmSolver};

// ---------------------------------------------------------------------------
/// LBM + IBM + FSI solver
///
/// Two output strategies run in parallel during the simulation loop:
///
/// 1. **High-frequency NPZ snapshots** (every `write_interval` steps):
///    the full Eulerian field (ρ, ux, uy) is written by Rust natively to
///    `<output.directory>/fluid_<NNNNNN>.npz`, readable by
///    `lbm_post.vtk_reader.NpzReader`.
///
/// 2. **Per-step CSV monitor log** (when `output.enable_csv_monitor = true`):
///    scalar quantities are appended to `<output.directory>/monitor.csv`
///    every step — lightweight time-series recording.
///
/// Additionally, when the `python-ffi` Cargo feature is active:
///
/// 3. **Low-frequency contour plots** (every `output.plot_interval` steps):
///    field arrays are passed in-memory to `lbm_post.bridge.plot_field_raw()`
///    via pyo3 — no intermediate `.npz` files are created for plotting.
///
/// 4. **IBM marker generation** (once at start-up, when `[ibm]` is present):
///    `lbm_pre.bridge.geometry_markers_raw()` returns marker coordinates
///    directly to Rust without producing a CSV file.
///
/// The `python.pre_script` / `python.post_script` subprocess mode is always
/// available regardless of the `python-ffi` feature flag.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Path to the TOML configuration file
    #[arg(short, long, default_value = "configs/lid_driven_cavity.toml")]
    config: PathBuf,

    /// Override the number of time steps
    #[arg(short, long)]
    steps: Option<u64>,
}

// ---------------------------------------------------------------------------
/// Run a Python script as a subprocess and wait for it to finish.
fn run_python_subprocess(interpreter: &str, script: &str, args: &[&str]) -> Result<()> {
    println!("  [python subprocess] {} {} {}", interpreter, script, args.join(" "));
    let status = std::process::Command::new(interpreter)
        .arg(script)
        .args(args)
        .status()
        .with_context(|| format!("Failed to launch Python script: {script}"))?;

    if !status.success() {
        anyhow::bail!("Python script `{script}` exited with status: {status}");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
fn main() -> Result<()> {
    let args = Args::parse();
    let mut cfg = Config::from_file(&args.config)?;

    if let Some(n) = args.steps {
        cfg.simulation.n_steps = n;
    }

    println!("=== LBM-IBM-FSI Solver ===");
    println!("Config : {}", args.config.display());
    println!("Grid   : {}×{}×{}", cfg.fluid.nx, cfg.fluid.ny, cfg.fluid.nz);
    println!("Steps  : {}", cfg.simulation.n_steps);
    println!("Model  : {} / {}", cfg.simulation.lattice_model, cfg.simulation.collision_model);
    println!("ω      : {:.6}", cfg.omega());

    // -----------------------------------------------------------------------
    // Plugin startup log
    // -----------------------------------------------------------------------
    if cfg.plugins.any_active() {
        println!("Plugins:");
        if !cfg.plugins.boundary.is_empty() {
            println!("  boundary  = \"{}\"  (IBoundaryPlugin)", cfg.plugins.boundary);
        }
        if !cfg.plugins.mesh.is_empty() {
            println!("  mesh      = \"{}\"  (IMeshPlugin)", cfg.plugins.mesh);
        }
        if !cfg.plugins.motion.is_empty() {
            println!("  motion    = \"{}\"  (IMotionPlugin)", cfg.plugins.motion);
        }
        if !cfg.plugins.flexible.is_empty() {
            println!("  flexible  = \"{}\"  (IFlexibleSolverPlugin)", cfg.plugins.flexible);
        }
        // NOTE: To register a plugin implementation, call
        //   lbm_bindings::register_plugins(PluginCallbacks { boundary_fn: Some(my_fn), .. })
        // before the simulation loop.  Plugin names above are informational only.
    }

    // -----------------------------------------------------------------------
    // Python FFI: extend sys.path so lbm_pre / lbm_post are importable
    // -----------------------------------------------------------------------
    if let Some(ref extra_path) = cfg.python.pythonpath {
        // add_python_path is a no-op (returns Ok) when python-ffi is disabled
        if let Err(e) = python_bridge::add_python_path(extra_path) {
            eprintln!("[python-ffi] sys.path extension failed: {e}");
        }
    }

    // -----------------------------------------------------------------------
    // Pre-processing: subprocess script
    // -----------------------------------------------------------------------
    if let Some(ref script) = cfg.python.pre_script.clone() {
        println!("\n--- Pre-processing (Python subprocess) ---");
        let config_str = args.config.to_string_lossy().into_owned();
        run_python_subprocess(&cfg.python.interpreter, script, &[&config_str])?;
    }

    // -----------------------------------------------------------------------
    // IBM marker generation via Python FFI (no CSV file produced)
    // -----------------------------------------------------------------------
    #[cfg(feature = "python-ffi")]
    {
        if let Some(ref ibm) = cfg.ibm {
            println!("\n[python-ffi] generating IBM markers in-process ...");
            match python_bridge::markers_from_geometry(
                &ibm.geometry, ibm.x0, ibm.y0, ibm.size, ibm.n_markers,
            ) {
                Ok((x, y, ds)) => {
                    let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
                    let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let y_min = y.iter().cloned().fold(f64::INFINITY, f64::min);
                    let y_max = y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    println!(
                        "  => {} markers  x=[{:.2},{:.2}]  y=[{:.2},{:.2}]  ds~={:.4}",
                        x.len(), x_min, x_max, y_min, y_max,
                        ds.first().copied().unwrap_or(0.0),
                    );
                    // TODO: forward (x, y, ds) to the C++ IBM kernel once
                    //       that interface is exposed through lbm_bindings.
                    let _ = (x, y, ds);
                }
                Err(e) => eprintln!("[python-ffi] marker generation skipped: {e}"),
            }
        }
    }

    let model = match cfg.simulation.lattice_model.as_str() {
        "D3Q19" => LatticeModel::D3Q19,
        "D3Q27" => LatticeModel::D3Q27,
        _       => LatticeModel::D2Q9,
    };

    let cm = match cfg.simulation.collision_model.as_str() {
        "MRT" => CollisionModel::Mrt,
        _     => CollisionModel::Bgk,
    };

    // Initialise fluid grid
    let mut grid = LbmGrid::new(
        cfg.fluid.nx as i32,
        cfg.fluid.ny as i32,
        cfg.fluid.nz as i32,
        model,
    );

    // Initialise solver
    let mut solver = LbmSolver::new(&mut grid, cfg.omega(), cm);

    // Create output directory
    std::fs::create_dir_all(&cfg.output.directory)?;

    let csv_path = format!("{}/monitor.csv", cfg.output.directory);

    println!("\nStarting time integration...");
    if cfg.simulation.n_steps == 0 {
        println!("n_steps is 0 — nothing to simulate.");
        return Ok(());
    }

    for step in 0..cfg.simulation.n_steps {
        solver.step(&mut grid);

        let time = (step + 1) as f64 * cfg.simulation.dt;

        // -- High-frequency: native Rust NPZ snapshot ----------------------
        if step % cfg.output.write_interval == 0 || step == cfg.simulation.n_steps - 1 {
            println!("  step {:>6} / {}  t = {:.3}", step + 1, cfg.simulation.n_steps, time);
            output::write_snapshot_npz(&grid, step + 1, time, &cfg.output.directory)
                .with_context(|| format!("Failed to write snapshot at step {}", step + 1))?;
        }

        // -- Per-step: lightweight CSV monitor log -------------------------
        if cfg.output.enable_csv_monitor {
            let n = (grid.nx() * grid.ny()) as usize;
            let ke: f64 = (0..n)
                .map(|i| {
                    let u = grid.ux(i as i32);
                    let v = grid.uy(i as i32);
                    u * u + v * v
                })
                .sum::<f64>()
                / (n as f64)
                * 0.5;
            output::append_monitor_csv(&csv_path, step + 1, time, &[("ke", ke)])
                .with_context(|| format!("Failed to write monitor CSV at step {}", step + 1))?;
        }

        // -- Low-frequency: Python FFI contour plot (python-ffi feature) ---
        #[cfg(feature = "python-ffi")]
        if let Some(pi) = cfg.output.plot_interval {
            if step % pi == 0 || step == cfg.simulation.n_steps - 1 {
                let nx = grid.nx() as usize;
                let ny = grid.ny() as usize;
                let n  = nx * ny;
                let rho: Vec<f64> = (0..n).map(|i| grid.rho(i as i32)).collect();
                let ux:  Vec<f64> = (0..n).map(|i| grid.ux (i as i32)).collect();
                let uy:  Vec<f64> = (0..n).map(|i| grid.uy (i as i32)).collect();

                for field_name in &["velocity_magnitude", "vorticity"] {
                    if let Err(e) = python_bridge::plot_field(
                        &rho, &ux, &uy, nx, ny,
                        step + 1, time,
                        &cfg.output.directory,
                        field_name,
                    ) {
                        eprintln!("[python-ffi] plot {field_name} failed at step {}: {e}", step + 1);
                    }
                }
            }
        }
    }

    println!("\nSimulation complete.");

    // -----------------------------------------------------------------------
    // Post-processing: subprocess script
    // -----------------------------------------------------------------------
    if let Some(ref script) = cfg.python.post_script.clone() {
        println!("\n--- Post-processing (Python subprocess) ---");
        run_python_subprocess(&cfg.python.interpreter, script, &[&cfg.output.directory])?;
    }

    Ok(())
}
