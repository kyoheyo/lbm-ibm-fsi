mod config;

use clap::Parser;
use std::path::PathBuf;
use anyhow::Result;

use config::Config;
use lbm_bindings::{LatticeModel, CollisionModel, LbmGrid, LbmSolver};

// ---------------------------------------------------------------------------
/// LBM + IBM + FSI solver
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

    println!("\nStarting time integration...");

    if cfg.simulation.n_steps == 0 {
        println!("n_steps is 0 — nothing to simulate.");
        return Ok(());
    }

    for step in 0..cfg.simulation.n_steps {
        solver.step(&mut grid);

        if step % cfg.output.write_interval == 0 || step == cfg.simulation.n_steps - 1 {
            println!("  step {:>6} / {}", step, cfg.simulation.n_steps);
            // TODO: write VTK output here
        }
    }

    println!("\nSimulation complete.");
    Ok(())
}
