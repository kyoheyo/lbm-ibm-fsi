mod config;
mod output;
mod python_bridge;

use clap::Parser;
use std::path::PathBuf;
use anyhow::{Context, Result};

use config::Config;
use lbm_bindings::{LatticeModel, CollisionModel, LbmGrid, LbmSolver};

// ---------------------------------------------------------------------------
/// LBM + IBM + FSI 求解器
///
/// 仿真循环中并行运行两种输出策略：
///
/// 1. **高频：NPZ 快照**（每 `write_interval` 步）：
///    完整欧拉场（ρ、ux、uy）由 Rust 原生写出到
///    `<output.directory>/fluid_<NNNNNN>.npz`，可被
///    `lbm_post.vtk_reader.NpzReader` 读取。
///
/// 2. **逐步：CSV 监控日志**（当 `output.enable_csv_monitor = true`）：
///    每步将标量量追加到 `<output.directory>/monitor.csv` ——
///    轻量级时间序列记录。
///
/// 此外，当 `python-ffi` Cargo 特性启用时：
///
/// 3. **低频：Python FFI 等值线图**（每 `output.plot_interval` 步）：
///    场数组在内存中传递给 `lbm_post.bridge.plot_field_raw()`
///    —— 不创建中间 `.npz` 文件。
///
/// 4. **IBM 标记点生成**（启动时，若存在 `[ibm]` 段）：
///    `lbm_pre.bridge.geometry_markers_raw()` 直接返回标记点坐标，
///    不生成 CSV 文件。
///
/// 无论是否启用 `python-ffi` 特性，
/// `python.pre_script` / `python.post_script` 子进程模式始终可用。
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// TOML 配置文件路径
    #[arg(short, long, default_value = "configs/lid_driven_cavity.toml")]
    config: PathBuf,

    /// 覆盖时间步数
    #[arg(short, long)]
    steps: Option<u64>,
}

// ---------------------------------------------------------------------------
/// 以子进程方式运行 Python 脚本并等待其结束。
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
    // 插件启动日志
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
        // 注意：若要注册插件实现，请在仿真循环前调用：
        //   lbm_bindings::register_plugins(PluginCallbacks { boundary_fn: Some(my_fn), .. })
        // 上方的插件名称仅供提示，不自动加载共享库。
    }

    // -----------------------------------------------------------------------
    // Python FFI：扩展 sys.path 使 lbm_pre / lbm_post 可导入
    // -----------------------------------------------------------------------
    if let Some(ref extra_path) = cfg.python.pythonpath {
        // 未启用 python-ffi 特性时，add_python_path 为无操作（返回 Ok）
        if let Err(e) = python_bridge::add_python_path(extra_path) {
            eprintln!("[python-ffi] sys.path extension failed: {e}");
        }
    }

    // -----------------------------------------------------------------------
    // 预处理：Python 子进程脚本
    // -----------------------------------------------------------------------
    if let Some(ref script) = cfg.python.pre_script.clone() {
        println!("\n--- Pre-processing (Python subprocess) ---");
        let config_str = args.config.to_string_lossy().into_owned();
        run_python_subprocess(&cfg.python.interpreter, script, &[&config_str])?;
    }

    // -----------------------------------------------------------------------
    // 通过 Python FFI 生成 IBM 标记点（不产生 CSV 文件）
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
                    // TODO: 待 lbm_bindings 暴露对应接口后，
                    //       将 (x, y, ds) 转发给 C++ IBM 核心。
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

    // 初始化流体格子网格
    let mut grid = LbmGrid::new(
        cfg.fluid.nx as i32,
        cfg.fluid.ny as i32,
        cfg.fluid.nz as i32,
        model,
    );

    // 初始化求解器
    let mut solver = LbmSolver::new(&mut grid, cfg.omega(), cm);

    // 创建输出目录
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

        // -- 高频：原生 Rust NPZ 快照 ----------------------------------------
        if step % cfg.output.write_interval == 0 || step == cfg.simulation.n_steps - 1 {
            println!("  step {:>6} / {}  t = {:.3}", step + 1, cfg.simulation.n_steps, time);
            output::write_snapshot_npz(&grid, step + 1, time, &cfg.output.directory)
                .with_context(|| format!("Failed to write snapshot at step {}", step + 1))?;
        }

        // -- 逐步：轻量级 CSV 监控日志 -----------------------------------------
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

        // -- 低频：Python FFI 等值线图（python-ffi 特性）-----------------------
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
    // 后处理：Python 子进程脚本
    // -----------------------------------------------------------------------
    if let Some(ref script) = cfg.python.post_script.clone() {
        println!("\n--- Post-processing (Python subprocess) ---");
        run_python_subprocess(&cfg.python.interpreter, script, &[&cfg.output.directory])?;
    }

    Ok(())
}
