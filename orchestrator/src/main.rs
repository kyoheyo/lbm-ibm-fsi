mod config;
mod output;
mod python_bridge;

use clap::Parser;
use std::path::PathBuf;
use anyhow::{Context, Result};

use config::Config;
use lbm_bindings::{LatticeModel, CollisionModel, LbmGrid, LbmSolver, BcType, Face,
                   mpi_rank, mpi_size, mpi_local_ny};

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

    println!("=== LBM-IBM-FSI 求解器 ===");
    println!("配置文件 : {}", args.config.display());
    println!("网格     : {}×{}×{}", cfg.fluid.nx, cfg.fluid.ny, cfg.fluid.nz);
    println!("步数     : {}", cfg.simulation.n_steps);
    println!("模型     : {} / {}", cfg.simulation.lattice_model, cfg.simulation.collision_model);
    println!("松弛频率 : ω = {:.6}", cfg.omega());
    println!("输出格式 : {} → {}", cfg.output.format, {
        match cfg.output.format.as_str() {
            "tecplot_asc" => "fluid_NNNNNN.dat（ASCII Tecplot）",
            "tecplot_bin" => "fluid_NNNNNN.plt（二进制 Tecplot TDV112）",
            _             => "fluid_NNNNNN.npz（NumPy 压缩归档）",
        }
    });

    // -----------------------------------------------------------------------
    // 应用并行配置：OpenMP 线程数
    // -----------------------------------------------------------------------
    if cfg.parallel.omp_num_threads > 0 {
        lbm_bindings::set_omp_num_threads(cfg.parallel.omp_num_threads as i32);
        println!("OpenMP   : 线程数 = {}（由 [parallel].omp_num_threads 设置）",
                 cfg.parallel.omp_num_threads);
    }

    lbm_bindings::print_parallel_status();

    // 打印 MPI 模式信息
    {
        let mpi_mode = cfg.mpi.mode.as_str();
        match mpi_mode {
            "2d_xy" => println!("MPI模式  : 二维块分解 {}×{}", cfg.mpi.nx_blocks, cfg.mpi.ny_blocks),
            "multi_grid" => println!("MPI模式  : 多网格独立（每进程独立仿真）"),
            _ => println!("MPI模式  : 一维 Y 方向切片"),
        }
    }

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

    // -----------------------------------------------------------------------
    // MPI 域分解：根据 [mpi] 配置计算本进程的本地网格尺寸
    //
    // 三种模式：
    //   "1d_y"       : 一维 Y 切片（默认）
    //   "2d_xy"      : 二维 XY 块分解（需要 nx_blocks * ny_blocks == nprocs）
    //   "multi_grid" : 每进程独立仿真，无通信
    // -----------------------------------------------------------------------
    let rank   = lbm_bindings::mpi_rank();
    let nprocs = lbm_bindings::mpi_size();

    // 确定本进程实际使用的网格尺寸
    let (grid_nx, grid_ny, grid_nz) = match cfg.mpi.mode.as_str() {
        "2d_xy" => {
            // 二维块分解：本进程的 nx/ny 由 LbmMpiDecomp2D 计算
            let px = cfg.mpi.nx_blocks as i32;
            let py = cfg.mpi.ny_blocks as i32;
            if px * py != nprocs {
                eprintln!(
                    "[warn] mpi.nx_blocks({}) * mpi.ny_blocks({}) = {} ≠ nprocs({}). \
                     Falling back to 1d_y decomposition.",
                    px, py, px * py, nprocs
                );
                // Fallback：使用 1D Y 分解
                let (gny, _, _) = lbm_bindings::mpi_local_ny(cfg.fluid.ny as i32);
                (cfg.fluid.nx as i32, gny, cfg.fluid.nz as i32)
            } else if let Some(d2) = lbm_bindings::LbmMpiDecomp2D::new(
                cfg.fluid.nx as i32, cfg.fluid.ny as i32, px, py)
            {
                (d2.grid_nx(), d2.grid_ny(), cfg.fluid.nz as i32)
            } else {
                // MPI 未启用时退化为全局网格
                (cfg.fluid.nx as i32, cfg.fluid.ny as i32, cfg.fluid.nz as i32)
            }
        }
        "multi_grid" => {
            // 多网格独立模式：每进程使用完整的全局网格，无幽灵层
            (cfg.fluid.nx as i32, cfg.fluid.ny as i32, cfg.fluid.nz as i32)
        }
        _ => {
            // "1d_y"（默认）：Y 方向一维切片
            let (gny, _, _) = lbm_bindings::mpi_local_ny(cfg.fluid.ny as i32);
            (cfg.fluid.nx as i32, gny, cfg.fluid.nz as i32)
        }
    };

    if nprocs > 1 {
        println!("本地网格  : rank={} → {}×{}×{}", rank, grid_nx, grid_ny, grid_nz);
    }

    // 初始化流体格子网格
    let mut grid = LbmGrid::new(grid_nx, grid_ny, grid_nz, model);

    // 初始化求解器
    let mut solver = LbmSolver::new(&mut grid, cfg.omega(), cm);

    // -----------------------------------------------------------------------
    // MPI 域分解绑定：根据模式创建并绑定 MpiDecomp 或 MpiDecomp2D
    // -----------------------------------------------------------------------
    // 声明持有者以延长生命周期到仿真循环结束
    let mut _decomp1d: Option<lbm_bindings::LbmMpiDecomp>   = None;
    let mut _decomp2d: Option<lbm_bindings::LbmMpiDecomp2D> = None;

    match cfg.mpi.mode.as_str() {
        "2d_xy" => {
            let px = cfg.mpi.nx_blocks as i32;
            let py = cfg.mpi.ny_blocks as i32;
            if px * py == nprocs {
                if let Some(mut d) = lbm_bindings::LbmMpiDecomp2D::new(
                    cfg.fluid.nx as i32, cfg.fluid.ny as i32, px, py)
                {
                    solver.attach_mpi2d(Some(&mut d));
                    _decomp2d = Some(d);
                }
            } else {
                // 已在上方打印警告，退化为 1D
                if let Some(mut d) = lbm_bindings::LbmMpiDecomp::new(
                    cfg.fluid.nx as i32, cfg.fluid.ny as i32)
                {
                    solver.attach_mpi(Some(&mut d));
                    _decomp1d = Some(d);
                }
            }
        }
        "multi_grid" => {
            // 无 MPI 通信，不绑定任何分解
        }
        _ => {
            // "1d_y"（默认）
            if let Some(mut d) = lbm_bindings::LbmMpiDecomp::new(
                cfg.fluid.nx as i32, cfg.fluid.ny as i32)
            {
                solver.attach_mpi(Some(&mut d));
                _decomp1d = Some(d);
            }
        }
    }

    // -----------------------------------------------------------------------
    // 注册边界条件（将 TOML 配置中的 [[fluid.boundary_conditions]] 传入 C++ 核心）
    //
    // 这是流场结果正确的关键步骤：若跳过此步骤，边界条件将不会被施加，
    // 所有节点保持初始平衡态（u=0），流场云图值均为零。
    // -----------------------------------------------------------------------
    for bc_cfg in &cfg.fluid.boundary_conditions {
        let bc_type = match bc_cfg.bc_type.to_lowercase().as_str() {
            // 反弹类
            "bounce_back"          => BcType::BounceBack,
            "bounce_back_full_way" => BcType::BounceBackFullWay,
            // Zou-He 非平衡反弹类
            "zou_he_velocity"      => BcType::ZouHeVelocity,
            "zou_he_pressure"      => BcType::ZouHePressure,
            // 出口类
            "fully_developed"      => BcType::FullyDeveloped,
            "free_outlet"          => BcType::FreeOutlet,
            // 非平衡外推类
            "guo_extrapolation"    => BcType::GuoExtrapolation,
            // 周期类（流式迁移中隐式处理）
            "periodic"             => BcType::Periodic,
            other => {
                eprintln!(
                    "  [warn] unknown bc_type {:?}; defaulting to BounceBack",
                    other
                );
                BcType::BounceBack
            }
        };
        let face = match bc_cfg.face.to_lowercase().as_str() {
            "east"   => Face::East,
            "south"  => Face::South,
            "north"  => Face::North,
            "bottom" => Face::Bottom,
            "top"    => Face::Top,
            "west"   => Face::West,
            other => {
                eprintln!(
                    "  [warn] unknown face {:?}; defaulting to West",
                    other
                );
                Face::West
            }
        };
        solver.add_boundary_condition(
            bc_type, face,
            bc_cfg.ux, bc_cfg.uy, bc_cfg.uz,
            bc_cfg.rho,
        );
        println!(
            "  BC registered: {:?} on {:?} face  (ux={:.4}, uy={:.4}, rho={:.4})",
            bc_type, face, bc_cfg.ux, bc_cfg.uy, bc_cfg.rho
        );
    }

    // 创建输出目录
    // 多网格独立模式下，每进程的输出写入各自的子目录 <output.directory>/rank_<N>/
    let output_dir = if cfg.mpi.mode == "multi_grid" && nprocs > 1 {
        format!("{}/rank_{}", cfg.output.directory, rank)
    } else {
        cfg.output.directory.clone()
    };
    std::fs::create_dir_all(&output_dir)?;

    let csv_path = format!("{}/monitor.csv", output_dir);

    println!("\nStarting time integration...");
    if cfg.simulation.n_steps == 0 {
        println!("n_steps is 0 — nothing to simulate.");
        return Ok(());
    }

    for step in 0..cfg.simulation.n_steps {
        solver.step(&mut grid);

        let time = (step + 1) as f64 * cfg.simulation.dt;

        // -- 高频：原生 Rust 快照（格式由 output.format 决定）-------------------
        if step % cfg.output.write_interval == 0 || step == cfg.simulation.n_steps - 1 {
            println!("  step {:>6} / {}  t = {:.3}", step + 1, cfg.simulation.n_steps, time);
            match cfg.output.format.as_str() {
                "tecplot_asc" => {
                    // ASCII Tecplot .dat 格式：人类可读，可用 Tecplot/ParaView 打开
                    output::write_snapshot_tecplot_asc(&grid, step + 1, time, &output_dir)
                        .with_context(|| format!("写出 Tecplot ASCII 快照失败（步数 {}）", step + 1))?;
                }
                "tecplot_bin" => {
                    // 二进制 Tecplot .plt 格式（TDV112）：体积最小，Tecplot 软件可直接打开
                    output::write_snapshot_tecplot_bin(&grid, step + 1, time, &output_dir)
                        .with_context(|| format!("写出 Tecplot 二进制快照失败（步数 {}）", step + 1))?;
                }
                _ => {
                    // 默认："npz"——NumPy .npz 压缩归档，Python 后处理首选格式
                    output::write_snapshot_npz(&grid, step + 1, time, &output_dir)
                        .with_context(|| format!("写出 NPZ 快照失败（步数 {}）", step + 1))?;
                }
            }
        }

        // -- 逐步：轻量级 CSV 监控日志 -----------------------------------------
        if cfg.output.enable_csv_monitor {            let n = (grid.nx() * grid.ny()) as usize;
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
                        &output_dir,
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
        run_python_subprocess(&cfg.python.interpreter, script, &[&output_dir])?;
    }

    Ok(())
}
