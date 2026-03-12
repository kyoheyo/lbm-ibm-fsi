mod config;
mod output;
mod python_bridge;

use clap::Parser;
use std::path::PathBuf;
use anyhow::{Context, Result};

use config::Config;
use lbm_bindings::{LatticeModel, CollisionModel, LbmGrid, LbmSolver, BcType, Face};
use output::PartitionInfo;

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
/// 真正的求解器入口（在 MPI init/finalize 包装内执行）。
fn run() -> Result<()> {
    let args = Args::parse();
    let mut cfg = Config::from_file(&args.config)?;

    if let Some(n) = args.steps {
        cfg.simulation.n_steps = n;
    }

    // 读取 rank 和进程数（mpi_init() 已在调用者 main() 中完成）。
    // 在非 MPI 模式下 rank=0, nprocs=1，此处调用安全。
    let rank   = lbm_bindings::mpi_rank();
    let nprocs = lbm_bindings::mpi_size();

    // -----------------------------------------------------------------------
    // 应用并行配置：OpenMP 线程数（必须在所有进程上生效，不只 rank-0）
    // -----------------------------------------------------------------------
    if cfg.parallel.omp_num_threads > 0 {
        lbm_bindings::set_omp_num_threads(cfg.parallel.omp_num_threads as i32);
    }

    // Print header from rank-0 only (avoid duplicate output in MPI mode).
    if rank == 0 {
        println!("=== LBM-IBM-FSI Solver ===");
        println!("Config   : {}", args.config.display());
        println!("Grid     : {}x{}x{}", cfg.fluid.nx, cfg.fluid.ny, cfg.fluid.nz);
        println!("Steps    : {}", cfg.simulation.n_steps);
        println!("Model    : {} / {}", cfg.simulation.lattice_model, cfg.simulation.collision_model);
        println!("Omega    : w = {:.6}", cfg.omega());
        println!("Output   : {} -> {}", cfg.output.format, {
            match cfg.output.format.as_str() {
                "tecplot_asc" => "fluid_NNNNNN.dat (ASCII Tecplot)",
                "tecplot_bin" => "fluid_NNNNNN.plt (binary Tecplot TDV112)",
                _             => "fluid_NNNNNN.npz (NumPy compressed archive)",
            }
        });
        if cfg.parallel.omp_num_threads > 0 {
            println!("OpenMP   : threads = {} (set by [parallel].omp_num_threads)",
                     cfg.parallel.omp_num_threads);
        }
        lbm_bindings::print_parallel_status();

        // Print MPI mode summary
        {
            let (norm_mode, _) = cfg.mpi.normalized_mode();
            match norm_mode {
                "block" => {
                    let (px, py) = cfg.mpi.effective_blocks(nprocs);
                    let pz = cfg.mpi.nz_blocks.max(1);
                    if pz > 1 {
                        println!("MPI mode : 3D block decomp {}x{}x{}", px, py, pz);
                    } else if px == 1 {
                        println!("MPI mode : 1D Y-slice (ny_blocks={})", py);
                    } else if py == 1 {
                        println!("MPI mode : 1D X-slice (nx_blocks={})", px);
                    } else {
                        println!("MPI mode : 2D block decomp {}x{}", px, py);
                    }
                }
                "multigrid"   => println!("MPI mode : nested multigrid (framework mode, currently degrades to independent)"),
                "independent" => println!("MPI mode : independent (each rank runs its own full simulation, no communication)"),
                _             => println!("MPI mode : {}", cfg.mpi.mode),
            }
        }

        // 插件启动日志
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
    } // end rank-0 header output

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
    // 预处理：Python 子进程脚本（仅 rank-0 运行，避免 MPI 多进程重复启动子进程）
    // -----------------------------------------------------------------------
    if rank == 0 {
        if let Some(ref script) = cfg.python.pre_script.clone() {
            println!("\n--- Pre-processing (Python subprocess) ---");
            let config_str = args.config.to_string_lossy().into_owned();
            run_python_subprocess(&cfg.python.interpreter, script, &[&config_str])?;
        }
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
    // 三种模式（均通过 MpiDecomp2D 实现，1D 是 2D 的特例）：
    //   "block"       : XY 块分解（nx_blocks=1 → 1D Y；ny_blocks=1 → 1D X）
    //   "independent" : 每进程独立仿真，无通信（旧名 "multi_grid"）
    //   "multigrid"   : 嵌套多重网格框架（当前版本退化为独立模式）
    //
    // 旧名称兼容：
    //   "1d_y"  → "block" (nx_blocks=1, ny_blocks=nprocs)
    //   "2d_xy" → "block"
    //   "multi_grid" → "independent"
    //
    // rank 和 nprocs 已在函数顶部通过 mpi_rank()/mpi_size() 初始化。
    // -----------------------------------------------------------------------
    // 归一化模式名称（旧名 → 新名）
    let (effective_mode, mode_was_renamed) = cfg.mpi.normalized_mode();
    if mode_was_renamed && nprocs > 1 {
        eprintln!(
            "[info] mpi.mode {:?} is deprecated; using {:?}. \
             See docs/MPI_parallel.md for the new mode names.",
            cfg.mpi.mode, effective_mode
        );
    }

    // -----------------------------------------------------------------------
    // 提前创建 MPI 分解对象（同时用于确定本地网格尺寸和绑定到求解器），
    // 声明持有者以延长生命周期到仿真循环结束
    // -----------------------------------------------------------------------
    let mut _decomp2d: Option<lbm_bindings::LbmMpiDecomp2D> = None;
    let mut _decomp3d: Option<lbm_bindings::LbmMpiDecomp3D> = None;

    // 是否启用三维 Z 方向分解（nz > 1 且 nz_blocks > 1）
    let use_3d_decomp = cfg.fluid.nz > 1 && cfg.mpi.nz_blocks > 1;

    match effective_mode {
        "block" => {
            if use_3d_decomp {
                // 三维 XYZ 块分解
                let (px, py, pz) = if cfg.mpi.mode == "1d_y" {
                    (1, nprocs as u32, 1)
                } else {
                    cfg.mpi.effective_blocks_3d(nprocs)
                };
                if px as i32 * py as i32 * pz as i32 != nprocs {
                    eprintln!(
                        "[warn] mpi nx_blocks({}) * ny_blocks({}) * nz_blocks({}) = {} ≠ nprocs({}). \
                         Falling back to 1D Y slice.",
                        px, py, pz, px as i32 * py as i32 * pz as i32, nprocs
                    );
                    _decomp2d = lbm_bindings::LbmMpiDecomp2D::new(
                        cfg.fluid.nx as i32, cfg.fluid.ny as i32, 1, nprocs);
                } else {
                    _decomp3d = lbm_bindings::LbmMpiDecomp3D::new(
                        cfg.fluid.nx as i32, cfg.fluid.ny as i32, cfg.fluid.nz as i32,
                        px as i32, py as i32, pz as i32);
                }
            } else {
                // 旧模式 "1d_y" → nx_blocks=1, ny_blocks=nprocs（1D Y 切片）
                let (px, py): (u32, u32) = if cfg.mpi.mode == "1d_y" {
                    (1, nprocs as u32)
                } else {
                    cfg.mpi.effective_blocks(nprocs)
                };
                if px as i32 * py as i32 != nprocs {
                    eprintln!(
                        "[warn] mpi.nx_blocks({}) * mpi.ny_blocks({}) = {} ≠ nprocs({}). \
                         Falling back to 1D Y slice (nx_blocks=1, ny_blocks=nprocs).",
                        px, py, px as i32 * py as i32, nprocs
                    );
                    _decomp2d = lbm_bindings::LbmMpiDecomp2D::new(
                        cfg.fluid.nx as i32, cfg.fluid.ny as i32, 1, nprocs);
                } else {
                    _decomp2d = lbm_bindings::LbmMpiDecomp2D::new(
                        cfg.fluid.nx as i32, cfg.fluid.ny as i32, px as i32, py as i32);
                }
            }
        }
        "independent" | "multigrid" => {
            // 无 MPI 通信，不创建任何分解对象
            // "multigrid" 当前版本退化为独立模式
            if effective_mode == "multigrid" && nprocs > 1 {
                eprintln!(
                    "[info] mode=\"multigrid\": MgTree framework is ready \
                     (LbmMgTree/MgNode); this version degrades to independent mode \
                     (each rank runs the full grid). Configure nesting via the \
                     lbm_bindings::LbmMgTree API."
                );
            }
        }
        _ => {
            eprintln!("[warn] unknown mpi.mode {:?}; defaulting to 1D Y slice.", effective_mode);
            _decomp2d = lbm_bindings::LbmMpiDecomp2D::new(
                cfg.fluid.nx as i32, cfg.fluid.ny as i32, 1, nprocs);
        }
    }

    // 确定本进程实际使用的网格尺寸（从已创建的分解对象中读取）
    let (grid_nx, grid_ny, grid_nz) = if let Some(ref d) = _decomp3d {
        (d.grid_nx(), d.grid_ny(), d.grid_nz())
    } else if let Some(ref d) = _decomp2d {
        (d.grid_nx(), d.grid_ny(), cfg.fluid.nz as i32)
    } else {
        // "independent" / "multigrid" / 未启用 MPI：使用完整全局网格
        (cfg.fluid.nx as i32, cfg.fluid.ny as i32, cfg.fluid.nz as i32)
    };

    // 各进程依次顺序打印本地网格信息，避免并发写入 stdout 导致 UTF-8 序列损坏
    // （MPI 并行时多进程同时写 stdout，多字节字符序列可能被截断/乱序）
    if nprocs > 1 {
        use std::io::Write;
        for r in 0..nprocs {
            if rank == r {
                print!("  rank {} local grid: {}x{}x{}\n", rank, grid_nx, grid_ny, grid_nz);
                let _ = std::io::stdout().flush();
            }
            lbm_bindings::mpi_barrier();
        }
    }

    // 初始化流体格子网格
    let mut grid = LbmGrid::new(grid_nx, grid_ny, grid_nz, model);

    // 初始化求解器
    let mut solver = LbmSolver::new(&mut grid, cfg.omega(), cm);

    // -----------------------------------------------------------------------
    // MPI 域分解绑定：将已创建的分解对象绑定到求解器
    // -----------------------------------------------------------------------
    if let Some(ref mut d) = _decomp3d {
        solver.attach_mpi3d(Some(d));
    } else if let Some(ref mut d) = _decomp2d {
        solver.attach_mpi2d(Some(d));
    }

    // -----------------------------------------------------------------------
    // Register boundary conditions
    // -----------------------------------------------------------------------
    // Collect per-rank BC log lines; printed in rank order after registration to
    // avoid stdout interleaving when running under MPI.
    let mut bc_log: Vec<String> = Vec::new();

    // Prepend a header showing this rank's global coordinate range.
    if effective_mode == "block" && nprocs > 1 {
        let region_str = if let Some(ref d) = _decomp3d {
            format!(
                "x=[{}, {}), y=[{}, {}), z=[{}, {})",
                d.x_start(), d.x_start() + d.local_nx(),
                d.y_start(), d.y_start() + d.local_ny(),
                d.z_start(), d.z_start() + d.local_nz(),
            )
        } else if let Some(ref d) = _decomp2d {
            format!(
                "x=[{}, {}), y=[{}, {})",
                d.x_start(), d.x_start() + d.local_nx(),
                d.y_start(), d.y_start() + d.local_ny(),
            )
        } else {
            format!("x=[0, {}), y=[0, {})", cfg.fluid.nx, cfg.fluid.ny)
        };
        bc_log.push(format!("  rank {:3}  region: {}", rank, region_str));
    }

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

        // MPI 块分解：仅在本进程持有该物理壁面时注册边界条件。
        // 判断依据：x_start/y_start/z_start + local_* 是否触及全局边界。
        let apply_bc = if effective_mode == "block" && nprocs > 1 {
            if let Some(ref d) = _decomp3d {
                let x_start  = d.x_start()  as u64;
                let y_start  = d.y_start()  as u64;
                let z_start  = d.z_start()  as u64;
                let local_nx = d.local_nx() as u64;
                let local_ny = d.local_ny() as u64;
                let local_nz = d.local_nz() as u64;
                let gnx = cfg.fluid.nx; let gny = cfg.fluid.ny; let gnz = cfg.fluid.nz;
                match face {
                    Face::South  => y_start == 0,
                    Face::North  => y_start + local_ny == gny as u64,
                    Face::West   => x_start == 0,
                    Face::East   => x_start + local_nx == gnx as u64,
                    Face::Bottom => z_start == 0,
                    Face::Top    => z_start + local_nz == gnz as u64,
                }
            } else if let Some(ref d) = _decomp2d {
                let x_start  = d.x_start()  as u64;
                let y_start  = d.y_start()  as u64;
                let local_nx = d.local_nx() as u64;
                let local_ny = d.local_ny() as u64;
                let gnx      = cfg.fluid.nx;
                let gny      = cfg.fluid.ny;
                match face {
                    Face::South  => y_start == 0,
                    Face::North  => y_start + local_ny == gny as u64,
                    Face::West   => x_start == 0,
                    Face::East   => x_start + local_nx == gnx as u64,
                    // Bottom/Top 用于三维，非 2D 分解方向，所有进程均注册
                    _            => true,
                }
            } else {
                true
            }
        } else {
            true
        };

        if apply_bc {
            solver.add_boundary_condition(
                bc_type, face,
                bc_cfg.ux, bc_cfg.uy, bc_cfg.uz,
                bc_cfg.rho,
            );
            bc_log.push(format!(
                "  rank {:3}, {:?}, {:?}  (ux={:.4}, uy={:.4}, rho={:.4})",
                rank, face, bc_type, bc_cfg.ux, bc_cfg.uy, bc_cfg.rho
            ));
        }
    }

    // Print BC log for every rank in rank order to avoid interleaving
    if nprocs > 1 {
        use std::io::Write;
        for r in 0..nprocs {
            if rank == r {
                for line in &bc_log {
                    println!("{}", line);
                }
                let _ = std::io::stdout().flush();
            }
            lbm_bindings::mpi_barrier();
        }
    } else {
        for line in &bc_log {
            println!("{}", line);
        }
    }

    // -----------------------------------------------------------------------
    // 构造 MPI 块分解分区信息（用于输出时剥离幽灵行/列并嵌入元数据）
    // -----------------------------------------------------------------------
    let partition: Option<PartitionInfo> = if effective_mode == "block" && nprocs > 1 {
        // 3D 分解时，使用 3D decomp 的 x/y 分区信息（z 方向暂由合并层处理）
        if let Some(ref d) = _decomp3d {
            Some(PartitionInfo {
                phys_x0:   d.phys_x0()  as usize,
                phys_y0:   d.phys_y0()  as usize,
                local_nx:  d.local_nx() as usize,
                local_ny:  d.local_ny() as usize,
                x_start:   d.x_start()  as usize,
                y_start:   d.y_start()  as usize,
                global_nx: cfg.fluid.nx as usize,
                global_ny: cfg.fluid.ny as usize,
            })
        } else {
            _decomp2d.as_ref().map(|d| PartitionInfo {
                phys_x0:   d.phys_x0()  as usize,
                phys_y0:   d.phys_y0()  as usize,
                local_nx:  d.local_nx() as usize,
                local_ny:  d.local_ny() as usize,
                x_start:   d.x_start()  as usize,
                y_start:   d.y_start()  as usize,
                global_nx: cfg.fluid.nx as usize,
                global_ny: cfg.fluid.ny as usize,
            })
        }
    } else {
        None
    };

    // 创建输出目录
    // MPI 块分解模式（mode="block"）下，每进程的分区快照写入各自的子目录
    // <output.directory>/rank_<N>/，避免多进程同时写同一文件引发竞态条件。
    // 多进程独立模式（"independent"/"multigrid"）同样写入各自子目录。
    let (eff_mode, _) = cfg.mpi.normalized_mode();
    let output_dir = if nprocs > 1
        && (eff_mode == "block" || eff_mode == "independent" || eff_mode == "multigrid")
    {
        format!("{}/rank_{}", cfg.output.directory, rank)
    } else {
        cfg.output.directory.clone()
    };
    std::fs::create_dir_all(&output_dir)?;

    // combine_blocks=true 时，rank-0 额外向全局输出目录写出合并快照
    let combine_blocks = cfg.output.combine_blocks && nprocs > 1 && eff_mode == "block";
    if combine_blocks && rank == 0 {
        std::fs::create_dir_all(&cfg.output.directory)?;
    }

    if nprocs > 1 && eff_mode == "block" {
        if rank == 0 {
            let combine_note = if combine_blocks {
                format!(" (+combined global snapshots -> {}/fluid_*.{})",
                        cfg.output.directory, cfg.output.format.replace("tecplot_", ""))
            } else {
                " (post-process: stitch partitions via x_start/y_start, or set combine_blocks=true)".to_string()
            };
            println!(
                "Output dir : rank=0 -> {}/rank_0/  (MPI block mode, each rank writes its partition{})",
                cfg.output.directory, combine_note
            );
        }
    }

    let csv_path = format!("{}/monitor.csv", output_dir);

    if rank == 0 {
        println!("\nStarting time integration...");
    }
    if cfg.simulation.n_steps == 0 {
        if rank == 0 {
            println!("n_steps is 0 — nothing to simulate.");
        }
        return Ok(());
    }

    for step in 0..cfg.simulation.n_steps {
        solver.step(&mut grid);

        let time = (step + 1) as f64 * cfg.simulation.dt;

        // -- 高频：原生 Rust 快照（格式由 output.format 决定）-------------------
        if step % cfg.output.write_interval == 0 || step == cfg.simulation.n_steps - 1 {
            if rank == 0 {
                println!("  step {:>6} / {}  t = {:.3}", step + 1, cfg.simulation.n_steps, time);
            }
            match cfg.output.format.as_str() {
                "tecplot_asc" => {
                    output::write_snapshot_tecplot_asc(&grid, step + 1, time, &output_dir, partition)
                        .with_context(|| format!("failed to write Tecplot ASCII snapshot (step {})", step + 1))?;
                }
                "tecplot_bin" => {
                    output::write_snapshot_tecplot_bin(&grid, step + 1, time, &output_dir, partition)
                        .with_context(|| format!("failed to write Tecplot binary snapshot (step {})", step + 1))?;
                }
                _ => {
                    output::write_snapshot_npz(&grid, step + 1, time, &output_dir, partition)
                        .with_context(|| format!("failed to write NPZ snapshot (step {})", step + 1))?;
                }
            }

            // -- combine_blocks: rank-0 gathers all partition data and writes a combined global snapshot --
            if combine_blocks {
                if let Some(p) = partition {
                    let grid_nx  = grid.nx() as usize;
                    let n_phys   = p.local_nx * p.local_ny;
                    let mut l_rho = Vec::with_capacity(n_phys);
                    let mut l_ux  = Vec::with_capacity(n_phys);
                    let mut l_uy  = Vec::with_capacity(n_phys);
                    for j in p.phys_y0..(p.phys_y0 + p.local_ny) {
                        for i in p.phys_x0..(p.phys_x0 + p.local_nx) {
                            let idx = (j * grid_nx + i) as i32;
                            l_rho.push(grid.rho(idx));
                            l_ux .push(grid.ux (idx));
                            l_uy .push(grid.uy (idx));
                        }
                    }
                    // Gather three fields to rank-0
                    if let (Some((g_rho, gnx, gny)), Some((g_ux, _, _)), Some((g_uy, _, _))) = (
                        output::gather_field_to_root(&l_rho, &p, 0),
                        output::gather_field_to_root(&l_ux,  &p, 0),
                        output::gather_field_to_root(&l_uy,  &p, 0),
                    ) {
                        // Only rank-0 writes (gather_field_to_root returns None for non-root ranks)
                        match cfg.output.format.as_str() {
                            "tecplot_asc" => {
                                output::write_global_snapshot_tecplot_asc(
                                    &g_rho, &g_ux, &g_uy, gnx, gny,
                                    step + 1, time, &cfg.output.directory,
                                ).with_context(|| format!(
                                    "failed to write combined Tecplot ASCII snapshot (step {})", step + 1))?;
                            }
                            "tecplot_bin" => {
                                output::write_global_snapshot_tecplot_bin(
                                    &g_rho, &g_ux, &g_uy, gnx, gny,
                                    step + 1, time, &cfg.output.directory,
                                ).with_context(|| format!(
                                    "failed to write combined Tecplot binary snapshot (step {})", step + 1))?;
                            }
                            _ => {
                                output::write_global_snapshot_npz(
                                    &g_rho, &g_ux, &g_uy, gnx, gny,
                                    step + 1, time, &cfg.output.directory,
                                ).with_context(|| format!(
                                    "failed to write combined NPZ snapshot (step {})", step + 1))?;
                            }
                        }
                    } else {
                        // Non-root ranks: already participated in MPI_Gatherv inside gather_field_to_root;
                        // no additional action needed here.
                    }
                }
            }
        }

        // -- 逐步：轻量级 CSV 监控日志 -----------------------------------------
        if cfg.output.enable_csv_monitor {
            // 仅统计物理节点的动能（排除幽灵行/列）
            let (px0, py0, pnx, pny, gnx) = if let Some(p) = partition {
                (p.phys_x0, p.phys_y0, p.local_nx, p.local_ny, grid.nx() as usize)
            } else {
                (0, 0, grid.nx() as usize, grid.ny() as usize, grid.nx() as usize)
            };
            let n_phys = pnx * pny;
            let ke: f64 = (0..pny).flat_map(|j| (0..pnx).map(move |i| (j, i)))
                .map(|(j, i)| {
                    let idx = ((py0 + j) * gnx + px0 + i) as i32;
                    let u = grid.ux(idx);
                    let v = grid.uy(idx);
                    u * u + v * v
                })
                .sum::<f64>()
                / (n_phys as f64)
                * 0.5;
            output::append_monitor_csv(&csv_path, step + 1, time, &[("ke", ke)])
                .with_context(|| format!("Failed to write monitor CSV at step {}", step + 1))?;
        }

        // -- 低频：Python FFI 等值线图（python-ffi 特性）-----------------------
        #[cfg(feature = "python-ffi")]
        if let Some(pi) = cfg.output.plot_interval {
            if step % pi == 0 || step == cfg.simulation.n_steps - 1 {
                // 准备物理节点数据（当 combine_blocks=true 时使用全局场；否则用本地分区场）
                let (plot_rho, plot_ux, plot_uy, plot_nx, plot_ny, plot_dir) =
                    if combine_blocks {
                        // combine_blocks 模式：gather 全局场并只在 rank-0 绘图
                        if let Some(p) = partition {
                            let grid_nx  = grid.nx() as usize;
                            let n_phys   = p.local_nx * p.local_ny;
                            let mut l_rho = Vec::with_capacity(n_phys);
                            let mut l_ux  = Vec::with_capacity(n_phys);
                            let mut l_uy  = Vec::with_capacity(n_phys);
                            for j in p.phys_y0..(p.phys_y0 + p.local_ny) {
                                for i in p.phys_x0..(p.phys_x0 + p.local_nx) {
                                    let idx = (j * grid_nx + i) as i32;
                                    l_rho.push(grid.rho(idx));
                                    l_ux .push(grid.ux (idx));
                                    l_uy .push(grid.uy (idx));
                                }
                            }
                            let grho = output::gather_field_to_root(&l_rho, &p, 0);
                            let gux  = output::gather_field_to_root(&l_ux,  &p, 0);
                            let guy  = output::gather_field_to_root(&l_uy,  &p, 0);
                            if let (Some((gr, gnx, gny)), Some((gu, _, _)), Some((gv, _, _))) =
                                (grho, gux, guy)
                            {
                                (gr, gu, gv, gnx, gny, cfg.output.directory.clone())
                            } else {
                                // 非 root 进程：不绘图（continue to next iter）
                                continue;
                            }
                        } else {
                            // partition 为 None（不应发生）
                            let (px0, py0, pnx, pny, gnx) =
                                (0, 0, grid.nx() as usize, grid.ny() as usize, grid.nx() as usize);
                            let n_phys = pnx * pny;
                            let mut rho = Vec::with_capacity(n_phys);
                            let mut ux  = Vec::with_capacity(n_phys);
                            let mut uy  = Vec::with_capacity(n_phys);
                            for j in py0..(py0 + pny) {
                                for i in px0..(px0 + pnx) {
                                    let idx = (j * gnx + i) as i32;
                                    rho.push(grid.rho(idx));
                                    ux .push(grid.ux (idx));
                                    uy .push(grid.uy (idx));
                                }
                            }
                            (rho, ux, uy, pnx, pny, output_dir.clone())
                        }
                    } else {
                        // 普通模式：使用本地分区场（原有行为）
                        let (px0, py0, pnx, pny, gnx) = if let Some(p) = partition {
                            (p.phys_x0, p.phys_y0, p.local_nx, p.local_ny, grid.nx() as usize)
                        } else {
                            (0, 0, grid.nx() as usize, grid.ny() as usize, grid.nx() as usize)
                        };
                        let n_phys = pnx * pny;
                        let mut rho = Vec::with_capacity(n_phys);
                        let mut ux  = Vec::with_capacity(n_phys);
                        let mut uy  = Vec::with_capacity(n_phys);
                        for j in py0..(py0 + pny) {
                            for i in px0..(px0 + pnx) {
                                let idx = (j * gnx + i) as i32;
                                rho.push(grid.rho(idx));
                                ux .push(grid.ux (idx));
                                uy .push(grid.uy (idx));
                            }
                        }
                        (rho, ux, uy, pnx, pny, output_dir.clone())
                    };

                for field_name in &["velocity_magnitude", "vorticity", "streamlines"] {
                    if let Err(e) = python_bridge::plot_field(
                        &plot_rho, &plot_ux, &plot_uy, plot_nx, plot_ny,
                        step + 1, time,
                        &plot_dir,
                        field_name,
                    ) {
                        eprintln!("[python-ffi] plot {field_name} failed at step {}: {e}", step + 1);
                    }
                }
            }
        }
    }

    if rank == 0 {
        println!("\nSimulation complete.");
    }

    // -----------------------------------------------------------------------
    // 后处理：Python 子进程脚本（仅 rank-0 运行，避免 MPI 多进程重复启动）
    // -----------------------------------------------------------------------
    if rank == 0 {
        if let Some(ref script) = cfg.python.post_script.clone() {
            println!("\n--- Post-processing (Python subprocess) ---");
            run_python_subprocess(&cfg.python.interpreter, script, &[&output_dir])?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
/// 顶层入口：初始化 MPI → 运行求解器 → 终结 MPI。
///
/// 将仿真逻辑封装在 [`run()`] 中，确保无论 `run()` 成功还是出错，
/// `mpi_finalize()` 总能被调用，避免 MPI 进程因未调用 `MPI_Finalize`
/// 而引发警告或资源泄漏。
///
/// 在非 MPI 模式（`ENABLE_MPI=OFF`）下，`mpi_init`/`mpi_finalize` 均为空操作，
/// 此函数仍正确地将错误码传递给操作系统。
fn main() {
    // mpi_init() 必须在所有 MPI 函数之前调用（包括 mpi_rank / mpi_size）。
    // 若未启用 LBM_ENABLE_MPI，此函数为空操作，安全调用。
    lbm_bindings::mpi_init();

    let result = run();

    // 确保 MPI 总能被正确终结（不论 run() 是否返回错误）
    lbm_bindings::mpi_finalize();

    if let Err(e) = result {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}
