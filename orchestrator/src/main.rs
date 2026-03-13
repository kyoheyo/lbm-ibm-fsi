mod config;
mod fsi;
mod output;
mod python_bridge;
mod sim;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use config::Config;
use lbm_bindings::{CollisionModel, LatticeModel, LbmGrid, LbmSolver};
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
/// 真正的求解器入口（在 MPI init/finalize 包装内执行）。
fn run() -> Result<()> {
    let args = Args::parse();
    let mut cfg = Config::from_file(&args.config)?;
    if let Some(n) = args.steps { cfg.simulation.n_steps = n; }

    // MPI rank / 进程数（mpi_init() 已在调用者 main() 中完成）
    let rank   = lbm_bindings::mpi_rank();
    let nprocs = lbm_bindings::mpi_size();

    // 应用并行配置（OpenMP 线程数必须在所有进程上生效）
    if cfg.parallel.omp_num_threads > 0 {
        lbm_bindings::set_omp_num_threads(cfg.parallel.omp_num_threads as i32);
    }

    // 打印头部信息（rank-0 only，避免 MPI 多进程重复输出）
    if rank == 0 { print_header(&cfg, &args.config, nprocs); }

    // Python sys.path 扩展（未启用 python-ffi 时为无操作）
    if let Some(ref extra_path) = cfg.python.pythonpath {
        if let Err(e) = python_bridge::add_python_path(extra_path) {
            eprintln!("[python-ffi] sys.path extension failed: {e}");
        }
    }

    // 预处理：Python 子进程（rank-0 only，避免多进程重复启动）
    if rank == 0 {
        if let Some(ref script) = cfg.python.pre_script.clone() {
            println!("\n--- Pre-processing (Python subprocess) ---");
            let config_str = args.config.to_string_lossy().into_owned();
            python_bridge::run_subprocess(&cfg.python.interpreter, script, &[&config_str])?;
        }
    }

    // IBM 标记点生成（python-ffi 特性，进程内不产生 CSV）
    #[cfg(feature = "python-ffi")]
    generate_ibm_markers_ffi(&cfg);

    // 解析格子模型与碰撞模型
    let model = parse_lattice_model(&cfg.simulation.lattice_model);
    let cm    = parse_collision_model(&cfg.simulation.collision_model);

    // MPI 域分解设置
    let mut mpi = sim::MpiDecomp::setup(&cfg, nprocs)?;

    // 本进程本地网格尺寸（MPI 块分解时 < 全局；单进程时 = 全局）
    let (grid_nx, grid_ny, grid_nz) = mpi.local_grid_size(&cfg);
    print_local_grid_sizes(rank, nprocs, grid_nx, grid_ny, grid_nz);

    // 初始化格子网格与求解器
    let mut grid   = LbmGrid::new(grid_nx, grid_ny, grid_nz, model);
    let mut solver = LbmSolver::new(&mut grid, cfg.omega(), cm);
    mpi.attach_to_solver(&mut solver);

    // 流固耦合模式解析与校验
    let has_solid = !cfg.solid.bodies.is_empty()
        && cfg.solid.bc_type.to_lowercase() != "none";
    let has_ibm = cfg.ibm.is_some();
    let coupling_mode = fsi::resolve_coupling_mode(&cfg.fsi, has_solid, has_ibm);
    fsi::validate_coupling_mode(
        coupling_mode, has_solid, has_ibm, cfg.fsi.normalized_coupling())?;

    // 固体标记与反弹方案（BB / IBB）
    if coupling_mode.needs_solid() {
        let (x_start, y_start, phys_x0, phys_y0) = mpi.global_coords();
        fsi::setup_solid_bodies(
            &cfg, &mut grid, &mut solver,
            x_start, y_start, phys_x0, phys_y0, rank,
        );
    }

    // 注册流体边界条件
    sim::register_boundary_conditions(&cfg, &mut solver, &mpi, nprocs, rank);

    // 构建分区信息与输出目录
    let partition  = mpi.partition_info(&cfg, nprocs);
    let output_dir = mpi.output_dir(&cfg, nprocs, rank);
    std::fs::create_dir_all(&output_dir)?;

    // combine_blocks 模式下 rank-0 额外创建全局输出目录
    let combine_blocks = cfg.output.combine_blocks && nprocs > 1
        && mpi.effective_mode == "block";
    if combine_blocks && rank == 0 {
        std::fs::create_dir_all(&cfg.output.directory)?;
    }

    // 输出目录路径提示（rank-0 only）
    if rank == 0 && nprocs > 1 && mpi.effective_mode == "block" {
        let combine_note = if combine_blocks {
            format!(" (+combined global snapshots -> {}/fluid_*.{})",
                    cfg.output.directory, cfg.output.format.replace("tecplot_", ""))
        } else {
            " (post-process: stitch partitions via x_start/y_start, or set combine_blocks=true)"
                .to_string()
        };
        println!(
            "Output dir : rank=0 -> {}/rank_0/  (MPI block mode, each rank writes its partition{})",
            cfg.output.directory, combine_note
        );
    }

    // IBM 体初始化（仅在需要 IBM 耦合时）
    let mut ibm_entries: Vec<fsi::IbmEntry> = if coupling_mode.needs_ibm() {
        fsi::setup_ibm_bodies(&cfg, rank)?
    } else {
        Vec::new()
    };

    if rank == 0 && coupling_mode != fsi::FsiCouplingMode::None {
        println!("  [FSI] 耦合模式: {}", coupling_mode.description());
    }
    if rank == 0 { println!("\nStarting time integration..."); }

    if cfg.simulation.n_steps == 0 {
        if rank == 0 { println!("n_steps is 0 — nothing to simulate."); }
        return Ok(());
    }

    // 时间循环
    let csv_path = format!("{}/monitor.csv", output_dir);
    run_time_loop(
        &cfg, &mut grid, &mut solver, &mut ibm_entries,
        partition, &output_dir, &csv_path, combine_blocks, rank,
    )?;

    if rank == 0 { println!("\nSimulation complete."); }

    // 后处理：Python 子进程（rank-0 only）
    if rank == 0 {
        if let Some(ref script) = cfg.python.post_script.clone() {
            println!("\n--- Post-processing (Python subprocess) ---");
            python_bridge::run_subprocess(&cfg.python.interpreter, script, &[&output_dir])?;
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
    lbm_bindings::mpi_init();
    let result = run();
    lbm_bindings::mpi_finalize();

    if let Err(e) = result {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// 时间循环
// ---------------------------------------------------------------------------

/// 主时间步循环：每步推进仿真并按配置频率写出各类结果。
///
/// 涵盖：
/// - LBM 步进 + IBM 步进
/// - 快照写出（NPZ / Tecplot ASC / Tecplot BIN）
/// - combine_blocks 全局快照聚合（MPI 块模式，可选）
/// - CSV 监控日志（动能等标量量）
/// - 固体受力统计（BB/IBB 动量交换法）
/// - IBM 固体受力统计（Lagrangian 力密度积分，多体逐体写出）
/// - Python FFI 等值线图（`python-ffi` 特性，可选）
fn run_time_loop(
    cfg: &Config,
    grid: &mut LbmGrid,
    solver: &mut LbmSolver,
    ibm_entries: &mut Vec<fsi::IbmEntry>,
    partition: Option<PartitionInfo>,
    output_dir: &str,
    csv_path: &str,
    combine_blocks: bool,
    rank: i32,
) -> Result<()> {
    for step in 0..cfg.simulation.n_steps {
        solver.step(grid);

        // IBM 力展布（step() 之后；力写入 grid.force，下一步 collide 时通过 Guo 格式加入）
        if !ibm_entries.is_empty() {
            step_ibm(cfg, grid, ibm_entries);
        }

        let time = (step + 1) as f64 * cfg.simulation.dt;

        // 高频：欧拉场快照（每 write_interval 步或最后一步）
        if step % cfg.output.write_interval == 0 || step == cfg.simulation.n_steps - 1 {
            if rank == 0 {
                println!("  step {:>6} / {}  t = {:.3}", step + 1, cfg.simulation.n_steps, time);
            }
            write_step_snapshot(cfg, grid, step + 1, time, output_dir, partition)?;
            if combine_blocks {
                write_combined_snapshot(cfg, grid, step + 1, time, partition)?;
            }
        }

        // 逐步：CSV 监控日志（动能等标量量）
        if cfg.output.enable_csv_monitor {
            write_monitor_csv(cfg, grid, step + 1, time, csv_path, partition)?;
        }

        // 固体受力输出（BB/IBB 动量交换法）
        if cfg.solid.force_output.enabled
            && (step % cfg.solid.force_output.interval == 0
                || step == cfg.simulation.n_steps - 1)
        {
            write_solid_force(cfg, grid, step + 1, time, output_dir, partition, rank)?;
        }

        // IBM 固体受力输出（Lagrangian 力密度积分，多体逐体写出）
        for entry in ibm_entries.iter() {
            if entry.force_cfg.enabled
                && (step % entry.force_cfg.interval == 0
                    || step == cfg.simulation.n_steps - 1)
            {
                let (ibm_fx, ibm_fy) = entry.ms.compute_body_force();
                if rank == 0 {
                    let force_csv = format!("{}/{}.csv", output_dir, entry.force_cfg.filename);
                    output::append_monitor_csv(
                        &force_csv, step + 1, time,
                        &[("ibm_fx", ibm_fx), ("ibm_fy", ibm_fy)],
                    ).with_context(|| format!(
                        "Failed to write IBM force CSV ({}) at step {}", entry.label, step + 1
                    ))?;
                }
            }
        }

        // 低频：Python FFI 等值线图（`python-ffi` 特性）
        #[cfg(feature = "python-ffi")]
        if let Some(pi) = cfg.output.plot_interval {
            if step % pi == 0 || step == cfg.simulation.n_steps - 1 {
                plot_step_ffi(cfg, grid, step + 1, time, partition, combine_blocks, output_dir)?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 时间循环内辅助函数
// ---------------------------------------------------------------------------

/// 对所有 IBM 体执行一步 IBM 力计算（方法由 `cfg.ibm.method` 指定）。
fn step_ibm(cfg: &Config, grid: &mut LbmGrid, ibm_entries: &mut [fsi::IbmEntry]) {
    let ibm_cfg = cfg.ibm.as_ref().unwrap();
    let dx = 1.0_f64;
    let dt = cfg.simulation.dt;
    for entry in ibm_entries.iter_mut() {
        match ibm_cfg.method.to_lowercase().as_str() {
            "penalty" => entry.ms.step_penalty(grid, dx, dt, ibm_cfg.alpha, ibm_cfg.beta),
            "mls"     => entry.ms.step_mls(grid, dx, dt),
            _         => entry.ms.step_mdf(grid, dx, dt, ibm_cfg.n_iter),
        }
    }
}

/// 将当前欧拉场写出为本进程分区快照（格式由 `cfg.output.format` 决定）。
fn write_step_snapshot(
    cfg: &Config,
    grid: &LbmGrid,
    step: u64,
    time: f64,
    output_dir: &str,
    partition: Option<PartitionInfo>,
) -> Result<()> {
    match cfg.output.format.as_str() {
        "tecplot_asc" => output::write_snapshot_tecplot_asc(grid, step, time, output_dir, partition)
            .with_context(|| format!("failed to write Tecplot ASCII snapshot (step {})", step)),
        "tecplot_bin" => output::write_snapshot_tecplot_bin(grid, step, time, output_dir, partition)
            .with_context(|| format!("failed to write Tecplot binary snapshot (step {})", step)),
        _ => output::write_snapshot_npz(grid, step, time, output_dir, partition)
            .with_context(|| format!("failed to write NPZ snapshot (step {})", step)),
    }
}

/// combine_blocks 模式：将各进程物理场 gather 到 rank-0 并写出全局快照。
///
/// 非 root 进程已在 [`output::gather_field_to_root`] 内参与 `MPI_Gatherv`，
/// 此处直接返回 `Ok(())`。
fn write_combined_snapshot(
    cfg: &Config,
    grid: &LbmGrid,
    step: u64,
    time: f64,
    partition: Option<PartitionInfo>,
) -> Result<()> {
    let Some(p) = partition else { return Ok(()); };

    let (l_rho, l_ux, l_uy, _, _) = output::extract_physical_fields(grid, Some(p));

    let (Some((g_rho, gnx, gny)), Some((g_ux, _, _)), Some((g_uy, _, _))) = (
        output::gather_field_to_root(&l_rho, &p, 0),
        output::gather_field_to_root(&l_ux,  &p, 0),
        output::gather_field_to_root(&l_uy,  &p, 0),
    ) else {
        // 非 root 进程：已参与 gather，无需写文件
        return Ok(());
    };

    match cfg.output.format.as_str() {
        "tecplot_asc" => output::write_global_snapshot_tecplot_asc(
            &g_rho, &g_ux, &g_uy, gnx, gny, step, time, &cfg.output.directory,
        ).with_context(|| format!("failed to write combined Tecplot ASCII snapshot (step {})", step)),
        "tecplot_bin" => output::write_global_snapshot_tecplot_bin(
            &g_rho, &g_ux, &g_uy, gnx, gny, step, time, &cfg.output.directory,
        ).with_context(|| format!("failed to write combined Tecplot binary snapshot (step {})", step)),
        _ => output::write_global_snapshot_npz(
            &g_rho, &g_ux, &g_uy, gnx, gny, step, time, &cfg.output.directory,
        ).with_context(|| format!("failed to write combined NPZ snapshot (step {})", step)),
    }
}

/// 计算平均动能并追加到 CSV 监控日志。
fn write_monitor_csv(
    _cfg: &Config,
    grid: &LbmGrid,
    step: u64,
    time: f64,
    csv_path: &str,
    partition: Option<PartitionInfo>,
) -> Result<()> {
    let (px0, py0, pnx, pny, gnx) = if let Some(p) = partition {
        (p.phys_x0, p.phys_y0, p.local_nx, p.local_ny, grid.nx() as usize)
    } else {
        (0, 0, grid.nx() as usize, grid.ny() as usize, grid.nx() as usize)
    };
    let n_phys = pnx * pny;
    let ke: f64 = (0..pny)
        .flat_map(|j| (0..pnx).map(move |i| (j, i)))
        .map(|(j, i)| {
            let idx = ((py0 + j) * gnx + px0 + i) as i32;
            let u = grid.ux(idx);
            let v = grid.uy(idx);
            u * u + v * v
        })
        .sum::<f64>()
        / (n_phys as f64)
        * 0.5;
    output::append_monitor_csv(csv_path, step, time, &[("ke", ke)])
        .with_context(|| format!("Failed to write monitor CSV at step {}", step))
}

/// 统计固体所受合力（动量交换法）并写入 CSV。
///
/// MPI 模式下各进程的局部贡献通过 `MPI_Allreduce` 求和；仅 rank-0 写文件。
fn write_solid_force(
    cfg: &Config,
    grid: &LbmGrid,
    step: u64,
    time: f64,
    output_dir: &str,
    partition: Option<PartitionInfo>,
    rank: i32,
) -> Result<()> {
    let (pi0, pj0, pi1, pj1) = if let Some(p) = partition {
        (p.phys_x0 as i32,
         p.phys_y0 as i32,
         (p.phys_x0 + p.local_nx - 1) as i32,
         (p.phys_y0 + p.local_ny - 1) as i32)
    } else {
        (0, 0, grid.nx() as i32 - 1, grid.ny() as i32 - 1)
    };

    let (local_fx, local_fy) = lbm_bindings::compute_solid_force(grid, pi0, pj0, pi1, pj1);
    let global_fx = lbm_bindings::mpi_allreduce_sum_f64(local_fx);
    let global_fy = lbm_bindings::mpi_allreduce_sum_f64(local_fy);

    if rank == 0 {
        let force_csv = format!("{}/{}.csv", output_dir, cfg.solid.force_output.filename);
        output::append_monitor_csv(
            &force_csv, step, time,
            &[("fx", global_fx), ("fy", global_fy)],
        ).with_context(|| format!("Failed to write solid force CSV at step {}", step))?;
    }
    Ok(())
}

/// Python FFI 等值线图（`python-ffi` 特性）。
///
/// combine_blocks 模式下，先将分区场 gather 到 rank-0，再由 rank-0 绘图。
/// 普通模式下直接使用本地物理分区场绘图。
#[cfg(feature = "python-ffi")]
fn plot_step_ffi(
    cfg: &Config,
    grid: &LbmGrid,
    step: u64,
    time: f64,
    partition: Option<PartitionInfo>,
    combine_blocks: bool,
    output_dir: &str,
) -> Result<()> {
    // 确定绘图所用场数据及输出目录
    let (plot_rho, plot_ux, plot_uy, plot_nx, plot_ny, plot_dir) = if combine_blocks {
        if let Some(p) = partition {
            let (l_rho, l_ux, l_uy, _, _) = output::extract_physical_fields(grid, Some(p));
            let grho = output::gather_field_to_root(&l_rho, &p, 0);
            let gux  = output::gather_field_to_root(&l_ux,  &p, 0);
            let guy  = output::gather_field_to_root(&l_uy,  &p, 0);
            if let (Some((gr, gnx, gny)), Some((gu, _, _)), Some((gv, _, _))) = (grho, gux, guy) {
                (gr, gu, gv, gnx, gny, cfg.output.directory.clone())
            } else {
                // 非 root 进程：已参与 gather，不绘图
                return Ok(());
            }
        } else {
            // partition 为 None（不应发生）：退化为本地全场
            let (rho, ux, uy, nx, ny) = output::extract_physical_fields(grid, None);
            (rho, ux, uy, nx, ny, output_dir.to_owned())
        }
    } else {
        let (rho, ux, uy, nx, ny) = output::extract_physical_fields(grid, partition);
        (rho, ux, uy, nx, ny, output_dir.to_owned())
    };

    for field_name in &["velocity_magnitude", "vorticity", "streamlines"] {
        if let Err(e) = python_bridge::plot_field(
            &plot_rho, &plot_ux, &plot_uy, plot_nx, plot_ny,
            step, time, &plot_dir, field_name,
        ) {
            eprintln!("[python-ffi] plot {field_name} failed at step {}: {e}", step);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 启动阶段辅助函数
// ---------------------------------------------------------------------------

/// 解析格子模型字符串为 [`LatticeModel`]（不认识的值退化为 D2Q9）。
fn parse_lattice_model(s: &str) -> LatticeModel {
    match s {
        "D3Q19" => LatticeModel::D3Q19,
        "D3Q27" => LatticeModel::D3Q27,
        _       => LatticeModel::D2Q9,
    }
}

/// 解析碰撞模型字符串为 [`CollisionModel`]（不认识的值退化为 BGK）。
fn parse_collision_model(s: &str) -> CollisionModel {
    match s {
        "MRT" => CollisionModel::Mrt,
        _     => CollisionModel::Bgk,
    }
}

/// 打印求解器启动头部信息（rank-0 only）。
fn print_header(cfg: &Config, config_path: &std::path::Path, nprocs: i32) {
    println!("=== LBM-IBM-FSI Solver ===");
    println!("Config   : {}", config_path.display());
    println!("Grid     : {}x{}x{}", cfg.fluid.nx, cfg.fluid.ny, cfg.fluid.nz);
    println!("Steps    : {}", cfg.simulation.n_steps);
    println!("Model    : {} / {}", cfg.simulation.lattice_model, cfg.simulation.collision_model);
    println!("Omega    : w = {:.6}", cfg.omega());
    println!("Output   : {} -> {}", cfg.output.format, match cfg.output.format.as_str() {
        "tecplot_asc" => "fluid_NNNNNN.dat (ASCII Tecplot)",
        "tecplot_bin" => "fluid_NNNNNN.plt (binary Tecplot TDV112)",
        _             => "fluid_NNNNNN.npz (NumPy compressed archive)",
    });
    if cfg.parallel.omp_num_threads > 0 {
        println!("OpenMP   : threads = {} (set by [parallel].omp_num_threads)",
                 cfg.parallel.omp_num_threads);
    }
    lbm_bindings::print_parallel_status();

    // MPI 模式摘要
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
        let ibm_halo = cfg.mpi.ibm_halo_width.max(1);
        if ibm_halo > 1 && cfg.ibm.is_some() {
            println!("MPI IBM  : ibm_halo_width={} (extended ghost layer for FourPoint δ kernel)",
                     ibm_halo);
        }
    }

    // 插件摘要
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
    }
}

/// 在多进程环境下，依次按 rank 顺序打印各进程的本地网格尺寸。
///
/// 顺序打印可避免多进程并发写入 stdout 时 UTF-8 多字节序列被截断/乱序。
fn print_local_grid_sizes(rank: i32, nprocs: i32, grid_nx: i32, grid_ny: i32, grid_nz: i32) {
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
}

/// 通过 Python FFI 生成 IBM 标记点（进程内，不产生 CSV 文件）。
///
/// 仅在启用 `python-ffi` 特性时编译；否则为空操作。
#[cfg(feature = "python-ffi")]
fn generate_ibm_markers_ffi(cfg: &Config) {
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

