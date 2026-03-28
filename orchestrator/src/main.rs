mod async_output;
mod config;
mod fsi;
mod motion;
mod output;
mod python_bridge;
mod sim;

use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;

use config::Config;
use lbm_bindings::{CollisionModel, Face, LatticeModel, LbmGrid, LbmMgTree, LbmSolver};
use crate::config::MotionType;
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
/// Solver entry point (executed inside the MPI init/finalize wrapper).
fn run() -> Result<()> {
    let args = Args::parse();
    let mut cfg = Config::from_file(&args.config)?;
    if let Some(n) = args.steps { cfg.simulation.n_steps = n; }

    // MPI rank / nprocs (mpi_init() has already been called by main())
    let rank   = lbm_bindings::mpi_rank();
    let nprocs = lbm_bindings::mpi_size();

    // When LBM_ENABLE_MPI is OFF, mpi_rank()/mpi_size() always return 0/1.
    // If the binary is launched under mpiexec anyway, every process would
    // think it is rank-0 and run a full independent simulation (allocating
    // grids, stepping, writing output — N redundant runs).
    // Detect the true process index via launcher env vars; any process that
    // is not rank-0 exits immediately so only one simulation is performed.
    let rank = if nprocs == 1 { launcher_rank(rank) } else { rank };
    if nprocs == 1 && rank > 0 {
        eprintln!(
            "[warn] Binary compiled without MPI support (LBM_ENABLE_MPI=OFF), \
             but launched with mpiexec (detected rank={rank}). \
             Only rank-0 runs the simulation; this process exits."
        );
        return Ok(());
    }

    // Apply parallel config (OpenMP thread count must be set on all ranks)
    if cfg.parallel.omp_num_threads > 0 {
        lbm_bindings::set_omp_num_threads(cfg.parallel.omp_num_threads as i32);
    }

    // MPI + OpenMP hybrid mode: prefer passive thread waiting so that OpenMP
    // worker threads yield the CPU (rather than spin-polling) while MPI
    // synchronisation is in progress.  This eliminates the artificial high-CPU
    // plateau visible on CPU-utilisation monitors between compute phases.
    //
    // OMP_WAIT_POLICY must be set before the first parallel region is entered.
    // Setting it here via std::env::set_var is safe because we are still
    // single-threaded at this point (mpi_init has been called but no OpenMP
    // threads have been spawned yet).
    if nprocs > 1 && std::env::var("OMP_WAIT_POLICY").is_err() {
        std::env::set_var("OMP_WAIT_POLICY", "passive");
    }

    // Print header (rank-0 only, prevents duplicate output in MPI mode)
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
    // 若用户配置了非单位初始密度，在求解器构造前写入（求解器构造时将 f 初始化为平衡态）
    if (cfg.fluid.rho0 - 1.0).abs() > 1e-15 {
        grid.fill_rho(cfg.fluid.rho0);
    }
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
    let (mut solid_entries, has_moving_solid) = if coupling_mode.needs_solid() {
        let (x_start, y_start, phys_x0, phys_y0) = mpi.global_coords();
        fsi::setup_solid_bodies(
            &cfg, &mut grid, &mut solver,
            x_start, y_start, phys_x0, phys_y0, rank,
        )
    } else {
        (Vec::new(), false)
    };

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

    // MPI 模式下：将 IBM 标记点坐标从全局格子坐标系变换到本地格子坐标系，
    // 并设置各进程"归属"的物理行/列范围，避免跨块双重计数。
    if !ibm_entries.is_empty() && nprocs > 1 {
        if let Some(ref d2) = mpi.decomp2d {
            let x_start  = d2.x_start();
            let y_start  = d2.y_start();
            let phys_x0  = d2.phys_x0();
            let phys_y0  = d2.phys_y0();
            let local_nx = d2.local_nx();
            let local_ny = d2.local_ny();
            for entry in ibm_entries.iter_mut() {
                entry.ms.adapt_to_partition(
                    x_start, y_start, phys_x0, phys_y0, local_nx, local_ny,
                );
            }
            if rank == 0 {
                println!(
                    "  [IBM] MPI partition adapt: rank-0 owns y=[{},{}), \
                     markers shifted to local coords",
                    phys_y0, phys_y0 + local_ny,
                );
            }
        }
    }

    if rank == 0 && coupling_mode != fsi::FsiCouplingMode::None {
        println!("  [FSI] coupling mode: {}", coupling_mode.description());
    }
    if rank == 0 { println!("\nStarting time integration..."); }

    if cfg.simulation.n_steps == 0 {
        if rank == 0 { println!("n_steps is 0 — nothing to simulate."); }
        return Ok(());
    }

    // 多重网格模式（[multigrid] 段存在且 enabled=true 且 levels 非空）
    if let Some(ref mg_cfg) = cfg.multigrid.clone() {
        if mg_cfg.enabled && !mg_cfg.levels.is_empty() {
            // 多重网格运行路径：以 mg_step_recursive 替代单层 solver.step()
            run_multigrid_loop(
                &cfg, mg_cfg, model, cm,
                &mut grid, &mut solver,
                &mut ibm_entries,
                &mut mpi, partition, combine_blocks,
                &output_dir, rank, nprocs,
            )?;
            if rank == 0 { println!("\nSimulation complete (multigrid mode)."); }
            if rank == 0 {
                if let Some(ref script) = cfg.python.post_script.clone() {
                    println!("\n--- Post-processing (Python subprocess) ---");
                    python_bridge::run_subprocess(&cfg.python.interpreter, script, &[&output_dir])?;
                }
            }
            return Ok(());
        }
    }

    // 时间循环
    let csv_path = format!("{}/monitor.csv", output_dir);
    run_time_loop(
        &cfg, &mut grid, &mut solver,
        &mut solid_entries, has_moving_solid,
        &mut ibm_entries,
        &mut mpi, partition, &output_dir, &csv_path, combine_blocks, rank,
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
/// Top-level entry point: initialise MPI → run solver → finalise MPI.
///
/// Wrapping simulation logic inside [`run()`] ensures that `mpi_finalize()`
/// is always called regardless of whether `run()` succeeds or returns an
/// error, preventing MPI processes from exiting without calling
/// `MPI_Finalize` (which would produce warnings or resource leaks).
///
/// When `LBM_ENABLE_MPI=OFF`, `mpi_init`/`mpi_finalize` are no-ops and
/// this function still propagates the exit code to the OS correctly.
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
/// ## 输出模式（由 `[output] async_io` 控制）
///
/// ### 阻塞模式（默认，`async_io = false`）
/// 每次写出操作在主线程内同步完成，行为与原始实现完全一致。
///
/// ### 异步模式（`async_io = true`）
/// 主线程从 `LbmGrid` 提取数据后立即提交给后台 `lbm-io` 线程写出，
/// 不阻塞仿真推进。MPI 集合通信仍在主线程同步执行。
fn run_time_loop(
    cfg: &Config,
    grid: &mut LbmGrid,
    solver: &mut LbmSolver,
    solid_entries: &mut Vec<fsi::SolidEntry>,
    has_moving_solid: bool,
    ibm_entries: &mut Vec<fsi::IbmEntry>,
    mpi: &mut sim::MpiDecomp,
    partition: Option<PartitionInfo>,
    output_dir: &str,
    csv_path: &str,
    combine_blocks: bool,
    rank: i32,
) -> Result<()> {
    let async_io = cfg.output.async_io;
    let writer: Option<async_output::AsyncWriter> = if async_io {
        Some(async_output::AsyncWriter::new(8))
    } else {
        None
    };

    for step in 0..cfg.simulation.n_steps {
        solver.step(grid);
        // solver.step() → stream() 在 compute_macroscopic() 之后自动完成幽灵层 u 交换，
        // 无需在此显式调用 ibm_halo_exchange_u_2d()。

        // 运动刚体 BB/IBB：手动施加 Ladd 移动壁面修正（在 solver.step 内已禁用自动 BC）
        if has_moving_solid && !solid_entries.is_empty() {
            let (pi0, pj0, pi1, pj1) = if let Some(p) = partition {
                (p.phys_x0 as i32, p.phys_y0 as i32,
                 (p.phys_x0 + p.local_nx - 1) as i32,
                 (p.phys_y0 + p.local_ny - 1) as i32)
            } else {
                (0, 0, grid.nx() as i32 - 1, grid.ny() as i32 - 1)
            };
            step_solid_moving(cfg, grid, solid_entries, step, pi0, pj0, pi1, pj1);
        }

        // IBM 力展布（step() 之后；力写入 grid.force，下一步 collide 时通过 Guo 格式加入）
        if !ibm_entries.is_empty() {
            step_ibm(cfg, grid, ibm_entries, step);

            // MPI 修正：spread_force() 可能向幽灵行写入力贡献；将这些贡献归还邻居并累加。
            if let Some(ref mut d2) = mpi.decomp2d {
                lbm_bindings::ibm_halo_reduce_force_2d(grid, d2);
            }
        }

        let time = (step + 1) as f64 * cfg.simulation.dt;
        let is_snapshot = step % cfg.output.write_interval == 0
            || step == cfg.simulation.n_steps - 1;

        // 高频：欧拉场快照
        if is_snapshot {
            if rank == 0 {
                println!("  step {:>6} / {}  t = {:.3}", step + 1, cfg.simulation.n_steps, time);
            }

            if let Some(ref w) = writer {
                let (rho, ux, uy, nx, ny) = output::extract_physical_fields(grid, partition);
                let fmt = cfg.output.format.clone();
                let dir = output_dir.to_owned();
                let p   = partition;
                w.submit(move || {
                    output::write_snapshot_raw(&fmt, &rho, &ux, &uy, nx, ny, step + 1, time, &dir, p)
                        .with_context(|| format!("failed to write snapshot (step {})", step + 1))
                });
            } else {
                write_step_snapshot(cfg, grid, step + 1, time, output_dir, partition)?;
            }

            // combine_blocks：MPI gather 在主线程；文件写出可异步
            if combine_blocks {
                if let Some(p) = partition {
                    let (l_rho, l_ux, l_uy, _, _) = output::extract_physical_fields(grid, Some(p));
                    let g_rho_r = output::gather_field_to_root(&l_rho, &p, 0);
                    let g_ux_r  = output::gather_field_to_root(&l_ux,  &p, 0);
                    let g_uy_r  = output::gather_field_to_root(&l_uy,  &p, 0);
                    if let (Some((g_rho, gnx, gny)), Some((g_ux, _, _)), Some((g_uy, _, _)))
                        = (g_rho_r, g_ux_r, g_uy_r)
                    {
                        let fmt = cfg.output.format.clone();
                        let dir = cfg.output.directory.clone();
                        if let Some(ref w) = writer {
                            w.submit(move || {
                                output::write_global_snapshot_raw(
                                    &fmt, &g_rho, &g_ux, &g_uy, gnx, gny, step + 1, time, &dir,
                                ).with_context(|| format!("failed to write combined snapshot (step {})", step + 1))
                            });
                        } else {
                            output::write_global_snapshot_raw(
                                &cfg.output.format, &g_rho, &g_ux, &g_uy, gnx, gny,
                                step + 1, time, &cfg.output.directory,
                            ).with_context(|| format!("failed to write combined snapshot (step {})", step + 1))?;
                        }
                    }
                    // non-root already participated in gather
                }
            }
        }

        // 逐步：CSV 监控日志（每步写出，保证时效性）
        if cfg.output.enable_csv_monitor {
            let ke = compute_monitor_ke(grid, partition);
            if let Some(ref w) = writer {
                let path = csv_path.to_owned();
                w.submit(move || {
                    output::append_monitor_csv(&path, step + 1, time, &[("ke", ke)])
                        .with_context(|| format!("Failed to write monitor CSV at step {}", step + 1))
                });
            } else {
                output::append_monitor_csv(csv_path, step + 1, time, &[("ke", ke)])
                    .with_context(|| format!("Failed to write monitor CSV at step {}", step + 1))?;
            }
        }

        // 固体受力输出（MPI allreduce 在主线程；文件写出可异步）
        if cfg.solid.force_output.enabled
            && (step % cfg.solid.force_output.interval == 0
                || step == cfg.simulation.n_steps - 1)
        {
            let (pi0, pj0, pi1, pj1) = if let Some(p) = partition {
                (p.phys_x0 as i32, p.phys_y0 as i32,
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
                if let Some(ref w) = writer {
                    w.submit(move || {
                        output::append_monitor_csv(
                            &force_csv, step + 1, time,
                            &[("fx", global_fx), ("fy", global_fy)],
                        ).with_context(|| format!("Failed to write solid force CSV at step {}", step + 1))
                    });
                } else {
                    output::append_monitor_csv(
                        &force_csv, step + 1, time,
                        &[("fx", global_fx), ("fy", global_fy)],
                    ).with_context(|| format!("Failed to write solid force CSV at step {}", step + 1))?;
                }
            }
        }

        // 逐体受力输出（BB/IBB 多固体体，各体独立 CSV）
        for entry in solid_entries.iter() {
            if entry.force_cfg.enabled
                && (step % entry.force_cfg.interval == 0
                    || step == cfg.simulation.n_steps - 1)
            {
                let (pi0, pj0, pi1, pj1) = if let Some(p) = partition {
                    (p.phys_x0 as i32, p.phys_y0 as i32,
                     (p.phys_x0 + p.local_nx - 1) as i32,
                     (p.phys_y0 + p.local_ny - 1) as i32)
                } else {
                    (0, 0, grid.nx() as i32 - 1, grid.ny() as i32 - 1)
                };
                let (local_fx, local_fy) = lbm_bindings::compute_solid_force(grid, pi0, pj0, pi1, pj1);
                let global_fx = lbm_bindings::mpi_allreduce_sum_f64(local_fx);
                let global_fy = lbm_bindings::mpi_allreduce_sum_f64(local_fy);
                if rank == 0 {
                    let force_csv = format!("{}/{}.csv", output_dir, entry.force_cfg.filename);
                    let label = entry.label.clone();
                    if let Some(ref w) = writer {
                        w.submit(move || {
                            output::append_monitor_csv(
                                &force_csv, step + 1, time,
                                &[("fx", global_fx), ("fy", global_fy)],
                            ).with_context(|| format!(
                                "Failed to write solid force CSV ({}) at step {}", label, step + 1
                            ))
                        });
                    } else {
                        output::append_monitor_csv(
                            &force_csv, step + 1, time,
                            &[("fx", global_fx), ("fy", global_fy)],
                        ).with_context(|| format!(
                            "Failed to write solid force CSV ({}) at step {}", entry.label, step + 1
                        ))?;
                    }
                }
            }
        }

        // IBM 固体受力输出（Lagrangian 力密度积分，多体逐体写出）
        for entry in ibm_entries.iter() {
            if entry.force_cfg.enabled
                && (step % entry.force_cfg.interval == 0
                    || step == cfg.simulation.n_steps - 1)
            {
                // MPI 模式下各进程仅处理归属自身的标记点子集，需 Allreduce 求全局合力
                let (local_ibm_fx, local_ibm_fy) = entry.ms.compute_body_force();
                let ibm_fx = lbm_bindings::mpi_allreduce_sum_f64(local_ibm_fx);
                let ibm_fy = lbm_bindings::mpi_allreduce_sum_f64(local_ibm_fy);
                // compute_body_force() 返回 Σ(mk.fx·mk.ds)，即 IBM 体力施加到流体上的合力。
                // 由牛顿第三定律，固体（圆柱）所受流体合力 = 负值，即阻力 = -ibm_fx。
                let drag_fx = -ibm_fx;
                let drag_fy = -ibm_fy;
                if rank == 0 {
                    let force_csv = format!("{}/{}.csv", output_dir, entry.force_cfg.filename);
                    let label = entry.label.clone();
                    if let Some(ref w) = writer {
                        w.submit(move || {
                            output::append_monitor_csv(
                                &force_csv, step + 1, time,
                                &[("ibm_fx", drag_fx), ("ibm_fy", drag_fy)],
                            ).with_context(|| format!(
                                "Failed to write IBM force CSV ({}) at step {}", label, step + 1
                            ))
                        });
                    } else {
                        output::append_monitor_csv(
                            &force_csv, step + 1, time,
                            &[("ibm_fx", drag_fx), ("ibm_fy", drag_fy)],
                        ).with_context(|| format!(
                            "Failed to write IBM force CSV ({}) at step {}", entry.label, step + 1
                        ))?;
                    }
                }
            }
        }

        // Python FFI 等值线图（Python GIL 限制，始终在主线程执行）
        #[cfg(feature = "python-ffi")]
        if let Some(pi) = cfg.output.plot_interval {
            if step % pi == 0 || step == cfg.simulation.n_steps - 1 {
                plot_step_ffi(cfg, grid, step + 1, time, partition, combine_blocks, output_dir)?;
            }
        }
    }

    // 异步模式：等待所有待写任务完成并收集错误
    if let Some(w) = writer {
        let errors = w.shutdown();
        if !errors.is_empty() {
            for e in errors.iter().skip(1) {
                eprintln!("[io-thread] {e:#}");
            }
            return Err(errors.into_iter().next().unwrap());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// 时间循环内辅助函数
// ---------------------------------------------------------------------------

/// 对所有 IBM 体执行一步 IBM 力计算。
///
/// 每个体的方法参数优先使用体级覆盖（`[[ibm.bodies]]` 中的 `method`/`n_iter`/
/// `alpha`/`beta` 字段），未设置时继承全局 `[ibm]` 设置。这允许同一仿真中
/// 不同 IBM 体使用不同的力计算方案，以便直接对比各方法的效果。
/// Apply IBM force computation for a **single** IBM body onto `grid`.
///
/// Unlike [`step_ibm`] this does NOT zero the force field first, so the
/// caller is responsible for zeroing each target grid exactly once before
/// calling this for all bodies assigned to that grid.
fn step_ibm_single(cfg: &Config, grid: &mut LbmGrid, entry: &mut fsi::IbmEntry, step: u64) {
    let dx = 1.0_f64;
    let dt = cfg.simulation.dt;
    let t  = step as f64 * dt;

    match entry.motion_type {
        MotionType::RigidFree => {
            if let Some(rb) = &entry.rigid_body {
                let (cx, cy, ux_cm, uy_cm, _theta, omega) = rb.state();
                entry.cx = cx;
                entry.cy = cy;
                entry.ms.set_rigid_body_targets(cx, cy, ux_cm, uy_cm, omega);
            }
        }
        MotionType::Prescribed => {
            if let Some(ref pm) = entry.prescribed {
                let (ux_cm, uy_cm, omega) = pm.eval_rigid(t);
                let cx = entry.cx;
                let cy = entry.cy;
                if omega != 0.0 {
                    entry.theta += omega * dt;
                    let cos_th = entry.theta.cos();
                    let sin_th = entry.theta.sin();
                    let n = entry.init_bx.len();
                    let bx: Vec<f64> = (0..n).map(|k| {
                        let rx = entry.init_bx[k] - entry.cx;
                        let ry = entry.init_by[k] - entry.cy;
                        entry.cx + rx * cos_th - ry * sin_th
                    }).collect();
                    let by: Vec<f64> = (0..n).map(|k| {
                        let rx = entry.init_bx[k] - entry.cx;
                        let ry = entry.init_by[k] - entry.cy;
                        entry.cy + rx * sin_th + ry * cos_th
                    }).collect();
                    entry.ms.update_positions(&bx, &by);
                }
                entry.ms.set_rigid_body_targets(cx, cy, ux_cm, uy_cm, omega);
            }
        }
        MotionType::Flexible => {
            if let Some(ref bs) = entry.beam_solver {
                let (vx, vy) = bs.marker_velocities(&entry.beam_s);
                entry.ms.set_marker_targets(&vx, &vy);
            } else if let Some(ref pm) = entry.prescribed {
                let (vx, vy) = pm.eval_markers(t, &entry.beam_s);
                entry.ms.set_marker_targets(&vx, &vy);
            }
        }
        _ => {}
    }

    match entry.method.to_lowercase().as_str() {
        "penalty"          => entry.ms.step_penalty(grid, dx, dt, entry.alpha, entry.beta),
        "mls"              => entry.ms.step_mls(grid, dx, dt),
        "mls_original"     => entry.ms.step_mls_original(grid, dx, dt),
        "mls_explicit"     => entry.ms.step_mls_explicit(grid, dx, dt),
        "ivc"              => entry.ms.step_ivc(grid, dx, dt),
        "ivc_stationary"   => entry.ms.step_ivc_stationary(grid, dx, dt),
        _                  => entry.ms.step_mdf(grid, dx, dt, entry.n_iter),
    }

    match entry.motion_type {
        MotionType::RigidFree => {
            if let Some(rb) = &mut entry.rigid_body {
                if rb.n_internal() > 0 {
                    let (ix, iy) = rb.internal_positions();
                    let (iux, iuy) = grid.interpolate_at_points(&ix, &iy, dx);
                    rb.set_internal_velocities(&iux, &iuy);
                    rb.compute_internal_momentum();
                }
                let cx = entry.cx;
                let cy = entry.cy;
                let (ftot_x, ftot_y, ttot) = entry.ms.compute_body_force_and_torque(cx, cy);
                rb.advance(-ftot_x, -ftot_y, -ttot, dt);
                let (bx, by) = rb.boundary_positions();
                entry.ms.update_positions(&bx, &by);
            }
        }
        MotionType::Prescribed => {
            if let Some(ref pm) = entry.prescribed {
                let (ux_cm, uy_cm, omega) = pm.eval_rigid(t);
                if omega == 0.0 && (ux_cm != 0.0 || uy_cm != 0.0) {
                    let (cur_bx, cur_by) = entry.ms.get_positions();
                    let new_bx: Vec<f64> = cur_bx.iter().map(|&x| x + ux_cm * dt).collect();
                    let new_by: Vec<f64> = cur_by.iter().map(|&y| y + uy_cm * dt).collect();
                    entry.ms.update_positions(&new_bx, &new_by);
                    entry.cx += ux_cm * dt;
                    entry.cy += uy_cm * dt;
                }
            }
        }
        MotionType::Flexible => {
            if let Some(ref mut bs) = entry.beam_solver {
                let (ibm_fx, ibm_fy) = entry.ms.get_forces();
                let arc_s = entry.beam_s.clone();
                bs.advance(&ibm_fx, &ibm_fy, &arc_s);
                let (bx, by) = bs.marker_positions(&arc_s);
                entry.ms.update_positions(&bx, &by);
            }
        }
        _ => {}
    }
}

fn step_ibm(cfg: &Config, grid: &mut LbmGrid, ibm_entries: &mut [fsi::IbmEntry], step: u64) {
    let dx = 1.0_f64;
    let dt = cfg.simulation.dt;
    let t  = step as f64 * dt;  // 当前物理时刻（格子步）
    // 每个 IBM 时间步开始前清零体力场，防止上一步的力场残留被 pre_force 机制
    // 意外累积到当前步（会导致 MLS/MDF 直接力方法逐步发散）。
    grid.zero_force();
    for entry in ibm_entries.iter_mut() {
        // --- Step A：更新标记点目标速度 ---
        match entry.motion_type {
            MotionType::RigidFree => {
                if let Some(rb) = &entry.rigid_body {
                    let (cx, cy, ux_cm, uy_cm, _theta, omega) = rb.state();
                    entry.cx = cx;
                    entry.cy = cy;
                    entry.ms.set_rigid_body_targets(cx, cy, ux_cm, uy_cm, omega);
                }
            }
            MotionType::Prescribed => {
                // 主动刚体：解析公式给出质心速度，再用刚体运动学分配到各标记点
                if let Some(ref pm) = entry.prescribed {
                    let (ux_cm, uy_cm, omega) = pm.eval_rigid(t);
                    let cx = entry.cx;
                    let cy = entry.cy;
                    if omega != 0.0 {
                        // 旋转模式：更新累计转角，重新计算标记点位置（从初始位置旋转）
                        entry.theta += omega * dt;
                        let cos_th = entry.theta.cos();
                        let sin_th = entry.theta.sin();
                        let n = entry.init_bx.len();
                        let bx: Vec<f64> = (0..n).map(|k| {
                            let rx = entry.init_bx[k] - entry.cx;
                            let ry = entry.init_by[k] - entry.cy;
                            entry.cx + rx * cos_th - ry * sin_th
                        }).collect();
                        let by: Vec<f64> = (0..n).map(|k| {
                            let rx = entry.init_bx[k] - entry.cx;
                            let ry = entry.init_by[k] - entry.cy;
                            entry.cy + rx * sin_th + ry * cos_th
                        }).collect();
                        entry.ms.update_positions(&bx, &by);
                    }
                    entry.ms.set_rigid_body_targets(cx, cy, ux_cm, uy_cm, omega);
                }
            }
            MotionType::Flexible => {
                if let Some(ref bs) = entry.beam_solver {
                    // 被动柔性体：将当前梁速度写入标记点目标速度
                    let (vx, vy) = bs.marker_velocities(&entry.beam_s);
                    entry.ms.set_marker_targets(&vx, &vy);
                } else if let Some(ref pm) = entry.prescribed {
                    // 主动柔性体（行波/振荡）：解析速度写入各标记点
                    let (vx, vy) = pm.eval_markers(t, &entry.beam_s);
                    entry.ms.set_marker_targets(&vx, &vy);
                }
            }
            _ => {}  // Fixed：不更新目标速度（保持零）
        }

        // --- Step B：IBM 力计算（所有方法均从 mk.ux_target/uy_target 读取目标速度）---
        match entry.method.to_lowercase().as_str() {
            "penalty"          => entry.ms.step_penalty(grid, dx, dt, entry.alpha, entry.beta),
            "mls"              => entry.ms.step_mls(grid, dx, dt),
            "mls_original"     => entry.ms.step_mls_original(grid, dx, dt),
            "mls_explicit"     => entry.ms.step_mls_explicit(grid, dx, dt),
            "ivc"              => entry.ms.step_ivc(grid, dx, dt),
            "ivc_stationary"   => entry.ms.step_ivc_stationary(grid, dx, dt),
            _                  => entry.ms.step_mdf(grid, dx, dt, entry.n_iter),
        }

        // --- Step C：根据运动类型推进结构状态 ---
        match entry.motion_type {
            MotionType::RigidFree => {
                if let Some(rb) = &mut entry.rigid_body {
                    // 方案 C（Lagrangian 内部点）：在 advance() 前插值内部点速度
                    if rb.n_internal() > 0 {
                        let (ix, iy) = rb.internal_positions();
                        let (iux, iuy) = grid.interpolate_at_points(&ix, &iy, dx);
                        rb.set_internal_velocities(&iux, &iuy);
                        rb.compute_internal_momentum();
                    }
                    let cx = entry.cx;
                    let cy = entry.cy;
                    let (ftot_x, ftot_y, ttot) = entry.ms.compute_body_force_and_torque(cx, cy);
                    rb.advance(-ftot_x, -ftot_y, -ttot, dt);
                    let (bx, by) = rb.boundary_positions();
                    entry.ms.update_positions(&bx, &by);
                }
            }
            MotionType::Prescribed => {
                // translate/oscillate 模式：按质心速度平移所有标记点
                if let Some(ref pm) = entry.prescribed {
                    let (ux_cm, uy_cm, omega) = pm.eval_rigid(t);
                    // 仅平移模式（无旋转）才平移标记点；旋转模式已在 Step A 处理
                    if omega == 0.0 && (ux_cm != 0.0 || uy_cm != 0.0) {
                        let n = entry.ms.len();
                        let (cur_bx, cur_by) = entry.ms.get_positions();
                        let new_bx: Vec<f64> = cur_bx.iter().map(|&x| x + ux_cm * dt).collect();
                        let new_by: Vec<f64> = cur_by.iter().map(|&y| y + uy_cm * dt).collect();
                        entry.ms.update_positions(&new_bx, &new_by);
                        // 同步质心位置
                        entry.cx += ux_cm * dt;
                        entry.cy += uy_cm * dt;
                        let _ = n; // used via new_bx/by
                    }
                }
            }
            MotionType::Flexible => {
                if let Some(ref mut bs) = entry.beam_solver {
                    // 被动柔性体：推进梁方程，然后更新标记点位置和速度
                    let (ibm_fx, ibm_fy) = entry.ms.get_forces();
                    let arc_s = entry.beam_s.clone();
                    bs.advance(&ibm_fx, &ibm_fy, &arc_s);
                    let (bx, by) = bs.marker_positions(&arc_s);
                    entry.ms.update_positions(&bx, &by);
                }
                // 主动柔性体：位置由外部解析公式给定，无需更新（标记点目标速度已在 Step A 设置）
            }
            _ => {}
        }
    }
}

/// 运动刚体 BB/IBB 每步更新：施加 Ladd 移动壁面修正，推进 Newton-Euler 刚体积分，
/// 并在步末清除旧标记、重新标记圆柱于新位置。
///
/// ## 时序（步 N 内）
/// 1. 按当前位置/速度对每个体调用 `apply_solid_bb_moving_rigid` 或
///    `apply_solid_ibb_moving_rigid`（Ladd 1994 移动壁面修正）。
/// 2. 通过动量交换法（MEA）计算各刚体受力。
/// 3. 用 Newton-Euler 方程推进刚体状态到步 N+1。
/// 4. `clear_solid()` + `mark_solid_cylinder()` → 重新标记步 N+1 位置。
///
/// 对静止体（`motion_type != RigidFree`）调用移动版本但传零速度，
/// 结果等价于标准半步长反弹（Ladd correction 为 0）。
fn step_solid_moving(
    cfg: &Config,
    grid: &mut LbmGrid,
    solid_entries: &mut [fsi::SolidEntry],
    step: u64,
    phys_i0: i32, phys_j0: i32,
    phys_i1: i32, phys_j1: i32,
) {
    let dt = cfg.simulation.dt;
    // 当前格子时刻（step 已完成流体推进；t = (step+1)*dt 为施加 Ladd 修正的时刻）
    let t = (step + 1) as f64 * dt;

    // --- Step A：对每个固体施加移动壁面 BB/IBB ---
    for entry in solid_entries.iter() {
        let (ux_cm, uy_cm, omega) = entry.wall_velocity_at(t);
        let cx = entry.cx;
        let cy = entry.cy;
        match entry.bc_mode {
            2 => lbm_bindings::apply_solid_ibb_moving_rigid(
                grid, cx, cy, ux_cm, uy_cm, omega,
                phys_i0, phys_j0, phys_i1, phys_j1),
            _ => lbm_bindings::apply_solid_bb_moving_rigid(
                grid, cx, cy, ux_cm, uy_cm, omega,
                phys_i0, phys_j0, phys_i1, phys_j1),
        }
    }

    // --- Step B + C：对运动刚体求力并推进状态 ---
    let any_rigid_free = solid_entries.iter().any(|e| e.motion_type == MotionType::RigidFree);

    if any_rigid_free {
        // 计算全局 MEA 合力（所有固体的总受力，通过 MPI_Allreduce 求和）
        let (local_fx, local_fy) = lbm_bindings::compute_solid_force(
            grid, phys_i0, phys_j0, phys_i1, phys_j1);
        // Note: 若多个刚体共存，目前简化为所有体共用总力；单体情形正确。
        // 多体场景的精确 MEA 需要逐体 mask，此处不作区分。
        let global_fx = lbm_bindings::mpi_allreduce_sum_f64(local_fx);
        let global_fy = lbm_bindings::mpi_allreduce_sum_f64(local_fy);

        for entry in solid_entries.iter_mut() {
            if entry.motion_type != MotionType::RigidFree { continue; }
            if let Some(rb) = &mut entry.rigid_body {
                // 固体所受流体合力 = MEA 合力（MEA 返回的是固体给流体的力，取负）
                rb.advance(-global_fx, -global_fy, 0.0, dt);
                let (cx, cy, _, _, _, _) = rb.state();
                entry.cx = cx;
                entry.cy = cy;
            }
        }
    }

    // --- Step B'：对主动 Prescribed 圆柱积分位置 ---
    for entry in solid_entries.iter_mut() {
        if entry.motion_type != MotionType::Prescribed { continue; }
        if entry.shape == "cylinder" {
            let (ux, uy, _omega) = entry.wall_velocity_at(t);
            entry.cx += ux * dt;
            entry.cy += uy * dt;
        }
    }

    // --- Step D：清除标记并在新位置重新标记 ---
    let any_moving = solid_entries.iter().any(|e| e.is_moving());
    if any_moving {
        lbm_bindings::clear_solid(grid);
        for entry in solid_entries.iter() {
            if entry.shape == "cylinder" {
                lbm_bindings::mark_solid_cylinder(grid, entry.cx, entry.cy, entry.radius);
                lbm_bindings::assign_solid_bc_unmarked(grid, entry.bc_mode);
            }
            // 矩形/mesh 不支持运动，但如果有标记也重新写入（静止位置）
            if entry.shape == "rectangle" {
                lbm_bindings::mark_solid_rectangle(grid, entry.i0, entry.j0, entry.i1, entry.j1);
                lbm_bindings::assign_solid_bc_unmarked(grid, entry.bc_mode);
            }
        }
    }
}

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

// ---------------------------------------------------------------------------
// 多重网格时间步主循环
// ---------------------------------------------------------------------------

/// 多重网格时间步主循环。
///
/// - `root_grid` / `root_solver`：最粗层（L0）网格与求解器，已完成 MPI 附加和 BC 注册。
/// - `ibm_entries`：IBM 体集合，力展布在每步 mg_step_recursive **之前**施加于根网格。
/// - 细化层的 LatticeGrid + Solver 在本函数内按 `mg_cfg.levels` 自动创建并绑定到树节点。
///
/// ## Omega 递推公式（Yu 2002 Eq.4）
///
/// `ω_f = 2·ω_c / (4 − ω_c)`（线性稳定性约束，每细化一层递推一次）。
///
/// ## IBM 策略
///
/// IBM 力当前施加于根（粗）网格；mg_step_recursive 内部通过 fringe 耦合将力效应
/// 传播到各细化层。如需在最细层施加 IBM，可扩展为选取覆盖圆柱的最细节点。
fn run_multigrid_loop(
    cfg: &Config,
    mg_cfg: &crate::config::MultigridConfig,
    model: LatticeModel,
    cm: CollisionModel,
    root_grid: &mut LbmGrid,
    root_solver: &mut LbmSolver,
    ibm_entries: &mut Vec<fsi::IbmEntry>,
    mpi: &mut sim::MpiDecomp,
    partition: Option<PartitionInfo>,
    combine_blocks: bool,
    output_dir: &str,
    rank: i32,
    nprocs: i32,
) -> Result<()> {
    use anyhow::anyhow;

    // -----------------------------------------------------------------------
    // Step 0: Determine root extent and coordinate conversion for MPI mode.
    //
    // In MPI block mode each rank holds a local partition of the global grid.
    // The root MgTree extent uses LOCAL grid coordinates (including ghost rows)
    // so that fringe coupling can safely read halo-exchanged neighbour data.
    // Fine-level extents are clipped to the local physical domain and then
    // converted from global → local grid coordinates.
    //
    // Coordinate convention (1D-Y block, n_ghost=1):
    //   physical row j_glob ∈ [y_start_glob, y_start_glob + local_ny - 1]
    //   local grid j = (j_glob - y_start_glob) + phys_y0  (1-based when phys_y0=1)
    //   root extent:  y0 = 0 (south ghost),  y1 = grid_ny-1 (north ghost)
    // -----------------------------------------------------------------------
    let is_block_mpi = mpi.effective_mode == "block" && nprocs > 1;

    // Collect local partition geometry (all values in local grid coords).
    let (root_x0, root_x1, root_y0, root_y1,
         x_start_glob, y_start_glob,
         phys_x0, phys_y0,
         local_phys_nx, local_phys_ny) =
        if is_block_mpi {
            if let Some(ref d2) = mpi.decomp2d {
                (0_i32, d2.grid_nx() - 1,
                 0_i32, d2.grid_ny() - 1,
                 d2.x_start(), d2.y_start(),
                 d2.phys_x0(), d2.phys_y0(),
                 d2.local_nx(), d2.local_ny())
            } else {
                (0, cfg.fluid.nx as i32 - 1, 0, cfg.fluid.ny as i32 - 1,
                 0, 0, 0, 0,
                 cfg.fluid.nx as i32, cfg.fluid.ny as i32)
            }
        } else {
            (0, cfg.fluid.nx as i32 - 1, 0, cfg.fluid.ny as i32 - 1,
             0, 0, 0, 0,
             cfg.fluid.nx as i32, cfg.fluid.ny as i32)
        };

    // -----------------------------------------------------------------------
    // Step 1: Create MgTree with the correct root extent.
    // -----------------------------------------------------------------------
    let mut tree = LbmMgTree::new(root_x0, root_x1, root_y0, root_y1, 0, 0, false)
        .ok_or_else(|| anyhow!(
            "Failed to create MgTree for root extent [{},{}]×[{},{}]",
            root_x0, root_x1, root_y0, root_y1))?;

    // -----------------------------------------------------------------------
    // Step 2: Compute per-level local extents and create fine grids/solvers.
    //
    // `level_extents[i]` is `Some(x_lo, x_hi, y_lo, y_hi)` in local grid
    // coords, or `None` when the level has no overlap with this rank's
    // physical domain.  A placeholder 1×1 grid is allocated for skipped
    // levels so that all three Vecs stay index-aligned.
    // -----------------------------------------------------------------------
    let mut level_extents: Vec<Option<(i32, i32, i32, i32)>> =
        Vec::with_capacity(mg_cfg.levels.len());
    let mut fine_grids:   Vec<LbmGrid>   = Vec::with_capacity(mg_cfg.levels.len());
    let mut fine_solvers: Vec<LbmSolver> = Vec::with_capacity(mg_cfg.levels.len());

    // For non-MPI: track root-coord extents and cumulative spatial scale per level,
    // needed to (a) convert parent-local config coords to root coords for add_child_level,
    // and (b) mark solid bodies / register BCs on fine grids at correct positions.
    // Each entry: (root_xs, root_xe, root_ys, root_ye, cumulative_scale).
    let mut level_root_info: Vec<(i32, i32, i32, i32, i32)> = Vec::with_capacity(mg_cfg.levels.len());

    let mut omega_parent = cfg.omega();
    for (i, level) in mg_cfg.levels.iter().enumerate() {
        let r = level.refine_ratio;
        if r < 1 {
            return Err(anyhow!("level[{}]: refine_ratio must be >= 1 (got {})", i, r));
        }
        // Advance omega chain regardless of whether this level is active on
        // this rank, so that subsequent levels get consistent relaxation rates.
        let omega_fine = 2.0 * omega_parent / (4.0 - omega_parent);
        omega_parent = omega_fine;

        // Compute extent in local grid coords, clipping to the local domain.
        let (extent_local, fine_nx, fine_ny) = if is_block_mpi {
            // MPI block mode: config coords are already in global/root coords.
            // Clip global level extent to local physical domain.
            let g_x_lo = level.x_start.max(x_start_glob);
            let g_x_hi = level.x_end  .min(x_start_glob + local_phys_nx - 1);
            let g_y_lo = level.y_start.max(y_start_glob);
            let g_y_hi = level.y_end  .min(y_start_glob + local_phys_ny - 1);

            if g_x_lo > g_x_hi || g_y_lo > g_y_hi {
                // No intersection: push a 1×1 placeholder and skip.
                level_extents.push(None);
                level_root_info.push((level.x_start, level.x_end, level.y_start, level.y_end, r));
                fine_grids.push(LbmGrid::new(1, 1, 1, model));
                let fs = LbmSolver::new(fine_grids.last_mut().unwrap(), omega_fine, cm);
                fine_solvers.push(fs);
                continue;
            }

            // Convert global intersection → local grid coords.
            let x_lo_l = (g_x_lo - x_start_glob) + phys_x0;
            let x_hi_l = (g_x_hi - x_start_glob) + phys_x0;
            let y_lo_l = (g_y_lo - y_start_glob) + phys_y0;
            let y_hi_l = (g_y_hi - y_start_glob) + phys_y0;

            let fnx = (g_x_hi - g_x_lo) * r + 1;
            let fny = (g_y_hi - g_y_lo) * r + 1;
            level_root_info.push((level.x_start, level.x_end, level.y_start, level.y_end, r));
            (Some((x_lo_l, x_hi_l, y_lo_l, y_hi_l)), fnx, fny)
        } else {
            // Serial / independent: config coords are in PARENT'S fine-grid local
            // coordinate system.  Convert to root coords using accumulated info.
            //
            // parent_level semantics (tree node index):
            //   0         → root grid
            //   k (k ≥ 1) → levels[k-1]  (tree node k was added as levels[k-1])
            //   -1        → auto linear chain: most recently added level (or root)
            let (p_xs, p_ys, p_scale) = if level.parent_level <= 0 {
                // Parent is root: root coords start at (0,0), scale = 1.
                (0_i32, 0_i32, 1_i32)
            } else {
                let pj = level.parent_level as usize - 1; // levels[] index of parent
                if pj < level_root_info.len() {
                    let pi = &level_root_info[pj];
                    (pi.0, pi.2, pi.4)  // (root_xs, root_ys, cumulative_scale)
                } else {
                    (0_i32, 0_i32, 1_i32)
                }
            };
            // For parent_level == -1 (auto), use the most recently added entry.
            let (p_xs, p_ys, p_scale) = if level.parent_level < 0 {
                if level_root_info.is_empty() {
                    (0_i32, 0_i32, 1_i32)
                } else {
                    let pi = level_root_info.last().unwrap();
                    (pi.0, pi.2, pi.4)
                }
            } else {
                (p_xs, p_ys, p_scale)
            };

            // Convert parent-local fine-grid coords to root coords.
            // Parent's fine-grid local ix → root coord: p_xs + ix / p_scale
            let root_xs = p_xs + level.x_start / p_scale;
            let root_xe = p_xs + level.x_end   / p_scale;
            let root_ys = p_ys + level.y_start / p_scale;
            let root_ye = p_ys + level.y_end   / p_scale;
            let cumulative_scale = p_scale * r;

            let fnx = (level.x_end - level.x_start) * r + 1;
            let fny = (level.y_end - level.y_start) * r + 1;
            level_root_info.push((root_xs, root_xe, root_ys, root_ye, cumulative_scale));
            (Some((root_xs, root_xe, root_ys, root_ye)), fnx, fny)
        };

        if fine_nx <= 0 || fine_ny <= 0 {
            return Err(anyhow!(
                "level[{}]: computed fine grid size {}×{} is invalid \
                 (x=[{},{}], y=[{},{}], r={})",
                i, fine_nx, fine_ny,
                level.x_start, level.x_end, level.y_start, level.y_end, r
            ));
        }

        level_extents.push(extent_local);
        fine_grids.push(LbmGrid::new(fine_nx, fine_ny, 1, model));
        let fs = LbmSolver::new(fine_grids.last_mut().unwrap(), omega_fine, cm);
        fine_solvers.push(fs);

        if rank == 0 {
            let (xl, xh, yl, yh) = extent_local.unwrap();
            if is_block_mpi {
                println!(
                    "  [MG] level {} : {}×{} local \
                     (global [{},{}]×[{},{}] → local [{},{}]×[{},{}], r={}, ω={:.4})",
                    i + 1, fine_nx, fine_ny,
                    level.x_start, level.x_end, level.y_start, level.y_end,
                    xl, xh, yl, yh, r, omega_fine
                );
            } else {
                println!(
                    "  [MG] level {} : {}×{} (root coords [{},{}]×[{},{}], r={}, ω={:.4})",
                    i + 1, fine_nx, fine_ny,
                    xl, xh, yl, yh, r, omega_fine
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Step 3: Bind root node's grid + solver.
    // -----------------------------------------------------------------------
    tree.set_grid_by_idx(0, root_grid);
    tree.set_solver_by_idx(0, root_solver);

    // -----------------------------------------------------------------------
    // Step 4: Add active fine levels to the tree.
    //
    // `node_map[i]` = actual tree node index for levels[i], or None if this
    // rank has no overlap with that level.  When a parent level was skipped,
    // the most recently active ancestor (or root) is used as parent.
    // -----------------------------------------------------------------------
    let mut node_map: Vec<Option<usize>> = vec![None; mg_cfg.levels.len()];
    for (i, level) in mg_cfg.levels.iter().enumerate() {
        let Some((x_lo, x_hi, y_lo, y_hi)) = level_extents[i] else {
            continue;
        };

        // Resolve parent tree node index.
        //
        // parent_level is a TREE NODE index (0 = root):
        //   0         → root
        //   k (k ≥ 1) → tree node k  (= levels[k-1], added before this level)
        //   -1        → auto linear chain (most recent active ancestor or root)
        let parent_idx = if level.parent_level < 0 {
            if i == 0 {
                0 // root
            } else {
                (0..i).rev().find_map(|k| node_map[k]).unwrap_or(0)
            }
        } else {
            // Direct tree-node-index lookup: parent_level IS the tree node index.
            level.parent_level as usize
        };

        let node_idx = tree.add_child_level(
            parent_idx, x_lo, x_hi, y_lo, y_hi, 0, 0, level.refine_ratio,
        ).ok_or_else(|| anyhow!(
            "Failed to add MgTree level {} (parent_idx={}, root_extent=[{},{}]×[{},{}]). \
             Ensure fine level extents are within parent extent (all in root coords).",
            i + 1, parent_idx, x_lo, x_hi, y_lo, y_hi
        ))?;
        tree.set_grid_by_idx(node_idx, &mut fine_grids[i]);
        tree.set_solver_by_idx(node_idx, &mut fine_solvers[i]);
        node_map[i] = Some(node_idx);

        // ----------------------------------------------------------------
        // Register fluid boundary conditions on fine grids that touch
        // global domain boundaries.  Only applies to non-MPI serial mode;
        // in MPI block mode BCs are handled by the root/partition grid.
        // ----------------------------------------------------------------
        if !is_block_mpi {
            let (root_xs, root_xe, root_ys, root_ye, _) = level_root_info[i];
            for bc_cfg in &cfg.fluid.boundary_conditions {
                let face = sim::parse_face(&bc_cfg.face);
                let touches = match face {
                    Face::West  => root_xs == 0,
                    Face::East  => root_xe >= cfg.fluid.nx as i32 - 1,
                    Face::South => root_ys == 0,
                    Face::North => root_ye >= cfg.fluid.ny as i32 - 1,
                    _           => false,
                };
                if touches {
                    let bc_type = sim::parse_bc_type(&bc_cfg.bc_type);
                    fine_solvers[i].add_boundary_condition(
                        bc_type, face, bc_cfg.ux, bc_cfg.uy, 0.0, bc_cfg.rho);
                }
            }

            // ----------------------------------------------------------------
            // Mark solid bounce-back / IBB bodies on fine grids.
            //
            // The solid body coordinates in the root grid are mapped to
            // fine-grid local positions using the cumulative scale and origin:
            //   fine_local = (root_coord - root_xs) * cumulative_scale
            // ----------------------------------------------------------------
            if !cfg.solid.bodies.is_empty() {
                let (root_xs, root_xe, root_ys, root_ye, cum_scale) = level_root_info[i];
                let fine_nx = fine_grids[i].nx();
                let fine_ny = fine_grids[i].ny();
                let sf = cum_scale as f64;

                for body in &cfg.solid.bodies {
                    // Per-body bc_type overrides global solid.bc_type.
                    let bc_str = body.bc_type.as_deref()
                        .unwrap_or(&cfg.solid.bc_type);
                    let body_bc_mode: i32 = match bc_str.to_lowercase().as_str() {
                        "bounce_back" | "bb"  => 1,
                        "interpolated_bounce_back" | "ibb" | "bouzidi" => 2,
                        _ => 0,
                    };
                    if body_bc_mode == 0 { continue; }

                    // ----------------------------------------------------------
                    // Cross-level detection: error if the solid body's bounding
                    // box partially overlaps this fine level without being fully
                    // contained inside it.  Coordinates are in root-grid units.
                    // ----------------------------------------------------------
                    let (bb_x0, bb_x1, bb_y0, bb_y1): (f64, f64, f64, f64) =
                        match body.shape.to_lowercase().as_str() {
                            "cylinder" | "circle" => (
                                body.cx - body.radius, body.cx + body.radius,
                                body.cy - body.radius, body.cy + body.radius,
                            ),
                            "rectangle" => (
                                body.i0 as f64, body.i1 as f64,
                                body.j0 as f64, body.j1 as f64,
                            ),
                            _ => continue,
                        };
                    let lx0 = root_xs as f64;
                    let lx1 = root_xe as f64;
                    let ly0 = root_ys as f64;
                    let ly1 = root_ye as f64;
                    let overlaps_x = bb_x0 < lx1 && bb_x1 > lx0;
                    let overlaps_y = bb_y0 < ly1 && bb_y1 > ly0;
                    let contained  = bb_x0 >= lx0 && bb_x1 <= lx1
                                  && bb_y0 >= ly0 && bb_y1 <= ly1;
                    if overlaps_x && overlaps_y && !contained {
                        return Err(anyhow!(
                            "[solid BB] body shape='{}' spans the boundary of MG level {} \
                             (level root extent x=[{},{}] y=[{},{}], body bbox x=[{:.2},{:.2}] \
                             y=[{:.2},{:.2}]). Move the body fully inside or outside the \
                             refinement region.",
                            body.shape, i + 1,
                            root_xs, root_xe, root_ys, root_ye,
                            bb_x0, bb_x1, bb_y0, bb_y1
                        ));
                    }

                    match body.shape.to_lowercase().as_str() {
                        "cylinder" | "circle" => {
                            let fine_cx = (body.cx - root_xs as f64) * sf;
                            let fine_cy = (body.cy - root_ys as f64) * sf;
                            let fine_r  = body.radius * sf;
                            // Only mark if the cylinder might overlap the fine grid.
                            if fine_cx + fine_r >= 0.0 && fine_cx - fine_r < fine_nx as f64
                                && fine_cy + fine_r >= 0.0 && fine_cy - fine_r < fine_ny as f64
                            {
                                lbm_bindings::mark_solid_cylinder(
                                    &mut fine_grids[i], fine_cx, fine_cy, fine_r);
                                lbm_bindings::assign_solid_bc_unmarked(
                                    &mut fine_grids[i], body_bc_mode);
                            }
                        }
                        "rectangle" => {
                            let fi0 = ((body.i0 - root_xs) * cum_scale).max(0);
                            let fj0 = ((body.j0 - root_ys) * cum_scale).max(0);
                            let fi1 = ((body.i1 - root_xs) * cum_scale).min(fine_nx - 1);
                            let fj1 = ((body.j1 - root_ys) * cum_scale).min(fine_ny - 1);
                            if fi0 <= fi1 && fj0 <= fj1 {
                                lbm_bindings::mark_solid_rectangle(
                                    &mut fine_grids[i], fi0, fj0, fi1, fj1);
                                lbm_bindings::assign_solid_bc_unmarked(
                                    &mut fine_grids[i], body_bc_mode);
                            }
                        }
                        _ => {}
                    }
                }

                // Activate solid BC on this fine solver.
                // Determine the global (non-moving) BC mode for this solver.
                let has_moving = cfg.solid.bodies.iter().any(|b| {
                    matches!(b.motion_type, MotionType::RigidFree | MotionType::Prescribed)
                });
                let global_bc_mode: i32 = if has_moving {
                    0
                } else {
                    match cfg.solid.bc_type.to_lowercase().as_str() {
                        "bounce_back" | "bb" => 1,
                        "interpolated_bounce_back" | "ibb" | "bouzidi" => 2,
                        _ => 0,
                    }
                };
                if global_bc_mode != 0 {
                    lbm_bindings::mark_solid_bc(&mut fine_solvers[i], global_bc_mode);
                }
            }
        }
    }

    if rank == 0 {
        println!(
            "  [MG] tree ready: depth={}, nodes={}, fringe_width={}",
            tree.max_level(), tree.node_count(), mg_cfg.fringe_width
        );
    }

    // -----------------------------------------------------------------------
    // Step 4.5: IBM level assignment (serial / independent MG mode only).
    //
    // For each IBM body:
    //   1. Compute bounding box of Lagrangian markers in ROOT grid coords.
    //   2. Find the finest MG level whose root-coord extent fully contains it.
    //   3. Error if the body straddles a level boundary (partial overlap).
    //   4. Scale marker positions (and centre coords) from root to fine-grid
    //      local coords so that step_ibm uses the correct lattice spacing.
    //
    // `ibm_level_assign[k]` = index into fine_grids/fine_solvers for body k,
    // or -1 meaning "apply on root grid".
    //
    // Note: moving IBM bodies (RigidFree / Prescribed) are kept on the root
    // grid; only stationary bodies are migrated to fine grids.
    // -----------------------------------------------------------------------
    let ibm_level_assign: Vec<i32> = if !is_block_mpi && !ibm_entries.is_empty() {
        let mut assign: Vec<i32> = vec![-1i32; ibm_entries.len()];
        for (body_idx, entry) in ibm_entries.iter_mut().enumerate() {
            // Moving bodies stay on root grid for now (scaling velocity/force
            // across grid levels for dynamic bodies is non-trivial).
            let is_moving = !matches!(entry.motion_type, MotionType::Fixed);
            if is_moving { continue; }

            let (bx, by) = entry.ms.get_positions();
            if bx.is_empty() { continue; }

            let root_bx_lo = bx.iter().cloned().fold(f64::INFINITY,     f64::min);
            let root_bx_hi = bx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let root_by_lo = by.iter().cloned().fold(f64::INFINITY,     f64::min);
            let root_by_hi = by.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Iterate from finest level (last) to coarsest (first).
            let mut finest: i32 = -1;
            let mut cross_level_err: Option<String> = None;

            for li in (0..mg_cfg.levels.len()).rev() {
                if level_extents[li].is_none() { continue; }
                let (lx0, lx1, ly0, ly1, _) = level_root_info[li];
                let (lx0, lx1, ly0, ly1) =
                    (lx0 as f64, lx1 as f64, ly0 as f64, ly1 as f64);

                let overlaps_x = root_bx_lo < lx1 && root_bx_hi > lx0;
                let overlaps_y = root_by_lo < ly1 && root_by_hi > ly0;
                let contained  = root_bx_lo >= lx0 && root_bx_hi <= lx1
                               && root_by_lo >= ly0 && root_by_hi <= ly1;

                if overlaps_x && overlaps_y && !contained {
                    cross_level_err = Some(format!(
                        "[IBM] body[{}] '{}' crosses the boundary of MG level {} \
                         (level root extent x=[{},{}] y=[{},{}], marker bbox \
                         x=[{:.2},{:.2}] y=[{:.2},{:.2}]). \
                         Move the body fully inside or outside the refinement region.",
                        body_idx, entry.label, li + 1,
                        lx0 as i32, lx1 as i32, ly0 as i32, ly1 as i32,
                        root_bx_lo, root_bx_hi, root_by_lo, root_by_hi
                    ));
                    break;
                }

                if contained && finest < 0 {
                    finest = li as i32;
                    // Don't break — check coarser levels for cross-level spans too.
                }
            }

            if let Some(msg) = cross_level_err {
                return Err(anyhow!("{}", msg));
            }

            if finest >= 0 {
                let li = finest as usize;
                let (root_xs, _, root_ys, _, cum_scale) = level_root_info[li];
                let sf = cum_scale as f64;

                // Scale all marker positions from root coords to fine-grid local coords.
                let fine_bx: Vec<f64> = bx.iter().map(|&x| (x - root_xs as f64) * sf).collect();
                let fine_by: Vec<f64> = by.iter().map(|&y| (y - root_ys as f64) * sf).collect();
                entry.ms.update_positions(&fine_bx, &fine_by);
                entry.cx = (entry.cx - root_xs as f64) * sf;
                entry.cy = (entry.cy - root_ys as f64) * sf;
                for x in entry.init_bx.iter_mut() { *x = (*x - root_xs as f64) * sf; }
                for y in entry.init_by.iter_mut() { *y = (*y - root_ys as f64) * sf; }

                assign[body_idx] = finest;

                if rank == 0 {
                    println!(
                        "  [IBM-MG] body[{}] '{}' → fine level {} \
                         (cum_scale={}, root_origin=[{},{}])",
                        body_idx, entry.label, li + 1, cum_scale, root_xs, root_ys
                    );
                }
            } else if rank == 0 {
                println!("  [IBM-MG] body[{}] '{}' → root grid", body_idx, entry.label);
            }
        }
        assign
    } else {
        // MPI block mode or no IBM: all bodies on root grid.
        vec![-1i32; ibm_entries.len()]
    };

    // -----------------------------------------------------------------------
    // Step 5: Time loop.
    // -----------------------------------------------------------------------
    for step in 0..cfg.simulation.n_steps {
        // IBM: spread body forces onto the appropriate grid level.
        if !ibm_entries.is_empty() {
            // In non-MPI MG mode we may need to route individual IBM bodies to
            // their assigned fine-grid levels.  In all other modes every body
            // goes to root_grid (ibm_level_assign is all -1).
            let any_on_fine = ibm_level_assign.iter().any(|&l| l >= 0);

            if !is_block_mpi && any_on_fine {
                // -- Zero force on every grid that will receive IBM forces. --
                root_grid.zero_force();
                for &lvl in ibm_level_assign.iter().filter(|&&l| l >= 0) {
                    fine_grids[lvl as usize].zero_force();
                }

                // -- Apply each IBM body to its assigned grid. --
                // Process root-grid bodies first (no borrow conflict).
                let dt  = cfg.simulation.dt;
                let _dx = 1.0_f64;
                for (body_idx, entry) in ibm_entries.iter_mut().enumerate() {
                    let lvl = ibm_level_assign[body_idx];
                    if lvl < 0 {
                        step_ibm_single(cfg, root_grid, entry, step);
                    }
                }
                // Then process fine-grid bodies (one level at a time to avoid
                // double-mutable-borrow on fine_grids).
                let n_levels = mg_cfg.levels.len();
                for li in 0..n_levels {
                    let any = ibm_level_assign.iter().any(|&l| l == li as i32);
                    if !any { continue; }
                    for (body_idx, entry) in ibm_entries.iter_mut().enumerate() {
                        if ibm_level_assign[body_idx] == li as i32 {
                            step_ibm_single(cfg, &mut fine_grids[li], entry, step);
                        }
                    }
                }
                let _ = dt;
            } else {
                step_ibm(cfg, root_grid, ibm_entries, step);

                // MPI block mode: ghost-row force contributions must be reduced
                // back to physical rows so each rank has a complete force field.
                if is_block_mpi {
                    if let Some(ref mut d2) = mpi.decomp2d {
                        lbm_bindings::ibm_halo_reduce_force_2d(root_grid, d2);
                    }
                }
            }
        }

        // Multigrid recursive step (collide-stream + C↔F coupling at every level).
        let rc = tree.mg_step_recursive(mg_cfg.fringe_width);
        if rc != 0 && rank == 0 {
            eprintln!("[MG] mg_step_recursive failed (rc={}) at step {}", rc, step);
        }

        let time = (step + 1) as f64 * cfg.simulation.dt;
        let is_snap = step % cfg.output.write_interval == 0
            || step + 1 == cfg.simulation.n_steps;

        if is_snap {
            if is_block_mpi {
                // Each MPI rank writes its own physical partition.
                write_step_snapshot(cfg, root_grid, step + 1, time, output_dir, partition)?;

                // Optional: gather all partitions to rank-0 for a combined file.
                if combine_blocks {
                    if let Some(p) = partition {
                        let (l_rho, l_ux, l_uy, _, _) =
                            output::extract_physical_fields(root_grid, Some(p));
                        let g_rho_r = output::gather_field_to_root(&l_rho, &p, 0);
                        let g_ux_r  = output::gather_field_to_root(&l_ux,  &p, 0);
                        let g_uy_r  = output::gather_field_to_root(&l_uy,  &p, 0);
                        if let (Some((g_rho, gnx, gny)),
                                Some((g_ux, _, _)),
                                Some((g_uy, _, _))) = (g_rho_r, g_ux_r, g_uy_r)
                        {
                            output::write_global_snapshot_raw(
                                &cfg.output.format, &g_rho, &g_ux, &g_uy, gnx, gny,
                                step + 1, time, &cfg.output.directory,
                            ).with_context(|| format!(
                                "failed to write combined MG snapshot (step {})", step + 1))?;
                        }
                    }
                }
            } else if rank == 0 {
                write_step_snapshot(cfg, root_grid, step + 1, time, output_dir, None)?;
            }
        }

        if rank == 0 && (step == 0 || (step + 1) % cfg.output.write_interval == 0
                         || step + 1 == cfg.simulation.n_steps) {
            println!("  step {:>6} / {}  t = {:.3}", step + 1, cfg.simulation.n_steps, time);
        }
    }

    Ok(())
}

/// combine_blocks 模式：将各进程物理场 gather 到 rank-0 并写出全局快照。
///
/// 非 root 进程已在 [`output::gather_field_to_root`] 内参与 `MPI_Gatherv`，
/// 此处直接返回 `Ok(())`。
#[allow(dead_code)]
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

    output::write_global_snapshot_raw(
        &cfg.output.format, &g_rho, &g_ux, &g_uy, gnx, gny, step, time, &cfg.output.directory,
    ).with_context(|| format!("failed to write combined snapshot (step {})", step))
}

/// 计算物理区域平均动能（纯内存操作，不涉及文件 I/O）。
fn compute_monitor_ke(grid: &LbmGrid, partition: Option<PartitionInfo>) -> f64 {
    let (px0, py0, pnx, pny, gnx) = if let Some(p) = partition {
        (p.phys_x0, p.phys_y0, p.local_nx, p.local_ny, grid.nx() as usize)
    } else {
        (0, 0, grid.nx() as usize, grid.ny() as usize, grid.nx() as usize)
    };
    let n_phys = pnx * pny;
    let ke_sum: f64 = (0..pny)
        .flat_map(|j| (0..pnx).map(move |i| (j, i)))
        .map(|(j, i)| {
            let idx = ((py0 + j) * gnx + px0 + i) as i32;
            let u = grid.ux(idx);
            let v = grid.uy(idx);
            u * u + v * v
        })
        .sum::<f64>();
    ke_sum / (n_phys as f64) * 0.5
}

/// 统计固体所受合力（动量交换法）并写入 CSV。
///
/// MPI 模式下各进程的局部贡献通过 `MPI_Allreduce` 求和；仅 rank-0 写文件。
#[allow(dead_code)]
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
// Startup helper functions
// ---------------------------------------------------------------------------

/// Parse a lattice model string into a [`LatticeModel`] (unknown values fall back to D2Q9).
fn parse_lattice_model(s: &str) -> LatticeModel {
    match s {
        "D3Q19" => LatticeModel::D3Q19,
        "D3Q27" => LatticeModel::D3Q27,
        _       => LatticeModel::D2Q9,
    }
}

/// Parse a collision model string into a [`CollisionModel`] (unknown values fall back to BGK).
fn parse_collision_model(s: &str) -> CollisionModel {
    match s {
        "MRT" => CollisionModel::Mrt,
        _     => CollisionModel::Bgk,
    }
}

/// Determine the effective MPI rank even when `LBM_ENABLE_MPI` is OFF.
///
/// When MPI support is not compiled in, `mpi_rank()` always returns 0 and
/// `mpi_size()` always returns 1.  If the binary is still launched via
/// `mpiexec -n N`, all N processes would believe they are rank-0 and each
/// run a full independent simulation (allocate grid, step, write output).
///
/// This function reads the standard MPI launcher environment variables set
/// by OpenMPI (`OMPI_COMM_WORLD_RANK`), MPICH/PMI (`PMI_RANK`),
/// MVAPICH2 (`MV2_COMM_WORLD_RANK`), and PMIx/Slurm (`PMIX_RANK`) as a
/// fallback.  If any variable is present and parses to a non-negative
/// integer, that value is returned; otherwise `fallback` (the value from
/// `mpi_rank()`) is returned unchanged.
///
/// The caller must exit non-zero processes immediately after this call when
/// `nprocs == 1 && rank > 0` to ensure only rank-0 runs the simulation.
fn launcher_rank(fallback: i32) -> i32 {
    for var in &[
        "OMPI_COMM_WORLD_RANK",
        "PMI_RANK",
        "MV2_COMM_WORLD_RANK",
        "PMIX_RANK",
    ] {
        if let Ok(val) = std::env::var(var) {
            if let Ok(r) = val.trim().parse::<i32>() {
                return r;
            }
        }
    }
    fallback
}

/// Print solver startup header (rank-0 only).
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
        if cfg.ibm.is_some() {
            println!(
                "MPI IBM  : ibm_halo_width={} \
                 (u-halo exchange {n_gh} layer(s) before interpolation + \
                 force halo reduce {n_gh} layer(s) after spread; \
                 {note})",
                ibm_halo,
                n_gh = ibm_halo,
                note = if ibm_halo >= 2 {
                    "FourPoint kernel fully supported at MPI boundaries"
                } else {
                    "TwoPoint kernel supported; FourPoint kernel requires ibm_halo_width=2"
                }
            );
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

/// Print each rank's local grid size in rank order.
///
/// Sequential printing avoids interleaved output when multiple processes
/// write to stdout concurrently (which can corrupt multi-byte UTF-8 sequences).
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

/// Generate IBM marker points via Python FFI (in-process, no CSV file produced).
///
/// Only compiled when the `python-ffi` feature is enabled; otherwise a no-op.
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
                // Forward (x, y, ds) to the C++ IBM core via lbm_bindings.
                let ms = lbm_bindings::LbmIbmMarkerSet::new_from_coords(&x, &y, &ds);
                println!("  => MarkerSet created ({} markers)", ms.len());
                // ms is dropped here; in a full simulation loop it would be passed
                // to step_mdf / step_mls / step_penalty each time step.
            }
            Err(e) => eprintln!("[python-ffi] marker generation skipped: {e}"),
        }
    }
}

