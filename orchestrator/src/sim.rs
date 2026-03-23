//! MPI 域分解上下文与边界条件注册
//!
//! 将 [`crate::main`] 中的 MPI 设置与边界条件注册逻辑提取为独立模块，
//! 使主仿真流程更加清晰。
//!
//! ## 主要类型与函数
//!
//! - [`MpiDecomp`]：封装 2D/3D MPI 分解对象和规范化后的模式名称
//! - [`register_boundary_conditions`]：按 rank 顺序注册 BC 并打印日志

use std::io::Write as IoWrite;

use anyhow::Result;
use lbm_bindings::{BcType, Face, LbmMpiDecomp2D, LbmMpiDecomp3D, LbmSolver};

use crate::config::Config;
use crate::output::PartitionInfo;

// ---------------------------------------------------------------------------
// MPI 域分解上下文
// ---------------------------------------------------------------------------

/// MPI 域分解上下文。
///
/// 封装 2D 或 3D MPI 分解对象（最多选用其一），以及规范化后的分解模式名称。
/// 调用 [`MpiDecomp::setup`] 创建；调用 [`MpiDecomp::attach_to_solver`]
/// 将分解对象绑定到 [`LbmSolver`]（在 `grid`/`solver` 初始化之后、时间循环之前）。
pub struct MpiDecomp {
    /// 二维 XY 块分解（与 `decomp3d` 互斥）
    pub decomp2d: Option<LbmMpiDecomp2D>,
    /// 三维 XYZ 块分解（与 `decomp2d` 互斥）
    pub decomp3d: Option<LbmMpiDecomp3D>,
    /// 归一化后的 MPI 模式名：`"block"` | `"independent"` | `"multigrid"`
    pub effective_mode: String,
}

impl MpiDecomp {
    /// 根据 `[mpi]` 配置创建 MPI 域分解上下文。
    ///
    /// - `"block"` 模式：按 `nx_blocks × ny_blocks (× nz_blocks)` 创建分解对象。
    ///   若乘积与 `nprocs` 不符，自动退化为 1D Y 切片（并打印警告）。
    /// - `"independent"` / `"multigrid"` 模式：不创建分解对象。
    /// - 未知模式：打印警告后退化为 1D Y 切片。
    pub fn setup(cfg: &Config, nprocs: i32) -> Result<Self> {
        let (effective_mode, mode_was_renamed) = cfg.mpi.normalized_mode();
        if mode_was_renamed && nprocs > 1 {
            eprintln!(
                "[info] mpi.mode {:?} is deprecated; using {:?}. \
                 See docs/MPI_parallel.md for the new mode names.",
                cfg.mpi.mode, effective_mode
            );
        }

        let use_3d_decomp = cfg.fluid.nz > 1 && cfg.mpi.nz_blocks > 1;
        let mut decomp2d: Option<LbmMpiDecomp2D> = None;
        let mut decomp3d: Option<LbmMpiDecomp3D> = None;
        // ibm_halo_width 控制每侧幽灵层数：1 = 默认（TwoPoint），2 = FourPoint
        let n_ghost = (cfg.mpi.ibm_halo_width.max(1)) as i32;

        match effective_mode {
            "block" => {
                if use_3d_decomp {
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
                        decomp2d = LbmMpiDecomp2D::new_n(
                            cfg.fluid.nx as i32, cfg.fluid.ny as i32, 1, nprocs, n_ghost);
                    } else {
                        decomp3d = LbmMpiDecomp3D::new(
                            cfg.fluid.nx as i32, cfg.fluid.ny as i32, cfg.fluid.nz as i32,
                            px as i32, py as i32, pz as i32);
                    }
                } else {
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
                        decomp2d = LbmMpiDecomp2D::new_n(
                            cfg.fluid.nx as i32, cfg.fluid.ny as i32, 1, nprocs, n_ghost);
                    } else {
                        decomp2d = LbmMpiDecomp2D::new_n(
                            cfg.fluid.nx as i32, cfg.fluid.ny as i32, px as i32, py as i32, n_ghost);
                    }
                }
            }
            "independent" | "multigrid" => {
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
                decomp2d = LbmMpiDecomp2D::new_n(
                    cfg.fluid.nx as i32, cfg.fluid.ny as i32, 1, nprocs, n_ghost);
            }
        }

        Ok(Self {
            decomp2d,
            decomp3d,
            effective_mode: effective_mode.to_owned(),
        })
    }

    /// 返回本进程本地网格尺寸 `(nx, ny, nz)`。
    ///
    /// MPI 块分解模式下从分解对象中读取；否则使用全局网格尺寸。
    pub fn local_grid_size(&self, cfg: &Config) -> (i32, i32, i32) {
        if let Some(ref d) = self.decomp3d {
            (d.grid_nx(), d.grid_ny(), d.grid_nz())
        } else if let Some(ref d) = self.decomp2d {
            (d.grid_nx(), d.grid_ny(), cfg.fluid.nz as i32)
        } else {
            (cfg.fluid.nx as i32, cfg.fluid.ny as i32, cfg.fluid.nz as i32)
        }
    }

    /// 将 MPI 分解对象绑定到求解器。
    ///
    /// 必须在 [`LbmSolver::new`] 之后、时间循环之前调用。
    pub fn attach_to_solver(&mut self, solver: &mut LbmSolver) {
        if let Some(ref mut d) = self.decomp3d {
            solver.attach_mpi3d(Some(d));
        } else if let Some(ref mut d) = self.decomp2d {
            solver.attach_mpi2d(Some(d));
        }
    }

    /// 返回全局坐标信息 `(x_start, y_start, phys_x0, phys_y0)`。
    ///
    /// 用于将全局固体坐标映射到本进程的本地网格坐标系。
    /// 非 MPI 模式下均返回 `(0, 0, 0, 0)`（恒等映射）。
    pub fn global_coords(&self) -> (i32, i32, i32, i32) {
        if let Some(ref d) = self.decomp3d {
            (d.x_start(), d.y_start(), d.phys_x0(), d.phys_y0())
        } else if let Some(ref d) = self.decomp2d {
            (d.x_start(), d.y_start(), d.phys_x0(), d.phys_y0())
        } else {
            (0, 0, 0, 0)
        }
    }

    /// 构建 MPI 块分解模式下的分区信息（用于输出时剥离幽灵行/列并嵌入元数据）。
    ///
    /// 仅在 `effective_mode == "block"` 且 `nprocs > 1` 时返回 `Some`；否则返回 `None`。
    pub fn partition_info(&self, cfg: &Config, nprocs: i32) -> Option<PartitionInfo> {
        if self.effective_mode != "block" || nprocs <= 1 {
            return None;
        }
        if let Some(ref d) = self.decomp3d {
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
            self.decomp2d.as_ref().map(|d| PartitionInfo {
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
    }

    /// 计算本进程的输出子目录路径。
    ///
    /// - MPI 块/独立/多重网格模式（`nprocs > 1`）：`<output.directory>/rank_<N>/`
    /// - 单进程模式：`<output.directory>`（与配置相同）
    pub fn output_dir(&self, cfg: &Config, nprocs: i32, rank: i32) -> String {
        let (eff_mode, _) = cfg.mpi.normalized_mode();
        if nprocs > 1
            && (eff_mode == "block" || eff_mode == "independent" || eff_mode == "multigrid")
        {
            format!("{}/rank_{}", cfg.output.directory, rank)
        } else {
            cfg.output.directory.clone()
        }
    }
}

// ---------------------------------------------------------------------------
// 边界条件注册
// ---------------------------------------------------------------------------

/// 注册流体边界条件到求解器，并按 rank 顺序打印注册日志。
///
/// MPI 块分解模式下，仅当本进程实际持有对应物理壁面时才注册该边界条件；
/// 非块分解模式下所有进程均注册全部边界条件。
///
/// 日志按 rank 顺序顺序打印，避免多进程并发写入 stdout 导致输出乱序。
///
/// # 参数
///
/// - `cfg`    — 全局配置（从中读取 `fluid.boundary_conditions`）
/// - `solver` — LBM 求解器（注册目标）
/// - `decomp` — MPI 域分解上下文（用于确定哪些壁面属于本进程）
/// - `nprocs` — 全局进程数
/// - `rank`   — 本进程序号
pub fn register_boundary_conditions(
    cfg: &Config,
    solver: &mut LbmSolver,
    decomp: &MpiDecomp,
    nprocs: i32,
    rank: i32,
) {
    let mut bc_log: Vec<String> = Vec::new();

    // 在 MPI 块模式下，prepend 本进程的全局坐标范围头部
    if decomp.effective_mode == "block" && nprocs > 1 {
        let region_str = if let Some(ref d) = decomp.decomp3d {
            format!(
                "x=[{}, {}), y=[{}, {}), z=[{}, {})",
                d.x_start(), d.x_start() + d.local_nx(),
                d.y_start(), d.y_start() + d.local_ny(),
                d.z_start(), d.z_start() + d.local_nz(),
            )
        } else if let Some(ref d) = decomp.decomp2d {
            format!(
                "x=[{}, {}), y=[{}, {})",
                d.x_start(), d.x_start() + d.local_nx(),
                d.y_start(), d.y_start() + d.local_ny(),
            )
        } else {
            format!("x=[0, {}), y=[0, {})", cfg.fluid.nx, cfg.fluid.ny)
        };
        bc_log.push(format!("  *** rank {:3}  global region: {} ***", rank, region_str));
    }

    for bc_cfg in &cfg.fluid.boundary_conditions {
        let bc_type = parse_bc_type(&bc_cfg.bc_type);
        let face    = parse_face(&bc_cfg.face);

        if should_apply_bc(decomp, face, cfg, nprocs) {
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

    // 按 rank 顺序打印日志行，避免并发写 stdout 时字节乱序
    if nprocs > 1 {
        for r in 0..nprocs {
            if rank == r {
                for line in &bc_log { println!("{}", line); }
                let _ = std::io::stdout().flush();
            }
            lbm_bindings::mpi_barrier();
        }
    } else {
        for line in &bc_log { println!("{}", line); }
    }
}

// ---------------------------------------------------------------------------
// 内部辅助函数
// ---------------------------------------------------------------------------

/// 将边界条件类型字符串转换为 [`BcType`]。
fn parse_bc_type(s: &str) -> BcType {
    match s.to_lowercase().as_str() {
        "bounce_back"          => BcType::BounceBack,
        "bounce_back_full_way" => BcType::BounceBackFullWay,
        "zou_he_velocity"      => BcType::ZouHeVelocity,
        "zou_he_pressure"      => BcType::ZouHePressure,
        "fully_developed"      => BcType::FullyDeveloped,
        "free_outlet"          => BcType::FreeOutlet,
        "guo_extrapolation"    => BcType::GuoExtrapolation,
        "periodic"             => BcType::Periodic,
        other => {
            eprintln!("  [warn] unknown bc_type {:?}; defaulting to BounceBack", other);
            BcType::BounceBack
        }
    }
}

/// 将面名称字符串转换为 [`Face`]。
fn parse_face(s: &str) -> Face {
    match s.to_lowercase().as_str() {
        "east"   => Face::East,
        "south"  => Face::South,
        "north"  => Face::North,
        "bottom" => Face::Bottom,
        "top"    => Face::Top,
        "west"   => Face::West,
        other => {
            eprintln!("  [warn] unknown face {:?}; defaulting to West", other);
            Face::West
        }
    }
}

/// 判断本进程是否应注册给定面的边界条件。
///
/// MPI 块分解模式下，只有当本进程的本地网格覆盖全局边界对应的壁面时才注册；
/// 否则（单进程或非块分解模式）所有进程均注册。
fn should_apply_bc(decomp: &MpiDecomp, face: Face, cfg: &Config, nprocs: i32) -> bool {
    if decomp.effective_mode != "block" || nprocs <= 1 {
        return true;
    }
    if let Some(ref d) = decomp.decomp3d {
        let (x_start, y_start, z_start) =
            (d.x_start() as u64, d.y_start() as u64, d.z_start() as u64);
        let (local_nx, local_ny, local_nz) =
            (d.local_nx() as u64, d.local_ny() as u64, d.local_nz() as u64);
        let (gnx, gny, gnz) =
            (cfg.fluid.nx as u64, cfg.fluid.ny as u64, cfg.fluid.nz as u64);
        match face {
            Face::South  => y_start == 0,
            Face::North  => y_start + local_ny == gny,
            Face::West   => x_start == 0,
            Face::East   => x_start + local_nx == gnx,
            Face::Bottom => z_start == 0,
            Face::Top    => z_start + local_nz == gnz,
        }
    } else if let Some(ref d) = decomp.decomp2d {
        let (x_start, y_start) = (d.x_start() as u64, d.y_start() as u64);
        let (local_nx, local_ny) = (d.local_nx() as u64, d.local_ny() as u64);
        let (gnx, gny) = (cfg.fluid.nx as u64, cfg.fluid.ny as u64);
        match face {
            Face::South  => y_start == 0,
            Face::North  => y_start + local_ny == gny,
            Face::West   => x_start == 0,
            Face::East   => x_start + local_nx == gnx,
            // Bottom/Top 是三维方向；2D 分解中所有进程均注册
            _            => true,
        }
    } else {
        true
    }
}
