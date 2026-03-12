//! FSI（流固耦合）方案模块
//!
//! 本模块将 FSI 流固耦合逻辑从 `main.rs` 中独立出来，提供：
//!
//! - [`FsiCouplingMode`]：三种耦合模式的类型安全枚举
//! - [`IbmEntry`]：IBM 体的运行时状态（标记点集 + 受力输出配置）
//! - [`resolve_coupling_mode`]：根据配置（显式或自动推断）确定耦合模式
//! - [`validate_coupling_mode`]：校验耦合模式与配置段是否一致
//! - [`setup_solid_bodies`]：设置 BB/IBB 固体体（标记 + 反弹方案）
//! - [`setup_ibm_bodies`]：创建 IBM 体（标记点集 + 日志）

use anyhow::{Result, bail};
use lbm_bindings::{LbmGrid, LbmSolver, LbmIbmMarkerSet};

use crate::config::{Config, FsiConfig, IbmBodyConfig, SolidForceOutputConfig};

// ---------------------------------------------------------------------------
// 耦合模式枚举
// ---------------------------------------------------------------------------

/// 流固耦合模式（类型安全枚举）
///
/// 由 [`resolve_coupling_mode`] 根据 TOML 配置确定：
/// - [`FsiCouplingMode::BounceBack`]：仅反弹格式（BB 或 IBB，由 `[solid]` 段配置）
/// - [`FsiCouplingMode::Ibm`]：仅 IBM 浸入边界（由 `[ibm]` 段配置）
/// - [`FsiCouplingMode::Hybrid`]：反弹 + IBM 混合（`[solid]` 和 `[ibm]` 两段均配置）
/// - [`FsiCouplingMode::None`]：无 FSI 耦合（纯流体仿真）
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FsiCouplingMode {
    /// ① 纯反弹格式（BB / IBB）
    BounceBack,
    /// ② 纯 IBM 浸入边界法
    Ibm,
    /// ③ 混合耦合（反弹 + IBM）
    Hybrid,
    /// 无 FSI 耦合（纯流体仿真）
    None,
}

impl FsiCouplingMode {
    /// 返回耦合模式的中文描述字符串（用于启动日志）
    pub fn description(&self) -> &'static str {
        match self {
            FsiCouplingMode::BounceBack => "① 纯 BB/IBB（反弹格式）",
            FsiCouplingMode::Ibm        => "② 纯 IBM（浸入边界法）",
            FsiCouplingMode::Hybrid     => "③ 混合 BB/IBB + IBM（反弹格式 + 浸入边界法）",
            FsiCouplingMode::None       => "无 FSI 耦合（纯流体仿真）",
        }
    }

    /// 是否需要 `[solid]` 配置段
    pub fn needs_solid(&self) -> bool {
        matches!(self, FsiCouplingMode::BounceBack | FsiCouplingMode::Hybrid)
    }

    /// 是否需要 `[ibm]` 配置段
    pub fn needs_ibm(&self) -> bool {
        matches!(self, FsiCouplingMode::Ibm | FsiCouplingMode::Hybrid)
    }
}

// ---------------------------------------------------------------------------
// IBM 运行时状态
// ---------------------------------------------------------------------------

/// 单个 IBM 体的运行时状态（每步 IBM 计算所需数据）
pub struct IbmEntry {
    /// Lagrangian 标记点集
    pub ms: LbmIbmMarkerSet,
    /// 体标签（用于输出文件名）
    pub label: String,
    /// 受力输出配置
    pub force_cfg: SolidForceOutputConfig,
}

// ---------------------------------------------------------------------------
// 耦合模式解析与校验
// ---------------------------------------------------------------------------

/// 根据 TOML 配置确定最终使用的耦合模式。
///
/// 若 `[fsi] coupling` 不为 `"auto"`，则直接使用显式指定的模式；
/// 否则根据 `[solid]`/`[ibm]` 段的存在自动推断。
///
/// # 参数
///
/// - `fsi_cfg`：`[fsi]` 配置段（含 `coupling` 字段）
/// - `has_solid`：是否存在非空且非 `"none"` 的 `[solid]` 段
/// - `has_ibm`：是否存在 `[ibm]` 段
pub fn resolve_coupling_mode(
    fsi_cfg: &FsiConfig,
    has_solid: bool,
    has_ibm: bool,
) -> FsiCouplingMode {
    let mode_str = fsi_cfg.normalized_coupling();
    match mode_str.to_lowercase().as_str() {
        "bounce_back" => FsiCouplingMode::BounceBack,
        "ibm"         => FsiCouplingMode::Ibm,
        "hybrid"      => FsiCouplingMode::Hybrid,
        "none"        => FsiCouplingMode::None,
        _             => {
            // "auto"（默认）：根据配置段推断
            match FsiConfig::infer_coupling(has_solid, has_ibm) {
                "hybrid"       => FsiCouplingMode::Hybrid,
                "ibm"          => FsiCouplingMode::Ibm,
                "bounce_back"  => FsiCouplingMode::BounceBack,
                _              => FsiCouplingMode::None,
            }
        }
    }
}

/// 校验显式指定的耦合模式与配置段是否一致。
///
/// 当 `[fsi] coupling` 不为 `"auto"` 时，检查所需的配置段是否已提供：
/// - `"bounce_back"` → 必须有 `[solid]` 段  
/// - `"ibm"` → 必须有 `[ibm]` 段  
/// - `"hybrid"` → 必须同时有 `[solid]` 和 `[ibm]` 段
///
/// 若配置不一致，返回 `Err` 以便提前报错而非静默忽略。
pub fn validate_coupling_mode(
    mode: FsiCouplingMode,
    has_solid: bool,
    has_ibm: bool,
    coupling_str: &str,
) -> Result<()> {
    // "auto" 模式不需要校验（推断结果总是与配置段一致）
    if coupling_str.to_lowercase() == "auto" {
        return Ok(());
    }

    if mode.needs_solid() && !has_solid {
        bail!(
            "[fsi] coupling={:?} 要求提供 [solid] 段并设置非空 bodies，\
             但配置中未找到有效的 [solid] 配置。\n\
             请添加 [solid] 段，或将 fsi.coupling 改为 \"ibm\" / \"auto\"。",
            coupling_str
        );
    }
    if mode.needs_ibm() && !has_ibm {
        bail!(
            "[fsi] coupling={:?} 要求提供 [ibm] 段，\
             但配置中未找到 [ibm] 配置。\n\
             请添加 [ibm] 段，或将 fsi.coupling 改为 \"bounce_back\" / \"auto\"。",
            coupling_str
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// 固体体标记与反弹方案设置
// ---------------------------------------------------------------------------

/// 设置 BB / IBB 固体体：依次标记几何体并配置反弹方案。
///
/// 遍历 `cfg.solid.bodies`，对每个体调用对应的几何标记函数，
/// 最后根据 `cfg.solid.bc_type` 设置求解器的反弹方案。
///
/// MPI 坐标转换参数 `(x_start, y_start, phys_x0, phys_y0)` 用于将全局坐标
/// 映射到本进程的本地网格坐标；单进程模式下均传 `(0, 0, 0, 0)`。
pub fn setup_solid_bodies(
    cfg: &Config,
    grid: &mut LbmGrid,
    solver: &mut LbmSolver,
    x_start: i32, y_start: i32,
    phys_x0: i32, phys_y0: i32,
    rank: i32,
) {
    if cfg.solid.bodies.is_empty() {
        return;
    }

    let to_local_i = |gi: i32| gi - x_start + phys_x0;
    let to_local_j = |gj: i32| gj - y_start + phys_y0;

    for body in &cfg.solid.bodies {
        match body.shape.to_lowercase().as_str() {
            "cylinder" => {
                let local_cx = body.cx - x_start as f64 + phys_x0 as f64;
                let local_cy = body.cy - y_start as f64 + phys_y0 as f64;
                lbm_bindings::mark_solid_cylinder(grid, local_cx, local_cy, body.radius);
                if rank == 0 {
                    println!(
                        "  [solid] 圆柱标记: center=({:.2}, {:.2}), radius={:.2}",
                        body.cx, body.cy, body.radius
                    );
                }
            }
            "rectangle" => {
                lbm_bindings::mark_solid_rectangle(
                    grid,
                    to_local_i(body.i0), to_local_j(body.j0),
                    to_local_i(body.i1), to_local_j(body.j1),
                );
                if rank == 0 {
                    println!(
                        "  [solid] 矩形标记: [{}, {}] × [{}, {}]",
                        body.i0, body.i1, body.j0, body.j1
                    );
                }
            }
            "mesh" => {
                if body.mesh_file.is_empty() {
                    if rank == 0 {
                        eprintln!("  [warn] solid.bodies shape=\"mesh\" 但 mesh_file 为空，跳过");
                    }
                } else {
                    lbm_bindings::mark_solid_from_mesh_file(grid, &body.mesh_file);
                    if rank == 0 {
                        println!(
                            "  [solid] 网格文件标记: file={:?}",
                            body.mesh_file
                        );
                    }
                }
            }
            other => {
                if rank == 0 {
                    eprintln!("  [warn] 未知固体形状 {:?}，跳过", other);
                }
            }
        }
    }

    let bc_mode = match cfg.solid.bc_type.to_lowercase().as_str() {
        "bounce_back"                                    => 1_i32,
        "interpolated_bounce_back" | "ibb" | "bouzidi"  => 2_i32,
        _                                                => 0_i32,
    };
    lbm_bindings::mark_solid_bc(solver, bc_mode);

    if rank == 0 {
        let scheme_name = match bc_mode {
            1 => "BounceBack（半步长反弹，Ladd 1994，一阶精度）",
            2 => "InterpolatedBounceBack（Bouzidi 插值反弹，2001，二阶精度）",
            _ => "None（固体节点已标记，但不施加反弹，仅用于调试）",
        };
        println!("  [solid] 反弹方案: {}", scheme_name);
        if cfg.solid.force_output.enabled {
            let nprocs = lbm_bindings::mpi_size();
            println!(
                "  [solid] 受力统计: 每 {} 步输出到 {}/{}.csv（动量交换法，MEA）",
                cfg.solid.force_output.interval,
                cfg.output.directory,
                cfg.solid.force_output.filename
            );
            if nprocs > 1 {
                println!("  [solid] MPI 受力统计: 各进程局部贡献通过 MPI_Allreduce 求和");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// IBM 体初始化
// ---------------------------------------------------------------------------

/// 创建所有 IBM 体的运行时状态（标记点集 + 受力输出配置）。
///
/// 根据 `cfg.ibm` 中的 bodies 列表（或顶层单体字段）逐体构造 [`IbmEntry`]。
/// 若 `[ibm]` 段不存在，则返回空列表（无 IBM 耦合）。
///
/// # 错误
///
/// 若某 IBM 体的 `geometry="file"` 但 `mesh_file` 未设置，返回 `Err`。
pub fn setup_ibm_bodies(cfg: &Config, rank: i32) -> Result<Vec<IbmEntry>> {
    let ibm_cfg = match cfg.ibm.as_ref() {
        Some(c) => c,
        None    => return Ok(Vec::new()),
    };

    // 确定有效 bodies 列表（多体优先，否则回退到顶层单体字段）
    let single_body_list: Vec<IbmBodyConfig>;
    let effective_bodies: &[IbmBodyConfig] = if ibm_cfg.bodies.is_empty() {
        if !ibm_cfg.geometry.is_empty() {
            single_body_list = vec![IbmBodyConfig {
                geometry:     ibm_cfg.geometry.clone(),
                x0:           ibm_cfg.x0,
                y0:           ibm_cfg.y0,
                size:         ibm_cfg.size,
                n_markers:    ibm_cfg.n_markers,
                mesh_file:    ibm_cfg.mesh_file.clone(),
                label:        String::new(),
                force_output: None,
            }];
            &single_body_list
        } else {
            single_body_list = vec![];
            &single_body_list
        }
    } else {
        &ibm_cfg.bodies
    };

    // 打印 IBM 全局方案说明（rank-0 只打印一次）
    if rank == 0 && !effective_bodies.is_empty() {
        let method_name = match ibm_cfg.method.to_lowercase().as_str() {
            "penalty" => format!(
                "Penalty-IBM（Goldstein 1993，α={:.2}, β={:.2}）",
                ibm_cfg.alpha, ibm_cfg.beta,
            ),
            "mls" => "MLS-IBM（移动最小二乘，Wang 2009）".to_string(),
            _     => format!("MDF-IBM（多重直接力，Luo 2007，n_iter={}）", ibm_cfg.n_iter),
        };
        println!(
            "  [IBM] 方案: {}  δ核: {}  固体体数: {}",
            method_name, ibm_cfg.delta_kernel, effective_bodies.len()
        );
    }

    let mut entries: Vec<IbmEntry> = Vec::with_capacity(effective_bodies.len());

    for (body_idx, body) in effective_bodies.iter().enumerate() {
        let ms: LbmIbmMarkerSet = match body.geometry.to_lowercase().as_str() {
            "file" => {
                if body.mesh_file.is_empty() {
                    bail!(
                        "[IBM] 第 {} 体 geometry=\"file\" 但 mesh_file 未设置，\
                         请在 [[ibm.bodies]] 中添加 mesh_file = \"path/to/markers.csv\"",
                        body_idx
                    );
                }
                LbmIbmMarkerSet::new_from_file(&body.mesh_file)
                    .map_err(|e| anyhow::anyhow!("{}", e))?
            }
            "filament" => LbmIbmMarkerSet::new_filament(
                body.x0, body.y0, body.size, body.n_markers,
            ),
            _ => LbmIbmMarkerSet::new_circle(
                body.x0, body.y0, body.size, body.n_markers,
            ),
        };

        let label = if body.label.is_empty() {
            format!("ibm_body_{}", body_idx)
        } else {
            body.label.clone()
        };

        let force_cfg = body.force_output.clone()
            .unwrap_or_else(|| ibm_cfg.force_output.clone());

        if rank == 0 {
            let geom_info = match body.geometry.to_lowercase().as_str() {
                "file"     => format!("file={:?}", body.mesh_file),
                "filament" => format!("起点: ({}, {})  长度: {}", body.x0, body.y0, body.size),
                _          => format!("圆心: ({}, {})  半径: {}", body.x0, body.y0, body.size),
            };
            println!(
                "  [IBM] 体[{}] 标签={:?}  标记点数: {}  几何: {}",
                body_idx, label, ms.len(), geom_info
            );
            if force_cfg.enabled {
                println!(
                    "  [IBM] 体[{}] 受力统计: 每 {} 步输出到 {}/{}.csv",
                    body_idx, force_cfg.interval,
                    cfg.output.directory, force_cfg.filename
                );
            }
        }

        entries.push(IbmEntry { ms, label, force_cfg });
    }

    Ok(entries)
}
