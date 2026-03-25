//! FSI（流固耦合）方案模块
//!
//! 本模块将 FSI 流固耦合逻辑从 `main.rs` 中独立出来，提供：
//!
//! - [`FsiCouplingMode`]：三种耦合模式的类型安全枚举
//! - [`IbmEntry`]：IBM 体的运行时状态（标记点集 + 受力输出配置 + 可选运动体）
//! - [`SolidEntry`]：BB/IBB 固体体运行时状态（含可选刚体运动）
//! - [`resolve_coupling_mode`]：根据配置（显式或自动推断）确定耦合模式
//! - [`validate_coupling_mode`]：校验耦合模式与配置段是否一致
//! - [`setup_solid_bodies`]：设置 BB/IBB 固体体（标记 + 反弹方案）
//! - [`setup_ibm_bodies`]：创建 IBM 体（标记点集 + 日志）

use anyhow::{Result, bail};
use lbm_bindings::{LbmGrid, LbmSolver, LbmIbmMarkerSet, LbmRigidBody2D, RigidBodyScheme};

use crate::config::{Config, FsiConfig, IbmBodyConfig, SolidForceOutputConfig, MotionType, RigidBodyMotionConfig};

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
    /// Returns a short English description string for startup logging.
    pub fn description(&self) -> &'static str {
        match self {
            FsiCouplingMode::BounceBack => "bounce-back / IBB",
            FsiCouplingMode::Ibm        => "IBM (immersed boundary)",
            FsiCouplingMode::Hybrid     => "hybrid BB/IBB + IBM",
            FsiCouplingMode::None       => "none (pure fluid)",
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
    /// IBM 力计算方法（`"mdf"` / `"mls"` / `"penalty"` / `"ivc"` / `"ivc_stationary"`）；来自体级覆盖或全局默认
    pub method: String,
    /// MDF 子迭代次数（`method="mdf"` 时有效）
    pub n_iter: i32,
    /// 罚函数比例增益（`method="penalty"` 时有效）
    pub alpha: f64,
    /// 罚函数积分增益（`method="penalty"` 时有效）
    pub beta: f64,
    /// 运动类型（`Fixed` / `RigidFree` / `Prescribed` / `Flexible`）
    pub motion_type: MotionType,
    /// 刚体求解器（仅 `motion_type == RigidFree` 时非 None）
    pub rigid_body: Option<LbmRigidBody2D>,
    /// 该体质心当前坐标（`rigid_free` 时跟踪）
    pub cx: f64,
    pub cy: f64,
}

// ---------------------------------------------------------------------------
// 固体体运行时状态
// ---------------------------------------------------------------------------

/// 单个 BB/IBB 固体体的运行时状态（每步固体更新所需数据）
pub struct SolidEntry {
    /// 体形状（"cylinder" / "rectangle" / "mesh"；小写）
    pub shape: String,
    /// 圆柱：质心 x 坐标（全局格子单位，运动时逐步更新）
    pub cx: f64,
    /// 圆柱：质心 y 坐标（全局格子单位）
    pub cy: f64,
    /// 圆柱半径（格子单位）
    pub radius: f64,
    /// 矩形：西南角（全局格子坐标）
    pub i0: i32, pub j0: i32,
    /// 矩形：东北角
    pub i1: i32, pub j1: i32,
    /// 网格文件路径（仅 mesh 形状）
    pub mesh_file: String,
    /// 反弹方案：1 = BB，2 = IBB
    pub bc_mode: i32,
    /// 运动类型（`Fixed` / `RigidFree`）
    pub motion_type: MotionType,
    /// 刚体求解器（仅 `motion_type == RigidFree` 时非 None）
    pub rigid_body: Option<LbmRigidBody2D>,
    /// 体标签
    pub label: String,
    /// 受力输出配置
    pub force_cfg: crate::config::SolidForceOutputConfig,
}

impl SolidEntry {
    /// 该体是否为运动刚体（需要 Ladd 移动壁面修正）。
    pub fn is_moving(&self) -> bool {
        self.motion_type == MotionType::RigidFree
    }

    /// 质心当前速度 (ux, uy, omega)；静止体返回 (0,0,0)。
    pub fn wall_velocity(&self) -> (f64, f64, f64) {
        if let Some(rb) = &self.rigid_body {
            let (_, _, ux, uy, _, omega) = rb.state();
            (ux, uy, omega)
        } else {
            (0.0, 0.0, 0.0)
        }
    }
}

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
            "[fsi] coupling={:?} requires a [solid] section with non-empty bodies, \
             but no valid [solid] config was found.\n\
             Please add a [solid] section, or set fsi.coupling = \"ibm\" / \"auto\".",
            coupling_str
        );
    }
    if mode.needs_ibm() && !has_ibm {
        bail!(
            "[fsi] coupling={:?} requires an [ibm] section, \
             but no [ibm] config was found.\n\
             Please add an [ibm] section, or set fsi.coupling = \"bounce_back\" / \"auto\".",
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
/// 根据 `cfg.solid.bc_type` 设置求解器的反弹方案。
///
/// - 若任一固体体 `motion_type = "rigid_free"`，求解器的自动 BB/IBB 被禁用
///   （设为 None），由调用方在每步手动调用 [`step_solid_moving`] 施加移动壁面 BC。
/// - 若所有固体体均为静止（`motion_type = "fixed"` 或缺省），则求解器自动处理 BC，
///   与旧行为完全兼容。
///
/// 返回 `(Vec<SolidEntry>, has_moving)` 其中 `has_moving` 标识是否有运动刚体。
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
) -> (Vec<SolidEntry>, bool) {
    if cfg.solid.bodies.is_empty() {
        return (Vec::new(), false);
    }

    let to_local_i = |gi: i32| gi - x_start + phys_x0;
    let to_local_j = |gj: i32| gj - y_start + phys_y0;

    // Pre-scan: check if any body is rigid_free
    let has_moving = cfg.solid.bodies.iter()
        .any(|b| b.motion_type == MotionType::RigidFree);

    let mut entries: Vec<SolidEntry> = Vec::with_capacity(cfg.solid.bodies.len());

    for (body_idx, body) in cfg.solid.bodies.iter().enumerate() {
        match body.shape.to_lowercase().as_str() {
            "cylinder" => {
                let local_cx = body.cx - x_start as f64 + phys_x0 as f64;
                let local_cy = body.cy - y_start as f64 + phys_y0 as f64;
                lbm_bindings::mark_solid_cylinder(grid, local_cx, local_cy, body.radius);
                if rank == 0 {
                    println!(
                        "  [solid] cylinder[{}]: center=({:.2}, {:.2}), radius={:.2}{}",
                        body_idx, body.cx, body.cy, body.radius,
                        if body.motion_type == MotionType::RigidFree { " [rigid_free]" } else { "" }
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
                        "  [solid] rectangle[{}]: [{}, {}] x [{}, {}]",
                        body_idx, body.i0, body.i1, body.j0, body.j1
                    );
                }
            }
            "mesh" => {
                if body.mesh_file.is_empty() {
                    if rank == 0 {
                        eprintln!("  [warn] solid.bodies[{}] shape=\"mesh\" but mesh_file is empty, skipping", body_idx);
                    }
                } else {
                    lbm_bindings::mark_solid_from_mesh_file(grid, &body.mesh_file);
                    if rank == 0 {
                        println!("  [solid] mesh[{}] file={:?}", body_idx, body.mesh_file);
                    }
                }
            }
            other => {
                if rank == 0 {
                    eprintln!("  [warn] unknown solid body shape {:?}, skipping", other);
                }
            }
        }

        // --- 逐体反弹方案：以局部 body.bc_type 为优先，缺省继承全局 ---
        let effective_bc_str = body.bc_type.as_deref()
            .unwrap_or(&cfg.solid.bc_type);
        let body_bc_mode = match effective_bc_str.to_lowercase().as_str() {
            "bounce_back" | "bb"                             => 1_i32,
            "interpolated_bounce_back" | "ibb" | "bouzidi"  => 2_i32,
            _                                                => 0_i32,
        };
        lbm_bindings::assign_solid_bc_unmarked(grid, body_bc_mode);

        // --- 构建刚体求解器（仅 rigid_free）---
        let (motion_type, rigid_body) = if body.motion_type == MotionType::RigidFree
            && body.shape.to_lowercase() == "cylinder"
        {
            let mc = &body.motion;
            let scheme = parse_rigid_scheme(&mc.internal_mass_scheme);
            let mass = if mc.mass > 0.0 { mc.mass }
                else { mc.body_density * std::f64::consts::PI * body.radius * body.radius };
            let inertia = if mc.inertia > 0.0 { mc.inertia }
                else { 0.5 * mass * body.radius * body.radius };
            // BB/IBB 刚体不需要边界/内部标记点参考坐标（不使用 IBM 力计算）
            let mut rb = LbmRigidBody2D::new(
                mass, inertia,
                mc.body_density.max(1.0), mc.rho_f,
                body.cx, body.cy,
                scheme, mc.is_closed,
                &[], &[], &[], &[],
            );
            if mc.vel_x0 != 0.0 || mc.vel_y0 != 0.0 || mc.omega0 != 0.0 {
                rb.set_velocity(mc.vel_x0, mc.vel_y0, mc.omega0);
            }
            if rank == 0 {
                println!(
                    "  [solid] body[{}] motion=rigid_free  mass={:.4}  inertia={:.4}  scheme={}",
                    body_idx, mass, inertia, &mc.internal_mass_scheme
                );
            }
            (MotionType::RigidFree, Some(rb))
        } else {
            (body.motion_type.clone(), None)
        };

        let label = if body.label.is_empty() {
            format!("solid_body_{}", body_idx)
        } else {
            body.label.clone()
        };

        entries.push(SolidEntry {
            shape:      body.shape.to_lowercase(),
            cx:         body.cx,
            cy:         body.cy,
            radius:     body.radius,
            i0: body.i0, j0: body.j0, i1: body.i1, j1: body.j1,
            mesh_file:  body.mesh_file.clone(),
            bc_mode:    body_bc_mode,
            motion_type,
            rigid_body,
            label,
            force_cfg:  Default::default(),  // global force_output is logged separately
        });
    }

    // --- 全局方案：若无运动体，注册到求解器（solver.step 自动施加 BC）---
    let global_bc_mode = match cfg.solid.bc_type.to_lowercase().as_str() {
        "bounce_back"                                    => 1_i32,
        "interpolated_bounce_back" | "ibb" | "bouzidi"  => 2_i32,
        _                                                => 0_i32,
    };

    if has_moving {
        // 运动刚体：禁用 solver 内的自动 BB/IBB；每步由 step_solid_moving 手动施加
        lbm_bindings::mark_solid_bc(solver, 0);
        if rank == 0 {
            println!("  [solid] BC mode: MANUAL (rigid_free body detected — Ladd moving wall correction enabled)");
        }
    } else {
        lbm_bindings::mark_solid_bc(solver, global_bc_mode);
        if rank == 0 {
            let scheme_name = match global_bc_mode {
                1 => "BounceBack (half-way, Ladd 1994, 1st-order)",
                2 => "InterpolatedBounceBack (Bouzidi 2001, 2nd-order)",
                _ => "None (solid nodes marked, no bounce-back applied; debug only)",
            };
            println!("  [solid] BC scheme: {}", scheme_name);
        }
    }

    if rank == 0 {
        if cfg.solid.force_output.enabled {
            let nprocs = lbm_bindings::mpi_size();
            println!(
                "  [solid] force output: every {} steps -> {}/{}.csv (momentum exchange, MEA)",
                cfg.solid.force_output.interval,
                cfg.output.directory,
                cfg.solid.force_output.filename
            );
            if nprocs > 1 {
                println!("  [solid] MPI force reduction: local contributions summed via MPI_Allreduce");
            }
        }
    }

    (entries, has_moving)
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
                method:       None,
                n_iter:       None,
                alpha:        None,
                beta:         None,
                motion_type:  MotionType::Fixed,
                motion:       Default::default(),
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
    // 首行显示全局默认方案；各体的具体方案在下方 [IBM] body[N] 行中逐一说明。
    if rank == 0 && !effective_bodies.is_empty() {
        let global_method_desc = match ibm_cfg.method.to_lowercase().as_str() {
            "penalty"          => format!(
                "Penalty-IBM (Goldstein 1993, α={:.2}, β={:.2})",
                ibm_cfg.alpha, ibm_cfg.beta,
            ),
            "mls"              => "MLS-IBM (implicit MLS, JCP 2025, n_iter=3)".to_string(),
            "mls_original"     => "MLS-IBM original (Algorithm 1, JCP 2025)".to_string(),
            "mls_explicit"     => "MLS-IBM explicit+Z (Algorithm 2, JCP 2025)".to_string(),
            "ivc"              => "IVC-IBM (implicit velocity correction, Wu & Shu 2009)".to_string(),
            "ivc_stationary"   => "IVC-IBM stationary (LU cache, Wu & Shu 2009)".to_string(),
            _                  => format!("MDF-IBM (multi-direct-forcing, Wang 2008, n_iter={})", ibm_cfg.n_iter),
        };
        println!(
            "  [IBM] scheme: {}  delta kernel: {}  bodies: {}",
            global_method_desc, ibm_cfg.delta_kernel, effective_bodies.len()
        );
    }

    let mut entries: Vec<IbmEntry> = Vec::with_capacity(effective_bodies.len());

    for (body_idx, body) in effective_bodies.iter().enumerate() {
        let ms: LbmIbmMarkerSet = match body.geometry.to_lowercase().as_str() {
            "file" => {
                if body.mesh_file.is_empty() {
                    bail!(
                        "[IBM] body {} has geometry=\"file\" but mesh_file is not set; \
                         please add mesh_file = \"path/to/markers.csv\" under [[ibm.bodies]]",
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

        // Resolve per-body method params (body-level overrides global [ibm] settings)
        let method = body.method.clone()
            .unwrap_or_else(|| ibm_cfg.method.clone());
        let n_iter = body.n_iter.unwrap_or(ibm_cfg.n_iter);
        let alpha  = body.alpha.unwrap_or(ibm_cfg.alpha);
        let beta   = body.beta.unwrap_or(ibm_cfg.beta);

        if rank == 0 {
            let geom_info = match body.geometry.to_lowercase().as_str() {
                "file"     => format!("file={:?}", body.mesh_file),
                "filament" => format!("start=({}, {})  length={}", body.x0, body.y0, body.size),
                _          => format!("center=({}, {})  radius={}", body.x0, body.y0, body.size),
            };
            let method_info = if body.method.is_some() {
                format!("  method={} (body override)", method)
            } else {
                format!("  method={} (inherited global)", method)
            };
            println!(
                "  [IBM] body[{}] label={:?}  markers={}  geometry: {}{}",
                body_idx, label, ms.len(), geom_info, method_info
            );
            if force_cfg.enabled {
                println!(
                    "  [IBM] body[{}] force output: every {} steps -> {}/{}.csv",
                    body_idx, force_cfg.interval,
                    cfg.output.directory, force_cfg.filename
                );
            }
        }

        // 构建可选刚体求解器
        let (motion_type, rigid_body, cx, cy) = match &body.motion_type {
            MotionType::RigidFree => {
                let mc = &body.motion;
                let scheme = parse_rigid_scheme(&mc.internal_mass_scheme);
                // 建立边界点参考坐标（体固系）
                let n = ms.len();
                let ref_x: Vec<f64> = (0..n).map(|k| {
                    let (x, _) = ms.get_marker_position(k);
                    x - body.x0
                }).collect();
                let ref_y: Vec<f64> = (0..n).map(|k| {
                    let (_, y) = ms.get_marker_position(k);
                    y - body.y0
                }).collect();
                // 自动推算质量（若未给出）：m = rho_b * pi * r^2（圆形）
                let mass = if mc.mass > 0.0 { mc.mass }
                    else { mc.body_density * std::f64::consts::PI * body.size * body.size };
                // 自动推算转动惯量（若未给出）：I = 0.5*m*r^2（实心圆柱）
                let inertia = if mc.inertia > 0.0 { mc.inertia }
                    else { 0.5 * mass * body.size * body.size };
                let mut rb = LbmRigidBody2D::new(
                    mass, inertia, mc.body_density.max(1.0), mc.rho_f,
                    body.x0, body.y0, scheme, mc.is_closed,
                    &ref_x, &ref_y, &[], &[],
                );
                if mc.vel_x0 != 0.0 || mc.vel_y0 != 0.0 || mc.omega0 != 0.0 {
                    rb.set_velocity(mc.vel_x0, mc.vel_y0, mc.omega0);
                }
                if rank == 0 {
                    println!(
                        "  [IBM] body[{}] motion=rigid_free  mass={:.4}  inertia={:.4}  is_closed={}  scheme={}",
                        body_idx, mass, inertia, mc.is_closed, &mc.internal_mass_scheme
                    );
                }
                (MotionType::RigidFree, Some(rb), body.x0, body.y0)
            }
            other => (other.clone(), None, body.x0, body.y0),
        };

        entries.push(IbmEntry {
            ms, label, force_cfg, method, n_iter, alpha, beta,
            motion_type, rigid_body, cx, cy,
        });
    }

    Ok(entries)
}

/// 解析内部质量方案字符串 → `RigidBodyScheme`。
fn parse_rigid_scheme(s: &str) -> RigidBodyScheme {
    match s.to_lowercase().as_str() {
        "uhlmann"       => RigidBodyScheme::UhlmannRigidBody,
        "feng"          => RigidBodyScheme::FengRigidBody,
        "lagrangian"    => RigidBodyScheme::LagrangianPoints,
        _               => RigidBodyScheme::None,
    }
}
