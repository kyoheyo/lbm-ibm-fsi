//! 主动/被动运动方案
//!
//! 本模块提供两种结构运动求解器：
//!
//! - [`PrescribedMotion`]：**主动运动**——根据解析公式（平移、振荡、旋转、行波）
//!   在每个时间步计算结构速度，直接写入 IBM 标记点目标速度。
//!   对应 `motion_type = "prescribed"`（刚体）或 `"flexible"` + `[prescribed]`（丝状体行波）。
//!
//! - [`BeamSolver`]：**被动柔性体**——Euler-Bernoulli 悬臂梁有限元（Newmark-β 时间积分）。
//!   每步接收 IBM 体力（`ms.get_forces()`），推进梁变形，输出标记点位置和速度。
//!   对应 `motion_type = "flexible"` + `[ibm.bodies.flexible_beam]`。

use std::f64::consts::PI;
use crate::config::{PrescribedMotionConfig, FlexibleBodyConfig};

// ===========================================================================
// PrescribedMotion — 主动指定运动
// ===========================================================================

/// 主动指定运动求解器。
///
/// 根据 [`PrescribedMotionConfig`] 中的解析公式，在给定时刻 `t`（格子步）
/// 计算质心速度 `(ux, uy, ω)` 或逐标记点速度（行波模式）。
#[derive(Debug, Clone)]
pub struct PrescribedMotion {
    mode:       String,
    amplitude:  f64,
    frequency:  f64,
    phase:      f64,
    velocity_x: f64,
    velocity_y: f64,
    omega:      f64,
    wavelength: f64,
}

impl PrescribedMotion {
    /// 从配置创建实例。
    pub fn new(cfg: &PrescribedMotionConfig) -> Self {
        Self {
            mode:       cfg.mode.clone(),
            amplitude:  cfg.amplitude,
            frequency:  cfg.frequency,
            phase:      cfg.phase,
            velocity_x: cfg.velocity_x,
            velocity_y: cfg.velocity_y,
            omega:      cfg.omega,
            wavelength: cfg.wavelength,
        }
    }

    /// 评估刚体质心速度 `(ux_cm, uy_cm, omega_z)` 在时刻 `t`（格子步数）。
    ///
    /// 返回格子单位速度（步^{-1}）。
    ///
    /// ## 模式说明
    ///
    /// | `mode` | 输出 |
    /// |--------|------|
    /// | `"translate"` | `(velocity_x, velocity_y, 0)` |
    /// | `"oscillate_x"` | `(A·2πf·cos(2πft+φ), 0, 0)` |
    /// | `"oscillate_y"` | `(0, A·2πf·cos(2πft+φ), 0)` |
    /// | `"oscillate_xy"` | x/y 同频，y 超前 90° |
    /// | `"rotate"` | `(0, 0, omega)` |
    /// | `"rotate_oscillate"` | `(0, 0, A·2πf·cos(2πft+φ))` |
    pub fn eval_rigid(&self, t: f64) -> (f64, f64, f64) {
        let a = self.amplitude;
        let f = self.frequency;
        let phi = self.phase;
        let two_pi_f = 2.0 * PI * f;
        let cos_val = (two_pi_f * t + phi).cos();

        match self.mode.to_lowercase().as_str() {
            "translate" => (self.velocity_x, self.velocity_y, 0.0),
            "oscillate_x" => (a * two_pi_f * cos_val, 0.0, 0.0),
            "oscillate_y" => (0.0, a * two_pi_f * cos_val, 0.0),
            "oscillate_xy" => {
                let ux = a * two_pi_f * cos_val;
                let uy = a * two_pi_f * (two_pi_f * t + phi + 0.5 * PI).cos();
                (ux, uy, 0.0)
            }
            "rotate" => (0.0, 0.0, self.omega),
            "rotate_oscillate" => (0.0, 0.0, a * two_pi_f * cos_val),
            // 其他模式视为 translate
            _ => (self.velocity_x, self.velocity_y, 0.0),
        }
    }

    /// 评估丝状体各标记点速度（用于主动柔性体行波/振荡运动）。
    ///
    /// # 参数
    /// - `t`：当前格子时刻（步）
    /// - `arc_s`：各标记点沿梁轴的弧坐标（格子单位，长度 N）
    ///
    /// # 返回
    /// `(ux, uy)` 各长度为 N 的向量（格子单位/步）。
    ///
    /// ## 行波模式（`"traveling_wave"`）
    ///
    /// `ẏ(s, t) = A · 2πf · cos(2π(ft − s/λ) + φ)`  
    /// 标记点 x 方向速度为 0（小变形假设）。
    pub fn eval_markers(&self, t: f64, arc_s: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = arc_s.len();
        let mut ux = vec![0.0_f64; n];
        let mut uy = vec![0.0_f64; n];
        let a = self.amplitude;
        let f = self.frequency;
        let phi = self.phase;
        let two_pi_f = 2.0 * PI * f;

        match self.mode.to_lowercase().as_str() {
            "traveling_wave" => {
                let lambda = if self.wavelength > 0.0 { self.wavelength } else { 1.0 };
                for (k, &s) in arc_s.iter().enumerate() {
                    let phase_k = two_pi_f * t - 2.0 * PI * s / lambda + phi;
                    uy[k] = a * two_pi_f * phase_k.cos();
                }
            }
            "oscillate_y" => {
                let v = a * two_pi_f * (two_pi_f * t + phi).cos();
                for k in 0..n { uy[k] = v; }
            }
            "oscillate_x" => {
                let v = a * two_pi_f * (two_pi_f * t + phi).cos();
                for k in 0..n { ux[k] = v; }
            }
            _ => {
                // 其他模式：所有标记点使用刚体质心速度
                let (vx, vy, _) = self.eval_rigid(t);
                for k in 0..n { ux[k] = vx; uy[k] = vy; }
            }
        }
        (ux, uy)
    }
}

// ===========================================================================
// BeamSolver — Euler-Bernoulli 悬臂梁 FEM + Newmark-β（被动柔性体）
// ===========================================================================

/// Euler-Bernoulli 悬臂梁有限元求解器（被动柔性体）。
///
/// ## 力学模型
///
/// 均匀截面悬臂梁（clamped-free），clamped 端固定于 `(anchor_x, anchor_y)`，
/// 方向角 θ（`orientation`）。  
/// 采用 Hermite 三次插值（每节点 2 个自由度：横向位移 w，转角 θ），
/// 以 Newmark-β（β=1/4，γ=1/2，无条件稳定）时间积分。
///
/// ## 坐标约定
///
/// - 梁轴方向：e₁ = (cos θ, sin θ)  
/// - 横向方向：e₂ = (−sin θ, cos θ)（w > 0 为正横向）  
/// - 标记点全局位置：`P_k = anchor + s_k·e₁ + w(s_k)·e₂`  
/// - 标记点速度：`V_k = ẇ(s_k)·e₂`（忽略轴向速度）
///
/// ## IBM 力约定
///
/// `ms.get_forces()` 返回 IBM 施加于**流体**的力 `(fk_x, fk_y)`；  
/// 结构所受流体力为其负值 `(−fk_x, −fk_y)`；  
/// 横向分量：`f⊥ = fk_x·sin θ − fk_y·cos θ`（等于 −(fk·e₂)）。
pub struct BeamSolver {
    // ---- 几何 ----
    pub anchor_x:    f64,
    pub anchor_y:    f64,
    pub cos_ori:     f64,   // cos(orientation)
    pub sin_ori:     f64,   // sin(orientation)
    pub elem_len:    f64,   // L_e = length / n_elements
    pub n_elem:      usize,

    // ---- FEM 参数 ----
    // ndof = 2 * n_elem  (自由自由度，去掉 clamped 端的 2 个约束自由度)
    k_eff_lu: Vec<f64>,   // K_eff = a0·M + a1·C + K 的 LU 因子（列主元，存于下三角+上三角，行优先）
    lu_piv:   Vec<usize>, // LU 置换向量
    m_mat:    Vec<f64>,   // 质量矩阵（行优先密集）
    c_mat:    Vec<f64>,   // 阻尼矩阵（Rayleigh，质量比例）
    #[allow(dead_code)]
    k_mat:    Vec<f64>,   // 刚度矩阵（去掉 clamped 约束后的缩减矩阵）
                          // 保留用于将来的残差计算或自适应时步方案中需重建 K_eff 的场景。
                          // TODO: 若长期不使用，可在 BeamSolver::new() 末尾 drop(k_mat)。
    ndof:     usize,

    // ---- 主动柔性体：锚点激励耦合向量（支持基础激励 base-excitation）----
    // 从全局 K/M 矩阵中提取的第 0 列（锚点横向 DOF w_0 对自由 DOF 的耦合）
    // 用于在 advance() 中组装等效载荷：F_eff -= K_fp·w_0 + M_fp·ẅ_0 + C_fp·ẇ_0
    k_fp: Vec<f64>,  // len = ndof
    m_fp: Vec<f64>,  // len = ndof
    c_fp: Vec<f64>,  // len = ndof

    // ---- 主动柔性体：激励参数 ----
    anchor_amplitude: f64,   // 锚点横向振幅（格子单位；0 = 不激励）
    anchor_frequency: f64,   // 锚点激励频率（步^{-1}）
    anchor_phase:     f64,   // 锚点激励初始相位（rad）
    tip_force_amp:    f64,   // 自由端横向外力幅值
    tip_force_freq:   f64,   // 外力频率（步^{-1}）
    tip_force_phase:  f64,   // 外力初始相位（rad）

    // ---- Newmark-β 参数 ----
    a0: f64, a1: f64, a2: f64, a3: f64, a4: f64, a5: f64,
    dt: f64,  // 时间步长（保存用于时刻追踪）

    // ---- 当前状态 ----
    q:         Vec<f64>,   // 位移 [w_1, θ_1, w_2, θ_2, ..., w_n, θ_n]
    qdot:      Vec<f64>,   // 速度
    qddot:     Vec<f64>,   // 加速度
    current_t: f64,        // 当前格子时刻（步数；初始 0，每次 advance() 递增 dt）
}

impl BeamSolver {
    /// 从配置创建梁求解器。
    ///
    /// # 参数
    /// - `cfg`：`FlexibleBodyConfig`（梁参数）  
    /// - `dt`：格子时间步长
    pub fn new(cfg: &FlexibleBodyConfig, dt: f64) -> Self {
        let n_elem = cfg.n_elements.max(1) as usize;
        let ndof   = 2 * n_elem;                   // 自由自由度数
        let le     = cfg.length / n_elem as f64;   // 单元长度
        let ei     = cfg.young_modulus * cfg.second_moment; // 弯曲刚度 EI
        let rho_a  = cfg.linear_density;           // 线密度 ρA
        let alpha_r = cfg.damping;                 // Rayleigh 质量阻尼系数

        // ── 组装全局刚度矩阵（ndof × ndof，含约束消去）─────────────────────
        // 全局 DOF 编号：[w_0,θ_0, w_1,θ_1, ..., w_n,θ_n]
        // clamped BC：DOF 0,1 固定 → 去掉后自由 DOF 从 index 2 起，自由 DOF j → 全局 j+2
        let n_full = 2 * (n_elem + 1);
        let mut k_full = vec![0.0_f64; n_full * n_full];
        let mut m_full = vec![0.0_f64; n_full * n_full];

        for e in 0..n_elem {
            // 单元节点全局 DOF（4 个）
            let dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1];

            // 单元刚度矩阵 k_e（4×4）
            let c1 = 12.0 * ei / (le * le * le);
            let c2 =  6.0 * ei / (le * le);
            let c3 =  4.0 * ei / le;
            let c4 =  2.0 * ei / le;
            let ke: [f64; 16] = [
                 c1,  c2, -c1,  c2,
                 c2,  c3, -c2,  c4,
                -c1, -c2,  c1, -c2,
                 c2,  c4, -c2,  c3,
            ];

            // 单元一致质量矩阵 m_e（4×4），系数 = ρA·Le/420
            let mc = rho_a * le / 420.0;
            let me: [f64; 16] = [
                 156.0*mc,  22.0*le*mc,   54.0*mc, -13.0*le*mc,
                22.0*le*mc,  4.0*le*le*mc, 13.0*le*mc, -3.0*le*le*mc,
                  54.0*mc,  13.0*le*mc,  156.0*mc, -22.0*le*mc,
               -13.0*le*mc, -3.0*le*le*mc, -22.0*le*mc,  4.0*le*le*mc,
            ];

            // 组装到全局矩阵
            for i in 0..4 {
                for j in 0..4 {
                    k_full[dofs[i] * n_full + dofs[j]] += ke[i * 4 + j];
                    m_full[dofs[i] * n_full + dofs[j]] += me[i * 4 + j];
                }
            }
        }

        // ── 提取自由 DOF（去掉前两行/列）─────────────────────────────────
        let mut k_mat = vec![0.0_f64; ndof * ndof];
        let mut m_mat = vec![0.0_f64; ndof * ndof];
        for i in 0..ndof {
            for j in 0..ndof {
                k_mat[i * ndof + j] = k_full[(i+2) * n_full + (j+2)];
                m_mat[i * ndof + j] = m_full[(i+2) * n_full + (j+2)];
            }
        }

        // ── 提取锚点（DOF 0 = w_0）与自由 DOF 的耦合列向量（基础激励用）──
        // k_fp[i] = K_full[(i+2)*n_full + 0]，即 K 第 0 列（w_0 列）中自由 DOF 部分
        // 在基础激励支持运动公式中：F_equiv = −(K_fp·w_0 + M_fp·ẅ_0 + C_fp·ẇ_0)
        let k_fp: Vec<f64> = (0..ndof).map(|i| k_full[(i+2) * n_full + 0]).collect();
        let m_fp: Vec<f64> = (0..ndof).map(|i| m_full[(i+2) * n_full + 0]).collect();

        // ── Rayleigh 阻尼矩阵 C = α·M（质量比例）──────────────────────────
        let c_mat: Vec<f64> = m_mat.iter().map(|&v| alpha_r * v).collect();
        let c_fp:  Vec<f64> = m_fp.iter().map(|&v| alpha_r * v).collect();

        // ── Newmark-β 参数（β=0.25，γ=0.5，无条件稳定）───────────────────
        let beta_nm  = 0.25_f64;
        let gamma_nm = 0.50_f64;
        let a0 = 1.0 / (beta_nm * dt * dt);
        let a1 = gamma_nm / (beta_nm * dt);
        let a2 = 1.0 / (beta_nm * dt);
        let a3 = 0.5 / beta_nm - 1.0;
        let a4 = gamma_nm / beta_nm - 1.0;
        let a5 = dt * (gamma_nm / (2.0 * beta_nm) - 1.0);

        // ── 组装有效刚度矩阵 K_eff = a0·M + a1·C + K ──────────────────────
        let mut k_eff = vec![0.0_f64; ndof * ndof];
        for i in 0..ndof * ndof {
            k_eff[i] = a0 * m_mat[i] + a1 * c_mat[i] + k_mat[i];
        }

        // ── LU 因子化（列主元 Gauss，原地修改）────────────────────────────
        let (k_eff_lu, lu_piv) = lu_factor(&k_eff, ndof);

        BeamSolver {
            anchor_x: cfg.anchor_x,
            anchor_y: cfg.anchor_y,
            cos_ori:  cfg.orientation.cos(),
            sin_ori:  cfg.orientation.sin(),
            elem_len: le,
            n_elem,
            k_eff_lu,
            lu_piv,
            m_mat,
            c_mat,
            k_mat,
            ndof,
            k_fp,
            m_fp,
            c_fp,
            anchor_amplitude: cfg.anchor_amplitude,
            anchor_frequency: cfg.anchor_frequency,
            anchor_phase:     cfg.anchor_phase,
            tip_force_amp:    cfg.tip_force_amplitude,
            tip_force_freq:   cfg.tip_force_frequency,
            tip_force_phase:  cfg.tip_force_phase,
            a0, a1, a2, a3, a4, a5,
            dt,
            q:         vec![0.0; ndof],
            qdot:      vec![0.0; ndof],
            qddot:     vec![0.0; ndof],
            current_t: 0.0,
        }
    }

    // ---- 查询接口 --------------------------------------------------------

    /// 当前标记点全局位置 `(x, y)`。
    ///
    /// 包含基础激励引起的锚点位移偏移：绝对横向位移 = w_anchor(t) + w_rel(s)。
    ///
    /// `arc_s`：各标记点弧坐标（格子单位，长度 N）。
    pub fn marker_positions(&self, arc_s: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = arc_s.len();
        let mut px = Vec::with_capacity(n);
        let mut py = Vec::with_capacity(n);
        let w_anc = self.anchor_transverse(self.current_t);
        for &s in arc_s {
            let w = w_anc + self.interp_w(s);
            // P = anchor + s·e₁ + w·e₂
            px.push(self.anchor_x + s * self.cos_ori - w * self.sin_ori);
            py.push(self.anchor_y + s * self.sin_ori + w * self.cos_ori);
        }
        (px, py)
    }

    /// 当前标记点速度 `(ux, uy)` = ẇ_abs(s)·e₂。
    ///
    /// 包含基础激励锚点速度：ẇ_abs = ẇ_anchor(t) + ẇ_rel(s)。
    ///
    /// `arc_s`：各标记点弧坐标（格子单位）。
    pub fn marker_velocities(&self, arc_s: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = arc_s.len();
        let mut vx = Vec::with_capacity(n);
        let mut vy = Vec::with_capacity(n);
        let wdot_anc = self.anchor_transverse_vel(self.current_t);
        for &s in arc_s {
            let wdot = wdot_anc + self.interp_wdot(s);
            vx.push(-wdot * self.sin_ori);
            vy.push( wdot * self.cos_ori);
        }
        (vx, vy)
    }

    // ---- 时间推进 --------------------------------------------------------

    /// 推进一个时间步。
    ///
    /// # 参数
    /// - `ibm_fx`, `ibm_fy`：IBM 施加于**流体**的逐标记点力（`ms.get_forces()` 输出）；
    ///   结构所受反作用力在内部取负。
    /// - `arc_s`：标记点弧坐标（长度与 `ibm_fx/fy` 相同）。
    /// - `dt`：时间步长（当前未使用，步进系数已在 `new()` 中预计算）。
    pub fn advance(&mut self, ibm_fx: &[f64], ibm_fy: &[f64], arc_s: &[f64]) {
        let ndof = self.ndof;

        // ── Step 1：组装外力向量（标记点 IBM 力 → 节点横向载荷）────────────
        let f_ext = self.assemble_force(ibm_fx, ibm_fy, arc_s);

        // ── Step 2：Newmark-β 预测值 ────────────────────────────────────────
        // q_pred  = q + dt·qdot + dt²·(0.5−β)·qddot  已内含在 K_eff 中的 a0/a1 系数里
        // 直接使用标准 Newmark 公式：
        //   K_eff · q_{n+1} = F_{n+1}
        //              + M·(a0·q_n + a2·qdot_n + a3·qddot_n)
        //              + C·(a1·q_n + a4·qdot_n + a5·qddot_n)
        let mut rhs = vec![0.0_f64; ndof];
        for i in 0..ndof {
            let mut m_term = 0.0_f64;
            let mut c_term = 0.0_f64;
            for j in 0..ndof {
                let m = self.m_mat[i * ndof + j];
                let c = self.c_mat[i * ndof + j];
                m_term += m * (self.a0 * self.q[j] + self.a2 * self.qdot[j] + self.a3 * self.qddot[j]);
                c_term += c * (self.a1 * self.q[j] + self.a4 * self.qdot[j] + self.a5 * self.qddot[j]);
            }
            rhs[i] = f_ext[i] + m_term + c_term;
        }

        // ── Step 3：解线性方程 K_eff · q_{n+1} = rhs ────────────────────────
        let q_new = lu_solve(&self.k_eff_lu, &self.lu_piv, &rhs, ndof);

        // ── Step 4：更新速度、加速度 ─────────────────────────────────────────
        // Newmark-β 公式（β=0.25，γ=0.5）：
        //   qddot_{n+1} = a0·(q_{n+1}−q_n) − a2·qdot_n − a3·qddot_n
        //   qdot_{n+1}  = qdot_n + dt·[(1−γ)·qddot_n + γ·qddot_{n+1}]
        //              = qdot_n + dt_step·[0.5·qddot_n + 0.5·qddot_{n+1}]
        // 其中 dt_step = a2/a0（= dt，由 a0=1/(β·dt²)，a2=1/(β·dt) 精确复原）
        let dt_step = self.a2 / self.a0;
        let mut qddot_new = vec![0.0_f64; ndof];
        let mut qdot_new  = vec![0.0_f64; ndof];
        for i in 0..ndof {
            qddot_new[i] = self.a0 * (q_new[i] - self.q[i])
                         - self.a2 * self.qdot[i]
                         - self.a3 * self.qddot[i];
        }
        for i in 0..ndof {
            qdot_new[i] = self.qdot[i]
                + dt_step * (0.5 * self.qddot[i] + 0.5 * qddot_new[i]);
        }

        self.q     = q_new;
        self.qdot  = qdot_new;
        self.qddot = qddot_new;
    }

    // ---- 内部辅助 --------------------------------------------------------

    /// 组装外力向量（IBM 标记点力 → 节点横向载荷，通过 Hermite 形函数分配）。
    fn assemble_force(&self, ibm_fx: &[f64], ibm_fy: &[f64], arc_s: &[f64]) -> Vec<f64> {
        let mut f = vec![0.0_f64; self.ndof];
        let le = self.elem_len;
        let cos_o = self.cos_ori;
        let sin_o = self.sin_ori;

        for (k, &s) in arc_s.iter().enumerate() {
            // IBM 施加于流体的力；结构反力取负
            // 结构所受横向分量：f⊥ = ibm_fx·sin θ − ibm_fy·cos θ
            let f_perp = ibm_fx[k] * sin_o - ibm_fy[k] * cos_o;

            // 定位元素 e 和局部坐标 ξ ∈ [0,1]
            let s_clamped = s.clamp(0.0, le * self.n_elem as f64);
            let e_raw = (s_clamped / le).floor() as usize;
            let e = e_raw.min(self.n_elem - 1);
            let xi = (s_clamped - e as f64 * le) / le;

            // Hermite 形函数 [N1, N2, N3, N4]
            let xi2 = xi * xi;
            let xi3 = xi2 * xi;
            let n1 = 1.0 - 3.0*xi2 + 2.0*xi3;
            let n2 = le * (xi - 2.0*xi2 + xi3);
            let n3 = 3.0*xi2 - 2.0*xi3;
            let n4 = le * (-xi2 + xi3);

            // 全局 DOF 对应节点 e 和 e+1（全局索引）
            //   节点 e   : full_dof = 2*e   → free_dof = 2*e − 2   (当 e ≥ 1)
            //   节点 e+1 : full_dof = 2*(e+1) → free_dof = 2*(e+1) − 2
            // 如果 e == 0（含 clamped 端），节点 0 的自由度已固定，跳过
            let n_vals = [n1, n2, n3, n4];
            for (local_i, &nval) in n_vals.iter().enumerate() {
                // local_i=0,1 → 节点 e；local_i=2,3 → 节点 e+1
                let (node, dof_in_node) = if local_i < 2 { (e, local_i) } else { (e+1, local_i-2) };
                if node == 0 { continue; }  // clamped 端，DOF 固定
                let free_idx = 2 * (node - 1) + dof_in_node;
                if free_idx < self.ndof {
                    f[free_idx] += nval * f_perp;
                }
            }
        }
        f
    }

    /// 横向位移插值 w(s)。
    fn interp_w(&self, s: f64) -> f64 {
        let le = self.elem_len;
        let s_clamped = s.clamp(0.0, le * self.n_elem as f64);
        let e_raw = (s_clamped / le).floor() as usize;
        let e = e_raw.min(self.n_elem - 1);
        let xi = (s_clamped - e as f64 * le) / le;
        let xi2 = xi * xi;
        let xi3 = xi2 * xi;
        let n1 = 1.0 - 3.0*xi2 + 2.0*xi3;
        let n2 = le * (xi - 2.0*xi2 + xi3);
        let n3 = 3.0*xi2 - 2.0*xi3;
        let n4 = le * (-xi2 + xi3);

        let w_e   = if e == 0 { 0.0 } else { self.q[2*(e-1)] };
        let th_e  = if e == 0 { 0.0 } else { self.q[2*(e-1)+1] };
        let w_e1  = if e+1 == 0 { 0.0 } else { self.q[2*e] };
        let th_e1 = if e+1 == 0 { 0.0 } else { self.q[2*e+1] };

        n1*w_e + n2*th_e + n3*w_e1 + n4*th_e1
    }

    /// 横向速度插值 ẇ(s)。
    fn interp_wdot(&self, s: f64) -> f64 {
        let le = self.elem_len;
        let s_clamped = s.clamp(0.0, le * self.n_elem as f64);
        let e_raw = (s_clamped / le).floor() as usize;
        let e = e_raw.min(self.n_elem - 1);
        let xi = (s_clamped - e as f64 * le) / le;
        let xi2 = xi * xi;
        let xi3 = xi2 * xi;
        let n1 = 1.0 - 3.0*xi2 + 2.0*xi3;
        let n2 = le * (xi - 2.0*xi2 + xi3);
        let n3 = 3.0*xi2 - 2.0*xi3;
        let n4 = le * (-xi2 + xi3);

        let wd_e   = if e == 0 { 0.0 } else { self.qdot[2*(e-1)] };
        let thd_e  = if e == 0 { 0.0 } else { self.qdot[2*(e-1)+1] };
        let wd_e1  = if e+1 == 0 { 0.0 } else { self.qdot[2*e] };
        let thd_e1 = if e+1 == 0 { 0.0 } else { self.qdot[2*e+1] };

        n1*wd_e + n2*thd_e + n3*wd_e1 + n4*thd_e1
    }
}

// ===========================================================================
// 密集矩阵 LU 分解（列主元 Gauss，用于小型方程组）
// ===========================================================================

/// 列主元 LU 分解（原地，行优先存储）。
///
/// 返回 `(lu, piv)`：`lu` 为含 L/U 因子的密集矩阵，`piv[i]` 为行置换。
fn lu_factor(a: &[f64], n: usize) -> (Vec<f64>, Vec<usize>) {
    let mut lu = a.to_vec();
    let mut piv: Vec<usize> = (0..n).collect();

    for k in 0..n {
        // 列主元：在 k..n 中找最大绝对值行
        let mut max_val = lu[k * n + k].abs();
        let mut max_row = k;
        for i in (k+1)..n {
            let v = lu[i * n + k].abs();
            if v > max_val { max_val = v; max_row = i; }
        }
        // 行交换
        if max_row != k {
            piv.swap(k, max_row);
            for j in 0..n {
                lu.swap(k * n + j, max_row * n + j);
            }
        }
        // 消去
        let pivot = lu[k * n + k];
        if pivot.abs() < 1e-300 { continue; }  // 奇异，跳过
        for i in (k+1)..n {
            lu[i * n + k] /= pivot;
            for j in (k+1)..n {
                let luk = lu[i * n + k];
                let ukj = lu[k * n + j];
                lu[i * n + j] -= luk * ukj;
            }
        }
    }
    (lu, piv)
}

/// 用已因子化的 LU 矩阵解线性方程 A·x = b。
fn lu_solve(lu: &[f64], piv: &[usize], b: &[f64], n: usize) -> Vec<f64> {
    let mut x: Vec<f64> = (0..n).map(|i| b[piv[i]]).collect();
    // 前代（L·y = b）
    for i in 0..n {
        for j in 0..i {
            x[i] -= lu[i * n + j] * x[j];
        }
    }
    // 回代（U·x = y）
    for i in (0..n).rev() {
        for j in (i+1)..n {
            x[i] -= lu[i * n + j] * x[j];
        }
        x[i] /= lu[i * n + i];
    }
    x
}

// ===========================================================================
// 辅助：计算丝状体标记点的弧坐标
// ===========================================================================

/// 计算均匀分布在梁上的 `n` 个标记点的弧坐标 `s_k = k * L / (n-1)`（格子单位）。
///
/// 用于在 `setup_ibm_bodies` 中初始化 `IbmEntry::beam_s`。
pub fn uniform_arc_s(n: usize, length: f64) -> Vec<f64> {
    if n <= 1 { return vec![0.0]; }
    (0..n).map(|k| k as f64 * length / (n - 1) as f64).collect()
}

// ===========================================================================
// 单元测试
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::PrescribedMotionConfig;

    #[test]
    fn test_prescribed_translate() {
        let cfg = PrescribedMotionConfig {
            mode: "translate".to_string(),
            velocity_x: 0.05,
            velocity_y: 0.0,
            ..Default::default()
        };
        let pm = PrescribedMotion::new(&cfg);
        let (ux, uy, om) = pm.eval_rigid(100.0);
        assert!((ux - 0.05).abs() < 1e-14);
        assert_eq!(uy, 0.0);
        assert_eq!(om, 0.0);
    }

    #[test]
    fn test_prescribed_oscillate_y() {
        let cfg = PrescribedMotionConfig {
            mode:      "oscillate_y".to_string(),
            amplitude: 1.0,
            frequency: 1.0 / (2.0 * PI),
            ..Default::default()
        };
        let pm = PrescribedMotion::new(&cfg);
        // t=0: cos(0)=1, velocity = A*2π*f = A*2π*(1/2π) = A = 1.0
        let (_, uy, _) = pm.eval_rigid(0.0);
        assert!((uy - 1.0).abs() < 1e-12, "uy={}", uy);
    }

    #[test]
    fn test_beam_solver_static_deflection() {
        // 悬臂梁端点集中载荷 P，端点挠度 = P·L³/(3EI)
        let ei = 1000.0_f64;
        let length = 10.0_f64;
        let n_elem = 10_u32;
        let rho_a = 1e-6;      // 接近无质量（静力学极限）
        let dt = 0.001;

        let cfg = FlexibleBodyConfig {
            anchor_x: 0.0,
            anchor_y: 0.0,
            orientation: 0.0,
            length,
            young_modulus: ei,  // E·I = ei (此处 second_moment=1)
            second_moment: 1.0,
            linear_density: rho_a,
            n_elements: n_elem,
            damping: 100.0,     // 大阻尼 → 快收敛到静态解
        };

        let mut beam = BeamSolver::new(&cfg, dt);
        let arc_s = uniform_arc_s(n_elem as usize + 1, length);

        // 在自由端施加纯横向力 P=1（只有一个标记点在末端）
        let p = 1.0_f64;
        let ibm_fx = vec![0.0_f64; arc_s.len()];
        // IBM 力取反得结构力（结构受 +y 力 → IBM_fy = −1）
        let ibm_fy = {
            let mut v = vec![0.0_f64; arc_s.len()];
            // 施力在自由端（最后一个标记点）
            // 结构受 +e₂ 力 p → IBM_fy = −p（取反）（e₂=(0,1)，θ=0）
            *v.last_mut().unwrap() = -p;
            v
        };

        // 积分足够多步使系统收敛（高阻尼快收敛）
        for _ in 0..50_000 {
            beam.advance(&ibm_fx, &ibm_fy, &arc_s);
        }

        let (_, py) = beam.marker_positions(&arc_s);
        let tip_w = py.last().copied().unwrap();
        let analytic = p * length * length * length / (3.0 * ei);
        let rel_err = (tip_w - analytic).abs() / analytic;
        assert!(
            rel_err < 0.02,
            "tip deflection: numeric={:.6}  analytic={:.6}  rel_err={:.4}",
            tip_w, analytic, rel_err
        );
    }
}
