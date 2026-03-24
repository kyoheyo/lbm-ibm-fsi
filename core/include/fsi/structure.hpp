#pragma once
#include <vector>
#include <cmath>

namespace fsi {

// ---------------------------------------------------------------------------
// 柔性 Euler-Bernoulli 梁的材料参数
// ---------------------------------------------------------------------------
struct BeamParams {
    double E;           ///< 杨氏模量
    double I;           ///< 截面惯性矩
    double rho_s;       ///< 结构密度
    double A;           ///< 截面面积
    double length;      ///< 静止长度
    int    n_elements;  ///< 梁单元数量
};

// ---------------------------------------------------------------------------
// 单个梁节点的自由度：(x, y, theta)
// ---------------------------------------------------------------------------
struct BeamDOF {
    double x, y;     ///< 当前位置
    double x0, y0;   ///< 参考（未变形）位置
    double theta;    ///< 转角
    double vx, vy;   ///< 速度
    double omega;    ///< 角速度
    double fx, fy;   ///< 外力（来自 IBM）
    double m;        ///< 外弯矩
};

// ---------------------------------------------------------------------------
// Euler-Bernoulli 梁有限元结构求解器
//   采用 Newmark-β 时间积分（β=0.25, γ=0.5 — 无条件稳定）
// ---------------------------------------------------------------------------
class BeamSolver {
public:
    explicit BeamSolver(const BeamParams& params);

    /// 在当前外力作用下将梁推进一个时间步 dt
    void step(double dt);

    /// 访问节点自由度
    [[nodiscard]] const std::vector<BeamDOF>& dofs() const { return dofs_; }
    [[nodiscard]]       std::vector<BeamDOF>& dofs()       { return dofs_; }

    /// 节点数  = n_elements + 1
    [[nodiscard]] int n_nodes() const {
        return params_.n_elements + 1;
    }

private:
    BeamParams           params_;
    std::vector<BeamDOF> dofs_;

    /// 集中质量矩阵（对角块存储为向量）
    std::vector<double> mass_diag_;
    /// 整体刚度矩阵（稀疏 CSR — 对小梁使用密集存储）
    std::vector<double> K_;

    void assemble_mass();
    void assemble_stiffness();

    /// 通过带部分主元的密集 Gauss 消元法求解  K_eff * x = rhs
    void solve_newmark(const std::vector<double>& K_eff,
                       const std::vector<double>& rhs,
                       std::vector<double>& x);
};


// ===========================================================================
// 2D 刚体运动结构求解器
//
// 参考：Suzuki K. & Inamuro T. (2011) Computers & Fluids 49:173–187
//       "Effect of internal mass in the simulation of a moving body
//        by the immersed boundary method"
//
// 实现四种内部质量效应处理方案（Section 3.3）：
//   (A)  无内部质量效应：Fin = 0, Tin = 0
//   (B-1) Uhlmann 刚体近似（隐式）：用 Meff = (1-qf/qb)*M 替代 M
//   (B-2) Feng 刚体近似（显式）：Fin ≈ (qf/qb)*M*(Uc(t)-Uc(t-dt))/dt
//   (C)  拉格朗日点近似：通过内部拉格朗日点直接计算内部动量
//
// 时间积分：一阶 Euler 法（Section 3.3）
//   Uc(t+dt) = Uc(t) + (dt/M_eff) * [Ftot + Fin]
//   omega(t+dt) = omega(t) + (dt/Izz_eff) * [Ttot + Tin]
//   Xc(t+dt) = Xc(t) + dt * Uc(t)
//   theta(t+dt) = theta(t) + dt * omega(t)
// ===========================================================================

/// 内部质量效应处理方案
enum class InternalMassScheme {
    None,               ///< (A)   忽略内部质量效应
    UhlmannRigidBody,   ///< (B-1) Uhlmann 隐式刚体近似
    FengRigidBody,      ///< (B-2) Feng 显式刚体近似
    LagrangianPoints,   ///< (C)   拉格朗日点近似（最精确）
};

// ---------------------------------------------------------------------------
// 2D 刚体参数
// ---------------------------------------------------------------------------
struct RigidBodyParams2D {
    double mass;         ///< 刚体质量 M（格子单位）
    double inertia;      ///< 转动惯量 Izz（格子单位）
    double rho_b;        ///< 刚体密度 ρ_b
    double rho_f;        ///< 流体密度 ρ_f（通常 = 1.0 格子单位）
    double cx0;          ///< 初始质心 x 坐标
    double cy0;          ///< 初始质心 y 坐标
    InternalMassScheme scheme = InternalMassScheme::None;
    /// 是否为封闭结构（封闭结构才有内部质量效应）
    bool is_closed = false;
};

// ---------------------------------------------------------------------------
// 2D 刚体内部拉格朗日点（仅方案 C 使用）
// ---------------------------------------------------------------------------
struct InternalPoint2D {
    double x0, y0;   ///< 参考坐标（相对质心，体固坐标系）
    double x,  y;    ///< 当前绝对坐标
    double ux, uy;   ///< 插值所得流体速度
    double dv;       ///< 体积元 = (Δx)^d（格子单位为 1.0）
};

// ---------------------------------------------------------------------------
// 2D 刚体状态（完整运动自由度）
// ---------------------------------------------------------------------------
struct RigidBodyState2D {
    // 质心运动
    double cx  = 0.0, cy  = 0.0;   ///< 质心位置
    double ux  = 0.0, uy  = 0.0;   ///< 质心速度
    double theta  = 0.0;            ///< 旋转角（弧度，逆时针）
    double omega  = 0.0;            ///< 角速度（rad/s）

    // 上一步状态（供 Feng 方案使用）
    double prev_ux = 0.0, prev_uy = 0.0;
    double prev_omega = 0.0;

    // 上一步内部动量（供 Lagrangian 方案使用）
    double prev_pin_x = 0.0, prev_pin_y = 0.0;
    double prev_lin   = 0.0;
};

// ===========================================================================
// 2D 刚体求解器
//
// 使用方法：
//   1. 构造时提供参数（包括方案选择）和边界标记点布局（相对质心的参考坐标）。
//   2. 每步调用流程：
//      a. [仅方案 C] interpolate_internal_velocity(grid, dx) — 插值内部点速度
//      b. set_ibm_forces(fx, fy, torque)                     — 设置 IBM 力/力矩
//      c. advance(dt)                                          — 推进刚体运动
//      d. update_boundary_markers(markers_x, markers_y, ...) — 更新标记点位置/速度
// ===========================================================================
class RigidBodySolver2D {
public:
    /// @param params       刚体参数（含内部质量方案选择）
    /// @param ref_x        边界标记点参考 x（相对质心，体固坐标系，长度 N）
    /// @param ref_y        边界标记点参考 y（相对质心，体固坐标系，长度 N）
    /// @param int_ref_x    内部标记点参考 x（方案 C，可为空）
    /// @param int_ref_y    内部标记点参考 y（方案 C，可为空）
    RigidBodySolver2D(const RigidBodyParams2D& params,
                      const std::vector<double>& ref_x,
                      const std::vector<double>& ref_y,
                      const std::vector<double>& int_ref_x = {},
                      const std::vector<double>& int_ref_y = {});

    // -------------------------------------------------------------------
    // 主调用接口
    // -------------------------------------------------------------------

    /// [仅方案 C] 从已插值的内部点速度（写入 internal_pts_[k].ux/uy）中
    /// 计算当前内部动量 Pin(t) 和角动量 Lin(t)。
    /// 调用方须先将插值速度写入 internal_pts_[k].ux/uy。
    void compute_internal_momentum();

    /// 设置本步由 IBM 体力算出的总力/总力矩（公式 28, 29）：
    ///   Ftot = -Σ_x g(x) * dx^2
    ///   Ttot = -Σ_x (x - Xc) × g(x) * dx^2
    void set_ibm_forces(double fx, double fy, double torque);

    /// 推进刚体一个时间步（公式 26, 27, A.6, 并更新边界标记位置）
    void advance(double dt);

    /// 更新所有边界标记点的绝对坐标与速度
    ///   Xk = Xc + R(theta) * BXk
    ///   Uk = Uc + omega × BXk  (2D：Uk = Uc + omega * [-BXk_y, BXk_x])
    void update_boundary_markers();

    // -------------------------------------------------------------------
    // 访问器
    // -------------------------------------------------------------------
    [[nodiscard]] const RigidBodyState2D& state() const { return state_; }
    [[nodiscard]]       RigidBodyState2D& state()       { return state_; }

    [[nodiscard]] const RigidBodyParams2D& params() const { return params_; }

    /// 边界标记点数量
    [[nodiscard]] int n_boundary() const {
        return static_cast<int>(bnd_ref_x_.size());
    }

    /// 边界标记点当前绝对坐标
    [[nodiscard]] const std::vector<double>& boundary_x()  const { return bnd_x_; }
    [[nodiscard]] const std::vector<double>& boundary_y()  const { return bnd_y_; }
    [[nodiscard]] const std::vector<double>& boundary_ux() const { return bnd_ux_; }
    [[nodiscard]] const std::vector<double>& boundary_uy() const { return bnd_uy_; }

    /// 内部拉格朗日点（方案 C）
    [[nodiscard]]       std::vector<InternalPoint2D>& internal_pts()       { return internal_pts_; }
    [[nodiscard]] const std::vector<InternalPoint2D>& internal_pts() const { return internal_pts_; }

    /// 当前步内部线动量（方案 C 用）
    [[nodiscard]] double pin_x() const { return pin_x_; }
    [[nodiscard]] double pin_y() const { return pin_y_; }
    [[nodiscard]] double lin()   const { return lin_; }

private:
    RigidBodyParams2D params_;
    RigidBodyState2D  state_;

    // 边界标记点（体固参考坐标）
    std::vector<double> bnd_ref_x_, bnd_ref_y_;
    // 边界标记点当前绝对坐标及速度
    std::vector<double> bnd_x_, bnd_y_, bnd_ux_, bnd_uy_;

    // 内部拉格朗日点（方案 C）
    std::vector<InternalPoint2D> internal_pts_;

    // 当前步内部动量（方案 C 更新）
    double pin_x_ = 0.0, pin_y_ = 0.0, lin_ = 0.0;

    // 本步 IBM 总力/力矩（由 set_ibm_forces 写入）
    double ibm_fx_ = 0.0, ibm_fy_ = 0.0, ibm_torque_ = 0.0;

    // 初始化标志：Feng/Lagrangian 方案的"上一步"在第一步前需要初始化
    bool first_step_ = true;
};

} // namespace fsi
