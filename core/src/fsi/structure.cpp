#include "fsi/structure.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace fsi {

// ---------------------------------------------------------------------------
// BeamSolver 构造函数：设置初始位置，组装质量矩阵和刚度矩阵
// ---------------------------------------------------------------------------
BeamSolver::BeamSolver(const BeamParams& params)
    : params_(params)
{
    const int n = n_nodes();
    dofs_.resize(n);

    // 沿 x 轴设置参考位置
    const double dl = params_.length / params_.n_elements;
    for (int i = 0; i < n; ++i) {
        dofs_[i].x = dofs_[i].x0 = i * dl;
        dofs_[i].y = dofs_[i].y0 = 0.0;
        dofs_[i].theta = 0.0;
        dofs_[i].vx    = 0.0;
        dofs_[i].vy    = 0.0;
        dofs_[i].omega = 0.0;
        dofs_[i].fx    = 0.0;
        dofs_[i].fy    = 0.0;
        dofs_[i].m     = 0.0;
    }

    assemble_mass();
    assemble_stiffness();
}

// ---------------------------------------------------------------------------
// 集中质量矩阵：m_i = ρ_s * A * l_element / 2（每侧分配一半给相邻节点）
// ---------------------------------------------------------------------------
void BeamSolver::assemble_mass()
{
    const int n = n_nodes();
    const double dl = params_.length / params_.n_elements;
    const double m_node = params_.rho_s * params_.A * dl;

    mass_diag_.assign(n * 3, 0.0);   // 每节点 3 个自由度：x、y、theta
    for (int i = 0; i < n; ++i) {
        mass_diag_[i * 3 + 0] = m_node;   // 平动 x 方向
        mass_diag_[i * 3 + 1] = m_node;   // 平动 y 方向
        // 转动惯量：I_rot ≈ ρ_s * I * dl（每单元，均分给两端节点）
        mass_diag_[i * 3 + 2] = params_.rho_s * params_.I * dl;
    }
}

// ---------------------------------------------------------------------------
// Euler-Bernoulli 梁的整体刚度矩阵（密集存储，大小 3n × 3n）
// 仅组装弯曲（EI）项；轴向（EA）项省略（不可伸长梁假设）
// ---------------------------------------------------------------------------
void BeamSolver::assemble_stiffness()
{
    const int n = n_nodes();
    const int dof = 3 * n;
    K_.assign(dof * dof, 0.0);

    const double dl = params_.length / params_.n_elements;
    const double EI = params_.E * params_.I;
    const double l  = dl;
    const double l2 = l * l;
    const double l3 = l2 * l;

    // 局部 4×4 Euler-Bernoulli 弯曲刚度（每单元两端各有 v、θ 两个自由度）
    // 映射到自由度 [y_i, theta_i, y_j, theta_j]
    const double ke[4][4] = {
        { 12*EI/l3,  6*EI/l2, -12*EI/l3,  6*EI/l2},
        {  6*EI/l2,  4*EI/l,   -6*EI/l2,  2*EI/l },
        {-12*EI/l3, -6*EI/l2,  12*EI/l3, -6*EI/l2},
        {  6*EI/l2,  2*EI/l,   -6*EI/l2,  4*EI/l },
    };

    for (int e = 0; e < params_.n_elements; ++e) {
        // 局部自由度在整体矩阵中的全局编号：[y_i, theta_i, y_j, theta_j]
        const int local_dof[4] = {
            e * 3 + 1,    // y_i
            e * 3 + 2,    // theta_i
            (e+1)*3 + 1,  // y_j
            (e+1)*3 + 2,  // theta_j
        };

        // 将单元刚度矩阵组装到整体刚度矩阵
        for (int r = 0; r < 4; ++r) {
            for (int c = 0; c < 4; ++c) {
                K_[local_dof[r] * dof + local_dof[c]] += ke[r][c];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Newmark-β 时间积分（β=0.25, γ=0.5 — 平均加速度法，无条件稳定）
// 求解：M*a_{n+1} + K*d_{n+1} = F_{n+1}
// 其中：d_{n+1} = d_n + dt*v_n + dt²[(0.5-β)*a_n + β*a_{n+1}]
//       v_{n+1} = v_n + dt[(1-γ)*a_n + γ*a_{n+1}]
// ---------------------------------------------------------------------------
void BeamSolver::step(double dt)
{
    const int n  = n_nodes();
    const int dof = 3 * n;

    constexpr double beta  = 0.25;
    constexpr double gamma = 0.5;

    // 将当前状态展开为平坦向量
    std::vector<double> d(dof), v(dof), a(dof), F(dof);
    for (int i = 0; i < n; ++i) {
        d[i*3+0] = dofs_[i].x;
        d[i*3+1] = dofs_[i].y;
        d[i*3+2] = dofs_[i].theta;
        v[i*3+0] = dofs_[i].vx;
        v[i*3+1] = dofs_[i].vy;
        v[i*3+2] = dofs_[i].omega;
        a[i*3+0] = 0.0;  // 上一步加速度（简化处理）
        a[i*3+1] = 0.0;
        a[i*3+2] = 0.0;
        F[i*3+0] = dofs_[i].fx;
        F[i*3+1] = dofs_[i].fy;
        F[i*3+2] = dofs_[i].m;
    }

    // 有效刚度：K_eff = M/(β*dt²) + K
    // 右端项：F_eff = F + M/β/dt² * (d + dt*v + dt²*(0.5-β)*a)
    std::vector<double> K_eff(dof * dof);
    for (int r = 0; r < dof; ++r) {
        for (int c = 0; c < dof; ++c) {
            K_eff[r * dof + c] = K_[r * dof + c];
        }
        // 在对角线上加上质量贡献
        K_eff[r * dof + r] += mass_diag_[r] / (beta * dt * dt);
    }

    // 预测位移和右端项
    std::vector<double> d_pred(dof);
    std::vector<double> rhs(dof);
    for (int i = 0; i < dof; ++i) {
        d_pred[i] = d[i] + dt * v[i] + dt * dt * (0.5 - beta) * a[i];
        rhs[i] = F[i] + mass_diag_[i] / (beta * dt * dt) * d_pred[i];
    }

    // 求解 K_eff * d_{n+1} = rhs（密集 Gauss 消元）
    std::vector<double> d_new(dof);
    solve_newmark(K_eff, rhs, d_new);

    // 更新加速度和速度
    std::vector<double> a_new(dof), v_new(dof);
    for (int i = 0; i < dof; ++i) {
        a_new[i] = (d_new[i] - d_pred[i]) / (beta * dt * dt);
        v_new[i] = v[i] + dt * ((1.0 - gamma) * a[i] + gamma * a_new[i]);
    }

    // 将结果写回节点自由度
    for (int i = 0; i < n; ++i) {
        dofs_[i].x     = d_new[i*3+0];
        dofs_[i].y     = d_new[i*3+1];
        dofs_[i].theta = d_new[i*3+2];
        dofs_[i].vx    = v_new[i*3+0];
        dofs_[i].vy    = v_new[i*3+1];
        dofs_[i].omega = v_new[i*3+2];
    }
}

// ---------------------------------------------------------------------------
// 密集 LU 分解求解：K_eff * x = rhs（带部分主元的 Gauss 消元）
// ---------------------------------------------------------------------------
void BeamSolver::solve_newmark(const std::vector<double>& K_eff,
                                const std::vector<double>& rhs,
                                std::vector<double>& x)
{
    const int dof = static_cast<int>(rhs.size());

    // K_eff 的工作副本
    std::vector<double> A(K_eff);

    // 将右端项复制到 x
    x = rhs;

    // 带部分主元的 Gauss 消元
    for (int k = 0; k < dof; ++k) {
        // 寻找主元
        int max_row = k;
        double max_val = std::abs(A[k * dof + k]);
        for (int i = k + 1; i < dof; ++i) {
            if (std::abs(A[i * dof + k]) > max_val) {
                max_val = std::abs(A[i * dof + k]);
                max_row = i;
            }
        }
        // 交换行
        if (max_row != k) {
            std::swap(x[k], x[max_row]);
            for (int j = 0; j < dof; ++j) {
                std::swap(A[k * dof + j], A[max_row * dof + j]);
            }
        }

        if (std::abs(A[k * dof + k]) < 1e-15) {
            // 矩阵奇异或接近奇异 — 将对应位移置零。
            // 注意：此实现仅支持齐次（零值）约束。
            // 若需要非零规定位移，应在调用本求解器前通过惩罚法
            // 或显式消元施加约束。
            x[k] = 0.0;
            continue;
        }

        // 消元
        for (int i = k + 1; i < dof; ++i) {
            const double factor = A[i * dof + k] / A[k * dof + k];
            x[i] -= factor * x[k];
            for (int j = k; j < dof; ++j) {
                A[i * dof + j] -= factor * A[k * dof + j];
            }
        }
    }

    // 回代
    for (int k = dof - 1; k >= 0; --k) {
        if (std::abs(A[k * dof + k]) < 1e-15) continue;
        for (int j = k + 1; j < dof; ++j) {
            x[k] -= A[k * dof + j] * x[j];
        }
        x[k] /= A[k * dof + k];
    }
}


} // namespace fsi

// =============================================================================
// RigidBodySolver2D 实现
// 参考：Suzuki & Inamuro (2011) Computers & Fluids 49:173-187
// =============================================================================

namespace fsi {

// ---------------------------------------------------------------------------
// 构造函数：初始化体固参考坐标系下的边界/内部标记点
// ---------------------------------------------------------------------------
RigidBodySolver2D::RigidBodySolver2D(
        const RigidBodyParams2D&   params,
        const std::vector<double>& ref_x,
        const std::vector<double>& ref_y,
        const std::vector<double>& int_ref_x,
        const std::vector<double>& int_ref_y)
    : params_(params)
{
    state_.cx = params_.cx0;
    state_.cy = params_.cy0;

    bnd_ref_x_ = ref_x;
    bnd_ref_y_ = ref_y;
    const int nb = static_cast<int>(ref_x.size());
    bnd_x_.resize(nb);
    bnd_y_.resize(nb);
    bnd_ux_.resize(nb);
    bnd_uy_.resize(nb);

    // 方案 (C)：初始化内部拉格朗日点
    const int ni = static_cast<int>(int_ref_x.size());
    internal_pts_.resize(ni);
    for (int k = 0; k < ni; ++k) {
        internal_pts_[k].x0  = int_ref_x[k];
        internal_pts_[k].y0  = int_ref_y[k];
        internal_pts_[k].x   = params_.cx0 + int_ref_x[k];
        internal_pts_[k].y   = params_.cy0 + int_ref_y[k];
        internal_pts_[k].ux  = 0.0;
        internal_pts_[k].uy  = 0.0;
        internal_pts_[k].dv  = 1.0;   // (Δx)^d = 1 格子单位
    }

    // 初始化边界标记点位置（theta=0，平移无旋转）
    update_boundary_markers();

    // 初始化上一步动量（供 Feng/Lagrangian 方案在第一步使用）
    state_.prev_ux    = state_.ux;
    state_.prev_uy    = state_.uy;
    state_.prev_omega = state_.omega;
    pin_x_ = pin_y_ = lin_ = 0.0;
    state_.prev_pin_x = 0.0;
    state_.prev_pin_y = 0.0;
    state_.prev_lin   = 0.0;
}

// ---------------------------------------------------------------------------
// 更新边界标记点绝对坐标与速度
//   Xk = Xc + R(theta) * BXk        (公式 A.8)
//   Uk = Uc + omega × BXk            (公式 A.9，2D 标量叉积)
// ---------------------------------------------------------------------------
void RigidBodySolver2D::update_boundary_markers()
{
    const double cos_t = std::cos(state_.theta);
    const double sin_t = std::sin(state_.theta);

    for (int k = 0; k < n_boundary(); ++k) {
        const double bx = bnd_ref_x_[k];
        const double by = bnd_ref_y_[k];

        // 旋转后绝对位置
        bnd_x_[k]  = state_.cx + cos_t * bx - sin_t * by;
        bnd_y_[k]  = state_.cy + sin_t * bx + cos_t * by;

        // 绝对速度：Uk = Uc + omega * (-rotated_y, rotated_x)
        // = Uc + R(theta)*(-omega*by_ref, omega*bx_ref) 等价写法：
        const double rx = cos_t * bx - sin_t * by;   // 旋转后相对位置 x
        const double ry = sin_t * bx + cos_t * by;   // 旋转后相对位置 y
        bnd_ux_[k] = state_.ux - state_.omega * ry;
        bnd_uy_[k] = state_.uy + state_.omega * rx;
    }

    // 同步内部拉格朗日点位置（速度由外部插值填写）
    const int ni = static_cast<int>(internal_pts_.size());
    for (int k = 0; k < ni; ++k) {
        const double bx = internal_pts_[k].x0;
        const double by = internal_pts_[k].y0;
        internal_pts_[k].x = state_.cx + cos_t * bx - sin_t * by;
        internal_pts_[k].y = state_.cy + sin_t * bx + cos_t * by;
    }
}

// ---------------------------------------------------------------------------
// 计算内部动量（方案 C）
//   Pin(t) = Σ_in u(Xin,t) * DVin         (公式 38)
//   Lin(t) = Σ_in (Xin-Xc) × u(Xin,t) * DVin   (公式 39，2D 标量)
// ---------------------------------------------------------------------------
void RigidBodySolver2D::compute_internal_momentum()
{
    pin_x_ = 0.0;
    pin_y_ = 0.0;
    lin_   = 0.0;
    for (const auto& pt : internal_pts_) {
        pin_x_ += pt.ux * pt.dv;
        pin_y_ += pt.uy * pt.dv;
        // 2D 叉积：(x-cx)*uy - (y-cy)*ux
        const double rx = pt.x - state_.cx;
        const double ry = pt.y - state_.cy;
        lin_ += (rx * pt.uy - ry * pt.ux) * pt.dv;
    }
}

// ---------------------------------------------------------------------------
// 设置本步 IBM 总力与总力矩（公式 28, 29）
// ---------------------------------------------------------------------------
void RigidBodySolver2D::set_ibm_forces(double fx, double fy, double torque)
{
    ibm_fx_     = fx;
    ibm_fy_     = fy;
    ibm_torque_ = torque;
}

// ---------------------------------------------------------------------------
// 推进刚体一个时间步（公式 26, 27, A.6）
//
// 四种内部质量效应方案：
//   (A)  Fin = 0, Tin = 0
//   (B-1) Uhlmann 隐式：Meff = (1-qf/qb)*M, Ieff = (1-qf/qb)*Izz，Fin=Tin=0
//   (B-2) Feng 显式：  Fin = (qf/qb)*M*(Uc(t)-Uc(t-dt))/dt，类似于 Tin
//   (C)  拉格朗日点：  Fin = qf*(Pin(t)-Pin(t-dt))/dt，需先调用
//                       compute_internal_momentum()
// ---------------------------------------------------------------------------
void RigidBodySolver2D::advance(double dt)
{
    const double qfqb = (params_.rho_b > 1e-30)
                        ? params_.rho_f / params_.rho_b
                        : 0.0;

    double M_eff   = params_.mass;
    double Izz_eff = params_.inertia;
    double fin_x   = 0.0, fin_y  = 0.0, tin = 0.0;

    // 是否应用内部质量效应（需要封闭结构且不是 None 方案）
    const bool apply_im = params_.is_closed &&
                          (params_.scheme != InternalMassScheme::None);

    if (apply_im) {
        switch (params_.scheme) {
        // ---------------------------------------------------------------
        case InternalMassScheme::UhlmannRigidBody:
            // (B-1) 公式 33–34：用有效质量替换实际质量，Fin=Tin=0
            M_eff   = (1.0 - qfqb) * params_.mass;
            Izz_eff = (1.0 - qfqb) * params_.inertia;
            fin_x   = 0.0;
            fin_y   = 0.0;
            tin     = 0.0;
            break;

        // ---------------------------------------------------------------
        case InternalMassScheme::FengRigidBody:
            // (B-2) 公式 35–36：显式向后差分
            // Fin(t) = (qf/qb)*M*(Uc(t)-Uc(t-dt))/dt
            // Tin(t) = (qf/qb)*Izz*(omega(t)-omega(t-dt))/dt
            if (!first_step_) {
                fin_x = qfqb * params_.mass    * (state_.ux    - state_.prev_ux)    / dt;
                fin_y = qfqb * params_.mass    * (state_.uy    - state_.prev_uy)    / dt;
                tin   = qfqb * params_.inertia * (state_.omega - state_.prev_omega) / dt;
            }
            break;

        // ---------------------------------------------------------------
        case InternalMassScheme::LagrangianPoints:
            // (C) 公式 40：内部拉格朗日点动量差分
            // Fin(t) = qf * (Pin(t) - Pin(t-dt)) / dt
            // Tin(t) = qf * (Lin(t) - Lin(t-dt)) / dt
            // 注意：调用方须在 advance() 前先调用 compute_internal_momentum()
            if (!first_step_) {
                fin_x = params_.rho_f * (pin_x_ - state_.prev_pin_x) / dt;
                fin_y = params_.rho_f * (pin_y_ - state_.prev_pin_y) / dt;
                tin   = params_.rho_f * (lin_   - state_.prev_lin)   / dt;
            }
            break;

        default:
            break;
        }
    }

    // 保存当前速度供下步 Feng/Lagrangian 使用
    state_.prev_ux    = state_.ux;
    state_.prev_uy    = state_.uy;
    state_.prev_omega = state_.omega;
    state_.prev_pin_x = pin_x_;
    state_.prev_pin_y = pin_y_;
    state_.prev_lin   = lin_;
    first_step_       = false;

    // 牛顿-欧拉积分（公式 26, 27）
    // Uc(t+dt) = Uc(t) + (dt/M_eff) * [Ftot + Fin]
    // omega(t+dt) = omega(t) + (dt/Izz_eff) * [Ttot + Tin]
    const double inv_M   = (M_eff   > 1e-30) ? 1.0 / M_eff   : 0.0;
    const double inv_Izz = (Izz_eff > 1e-30) ? 1.0 / Izz_eff : 0.0;

    state_.ux    += dt * inv_M   * (ibm_fx_     + fin_x);
    state_.uy    += dt * inv_M   * (ibm_fy_     + fin_y);
    state_.omega += dt * inv_Izz * (ibm_torque_ + tin);

    // 运动学积分（公式 A.6）
    // Xc(t+dt) = Xc(t) + dt * Uc(t)
    // theta(t+dt) = theta(t) + dt * omega(t)
    state_.cx    += dt * state_.prev_ux;      // 用更新前速度（Euler）
    state_.cy    += dt * state_.prev_uy;
    state_.theta += dt * state_.prev_omega;

    // 用新位置/角度更新所有标记点坐标
    update_boundary_markers();
}

} // namespace fsi
