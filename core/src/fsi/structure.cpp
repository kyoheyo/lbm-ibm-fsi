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
