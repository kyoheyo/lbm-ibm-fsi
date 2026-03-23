#pragma once
#include <vector>

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

} // namespace fsi
