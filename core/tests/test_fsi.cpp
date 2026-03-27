// 测试：FSI 结构求解器基本特性
#include "fsi/structure.hpp"
#include <cmath>
#include <cstdio>

// 测试 1：BeamSolver 初始化后节点数正确
static int test_beam_node_count()
{
    fsi::BeamParams p;
    p.E          = 1.0e6;
    p.I          = 1.0e-8;
    p.rho_s      = 1000.0;
    p.A          = 1.0e-4;
    p.length     = 1.0;
    p.n_elements = 8;

    fsi::BeamSolver beam(p);
    const bool ok = (beam.n_nodes() == 9);
    std::printf("[FSI] beam node count: %s  (n_nodes=%d)\n",
                ok ? "PASS" : "FAIL", beam.n_nodes());
    return ok ? 0 : 1;
}

// 测试 2：无外力时梁保持静止
static int test_beam_at_rest()
{
    fsi::BeamParams p;
    p.E          = 1.0e4;
    p.I          = 1.0e-6;
    p.rho_s      = 1.0;
    p.A          = 0.01;
    p.length     = 1.0;
    p.n_elements = 4;

    fsi::BeamSolver beam(p);

    const double dt = 0.01;
    for (int t = 0; t < 100; ++t) beam.step(dt);

    bool ok = true;
    for (const auto& dof : beam.dofs()) {
        if (std::abs(dof.y) > 1e-10 || std::abs(dof.vy) > 1e-10) {
            ok = false;
            break;
        }
    }
    std::printf("[FSI] beam at rest (no forces): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试 3：梁对集中载荷的响应（尖端挠度方向正确）
static int test_beam_tip_deflection()
{
    fsi::BeamParams p;
    p.E          = 2.1e11;  // 类钢材料
    p.I          = 1.0e-8;
    p.rho_s      = 7850.0;
    p.A          = 1.0e-4;
    p.length     = 0.5;
    p.n_elements = 4;

    fsi::BeamSolver beam(p);

    // 在尖端节点施加向下的力
    beam.dofs().back().fy = -100.0;

    const double dt = 1e-5;
    for (int t = 0; t < 10; ++t) beam.step(dt);

    // 向下的尖端力作用后，尖端 y 坐标应为负值（确认挠度方向）
    const double tip_y = beam.dofs().back().y;
    const bool ok = (tip_y < 1e-10);  // 严格小于零以确认向下运动
    std::printf("[FSI] beam tip deflection direction: %s  (tip_y=%.3e)\n",
                ok ? "PASS" : "FAIL", tip_y);
    return ok ? 0 : 1;
}

// 前向声明：RigidBodySolver2D 测试（定义在 test_fsi_main 之后）
static int test_rigid_body_at_rest_no_force();
static int test_rigid_body_free_fall();
static int test_rigid_body_rotation();

int test_fsi_main()
{
    int failures = 0;
    failures += test_beam_node_count();
    failures += test_beam_at_rest();
    failures += test_beam_tip_deflection();
    // RigidBodySolver2D 新测试
    failures += test_rigid_body_at_rest_no_force();
    failures += test_rigid_body_free_fall();
    failures += test_rigid_body_rotation();
    return failures;
}

// ===========================================================================
// RigidBodySolver2D 测试
// ===========================================================================

// 测试 4：无外力时刚体保持静止
static int test_rigid_body_at_rest_no_force()
{
    fsi::RigidBodyParams2D p;
    p.mass    = 1.0;
    p.inertia = 0.5;
    p.rho_b   = 2.0;
    p.rho_f   = 1.0;
    p.cx0 = 10.0;  p.cy0 = 10.0;
    p.scheme  = fsi::InternalMassScheme::None;

    fsi::RigidBodySolver2D rb(p, {}, {});
    rb.set_ibm_forces(0.0, 0.0, 0.0);

    const double dt = 0.1;
    for (int t = 0; t < 100; ++t) rb.advance(dt);

    const auto& s = rb.state();
    const bool ok = (std::abs(s.cx - 10.0) < 1e-12 &&
                     std::abs(s.cy - 10.0) < 1e-12 &&
                     std::abs(s.ux)  < 1e-12 &&
                     std::abs(s.uy)  < 1e-12 &&
                     std::abs(s.omega) < 1e-12);
    std::printf("[FSI] rigid body at rest (no force): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试 5：恒定力作用下刚体做匀加速运动（a = F/m）
static int test_rigid_body_free_fall()
{
    fsi::RigidBodyParams2D p;
    p.mass    = 2.0;
    p.inertia = 1.0;
    p.rho_b   = 2.0;
    p.rho_f   = 1.0;
    p.cx0 = 0.0;  p.cy0 = 0.0;
    p.scheme  = fsi::InternalMassScheme::None;

    fsi::RigidBodySolver2D rb(p, {}, {});

    const double F = 1.0;   // 恒力（方案 None：直接 F_total = F）
    const double dt = 0.01;
    const int    N  = 100;

    for (int t = 0; t < N; ++t) {
        rb.set_ibm_forces(F, 0.0, 0.0);
        rb.advance(dt);
    }
    const double T = N * dt;
    const auto& s = rb.state();
    const double expected_ux = (F / p.mass) * T;
    const double expected_cx = 0.5 * (F / p.mass) * T * T;

    const bool ok = (std::abs(s.ux - expected_ux) < 1e-6 &&
                     std::abs(s.cx - expected_cx)  < 1e-6);
    std::printf("[FSI] rigid body free fall (a=F/m): %s  (cx=%.4f expected=%.4f)\n",
                ok ? "PASS" : "FAIL", s.cx, expected_cx);
    return ok ? 0 : 1;
}

// 测试 6：恒力矩作用下刚体做匀角加速度旋转（α = T/I）
static int test_rigid_body_rotation()
{
    fsi::RigidBodyParams2D p;
    p.mass    = 1.0;
    p.inertia = 2.0;
    p.rho_b   = 1.5;
    p.rho_f   = 1.0;
    p.cx0 = 5.0;  p.cy0 = 5.0;
    p.scheme  = fsi::InternalMassScheme::None;

    fsi::RigidBodySolver2D rb(p, {}, {});

    const double Torque = 1.0;  // 恒力矩
    const double dt     = 0.01;
    const int    N      = 100;

    for (int t = 0; t < N; ++t) {
        rb.set_ibm_forces(0.0, 0.0, Torque);
        rb.advance(dt);
    }

    // α = T/I = 0.5; omega = α*T = 0.5; theta = 0.5*α*T^2 = 0.25 (T=1.0)
    const double T = N * dt;
    const auto& s = rb.state();
    const double expected_omega = (Torque / p.inertia) * T;
    const double expected_theta = 0.5 * (Torque / p.inertia) * T * T;

    const bool ok = (std::abs(s.omega - expected_omega) < 1e-6 &&
                     std::abs(s.theta - expected_theta) < 1e-6);
    std::printf("[FSI] rigid body rotation (alpha=T/I): %s  (omega=%.4f expected=%.4f)\n",
                ok ? "PASS" : "FAIL", s.omega, expected_omega);
    return ok ? 0 : 1;
}
