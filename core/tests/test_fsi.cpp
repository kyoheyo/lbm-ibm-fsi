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

int test_fsi_main()
{
    int failures = 0;
    failures += test_beam_node_count();
    failures += test_beam_at_rest();
    failures += test_beam_tip_deflection();
    return failures;
}
