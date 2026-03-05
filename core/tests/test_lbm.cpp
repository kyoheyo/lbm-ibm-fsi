// 测试：LBM 求解器守恒质量并趋向平衡态
#include "lbm/lattice.hpp"
#include "lbm/solver.hpp"
#include "lbm/boundary.hpp"
#include <cmath>
#include <cstdio>
#include <numeric>

// 辅助函数：近似相等判断
static bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

// 测试 1：碰撞 + 流式迁移后总质量（ρ 之和）守恒
static int test_mass_conservation()
{
    const int nx = 16, ny = 16;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);
    // 轻微扰动密度
    for (int i = 0; i < g.size(); ++i) g.rho[i] = 1.0 + 0.01 * (i % 5);

    const double omega = 1.0;
    lbm::Solver solver(g, omega);

    const double rho_init = std::accumulate(g.rho.begin(), g.rho.end(), 0.0);

    for (int t = 0; t < 100; ++t) solver.step();

    const double rho_final = std::accumulate(g.rho.begin(), g.rho.end(), 0.0);
    const bool ok = approx(rho_init, rho_final, 1e-8 * rho_init);
    std::printf("[LBM] mass conservation: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试 2：f_eq 对所有方向求和等于 rho（归一化验证）
static int test_feq_normalisation()
{
    const double rho = 1.2;
    const double u[2] = {0.05, 0.03};
    double sum = 0.0;
    for (int a = 0; a < lbm::d2q9::Q; ++a) {
        const double c[2] = {
            static_cast<double>(lbm::d2q9::C[a][0]),
            static_cast<double>(lbm::d2q9::C[a][1])
        };
        sum += lbm::f_eq(lbm::d2q9::W[a], rho, c, u, 2);
    }
    const bool ok = approx(sum, rho, 1e-14);
    std::printf("[LBM] f_eq normalisation (sum==rho): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试 3：施加 Zou-He 速度边界条件后，流场应出现非零速度
// （回归测试：确保边界条件被真正施加，而非被忽略导致流场永远为零）
static int test_bc_drives_flow()
{
    const int nx = 16, ny = 16;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);

    const double omega = 1.0;  // ν = 1/6，对应 Re ≈ 1.6（小雷诺数）
    lbm::Solver solver(g, omega);

    // 北壁施加 Zou-He 速度 BC：ux = 0.1（驱动盖板）
    lbm::BoundaryCondition bc_north;
    bc_north.type = lbm::BCType::ZouHe_Velocity;
    bc_north.face = lbm::Face::North;
    bc_north.ux   = 0.1;
    bc_north.uy   = 0.0;
    solver.add_boundary_condition(bc_north);

    // 南/西/东壁施加反弹 BC（无滑移固壁）
    for (auto face : {lbm::Face::South, lbm::Face::West, lbm::Face::East}) {
        lbm::BoundaryCondition bc_wall;
        bc_wall.type = lbm::BCType::BounceBack;
        bc_wall.face = face;
        solver.add_boundary_condition(bc_wall);
    }

    // 运行足够多步使流场充分发展
    for (int t = 0; t < 200; ++t) solver.step();

    // 验证：北壁附近应出现接近盖板速度的 ux
    // 检查北壁内侧节点（j = ny-2）的平均速度
    double ux_sum = 0.0;
    for (int i = 0; i < nx; ++i) {
        ux_sum += g.u[g.idx(i, ny - 2) * 2 + 0];
    }
    const double ux_avg = ux_sum / nx;

    // 北壁内侧平均速度应显著大于零（约为盖板速度的一半以上）
    const bool flow_nonzero = ux_avg > 0.01;

    // 验证：远离驱动壁的中间层也应存在流动
    double ux_mid_sum = 0.0;
    for (int i = 0; i < nx; ++i) {
        ux_mid_sum += g.u[g.idx(i, ny / 2) * 2 + 0];
    }
    const bool mid_nonzero = (ux_mid_sum / nx) > 1e-6;

    const bool ok = flow_nonzero && mid_nonzero;
    std::printf("[LBM] BC drives non-zero flow (ux_near_lid=%.5f): %s\n",
                ux_avg, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int test_lbm_main()
{
    int failures = 0;
    failures += test_mass_conservation();
    failures += test_feq_normalisation();
    failures += test_bc_drives_flow();
    return failures;
}