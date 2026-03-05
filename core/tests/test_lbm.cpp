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

// 测试 3：方腔流物理正确性——施加正确的 Zou-He + 反弹 BC 后，
//         流场应在北壁产生接近盖板速度的水平流动，
//         南壁 ux 趋近于零，并且所有 f 值有限（无发散）。
// 这是对 Bug 1（Zou-He 已知/未知对换）和 Bug 2（反弹 bb_node 错误）的回归测试。
static int test_bc_drives_flow()
{
    const int nx = 16, ny = 16;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);

    const double omega = 1.0;  // ν = 1/6，对应低雷诺数（Re ≈ 1.6）
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

    // Re ≈ 1.6，扩散时间尺度 L²/ν = 16²×6 ≈ 1536 步；
    // 运行 500 步（≈1/3 扩散时间），流场已充分由盖板驱动
    constexpr int TIMESTEPS_FOR_STEADY_STATE = 500;
    for (int t = 0; t < TIMESTEPS_FOR_STEADY_STATE; ++t) solver.step();

    // 物理一致性检查 1：北壁内侧（j=ny-2）中线节点 ux 应接近盖板速度
    // 在低 Re 方腔中，盖板正下方中线 ux 约为盖板速度的 50-70%
    // 阈值 40% 保守留有余量（角点速度较低会拉低平均值）
    constexpr double MIN_CENTERLINE_VELOCITY_RATIO = 0.04;  // 盖板速度的 40%
    const double ux_centerline = g.u[g.idx(nx / 2, ny - 2) * 2 + 0];
    const bool lid_ok = ux_centerline > MIN_CENTERLINE_VELOCITY_RATIO;

    // 物理一致性检查 2：南壁节点（j=1，第一内部层）ux 应接近零（无滑移壁）
    // 允许公差 0.02（为盖板速度的 20%），反映格子分辨率导致的有限厚度效应
    constexpr double MAX_NOSLIP_VELOCITY = 0.02;
    double ux_near_south = 0.0;
    for (int i = 1; i < nx - 1; ++i) {
        ux_near_south += std::abs(g.u[g.idx(i, 1) * 2 + 0]);
    }
    ux_near_south /= (nx - 2);
    const bool south_ok = ux_near_south < MAX_NOSLIP_VELOCITY;

    // 物理一致性检查 3：腔体中下部应有反向环流（ux < 0）——这是方腔流的特征涡
    // 阈值对应盖板速度的 5%，保守设置
    constexpr double MIN_RECIRCULATION_VELOCITY = -0.005;
    const double ux_recirculation = g.u[g.idx(nx / 2, ny / 4) * 2 + 0];
    const bool recirculation_ok = ux_recirculation < MIN_RECIRCULATION_VELOCITY;

    // 物理一致性检查 4：所有分布函数有限（无 NaN/Inf，未发散）
    bool finite_ok = true;
    for (int idx = 0; idx < g.size() * lbm::d2q9::Q; ++idx) {
        if (!std::isfinite(g.f[idx])) { finite_ok = false; break; }
    }

    const bool ok = lid_ok && south_ok && recirculation_ok && finite_ok;
    std::printf("[LBM] lid-driven cavity physics: "
                "ux_centerline=%.4f(>%.3f) ux_south=%.4f(<%.3f) "
                "ux_recirc=%.4f(<%.3f) finite=%s → %s\n",
                ux_centerline, MIN_CENTERLINE_VELOCITY_RATIO,
                ux_near_south, MAX_NOSLIP_VELOCITY,
                ux_recirculation, MIN_RECIRCULATION_VELOCITY,
                finite_ok ? "yes" : "NO",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试 4：反弹 BC 使用碰后迁移前（f_tmp）的值设置未知方向
//
// 正确的 halfway bounce-back 应满足：
//   f[opp[a]](x_wall, t+dt) = f*[a](x_wall, t)
// 其中 f* 是碰撞后、流式迁移前的分布函数，stream() 后存于 g.f_tmp。
//
// 施加反弹后（仍在同一步内），验证：
//   f[2](i,0) == f_tmp[4](i,0)   （N ← 碰前 S）
//   f[5](i,0) == f_tmp[7](i,0)   （NE ← 碰前 SW）
//   f[6](i,0) == f_tmp[8](i,0)   （NW ← 碰前 SE）
// 精度公差：IEEE 754 双精度精确赋值，应严格相等（tol = 1e-14）
static int test_bounce_back_uses_precollision_values()
{
    const int nx = 8, ny = 8;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);

    const double omega = 1.0;
    lbm::Solver solver(g, omega);

    // 仅在南壁施加反弹（不添加其他 BC，让流场自然发展）
    lbm::BoundaryCondition bc_south;
    bc_south.type = lbm::BCType::BounceBack;
    bc_south.face = lbm::Face::South;
    solver.add_boundary_condition(bc_south);

    // 运行几步产生非均匀分布
    for (int t = 0; t < 5; ++t) solver.step();

    // 步骤结束后：
    //   g.f     = 施加 BC + compute_macroscopic() 后的分布函数
    //   g.f_tmp = 本次 stream() 之前的碰后分布函数（即反弹 BC 的正确来源）
    // 验证南壁反弹赋值使用了 f_tmp（碰后迁移前）而非 f（迁移后）的值
    constexpr double BOUNCE_BACK_TOLERANCE = 1e-14;
    bool ok = true;
    for (int i = 1; i < nx - 1; ++i) {
        const int n = g.idx(i, 0);
        const double* f  = &g.f    [n * lbm::d2q9::Q];
        const double* fp = &g.f_tmp[n * lbm::d2q9::Q];  // post-collision, pre-streaming values
        // 正确反弹：f[2] == f_tmp[4], f[5] == f_tmp[7], f[6] == f_tmp[8]
        if (std::abs(f[2] - fp[4]) > BOUNCE_BACK_TOLERANCE) { ok = false; break; }
        if (std::abs(f[5] - fp[7]) > BOUNCE_BACK_TOLERANCE) { ok = false; break; }
        if (std::abs(f[6] - fp[8]) > BOUNCE_BACK_TOLERANCE) { ok = false; break; }
    }

    std::printf("[LBM] bounce-back uses pre-streaming (post-collision) values (south wall): %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int test_lbm_main()
{
    int failures = 0;
    failures += test_mass_conservation();
    failures += test_feq_normalisation();
    failures += test_bc_drives_flow();
    failures += test_bounce_back_uses_precollision_values();
    return failures;
}