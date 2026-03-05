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

// 测试 5：全步长反弹（BounceBackFullWay）— 南壁
// 正确的全步长（on-node）反弹应满足：
//   f[2](i,0) == 施加前的 f[4](i,0)（使用当前迁移后值，而非 f_tmp）
// 实现上通过临时变量先存储入射方向，再写入，避免读写混用。
static int test_bounce_back_fullway_south()
{
    const int nx = 8, ny = 8;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);

    const double omega = 1.0;
    lbm::Solver solver(g, omega);

    lbm::BoundaryCondition bc_south;
    bc_south.type = lbm::BCType::BounceBackFullWay;
    bc_south.face = lbm::Face::South;
    solver.add_boundary_condition(bc_south);

    for (int t = 0; t < 5; ++t) solver.step();

    // 全步长反弹后：f[2] 必须等于 施加 BC 时 f[4] 的值（即迁移后 f[4]）
    // 因 BC 施加后宏观量也被更新，此处逆向验证：
    //   若全步长反弹正确，则 f[2] == 施加前 f[4]
    // 通过对比 f[2] 与 f[4] 来检验（施加 BC 后 f[4] 未被修改，故 f[4] 仍为入射值）
    constexpr double TOL = 1e-14;
    bool ok = true;
    for (int i = 1; i < nx - 1; ++i) {
        const int n = g.idx(i, 0);
        const double* f = &g.f[n * lbm::d2q9::Q];
        // 全步长反弹：f[2] == f[4]（均为当前迁移后值）
        if (std::abs(f[2] - f[4]) > TOL) { ok = false; break; }
        if (std::abs(f[5] - f[7]) > TOL) { ok = false; break; }
        if (std::abs(f[6] - f[8]) > TOL) { ok = false; break; }
    }
    std::printf("[LBM] full-way bounce-back (south wall): %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}


// 测试 6：Zou-He 压力边界条件 — East 面（背压出口）
// 物理期望：在 East 面规定 ρ_b=1.0，West 面 Zou-He 速度（ux=0.05）驱动后，
// 求解器不发散，East 面密度趋近于规定值。
static int test_zou_he_pressure_east()
{
    const int nx = 16, ny = 8;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);

    const double omega = 1.0;
    lbm::Solver solver(g, omega);

    // West 入口：Zou-He 速度
    lbm::BoundaryCondition bc_west;
    bc_west.type = lbm::BCType::ZouHe_Velocity;
    bc_west.face = lbm::Face::West;
    bc_west.ux   = 0.05;
    bc_west.uy   = 0.0;
    solver.add_boundary_condition(bc_west);

    // East 出口：Zou-He 压力（规定 ρ=1.0）
    lbm::BoundaryCondition bc_east;
    bc_east.type = lbm::BCType::ZouHe_Pressure;
    bc_east.face = lbm::Face::East;
    bc_east.rho  = 1.0;
    solver.add_boundary_condition(bc_east);

    // South/North 无滑移壁
    for (auto face : {lbm::Face::South, lbm::Face::North}) {
        lbm::BoundaryCondition bc_wall;
        bc_wall.type = lbm::BCType::BounceBack;
        bc_wall.face = face;
        solver.add_boundary_condition(bc_wall);
    }

    for (int t = 0; t < 300; ++t) solver.step();

    // 检查：所有分布函数有限
    bool finite_ok = true;
    for (int idx = 0; idx < g.size() * lbm::d2q9::Q; ++idx) {
        if (!std::isfinite(g.f[idx])) { finite_ok = false; break; }
    }

    // East 面平均密度应接近 1.0
    double rho_east_avg = 0.0;
    for (int j = 0; j < ny; ++j)
        rho_east_avg += g.rho[g.idx(nx - 1, j)];
    rho_east_avg /= ny;

    const bool rho_ok = std::abs(rho_east_avg - 1.0) < 0.01;
    const bool ok = finite_ok && rho_ok;
    std::printf("[LBM] Zou-He pressure BC (east outlet): finite=%s rho_east=%.5f → %s\n",
                finite_ok ? "yes" : "NO", rho_east_avg, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}


// 测试 7：充分发展出口（FullyDeveloped）— East 面
// 物理期望：West 驱动 + East 充分发展出口，流场有限无发散，
// East 面分布函数与 i=nx-2 面相同（零梯度条件）。
static int test_fully_developed_east()
{
    const int nx = 16, ny = 8;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);

    const double omega = 1.0;
    lbm::Solver solver(g, omega);

    lbm::BoundaryCondition bc_west;
    bc_west.type = lbm::BCType::ZouHe_Velocity;
    bc_west.face = lbm::Face::West;
    bc_west.ux   = 0.05;
    solver.add_boundary_condition(bc_west);

    lbm::BoundaryCondition bc_east;
    bc_east.type = lbm::BCType::FullyDeveloped;
    bc_east.face = lbm::Face::East;
    solver.add_boundary_condition(bc_east);

    for (auto face : {lbm::Face::South, lbm::Face::North}) {
        lbm::BoundaryCondition bc_wall;
        bc_wall.type = lbm::BCType::BounceBack;
        bc_wall.face = face;
        solver.add_boundary_condition(bc_wall);
    }

    for (int t = 0; t < 300; ++t) solver.step();

    // 充分发展条件：f[a](nx-1, j) == f[a](nx-2, j)
    constexpr double TOL = 1e-12;
    bool zero_grad_ok = true;
    for (int j = 1; j < ny - 1 && zero_grad_ok; ++j) {
        const int n_out = g.idx(nx - 1, j);
        const int n_in  = g.idx(nx - 2, j);
        for (int a = 0; a < lbm::d2q9::Q; ++a) {
            if (std::abs(g.f[n_out * lbm::d2q9::Q + a]
                       - g.f[n_in  * lbm::d2q9::Q + a]) > TOL) {
                zero_grad_ok = false; break;
            }
        }
    }

    bool finite_ok = true;
    for (int idx = 0; idx < g.size() * lbm::d2q9::Q; ++idx) {
        if (!std::isfinite(g.f[idx])) { finite_ok = false; break; }
    }

    const bool ok = zero_grad_ok && finite_ok;
    std::printf("[LBM] fully-developed outlet BC (east): zero_grad=%s finite=%s → %s\n",
                zero_grad_ok ? "yes" : "NO", finite_ok ? "yes" : "NO",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}


// 测试 8：郭照立非平衡外推（Guo_Extrapolation）— West 入口速度 BC
// 物理期望：与 Zou-He 速度 BC 类似，流场有限，West 面速度接近指定值。
// Guo 外推速度模式：bc.rho = 0.0 以触发速度模式（规定 ux=0.05，密度从内部外推）。
static int test_guo_extrapolation_west_velocity()
{
    const int nx = 16, ny = 8;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);

    const double omega = 1.0;
    lbm::Solver solver(g, omega);

    lbm::BoundaryCondition bc_west;
    bc_west.type = lbm::BCType::Guo_Extrapolation;
    bc_west.face = lbm::Face::West;
    bc_west.ux   = 0.05;
    bc_west.uy   = 0.0;
    bc_west.rho  = 0.0;   // 速度模式：rho=0 触发从内部外推密度，直接规定 ux/uy
    solver.add_boundary_condition(bc_west);

    lbm::BoundaryCondition bc_east;
    bc_east.type = lbm::BCType::FullyDeveloped;
    bc_east.face = lbm::Face::East;
    solver.add_boundary_condition(bc_east);

    for (auto face : {lbm::Face::South, lbm::Face::North}) {
        lbm::BoundaryCondition bc_wall;
        bc_wall.type = lbm::BCType::BounceBack;
        bc_wall.face = face;
        solver.add_boundary_condition(bc_wall);
    }

    for (int t = 0; t < 300; ++t) solver.step();

    // 检查所有 f 有限
    bool finite_ok = true;
    for (int idx = 0; idx < g.size() * lbm::d2q9::Q; ++idx) {
        if (!std::isfinite(g.f[idx])) { finite_ok = false; break; }
    }

    // West 面平均 ux 应接近指定值 0.05（±20% 公差，因为 Guo 外推在初始阶段允许微量偏差）
    double ux_west_avg = 0.0;
    for (int j = 1; j < ny - 1; ++j)
        ux_west_avg += g.u[g.idx(0, j) * 2 + 0];
    ux_west_avg /= (ny - 2);
    const bool ux_ok = (ux_west_avg > 0.04) && (ux_west_avg < 0.06);

    const bool ok = finite_ok && ux_ok;
    std::printf("[LBM] Guo-extrapolation BC (west velocity): finite=%s ux_west=%.4f → %s\n",
                finite_ok ? "yes" : "NO", ux_west_avg, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}


int test_lbm_main()
{
    int failures = 0;
    failures += test_mass_conservation();
    failures += test_feq_normalisation();
    failures += test_bc_drives_flow();
    failures += test_bounce_back_uses_precollision_values();
    failures += test_bounce_back_fullway_south();
    failures += test_zou_he_pressure_east();
    failures += test_fully_developed_east();
    failures += test_guo_extrapolation_west_velocity();
    return failures;
}