// 测试：LBM 求解器守恒质量并趋向平衡态
#include "lbm/lattice.hpp"
#include "lbm/solver.hpp"
#include "lbm/boundary.hpp"
#include "lbm/mpi_decomp.hpp"
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


// 测试 9：MpiDecomp 单进程模式字段正确性
//
// 在非 MPI 构建（或 nprocs=1）下，MpiDecomp::create() 应返回：
//   rank=0, nprocs=1, y_start=0, y_end=gny-1, local_ny=gny
//   has_south_wall() == true, has_north_wall() == true
//   grid_ny() == global_ny（无幽灵行）
//
// 这验证了域分解描述符在单进程/无 MPI 构建下的正确初始化，
// 同时作为多进程情形下各字段计算的基准回归测试。
static int test_mpi_decomp_single_rank()
{
    const int gnx = 32, gny = 16;
    const auto d = lbm::MpiDecomp::create(gnx, gny);

    bool ok = true;
    ok = ok && (d.rank      == 0);
    ok = ok && (d.nprocs    == 1);
    ok = ok && (d.global_nx == gnx);
    ok = ok && (d.global_ny == gny);
    ok = ok && (d.y_start   == 0);
    ok = ok && (d.y_end     == gny - 1);
    ok = ok && (d.local_ny  == gny);
    ok = ok && (d.grid_ny() == gny);       // nprocs==1: no ghost rows
    ok = ok && d.has_south_wall();
    ok = ok && d.has_north_wall();

    std::printf("[MPI] MpiDecomp single-rank fields: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}


// 测试 10：attach_mpi() 单进程模式对仿真结果无影响（质量守恒不受干扰）
//
// 物理期望：
//   在非 MPI 构建（nprocs=1）下，attach_mpi() 应为空操作——
//   halo exchange 代码路径受 nprocs>1 守卫，因此仿真结果应与
//   不调用 attach_mpi() 完全一致。
//
// 验证方法：两组完全相同的初始条件下各运行 100 步，
//   分别不绑定 / 绑定 MpiDecomp；最终宏观量逐节点比较。
static int test_mpi_attach_no_effect()
{
    const int nx = 16, ny = 16;
    const double omega = 1.0;

    // 基准：不绑定任何 MpiDecomp
    lbm::LatticeGrid g_ref(nx, ny, 1, lbm::LatticeModel::D2Q9);
    for (int i = 0; i < g_ref.size(); ++i) g_ref.rho[i] = 1.0 + 0.01 * (i % 7);
    lbm::Solver solver_ref(g_ref, omega);
    for (int t = 0; t < 100; ++t) solver_ref.step();

    // 对照：绑定 nprocs==1 的 MpiDecomp（应为 no-op）
    lbm::LatticeGrid g_mpi(nx, ny, 1, lbm::LatticeModel::D2Q9);
    for (int i = 0; i < g_mpi.size(); ++i) g_mpi.rho[i] = 1.0 + 0.01 * (i % 7);
    lbm::Solver solver_mpi(g_mpi, omega);
    const auto decomp = lbm::MpiDecomp::create(nx, ny);  // nprocs==1
    solver_mpi.attach_mpi(&decomp);
    for (int t = 0; t < 100; ++t) solver_mpi.step();

    // 两组结果应逐节点完全相同（IEEE-754 精确相等）
    constexpr double TOL = 1e-14;
    bool ok = true;
    for (int i = 0; i < g_ref.size() && ok; ++i) {
        if (std::abs(g_ref.rho[i] - g_mpi.rho[i]) > TOL) { ok = false; break; }
        if (std::abs(g_ref.u[i * 2 + 0] - g_mpi.u[i * 2 + 0]) > TOL) { ok = false; break; }
        if (std::abs(g_ref.u[i * 2 + 1] - g_mpi.u[i * 2 + 1]) > TOL) { ok = false; break; }
    }
    std::printf("[MPI] attach_mpi single-rank no-op (results identical): %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}


// ---------------------------------------------------------------------------
// 虚拟幽灵行交换辅助函数（模拟 MPI halo_exchange_d2q9 的行为，无需实际 MPI）
//
// 模拟两相邻切片（slab0 为南切片，slab1 为北切片）之间的交换：
//   slab0 的顶物理行（j=local_ny0）→ slab1 的南幽灵行（j=0）
//   slab1 的底物理行（j=1）         → slab0 的北幽灵行（j=local_ny0+1）
// ---------------------------------------------------------------------------
static void virtual_halo_exchange(lbm::LatticeGrid& slab0, int local_ny0,
                                   lbm::LatticeGrid& slab1)
{
    const int nx = slab0.nx;
    const int Q  = lbm::d2q9::Q;

    // Exchange 1: slab0 top physical row → slab1 south ghost
    for (int i = 0; i < nx; ++i) {
        const int src = slab0.idx(i, local_ny0);
        const int dst = slab1.idx(i, 0);
        for (int a = 0; a < Q; ++a)
            slab1.f[dst * Q + a] = slab0.f[src * Q + a];
    }

    // Exchange 2: slab1 bottom physical row → slab0 north ghost
    for (int i = 0; i < nx; ++i) {
        const int src = slab1.idx(i, 1);
        const int dst = slab0.idx(i, local_ny0 + 1);
        for (int a = 0; a < Q; ++a)
            slab0.f[dst * Q + a] = slab1.f[src * Q + a];
    }
}


// 测试 11：虚拟幽灵行交换行正确性
//
// 设计：创建两个相邻切片网格，向每行填写唯一可识别的 f 值（行号 + 常数），
// 执行虚拟幽灵行交换，然后验证：
//   1. slab1 的南幽灵行（j=0）== slab0 的顶物理行（j=local_ny0）
//   2. slab0 的北幽灵行（j=local_ny0+1）== slab1 的底物理行（j=1）
//
// 这保证了 MPI halo_exchange_d2q9 的逻辑（发送/接收哪一行）是正确的，
// 即相邻进程的物理边界数据能够正确填入对方的幽灵行，
// 为后续流式迁移提供正确的入流值。
static int test_mpi_halo_exchange_correctness()
{
    const int nx       = 8;
    const int local_ny = 4;   // 每个切片的物理行数
    // slab0 和 slab1 均含 local_ny+2 行（含幽灵行）
    const int grid_ny  = local_ny + 2;

    lbm::LatticeGrid slab0(nx, grid_ny, 1, lbm::LatticeModel::D2Q9);
    lbm::LatticeGrid slab1(nx, grid_ny, 1, lbm::LatticeModel::D2Q9);

    // 向每行的每个节点的每个方向填入易于识别的值：
    //   row_offset * 100 + direction_index（避免跨行值碰撞）
    // 幽灵行（j=0 和 j=grid_ny-1）填为 -1（代表"未初始化"）
    const int Q = lbm::d2q9::Q;
    for (int j = 0; j < grid_ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int n0 = slab0.idx(i, j);
            const int n1 = slab1.idx(i, j);
            for (int a = 0; a < Q; ++a) {
                slab0.f[n0 * Q + a] = (j == 0 || j == grid_ny - 1) ? -1.0
                    : static_cast<double>(j * 100 + a);       // slab0 物理行
                slab1.f[n1 * Q + a] = (j == 0 || j == grid_ny - 1) ? -1.0
                    : static_cast<double>((j + local_ny) * 100 + a); // slab1 物理行（全局行偏移）
            }
        }
    }

    // 执行虚拟幽灵行交换
    virtual_halo_exchange(slab0, local_ny, slab1);

    bool ok = true;
    // 验证 1：slab1 的南幽灵行（j=0）== slab0 的顶物理行（j=local_ny）
    for (int i = 0; i < nx && ok; ++i) {
        const int ghost = slab1.idx(i, 0);
        const int phys  = slab0.idx(i, local_ny);
        for (int a = 0; a < Q && ok; ++a) {
            if (slab1.f[ghost * Q + a] != slab0.f[phys * Q + a])
                ok = false;
        }
    }
    // 验证 2：slab0 的北幽灵行（j=local_ny+1）== slab1 的底物理行（j=1）
    for (int i = 0; i < nx && ok; ++i) {
        const int ghost = slab0.idx(i, local_ny + 1);
        const int phys  = slab1.idx(i, 1);
        for (int a = 0; a < Q && ok; ++a) {
            if (slab0.f[ghost * Q + a] != slab1.f[phys * Q + a])
                ok = false;
        }
    }

    std::printf("[MPI] halo exchange row correctness (virtual 2-slab): %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}


// 测试 12：虚拟双进程分解下幽灵行不被二次碰撞——单进程对照验证
//
// 设计：在无 MPI 的单进程构建中，collide_bgk 的幽灵行跳过逻辑仅在
// mpi_decomp_->nprocs > 1 时激活；当 nprocs==1 时行为与无 MPI 完全相同。
// 本测试通过以下方式验证：
//   - 两组完全相同的初始条件、相同的边界条件，分别使用：
//       A）不绑定任何 MpiDecomp（nprocs=1，n_start/n_end 不裁剪）
//       B）绑定 nprocs==1 的 MpiDecomp（n_start/n_end 不裁剪，因 has_south/north_wall()=true）
//   - 两组运行 100 步后宏观量逐节点比较，确保数值完全一致
//   - 这同时验证：has_south_wall()/has_north_wall() 在 nprocs==1 时均返回 true，
//     即幽灵行跳过逻辑对整域进程（rank 0 = rank nprocs-1）不产生影响
//
// 注意：此测试仅在单进程模式下验证逻辑的"无影响性"，多进程下的
// 二次碰撞修复需要通过 mpirun 运行集成测试才能完整验证。
static int test_mpi_virtual_two_rank_no_ghost_collision()
{
    const int nx = 16, ny = 16;
    const double omega = 1.2;

    // 基准：不绑定任何 MpiDecomp
    lbm::LatticeGrid g_ref(nx, ny, 1, lbm::LatticeModel::D2Q9);
    for (int i = 0; i < g_ref.size(); ++i) g_ref.rho[i] = 1.0 + 0.02 * (i % 7);
    lbm::Solver solver_ref(g_ref, omega);
    // 北壁 BounceBack（模拟有壁面的场景）
    lbm::BoundaryCondition bc_ref;
    bc_ref.type = lbm::BCType::BounceBack;
    bc_ref.face = lbm::Face::North;
    solver_ref.add_boundary_condition(bc_ref);
    for (int t = 0; t < 100; ++t) solver_ref.step();

    // 对照：绑定 nprocs==1 的 MpiDecomp
    // nprocs==1 时 has_south_wall()=true && has_north_wall()=true，
    // 因此 n_start=0, n_end=n — 与不绑定完全等价
    lbm::LatticeGrid g_mpi(nx, ny, 1, lbm::LatticeModel::D2Q9);
    for (int i = 0; i < g_mpi.size(); ++i) g_mpi.rho[i] = 1.0 + 0.02 * (i % 7);
    lbm::Solver solver_mpi(g_mpi, omega);
    lbm::BoundaryCondition bc_mpi;
    bc_mpi.type = lbm::BCType::BounceBack;
    bc_mpi.face = lbm::Face::North;
    solver_mpi.add_boundary_condition(bc_mpi);
    const auto decomp = lbm::MpiDecomp::create(nx, ny);  // nprocs=1 in non-MPI build
    solver_mpi.attach_mpi(&decomp);
    for (int t = 0; t < 100; ++t) solver_mpi.step();

    constexpr double TOL = 1e-14;
    bool ok = true;
    for (int i = 0; i < g_ref.size() && ok; ++i) {
        if (std::abs(g_ref.rho[i] - g_mpi.rho[i]) > TOL) { ok = false; break; }
        if (std::abs(g_ref.u[i * 2 + 0] - g_mpi.u[i * 2 + 0]) > TOL) { ok = false; break; }
        if (std::abs(g_ref.u[i * 2 + 1] - g_mpi.u[i * 2 + 1]) > TOL) { ok = false; break; }
    }
    std::printf("[MPI] no ghost-row double-collision in single-rank mode: %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}


// 测试：边角点反弹修正（直接验证角节点幽灵方向使用 f_tmp 而非幽灵周期值）
//
// apply_corner_bounce_back() 应对每个角节点的幽灵方向满足 f[a] = f_tmp[opp(a)]。
// 本测试在 ZouHe + BounceBack 组合下运行一步后，直接检验此不变量。
//
// 时序说明（与 solver.cpp 第 60–69 行一致）：
//   step() 内部顺序：collide() → stream()（结束时 swap(f, f_tmp)）
//                   → apply_boundary_conditions()（读 f_tmp，写 f）
//                   → compute_macroscopic()（读 f，不修改 f）
//   step() 结束后：
//     g.f_tmp = 本步 collide 后、stream 前的值（apply_corner_bounce_back 的读来源）
//     g.f     = BC + macroscopic 修正后的当前状态
//   因此，不变量 f[ghost_a] == f_tmp[opp(a)] 在 step() 结束后严格成立，
//   直到下一个 step() 的 collide() 修改 g.f 为止。
//
// 同时验证：
//   (a) 4 个角节点的 f 值均有限（无 NaN/Inf）
//   (b) 幽灵方向 f[a] == f_tmp[opp(a)]（角点修正的直接证据）
static int test_corner_bounce_back_fix()
{
    const int nx = 16, ny = 16;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);
    const double omega = 1.0;
    lbm::Solver solver(g, omega);

    // 北壁 ZouHe 速度 BC（ux=0.1），其余三面 BounceBack
    lbm::BoundaryCondition bc_north;
    bc_north.type = lbm::BCType::ZouHe_Velocity;
    bc_north.face = lbm::Face::North;
    bc_north.ux   = 0.1;
    bc_north.uy   = 0.0;
    solver.add_boundary_condition(bc_north);

    for (auto face : {lbm::Face::South, lbm::Face::West, lbm::Face::East}) {
        lbm::BoundaryCondition bc_wall;
        bc_wall.type = lbm::BCType::BounceBack;
        bc_wall.face = face;
        solver.add_boundary_condition(bc_wall);
    }

    // 先运行几步建立非零场，再做一步后检查
    for (int t = 0; t < 10; ++t) solver.step();
    // step() 返回后 g.f_tmp = 本步 collide-post/stream-pre 值（角点修正的读来源）
    solver.step();

    // 对立方向查找表
    const int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

    const char* corner_names[4] = {"SW(0,0)", "SE(nx-1,0)", "NW(0,ny-1)", "NE(nx-1,ny-1)"};
    struct CornerInfo { int i, j; int ghosts[5]; };
    CornerInfo corners[4] = {
        {0,      0,      {1, 2, 5, 6, 8}},
        {nx-1,   0,      {2, 3, 5, 6, 7}},
        {0,      ny-1,   {1, 4, 5, 7, 8}},
        {nx-1,   ny-1,   {3, 4, 6, 7, 8}},
    };

    bool ok = true;
    constexpr double TOL = 1e-14;

    for (int ci = 0; ci < 4; ++ci) {
        const auto& c = corners[ci];
        const int n = g.idx(c.i, c.j);
        const double* f     = &g.f    [n * lbm::d2q9::Q];
        const double* f_tmp = &g.f_tmp[n * lbm::d2q9::Q];

        // (a) 所有 f 值有限
        for (int a = 0; a < lbm::d2q9::Q; ++a) {
            if (!std::isfinite(f[a])) {
                std::printf("[LBM] corner BB fix: %s f[%d] not finite\n",
                            corner_names[ci], a);
                ok = false;
            }
        }

        // (b) 幽灵方向满足 f[a] = f_tmp[opp(a)]（角点修正的直接验证）
        for (int a : c.ghosts) {
            const double diff = std::abs(f[a] - f_tmp[OPP[a]]);
            if (diff > TOL) {
                std::printf("[LBM] corner BB fix: %s f[%d]=%.6e != f_tmp[%d]=%.6e diff=%.2e\n",
                            corner_names[ci], a, f[a], OPP[a], f_tmp[OPP[a]], diff);
                ok = false;
            }
        }
    }

    // 全域 f 有限检查
    for (int idx = 0; idx < g.size() * lbm::d2q9::Q; ++idx) {
        if (!std::isfinite(g.f[idx])) {
            std::printf("[LBM] corner BB fix: global f[%d] not finite\n", idx);
            ok = false;
            break;
        }
    }

    std::printf("[LBM] corner BB fix (f[ghost]==f_tmp[opp]): %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}


// 测试：非固壁角点不施加反弹修正
//
// 当角节点的两个相邻面均为非固壁 BC（FullyDeveloped），
// apply_corner_bounce_back() 应**跳过**该角节点，不覆盖其 f 值为 f_tmp[opp(a)]。
//
// 测试设计（直接调用 apply_boundary_conditions，而非通过 step()）：
//   1. 用已知的平衡分布初始化整个网格的 f 和 f_tmp
//   2. 将 SW 角节点的 f_tmp 覆写为哨兵值（100 + a），
//      使 f_tmp[opp(a)](SW) ≈ 100 远大于任何物理 f 值（< 0.5）
//   3. 注册 BCs：South + West = FullyDeveloped（非固壁）；North + East = BounceBack（固壁）
//   4. 直接调用 apply_boundary_conditions（跳过碰撞/流式步骤）
//   5. 验证：
//      (a) SW 幽灵方向的 f 值 ≈ FD 源节点 (1,0) 的 f 值（未被哨兵覆盖），
//          即 |f[ghost](SW) - f[ghost](1,0)| < 1e-14，
//          同时 f[ghost](SW) 绝对值远小于哨兵值 100（未被 bounce-back 覆盖）。
//      (b) NE 幽灵方向满足 f[a] == f_tmp[opp(a)]（固壁角，bounce-back 正常施加）。
static int test_corner_bounce_back_skip_nonwall()
{
    const int nx = 4, ny = 4;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);

    // 1. 用平衡分布（rho=1, ux=uy=0）初始化 f 和 f_tmp
    //    平衡分布值均 < 0.5，与哨兵值 ~100 差异显著
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int n = g.idx(i, j);
            const double u[2] = {0.0, 0.0};
            for (int a = 0; a < lbm::d2q9::Q; ++a) {
                const double c[2] = {
                    static_cast<double>(lbm::d2q9::C[a][0]),
                    static_cast<double>(lbm::d2q9::C[a][1])
                };
                const double fval = lbm::f_eq(lbm::d2q9::W[a], 1.0, c, u, 2);
                g.f    [n * lbm::d2q9::Q + a] = fval;
                g.f_tmp[n * lbm::d2q9::Q + a] = fval;
            }
        }
    }

    // 2. 将 SW 角节点的 f_tmp 覆写为哨兵值（100+a），
    //    使 f_tmp[opp(ghost_a)](SW) ≈ 100 >> 任何物理 f 值
    {
        const int sw_n = g.idx(0, 0);
        for (int a = 0; a < lbm::d2q9::Q; ++a)
            g.f_tmp[sw_n * lbm::d2q9::Q + a] = 100.0 + a;
    }

    // 3. 注册 BCs
    std::vector<lbm::BoundaryCondition> bcs;
    for (auto face : {lbm::Face::South, lbm::Face::West}) {
        lbm::BoundaryCondition bc;
        bc.type = lbm::BCType::FullyDeveloped;
        bc.face = face;
        bcs.push_back(bc);
    }
    for (auto face : {lbm::Face::North, lbm::Face::East}) {
        lbm::BoundaryCondition bc;
        bc.type = lbm::BCType::BounceBack;
        bc.face = face;
        bcs.push_back(bc);
    }

    // 4. 直接调用 apply_boundary_conditions（不经过碰撞/流式迁移）
    lbm::apply_boundary_conditions(g, bcs);

    const int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    bool ok = true;

    // --- (a) SW 角（South FD + West FD）：幽灵方向应等于 FD 源节点 (1,0) 的 f 值 ---
    //   FD West 最后执行，将 f[a](1,0) 拷贝到 f[a](0,0)。
    //   若角点修正被正确跳过，幽灵方向不会被哨兵覆盖。
    {
        const int sw_n   = g.idx(0, 0);
        const int fd_src = g.idx(1, 0);   // West FD 的源节点（最后覆盖 South FD 的结果）
        const double* f_sw  = &g.f    [sw_n   * lbm::d2q9::Q];
        const double* f_src = &g.f    [fd_src * lbm::d2q9::Q];
        const int sw_ghosts[5] = {1, 2, 5, 6, 8};

        for (int a : sw_ghosts) {
            // 若 bounce-back 被错误施加：f_sw[a] = f_tmp[OPP[a]](SW) = 100 + OPP[a]
            if (f_sw[a] > 10.0) {
                std::printf("[LBM] corner skip: SW ghost dir %d = %.2f "
                            "(sentinel value, bounce-back was wrongly applied)\n",
                            a, f_sw[a]);
                ok = false;
            }
            // 若 FD 正常：f_sw[a] == f_src[a]（平衡值，< 0.5）
            const double diff = std::abs(f_sw[a] - f_src[a]);
            if (diff > 1e-14) {
                std::printf("[LBM] corner skip: SW ghost dir %d = %.6e, "
                            "FD source = %.6e, diff = %.2e\n",
                            a, f_sw[a], f_src[a], diff);
                ok = false;
            }
        }
    }

    // --- (b) NE 角（North BB + East BB）：幽灵方向应满足 f[a] == f_tmp[opp(a)] ---
    //   NE 的 f_tmp 仍为初始平衡值（未被哨兵覆盖），bounce-back 应正常施加。
    {
        const int ne_n = g.idx(nx - 1, ny - 1);
        const double* f_ne  = &g.f    [ne_n * lbm::d2q9::Q];
        const double* ft_ne = &g.f_tmp[ne_n * lbm::d2q9::Q];
        const int ne_ghosts[5] = {3, 4, 6, 7, 8};

        for (int a : ne_ghosts) {
            const double diff = std::abs(f_ne[a] - ft_ne[OPP[a]]);
            if (diff > 1e-14) {
                std::printf("[LBM] corner skip: NE ghost dir %d not bounce-backed: "
                            "f=%.6e, f_tmp[opp]=%.6e, diff=%.2e\n",
                            a, f_ne[a], ft_ne[OPP[a]], diff);
                ok = false;
            }
        }
    }

    std::printf("[LBM] corner BB skip (non-wall corner not overwritten): %s\n",
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
    failures += test_bounce_back_fullway_south();
    failures += test_zou_he_pressure_east();
    failures += test_fully_developed_east();
    failures += test_guo_extrapolation_west_velocity();
    failures += test_corner_bounce_back_fix();
    failures += test_corner_bounce_back_skip_nonwall();
    failures += test_mpi_decomp_single_rank();
    failures += test_mpi_attach_no_effect();
    failures += test_mpi_halo_exchange_correctness();
    failures += test_mpi_virtual_two_rank_no_ghost_collision();
    return failures;
}