// 测试：LBM 求解器守恒质量并趋向平衡态
#include "lbm/lattice.hpp"
#include "lbm/solver.hpp"
#include "lbm/boundary.hpp"
#include "lbm/mpi_decomp.hpp"
#include "lbm/mg_tree.hpp"
#include "lbm/stretched_grid.hpp"
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
    // 非 MPI 模式：物理边界就是整个网格，PhysicalBounds 使用默认（全格覆盖）
    lbm::PhysicalBounds pb;
    pb.j_s = 0; pb.j_n = g.ny - 1;
    pb.i_w = 0; pb.i_e = g.nx - 1;
    lbm::apply_boundary_conditions(g, bcs, pb);

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


// 测试：速度入口角节点不被相邻 FullyDeveloped BC 覆盖
//
// 问题描述（已修复）：
//   若按注册顺序施加边界条件，West=ZouHe_Velocity 先执行后，
//   South/North=FullyDeveloped 会将角节点 (0,0) 和 (0,ny-1) 的 ALL f[a]
//   替换为 j=1 / j=ny-2 行的值，销毁 ZouHe 的入口约束。
//   修复方案：三轮优先级排序（FD/FO → BB → ZouHe/Guo），ZouHe 最后施加。
//
// 测试方案：
//   配置：West=ZouHe_Velocity(ux=0.05), East=FullyDeveloped,
//         South=FullyDeveloped, North=FullyDeveloped。
//   运行 500 步后，West 面所有节点（含角节点 j=0, j=ny-1）的
//   g.u[i=0, j] 均应满足 ux ≈ 0.05 (±1%容差)。
//   若优先级排序缺失，角节点的 f[1,5,8] 被 FD 覆盖，ZouHe 公式失效，
//   角节点的 ux 会偏离 0.05（通常偏高或不稳定）。
static int test_inlet_corner_not_overwritten_by_fd()
{
    const int nx = 20, ny = 10;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);

    const double omega = 1.2;  // ν = (1/omega - 0.5)/3 ≈ 0.083
    lbm::Solver solver(g, omega);

    // West: Zou-He 速度入口
    lbm::BoundaryCondition bc_west;
    bc_west.type = lbm::BCType::ZouHe_Velocity;
    bc_west.face = lbm::Face::West;
    bc_west.ux   = 0.05;
    bc_west.uy   = 0.0;
    solver.add_boundary_condition(bc_west);

    // East / South / North: 充分发展出口（自由出口，零法向梯度）
    for (auto face : {lbm::Face::East, lbm::Face::South, lbm::Face::North}) {
        lbm::BoundaryCondition bc;
        bc.type = lbm::BCType::FullyDeveloped;
        bc.face = face;
        solver.add_boundary_condition(bc);
    }

    for (int t = 0; t < 500; ++t) solver.step();

    // 验证 West 面（i=0）所有行的 ux 接近 0.05（含角节点 j=0 和 j=ny-1）
    // 容差选取依据：ny=10, omega=1.2（ν≈0.083）, Re=ux*ny/ν≈6；
    // 流动发展长度 ~0.06*Re*ny ≈ 4 格点 << nx=20，500 步已充分收敛。
    // 全域 FD 出口（无压力参考）的稳态速度应精确等于 ZouHe 规定值（1% 容差）。
    const double UX_TARGET = 0.05;
    const double TOL = UX_TARGET * 0.01;   // 1% 容差，经收敛分析验证合理
    bool inlet_ok = true;
    for (int j = 0; j < ny; ++j) {
        const int n   = g.idx(0, j);
        const double ux = g.u[n * 2 + 0];
        if (std::abs(ux - UX_TARGET) > TOL) {
            std::printf("[LBM] inlet corner: j=%d ux=%.6f (expected %.4f ± %.4f)\n",
                        j, ux, UX_TARGET, TOL);
            inlet_ok = false;
        }
    }

    // 验证所有 f 有限
    bool finite_ok = true;
    for (int idx = 0; idx < g.size() * lbm::d2q9::Q; ++idx) {
        if (!std::isfinite(g.f[idx])) { finite_ok = false; break; }
    }

    const bool ok = inlet_ok && finite_ok;
    std::printf("[LBM] inlet corner not overwritten by FD: "
                "inlet_ux_ok=%s finite=%s → %s\n",
                inlet_ok ? "yes" : "NO", finite_ok ? "yes" : "NO",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}


// 测试：MpiDecomp2D 在单进程（无 MPI 构建）下的正确初始化
//
// 验证点：
//   - nprocs == 1, rank == 0, px == 1, py == 1
//   - col_rank == 0, row_rank == 0
//   - local_nx == global_nx, local_ny == global_ny
//   - has_south_wall() == true, has_north_wall() == true
//   - has_west_wall()  == true, has_east_wall()  == true
//   - 无任何幽灵层（has_*_ghost() == false）
//   - grid_nx() == global_nx, grid_ny() == global_ny
static int test_mpi_decomp2d_single_rank()
{
    const int gnx = 32, gny = 24;
    const auto d = lbm::MpiDecomp2D::create(gnx, gny, 1, 1);

    bool ok = true;
    ok = ok && (d.nprocs   == 1);
    ok = ok && (d.rank     == 0);
    ok = ok && (d.px       == 1);
    ok = ok && (d.py       == 1);
    ok = ok && (d.col_rank == 0);
    ok = ok && (d.row_rank == 0);
    ok = ok && (d.local_nx == gnx);
    ok = ok && (d.local_ny == gny);
    ok = ok && (d.x_start  == 0);
    ok = ok && (d.y_start  == 0);
    ok = ok && (d.x_end    == gnx - 1);
    ok = ok && (d.y_end    == gny - 1);
    ok = ok && d.has_south_wall() && d.has_north_wall();
    ok = ok && d.has_west_wall()  && d.has_east_wall();
    ok = ok && !d.has_south_ghost() && !d.has_north_ghost();
    ok = ok && !d.has_west_ghost()  && !d.has_east_ghost();
    ok = ok && (d.grid_nx() == gnx);
    ok = ok && (d.grid_ny() == gny);

    std::printf("[MPI] MpiDecomp2D single-rank fields: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试：attach_mpi2d() 在单进程模式下对仿真结果无影响
//
// 验证：attach_mpi2d() 在 nprocs==1 时是无操作，
//   100 步仿真后与不绑定 2D 分解的结果完全一致（逐节点差异 < 1e-14）。
static int test_mpi_attach_mpi2d_no_effect()
{
    const int nx = 16, ny = 16;
    const double omega = 1.2;

    // 基准：不绑定任何分解
    lbm::LatticeGrid g_ref(nx, ny, 1, lbm::LatticeModel::D2Q9);
    for (int i = 0; i < g_ref.size(); ++i) g_ref.rho[i] = 1.0 + 0.01 * (i % 7);
    lbm::Solver solver_ref(g_ref, omega);
    for (int t = 0; t < 100; ++t) solver_ref.step();

    // 对照：绑定 nprocs==1 的 MpiDecomp2D（px=1, py=1）
    lbm::LatticeGrid g_2d(nx, ny, 1, lbm::LatticeModel::D2Q9);
    for (int i = 0; i < g_2d.size(); ++i) g_2d.rho[i] = 1.0 + 0.01 * (i % 7);
    lbm::Solver solver_2d(g_2d, omega);
    const auto decomp2d = lbm::MpiDecomp2D::create(nx, ny, 1, 1);
    solver_2d.attach_mpi2d(&decomp2d);
    for (int t = 0; t < 100; ++t) solver_2d.step();

    constexpr double TOL = 1e-14;
    bool ok = true;
    for (int i = 0; i < g_ref.size() && ok; ++i) {
        if (std::abs(g_ref.rho[i] - g_2d.rho[i]) > TOL) { ok = false; break; }
        if (std::abs(g_ref.u[i * 2 + 0] - g_2d.u[i * 2 + 0]) > TOL) { ok = false; break; }
        if (std::abs(g_ref.u[i * 2 + 1] - g_2d.u[i * 2 + 1]) > TOL) { ok = false; break; }
    }
    std::printf("[MPI] attach_mpi2d single-rank no-op (results identical): %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// 测试：MgTree 基本结构（单进程，2D，三层嵌套）
// ---------------------------------------------------------------------------
static int test_mg_tree_basic()
{
    // 创建根节点（粗网格 256×256，2D）
    lbm::MgExtent root_ext{0, 255, 0, 255, 0, 0};
    lbm::MgTree tree(root_ext, lbm::MgDim::D2);

    bool ok = true;

    // 根节点验证
    auto* root = tree.root();
    ok &= (root != nullptr);
    ok &= (root->level == 0);
    ok &= (root->refine_ratio == 1);
    ok &= root->is_root();
    ok &= root->is_leaf();
    ok &= !root->has_grid();
    ok &= (root->extent.nx() == 256);
    ok &= (root->extent.ny() == 256);
    ok &= (root->extent.nz() == 1);
    ok &= !root->extent.is_3d();
    ok &= (tree.max_level() == 0);
    ok &= (tree.node_count() == 1);

    // 在粗网格中嵌套第一层细化（中心 64×64 区域，加密比 2）
    lbm::MgExtent child1_ext{96, 159, 96, 159, 0, 0};
    auto* child1 = tree.add_level(root, child1_ext, 2);
    ok &= (child1 != nullptr);
    ok &= (child1->level == 1);
    ok &= (child1->refine_ratio == 2);
    ok &= !child1->is_root();
    ok &= child1->is_leaf();
    ok &= (child1->parent == root);
    ok &= !root->is_leaf();
    ok &= ((int)root->children.size() == 1);
    ok &= (tree.max_level() == 1);
    ok &= (tree.node_count() == 2);

    // 在第一层细化中嵌套第二层（更细，加密比 4）
    lbm::MgExtent child2_ext{112, 143, 112, 143, 0, 0};
    auto* child2 = tree.add_level(child1, child2_ext, 4);
    ok &= (child2 != nullptr);
    ok &= (child2->level == 2);
    ok &= (child2->refine_ratio == 4);
    ok &= !child2->is_root();
    ok &= child2->is_leaf();
    ok &= (child2->parent == child1);
    ok &= (tree.max_level() == 2);
    ok &= (tree.node_count() == 3);

    // nodes_at_level
    auto lvl0 = tree.nodes_at_level(0);
    auto lvl1 = tree.nodes_at_level(1);
    auto lvl2 = tree.nodes_at_level(2);
    ok &= (lvl0.size() == 1 && lvl0[0] == root);
    ok &= (lvl1.size() == 1 && lvl1[0] == child1);
    ok &= (lvl2.size() == 1 && lvl2[0] == child2);

    // volume_ratio
    ok &= (child1->volume_ratio() == 4);   // 2D: refine_ratio^2 = 4
    ok &= (child2->volume_ratio() == 16);  // 2D: 4^2 = 16

    // 绑定 LatticeGrid（仅验证 has_grid 状态）
    lbm::LatticeGrid g1(256, 256, 1, lbm::LatticeModel::D2Q9);
    root->grid = &g1;
    ok &= root->has_grid();
    root->grid = nullptr;
    ok &= !root->has_grid();

    std::printf("[MgTree] basic structure (2D 3-level): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// 测试：MgTree 遍历顺序（粗→细 和 细→粗）
// ---------------------------------------------------------------------------
static int test_mg_tree_traversal()
{
    // 构建三层树：root → child1a, child1b → child2（child1b 的子节点）
    lbm::MgExtent root_ext{0, 127, 0, 127, 0, 0};
    lbm::MgTree tree(root_ext);
    auto* root   = tree.root();
    auto* c1a    = tree.add_level(root, {0, 63, 0, 63, 0, 0}, 2);
    auto* c1b    = tree.add_level(root, {64, 127, 64, 127, 0, 0}, 2);
    auto* c2     = tree.add_level(c1b, {80, 111, 80, 111, 0, 0}, 2);
    (void)c1a; (void)c2;

    // 粗→细遍历（BFS）：应为 root, c1a/c1b（任意顺序），c2
    std::vector<int> visit_ctf;
    tree.traverse_coarse_to_fine([&](lbm::MgNode* n) {
        visit_ctf.push_back(n->level);
    });
    bool ok = true;
    ok &= (visit_ctf.size() == 4);
    ok &= (visit_ctf[0] == 0);                       // root 第一
    ok &= (visit_ctf[1] == 1 && visit_ctf[2] == 1);  // 两个 level-1 节点
    ok &= (visit_ctf[3] == 2);                        // c2 最后

    // 细→粗遍历（BFS 逆序）：应为 c2, c1b, c1a, root
    std::vector<int> visit_ftc;
    tree.traverse_fine_to_coarse([&](lbm::MgNode* n) {
        visit_ftc.push_back(n->level);
    });
    ok &= (visit_ftc.size() == 4);
    ok &= (visit_ftc[0] == 2);                        // c2 第一
    ok &= (visit_ftc[1] == 1 && visit_ftc[2] == 1);  // 两个 level-1 节点
    ok &= (visit_ftc[3] == 0);                        // root 最后

    std::printf("[MgTree] traversal order (coarse-to-fine / fine-to-coarse): %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// 测试：MpiDecomp3D 在单进程（无 MPI 构建）下的正确初始化
// ---------------------------------------------------------------------------
static int test_mpi_decomp3d_single_rank()
{
    const int gnx = 64, gny = 48, gnz = 32;
    bool ok = true;
    try {
        const auto d = lbm::MpiDecomp3D::create(gnx, gny, gnz, 1, 1, 1);
        ok &= (d.rank == 0 && d.nprocs == 1);
        ok &= (d.px == 1 && d.py == 1 && d.pz == 1);
        ok &= (d.col_rank == 0 && d.row_rank == 0 && d.pz_rank == 0);
        ok &= (d.local_nx == gnx && d.local_ny == gny && d.local_nz == gnz);
        ok &= (d.x_start == 0 && d.x_end == gnx - 1);
        ok &= (d.y_start == 0 && d.y_end == gny - 1);
        ok &= (d.z_start == 0 && d.z_end == gnz - 1);
        // 无幽灵层（单进程）
        ok &= !d.has_west_ghost() && !d.has_east_ghost();
        ok &= !d.has_south_ghost() && !d.has_north_ghost();
        ok &= !d.has_bottom_ghost() && !d.has_top_ghost();
        // 有全局壁面
        ok &= d.has_west_wall() && d.has_east_wall();
        ok &= d.has_south_wall() && d.has_north_wall();
        ok &= d.has_bottom_wall() && d.has_top_wall();
        // grid_n* == local_n*（无幽灵）
        ok &= (d.grid_nx() == gnx && d.grid_ny() == gny && d.grid_nz() == gnz);
    } catch (...) {
        ok = false;
    }
    std::printf("[MPI] MpiDecomp3D single-rank fields: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}



// ---------------------------------------------------------------------------
// 测试：mg_prolong_rho_u — 双线性延拓（粗→细）
//
// 常数场验证：若粗格 ρ/u 处处相等，延拓后细格应精确还原该常数（插值不改变常数场）。
// 线性场验证：ρ(i,j) = a*i + b*j + c，双线性插值应精确还原（线性场是双线性的特例）。
// ---------------------------------------------------------------------------
static int test_mg_prolong_rho_u()
{
    bool ok = true;

    // --- 子测试 1：常数场 ---
    {
        lbm::LatticeGrid coarse(4, 4, 1, lbm::LatticeModel::D2Q9);
        for (int k = 0; k < coarse.size(); ++k) {
            coarse.rho[k]       = 1.25;
            coarse.u[k * 2 + 0] = 0.03;
            coarse.u[k * 2 + 1] = -0.01;
        }
        lbm::LatticeGrid fine(4, 4, 1, lbm::LatticeModel::D2Q9);

        lbm::MgTree tree({0, 3, 0, 3, 0, 0}, lbm::MgDim::D2);
        auto* root_node = tree.root();
        root_node->grid  = &coarse;
        auto* fine_node  = tree.add_level(root_node, {0, 1, 0, 1, 0, 0}, 2);
        fine_node->grid  = &fine;
        lbm::mg_prolong_rho_u(*root_node, *fine_node);

        for (int k = 0; k < fine.size(); ++k) {
            ok &= (std::fabs(fine.rho[k] - 1.25) < 1e-12);
            ok &= (std::fabs(fine.u[k * 2 + 0] - 0.03)  < 1e-12);
            ok &= (std::fabs(fine.u[k * 2 + 1] - (-0.01)) < 1e-12);
        }
    }

    // --- 子测试 2：线性场 ρ(i,j) = 1.0 + 0.1*i + 0.05*j（以粗节点整数坐标为参考）---
    // 双线性插值对线性函数精确：ρ_f = ρ(px, py) 精确（无边界夹持时）
    {
        lbm::LatticeGrid coarse(4, 4, 1, lbm::LatticeModel::D2Q9);
        for (int j = 0; j < 4; ++j) {
            for (int i = 0; i < 4; ++i) {
                coarse.rho[coarse.idx(i, j)] = 1.0 + 0.1 * i + 0.05 * j;
                coarse.u[coarse.idx(i, j) * 2 + 0] = 0.0;
                coarse.u[coarse.idx(i, j) * 2 + 1] = 0.0;
            }
        }
        // 细网格覆盖粗格 {1,2,1,2}（远离边界，避免夹持），加密比 2 → 3×3 细节点
        // fine_extent.nx() = 2, r=2 → 4 细节点 (0..3) 覆盖粗坐标 1.0..2.0
        lbm::LatticeGrid fine(4, 4, 1, lbm::LatticeModel::D2Q9);

        lbm::MgTree tree2({0, 3, 0, 3, 0, 0}, lbm::MgDim::D2);
        auto* root2 = tree2.root();
        root2->grid  = &coarse;
        // fine extent {1,2,1,2}: fine_nx = (2-1+1)*2 = 4, fine_ny = 4
        auto* fine2  = tree2.add_level(root2, {1, 2, 1, 2, 0, 0}, 2);
        fine2->grid  = &fine;
        lbm::mg_prolong_rho_u(*root2, *fine2);

        // fine cell (if,jf): px = 1 + if/2, py = 1 + jf/2
        // 在 0 <= if <= 4-1=3 时，px in [1.0, 2.5]；至多夹持在 coarse x_end=3
        // 对 if=0..3：px = 1.0, 1.5, 2.0, 2.5 → 无夹持（2.5 < 3）
        const int r2 = 2;
        for (int jf = 0; jf < 4; ++jf) {
            for (int ix = 0; ix < 4; ++ix) {
                const double px = 1 + static_cast<double>(ix) / r2;
                const double py = 1 + static_cast<double>(jf)  / r2;
                // 双线性对线性函数精确（节点坐标无需夹持）
                const double expected = 1.0 + 0.1 * px + 0.05 * py;
                const double actual   = fine.rho[fine.idx(ix, jf)];
                ok &= (std::fabs(actual - expected) < 1e-10);
            }
        }
    }

    std::printf("[MgTree] mg_prolong_rho_u bilinear: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// 测试：mg_restrict_rho_u — 体积平均限制（细→粗）
//
// 细网格 4×4（覆盖粗格 {0,1, 0,1}，加密比 2）。
// 细网格赋常数 ρ=1.5，限制后粗格对应区域应得 ρ=1.5（精确）。
// ---------------------------------------------------------------------------
static int test_mg_restrict_rho_u()
{
    // 粗网格 4×4
    lbm::LatticeGrid coarse(4, 4, 1, lbm::LatticeModel::D2Q9);

    // 细网格：覆盖粗格 {0,1, 0,1}，加密比 2 → 4×4 细格
    lbm::LatticeGrid fine(4, 4, 1, lbm::LatticeModel::D2Q9);
    // 赋常数宏观量
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            const int fi = fine.idx(i, j);
            fine.rho[fi]       = 1.5;
            fine.u[fi * 2 + 0] = 0.01;
            fine.u[fi * 2 + 1] = 0.02;
        }
    }

    lbm::MgTree tree({0, 3, 0, 3, 0, 0}, lbm::MgDim::D2);
    auto* root_node = tree.root();
    root_node->grid  = &coarse;
    auto* fine_node  = tree.add_level(root_node, {0, 1, 0, 1, 0, 0}, 2);
    fine_node->grid  = &fine;

    // 初始化粗格为0
    for (int k = 0; k < coarse.size(); ++k) coarse.rho[k] = 0.0;
    lbm::mg_restrict_rho_u(*fine_node, *root_node);

    bool ok = true;
    // 粗格 (0,0) 和 (1,0), (0,1), (1,1) 都被 4 个细格平均 → ρ = 1.5
    for (int jc = 0; jc <= 1; ++jc) {
        for (int ic = 0; ic <= 1; ++ic) {
            const double rho_c = coarse.rho[coarse.idx(ic, jc)];
            ok &= (std::fabs(rho_c - 1.5) < 1e-12);
            const double ux_c  = coarse.u[coarse.idx(ic, jc) * 2 + 0];
            const double uy_c  = coarse.u[coarse.idx(ic, jc) * 2 + 1];
            ok &= (std::fabs(ux_c - 0.01) < 1e-12);
            ok &= (std::fabs(uy_c - 0.02) < 1e-12);
        }
    }
    // 粗格 (2..3, 2..3) 未被修改，保持 0
    ok &= (std::fabs(coarse.rho[coarse.idx(2, 2)]) < 1e-12);

    std::printf("[MgTree] mg_restrict_rho_u volume-avg: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ---------------------------------------------------------------------------
// 测试：MgTree overflow guard（refine_ratio 超限时抛出异常）
// 和 void* → std::variant decomp 字段
// ---------------------------------------------------------------------------
static int test_mg_tree_guards()
{
    lbm::MgTree tree({0, 63, 0, 63, 0, 0}, lbm::MgDim::D2);
    auto* root = tree.root();
    bool ok = true;

    // refine_ratio > 1024 应抛出异常
    bool threw = false;
    try {
        tree.add_level(root, {0, 31, 0, 31, 0, 0}, 1025);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ok &= threw;

    // refine_ratio < 1 应抛出异常
    threw = false;
    try {
        tree.add_level(root, {0, 31, 0, 31, 0, 0}, 0);
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    ok &= threw;

    // refine_ratio == 1024 应正常通过
    bool noThrow = true;
    try {
        tree.add_level(root, {0, 31, 0, 31, 0, 0}, 1024);
    } catch (...) {
        noThrow = false;
    }
    ok &= noThrow;

    // std::variant decomp 默认为 monostate（未绑定）
    ok &= !root->has_decomp();

    // 绑定 MpiDecomp2D*（仅测试类型安全，不测试 MPI 行为）
    auto decomp2d = lbm::MpiDecomp2D::create(64, 64, 1, 1);
    root->decomp = &decomp2d;
    ok &= root->has_decomp();
    ok &= std::holds_alternative<lbm::MpiDecomp2D*>(root->decomp);

    // volume_ratio 用 long long 防溢出
    auto* child = tree.nodes_at_level(1)[0];
    const long long vr = child->volume_ratio();
    ok &= (vr == 1024LL * 1024LL);  // D2: 1024^2


    std::printf("[MgTree] overflow guards + std::variant decomp: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ===========================================================================
// 拉伸网格（StretchedGrid）测试
// ===========================================================================

// 测试：等比拉伸网格构造 — 坐标单调递增、端点精确、间距比正确
static int test_stretched_grid_geometric()
{
    // 构建 x 方向等比拉伸（ratio=1.1）、y 方向均匀（ratio=1.0）网格
    const int nx = 20, ny = 10;
    auto sg = lbm::StretchedGrid::build_geometric(
        nx, ny, 1,
        0.0, 1.0,   // x 范围
        0.0, 1.0,   // y 范围
        1.1, 1.0,   // x 等比 1.1, y 均匀
        lbm::LatticeModel::D2Q9);

    bool ok = true;

    // 检查节点数
    ok &= (sg.lattice.nx == nx);
    ok &= (sg.lattice.ny == ny);

    // 检查端点精确
    ok &= (std::abs(sg.x_phys[0]      - 0.0) < 1e-12);
    ok &= (std::abs(sg.x_phys[nx - 1] - 1.0) < 1e-12);
    ok &= (std::abs(sg.y_phys[0]      - 0.0) < 1e-12);
    ok &= (std::abs(sg.y_phys[ny - 1] - 1.0) < 1e-12);

    // 检查 x 方向单调递增
    for (int i = 0; i + 1 < nx; ++i) {
        if (sg.x_phys[i + 1] <= sg.x_phys[i]) { ok = false; break; }
    }

    // 检查 y 方向均匀（间距误差 < 1e-10）
    const double dy_expected = 1.0 / (ny - 1);
    for (int j = 0; j + 1 < ny; ++j) {
        if (std::abs(sg.y_phys[j + 1] - sg.y_phys[j] - dy_expected) > 1e-10) {
            ok = false; break;
        }
    }

    // 检查拉伸比
    const double sr_x = sg.max_stretch_ratio_x();
    const double sr_y = sg.max_stretch_ratio_y();
    ok &= (sr_x > 1.0 && sr_x < 1.15);   // 等比 1.1，最大比略大于 1.0
    ok &= (sr_y < 1.001);                 // 均匀时 ≈ 1.0

    // 检查 dx/dy 数组大小和最后一个值（外推）
    ok &= (static_cast<int>(sg.dx.size()) == nx);
    ok &= (static_cast<int>(sg.dy.size()) == ny);
    ok &= (sg.dx[nx - 1] == sg.dx[nx - 2]);
    ok &= (sg.dy[ny - 1] == sg.dy[ny - 2]);

    std::printf("[StretchedGrid] geometric construction: sr_x=%.3f sr_y=%.3f → %s\n",
                sr_x, sr_y, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试：tanh 拉伸网格 — 两端节点比中央密（边界层效果）
static int test_stretched_grid_tanh()
{
    const int nx = 50, ny = 50;
    // delta=0.8 → 适度拉伸，两端约 2× 比中央密
    auto sg = lbm::StretchedGrid::build_tanh(
        nx, ny, 1,
        0.0, 1.0,
        0.0, 1.0,
        0.8, 0.8,
        lbm::LatticeModel::D2Q9);

    bool ok = true;

    // 端点精确
    ok &= (std::abs(sg.x_phys[0]      - 0.0) < 1e-12);
    ok &= (std::abs(sg.x_phys[nx - 1] - 1.0) < 1e-12);
    ok &= (std::abs(sg.y_phys[0]      - 0.0) < 1e-12);
    ok &= (std::abs(sg.y_phys[ny - 1] - 1.0) < 1e-12);

    // 单调递增
    for (int i = 0; i + 1 < nx; ++i) {
        if (sg.x_phys[i + 1] <= sg.x_phys[i]) { ok = false; break; }
    }

    // tanh 拉伸：两端间距 < 中央间距（边界层效果）
    const double dx_left   = sg.dx[0];
    const double dx_center = sg.dx[nx / 2];
    ok &= (dx_left < dx_center);   // 左端比中央更密

    // delta=0 时应退化为均匀
    auto sg_uniform = lbm::StretchedGrid::build_tanh(
        20, 20, 1,
        0.0, 1.0, 0.0, 1.0,
        0.0, 0.0);
    const double dy_u = sg_uniform.y_phys[1] - sg_uniform.y_phys[0];
    for (int j = 0; j + 1 < 20; ++j) {
        if (std::abs((sg_uniform.y_phys[j + 1] - sg_uniform.y_phys[j]) - dy_u) > 1e-10) {
            ok = false; break;
        }
    }

    std::printf("[StretchedGrid] tanh construction: dx_left=%.4f dx_center=%.4f → %s\n",
                dx_left, dx_center, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试：自定义坐标 + 错误检查
static int test_stretched_grid_custom()
{
    bool ok = true;

    // 正常构建
    std::vector<double> xc = {0.0, 0.1, 0.3, 0.6, 1.0};
    std::vector<double> yc = {0.0, 0.5, 1.0};
    auto sg = lbm::StretchedGrid::build_custom(1, xc, yc, lbm::LatticeModel::D2Q9);
    ok &= (sg.lattice.nx == 5);
    ok &= (sg.lattice.ny == 3);
    ok &= (std::abs(sg.x_phys[2] - 0.3) < 1e-14);
    ok &= (std::abs(sg.dx[0] - 0.1) < 1e-14);

    // 非单调坐标应抛出异常
    bool threw_x = false;
    try {
        std::vector<double> bad = {0.0, 0.5, 0.3, 1.0};  // 非单调
        lbm::StretchedGrid::build_custom(1, bad, {0.0, 1.0});
    } catch (const std::invalid_argument&) {
        threw_x = true;
    }
    ok &= threw_x;

    // 空坐标应抛出异常
    bool threw_empty = false;
    try {
        lbm::StretchedGrid::build_custom(1, {}, {0.0, 1.0});
    } catch (const std::invalid_argument&) {
        threw_empty = true;
    }
    ok &= threw_empty;

    std::printf("[StretchedGrid] custom coordinates: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试：IS-LBM 修正基本性质
//   - 在均匀网格上，修正应产生与标准流式完全相同（幂等）的结果
//   - 拉伸比低于阈值时，修正自动跳过（f 不变）
//   - 拉伸网格上，修正后 f 值有限（无 NaN/Inf）
static int test_stretched_grid_islbm()
{
    bool ok = true;

    // ---- 1. 均匀网格（ratio=1.0）：IS-LBM 修正后结果应与修正前相同（幂等）
    {
        const int nx = 16, ny = 16;
        auto sg = lbm::StretchedGrid::build_geometric(
            nx, ny, 1, 0.0, 1.0, 0.0, 1.0,
            1.0, 1.0);

        // 初始化为有值的 f（非零速度）
        lbm::Solver solver(sg.lattice, 1.0);
        lbm::BoundaryCondition bc;
        bc.type = lbm::BCType::BounceBack; bc.face = lbm::Face::South;
        solver.add_boundary_condition(bc);
        bc.face = lbm::Face::North; solver.add_boundary_condition(bc);
        solver.step();

        // 记录 step() 后的 f 值
        std::vector<double> f_before = sg.lattice.f;

        // 施加 IS-LBM 修正（均匀网格，拉伸比 = 1.0 < 阈值 1.05 → 应自动跳过）
        lbm::islbm_interpolation_correction(sg, 1.05);
        const std::vector<double>& f_after = sg.lattice.f;

        for (int i = 0; i < static_cast<int>(f_before.size()); ++i) {
            if (std::abs(f_before[i] - f_after[i]) > 1e-12) { ok = false; break; }
        }
    }

    // ---- 2. 拉伸网格（ratio=1.2 > 阈值）：修正后 f 有限、质量守恒
    {
        const int nx = 32, ny = 32;
        auto sg = lbm::StretchedGrid::build_geometric(
            nx, ny, 1, 0.0, 1.0, 0.0, 1.0,
            1.2, 1.2);

        lbm::Solver solver(sg.lattice, 1.0);
        // 运行几步使 f 有一定变化
        for (int t = 0; t < 5; ++t) solver.step();

        // 修正前总质量
        double mass_before = 0.0;
        for (double r : sg.lattice.rho) mass_before += r;

        // 施加 IS-LBM 修正（拉伸比 > 阈值，实际修正）
        lbm::islbm_interpolation_correction(sg, 1.05);

        // 检查 f 值有限
        for (int i = 0; i < static_cast<int>(sg.lattice.f.size()); ++i) {
            if (!std::isfinite(sg.lattice.f[i])) { ok = false; break; }
        }

        // 修正后总质量（双线性插值保持连续性，质量应基本守恒，允许 0.1% 偏差）
        double mass_after = 0.0;
        for (double r : sg.lattice.rho) mass_after += r;
        ok &= (std::abs(mass_after - mass_before) / mass_before < 0.001);
    }

    std::printf("[StretchedGrid] IS-LBM correction (idempotent uniform, finite stretched): %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ===========================================================================
// 多重网格改进：mg_prolong_f、mg_apply_fringe_bc、AMR 指标
// ===========================================================================

// 测试：mg_prolong_f — 延拓后细网格 f 为有效平衡分布（sum_a f_a = rho, f finite）
static int test_mg_prolong_f()
{
    // 创建 2 层网格：粗 32×32，细 16×16（覆盖粗网格中心区域 8..23）
    lbm::MgTree tree({0, 31, 0, 31, 0, 0});
    auto* fine_node = tree.add_level(tree.root(), {8, 23, 8, 23, 0, 0}, 2);

    lbm::LatticeGrid coarse_g(32, 32, 1, lbm::LatticeModel::D2Q9);
    lbm::LatticeGrid fine_g  (32, 32, 1, lbm::LatticeModel::D2Q9);

    // 初始化粗网格：均匀密度 + 非零速度
    const double omega = 1.0;
    lbm::Solver coarse_solver(coarse_g, omega);
    for (int t = 0; t < 10; ++t) coarse_solver.step();

    tree.root()->grid = &coarse_g;
    fine_node->grid   = &fine_g;

    // 执行 f 延拓
    lbm::mg_prolong_f(*tree.root(), *fine_node);

    bool ok = true;

    // 验证细网格 f 有限
    for (double v : fine_g.f) {
        if (!std::isfinite(v)) { ok = false; break; }
    }

    // 验证细网格各节点 sum_a f_a = rho（平衡分布守恒性）
    for (int n = 0; n < fine_g.size(); ++n) {
        double sum = 0.0;
        for (int a = 0; a < lbm::d2q9::Q; ++a) {
            sum += fine_g.f[n * lbm::d2q9::Q + a];
        }
        if (std::abs(sum - fine_g.rho[n]) > 1e-10) { ok = false; break; }
    }

    // 验证 f_tmp 已同步（首步流式迁移时不会使用未初始化数据）
    for (int i = 0; i < static_cast<int>(fine_g.f.size()); ++i) {
        if (std::abs(fine_g.f[i] - fine_g.f_tmp[i]) > 1e-14) { ok = false; break; }
    }

    std::printf("[MgTree] mg_prolong_f (f = f_eq(rho,u) + f_tmp synced): %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试：mg_apply_fringe_bc — fringe 区域 f 从粗网格更新，内部区域不变
static int test_mg_apply_fringe_bc()
{
    lbm::MgTree tree({0, 31, 0, 31, 0, 0});
    auto* fine_node = tree.add_level(tree.root(), {4, 27, 4, 27, 0, 0}, 2);

    lbm::LatticeGrid coarse_g(32, 32, 1, lbm::LatticeModel::D2Q9);
    lbm::LatticeGrid fine_g  (48, 48, 1, lbm::LatticeModel::D2Q9);

    // 粗网格 rho 设为非均匀（便于检测是否被插值进 fringe）
    for (int n = 0; n < coarse_g.size(); ++n) {
        coarse_g.rho[n] = 1.0 + 0.01 * (n % 7);
        // u 保持 0（默认已初始化为 0）
    }
    // 不调用 compute_macroscopic()——它需要有效的 f 值才能工作。
    // mg_apply_fringe_bc 只读 rho 和 u，无需 f。

    // 细网格 f 设为已知值（便于检测 fringe 是否被覆盖）
    for (double& v : fine_g.f) v = 42.0;

    tree.root()->grid = &coarse_g;
    fine_node->grid   = &fine_g;

    // 施加 fringe BC（宽度 2 格）
    lbm::mg_apply_fringe_bc(*tree.root(), *fine_node, 2);

    bool ok = true;
    const int fnx = fine_g.nx;  // = 48

    // fringe 区域（前2列/后2列/前2行/后2行）：f 应已被覆盖（≠ 42）
    // 检查 (0,0)（左下角，在 fringe 内）
    const int n_fringe = fine_g.idx(0, 0);
    ok &= (fine_g.f[n_fringe * lbm::d2q9::Q] != 42.0);

    // 内部节点 (10,10)：f 应仍为 42（未被修改）
    const int n_inner = fine_g.idx(10, 10);
    ok &= (fine_g.f[n_inner * lbm::d2q9::Q] == 42.0);

    // fringe 区域的 f 值有限
    for (int j = 0; j < 2; ++j) {
        for (int i = 0; i < fnx; ++i) {
            const int n = fine_g.idx(i, j);
            for (int a = 0; a < lbm::d2q9::Q; ++a) {
                if (!std::isfinite(fine_g.f[n * lbm::d2q9::Q + a])) { ok = false; }
            }
        }
    }

    std::printf("[MgTree] mg_apply_fringe_bc (fringe updated, interior unchanged): %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试：AMR 细化指标 — mg_compute_refinement_indicator（密度梯度正确性）
static int test_mg_refinement_indicator()
{
    // 创建含密度梯度的网格（右半部分 rho=1.1，左半部分 rho=1.0）
    const int nx = 20, ny = 10;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);

    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            g.rho[g.idx(i, j)] = (i >= nx / 2) ? 1.1 : 1.0;
        }
    }

    lbm::MgTree tree({0, nx - 1, 0, ny - 1, 0, 0});
    tree.root()->grid = &g;

    std::vector<double> indicator;
    lbm::mg_compute_refinement_indicator(*tree.root(), indicator);

    bool ok = true;
    ok &= (static_cast<int>(indicator.size()) == g.size());

    // 分界面附近（i = nx/2 - 1 或 i = nx/2）的指标值应比均匀区域大
    double max_indicator = *std::max_element(indicator.begin(), indicator.end());
    // 远离分界面的节点（e.g. i=2, j=0）指标应约为 0
    const double ind_interior = indicator[g.idx(2, ny / 2)];
    ok &= (max_indicator > ind_interior * 10.0);
    ok &= (max_indicator > 0.0);

    // 均匀区域指标 = 0（中心差分后 dρ/dx = 0）
    ok &= (std::abs(ind_interior) < 1e-14);

    // mg_total_subcycle_steps 测试（顺便放在这里）
    lbm::MgTree tree2({0, 63, 0, 63, 0, 0});
    auto* l1 = tree2.add_level(tree2.root(), {16, 47, 16, 47, 0, 0}, 2);
    auto* l2 = tree2.add_level(l1,            {24, 39, 24, 39, 0, 0}, 4);
    ok &= (lbm::mg_total_subcycle_steps(*tree2.root()) == 1);
    ok &= (lbm::mg_total_subcycle_steps(*l1) == 2);
    ok &= (lbm::mg_total_subcycle_steps(*l2) == 8);

    std::printf("[MgTree] mg_compute_refinement_indicator + total_subcycle_steps: "
                "max_ind=%.4f interior_ind=%.2e → %s\n",
                max_indicator, ind_interior, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ===========================================================================
// MPI BC 壁面所有权过滤测试（修复分块边界速度阶跃）
// ===========================================================================

// 测试：PhysicalBounds::has_south_wall/has_north_wall=false 时
//       South/North BC 应完全跳过（不修改内部物理行），防止 MPI 分块边界速度阶跃。
//
// 复现场景：4 进程 1D Y 分解中，rank 1（内部进程）的 has_south_wall=false，
//           因此 South BounceBack 不应改变 j=1 行的 f 值。
//           若改变，则在分块边界处产生速度 kink（即问题所描述的物理失真）。
static int test_bc_wall_ownership_filter()
{
    // 模拟内部进程（non-wall rank）：local_ny=15, grid_ny=17（含南北幽灵）
    const int nx = 16, local_ny = 15, grid_ny = local_ny + 2;
    lbm::LatticeGrid g(nx, grid_ny, 1, lbm::LatticeModel::D2Q9);

    // 将所有物理行 f 设为已知值，以便检测是否被意外修改
    for (int j = 1; j <= local_ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int n = g.idx(i, j);
            for (int a = 0; a < lbm::d2q9::Q; ++a) {
                g.f    [n * lbm::d2q9::Q + a] = 1.0 + 0.01 * a + 0.001 * j;
                g.f_tmp[n * lbm::d2q9::Q + a] = g.f[n * lbm::d2q9::Q + a];
            }
        }
    }

    // 内部进程的 PhysicalBounds：不拥有南/北物理壁
    lbm::PhysicalBounds pb;
    pb.j_s = 1;        pb.j_n = local_ny;
    pb.i_w = 0;        pb.i_e = nx - 1;
    pb.has_south_wall = false;   // ← 关键：内部进程不持有南壁
    pb.has_north_wall = false;   // ← 内部进程不持有北壁
    pb.has_west_wall  = true;
    pb.has_east_wall  = true;

    // 注册 South + North BounceBack（与实际通道流配置相同）
    std::vector<lbm::BoundaryCondition> bcs;
    lbm::BoundaryCondition bc_s, bc_n;
    bc_s.type = lbm::BCType::BounceBack; bc_s.face = lbm::Face::South;
    bc_n.type = lbm::BCType::BounceBack; bc_n.face = lbm::Face::North;
    bcs.push_back(bc_s);
    bcs.push_back(bc_n);

    // 记录 BC 施加前 j=1（紧邻南幽灵）和 j=local_ny（紧邻北幽灵）的 f 值
    std::vector<double> f_row1_before(nx * lbm::d2q9::Q);
    std::vector<double> f_rowN_before(nx * lbm::d2q9::Q);
    for (int i = 0; i < nx; ++i) {
        for (int a = 0; a < lbm::d2q9::Q; ++a) {
            f_row1_before[i * lbm::d2q9::Q + a] =
                g.f[g.idx(i, 1)        * lbm::d2q9::Q + a];
            f_rowN_before[i * lbm::d2q9::Q + a] =
                g.f[g.idx(i, local_ny) * lbm::d2q9::Q + a];
        }
    }

    // 施加 BC（has_*_wall=false 的面应被完全跳过）
    lbm::apply_boundary_conditions(g, bcs, pb);

    bool ok = true;
    // j=1 的 f 值应完全不变（South BounceBack 已跳过）
    for (int i = 0; i < nx && ok; ++i) {
        for (int a = 0; a < lbm::d2q9::Q && ok; ++a) {
            if (g.f[g.idx(i, 1) * lbm::d2q9::Q + a] !=
                f_row1_before[i * lbm::d2q9::Q + a])
                ok = false;
        }
    }
    // j=local_ny 的 f 值应完全不变（North BounceBack 已跳过）
    for (int i = 0; i < nx && ok; ++i) {
        for (int a = 0; a < lbm::d2q9::Q && ok; ++a) {
            if (g.f[g.idx(i, local_ny) * lbm::d2q9::Q + a] !=
                f_rowN_before[i * lbm::d2q9::Q + a])
                ok = false;
        }
    }

    std::printf("[MPI] BC wall-ownership filter (interior rank skips S/N BC): %s\n",
                ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 互补测试：has_south_wall=true 时 South BounceBack 应实际修改 j=pb.j_s 的 f 值
static int test_bc_wall_ownership_south_applied()
{
    const int nx = 8, ny = 8;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);
    // f 和 f_tmp 设为不同的已知值，以便通过读取 f_tmp[opp] 来检测反弹是否执行
    for (auto& v : g.f)     v = 2.0;
    for (auto& v : g.f_tmp) v = 3.0;  // BounceBack 从 f_tmp 读取

    lbm::PhysicalBounds pb;
    pb.j_s = 0;        pb.j_n = ny - 1;
    pb.i_w = 0;        pb.i_e = nx - 1;
    pb.has_south_wall = true;  // 持有南壁 → BC 应被施加
    pb.has_north_wall = true;
    pb.has_west_wall  = true;
    pb.has_east_wall  = true;

    std::vector<lbm::BoundaryCondition> bcs;
    lbm::BoundaryCondition bc;
    bc.type = lbm::BCType::BounceBack; bc.face = lbm::Face::South;
    bcs.push_back(bc);

    lbm::apply_boundary_conditions(g, bcs, pb);

    // 半步反弹（South 面）：南壁节点的"朝北"幽灵方向 ← f_tmp 中对应的"朝南"方向
    // D2Q9 方向约定：2=N(0,+1), 5=NE(+1,+1), 6=NW(-1,+1) — 均为南壁的未知（幽灵）方向
    //                4=S(0,-1), 7=SW(-1,-1), 8=SE(+1,-1) — 对应的已知（反射来源）方向
    // 反弹：f[2]←f_tmp[4]=3, f[5]←f_tmp[7]=3, f[6]←f_tmp[8]=3（原值均为 2.0）
    bool ok = true;
    for (int i = 0; i < nx && ok; ++i) {
        const double* f = &g.f[g.idx(i, 0) * lbm::d2q9::Q];
        if (std::abs(f[2] - 3.0) > 1e-14) ok = false;   // N  ← f_tmp[S=4]  = 3.0
        if (std::abs(f[5] - 3.0) > 1e-14) ok = false;   // NE ← f_tmp[SW=7] = 3.0
        if (std::abs(f[6] - 3.0) > 1e-14) ok = false;   // NW ← f_tmp[SE=8] = 3.0
    }

    std::printf("[MPI] BC wall-ownership: south wall owner applies BB (f changed): %s\n",
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
    failures += test_inlet_corner_not_overwritten_by_fd();
    failures += test_mpi_decomp_single_rank();
    failures += test_mpi_attach_no_effect();
    failures += test_mpi_halo_exchange_correctness();
    failures += test_mpi_virtual_two_rank_no_ghost_collision();
    failures += test_mpi_decomp2d_single_rank();
    failures += test_mpi_attach_mpi2d_no_effect();
    failures += test_mg_tree_basic();
    failures += test_mg_tree_traversal();
    failures += test_mpi_decomp3d_single_rank();
    failures += test_mg_prolong_rho_u();
    failures += test_mg_restrict_rho_u();
    failures += test_mg_tree_guards();
    failures += test_stretched_grid_geometric();
    failures += test_stretched_grid_tanh();
    failures += test_stretched_grid_custom();
    failures += test_stretched_grid_islbm();
    failures += test_mg_prolong_f();
    failures += test_mg_apply_fringe_bc();
    failures += test_mg_refinement_indicator();
    failures += test_bc_wall_ownership_filter();
    failures += test_bc_wall_ownership_south_applied();
    if (failures == 0)
        std::printf("1: All tests PASSED\n");
    return failures;
}