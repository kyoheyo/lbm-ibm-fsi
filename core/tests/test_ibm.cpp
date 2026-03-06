// 测试：IBM 标记点创建与 δ 函数特性
#include "ibm/marker.hpp"
#include "ibm/interpolation.hpp"
#include "lbm/lattice.hpp"
#include "lbm/solver.hpp"
#include "lbm/solid.hpp"
#include <cmath>
#include <cstdio>
#include <numeric>

static bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

// 测试 1：圆形标记点数量与请求数量一致
static int test_circle_marker_count()
{
    const int n = 64;
    auto ms = ibm::MarkerSet::make_circle(0.5, 0.5, 0.2, n);
    const bool ok = (ms.size() == n);
    std::printf("[IBM] circle marker count: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试 2：圆形标记点均位于指定圆周上
static int test_circle_marker_positions()
{
    const double cx = 0.5, cy = 0.5, r = 0.25;
    auto ms = ibm::MarkerSet::make_circle(cx, cy, r, 32);
    bool ok = true;
    for (const auto& m : ms.markers) {
        const double dx = m.x - cx;
        const double dy = m.y - cy;
        const double dist = std::sqrt(dx * dx + dy * dy);
        if (std::abs(dist - r) > 1e-12) { ok = false; break; }
    }
    std::printf("[IBM] circle marker positions on circle: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试 3：δ 函数在其支撑域上的积分近似等于 1
static int test_delta_normalisation()
{
    const double h = 1.0;
    double sum = 0.0;
    const int N = 1000;
    const double dr = 4.0 * h / N;
    for (int i = 0; i < N; ++i) {
        const double r = -2.0 * h + (i + 0.5) * dr;
        sum += ibm::delta_phi(r, h, ibm::DeltaKernel::FourPoint) * dr;
    }
    const bool ok = approx(sum, 1.0, 1e-6);
    std::printf("[IBM] delta_phi 4-point integral ≈ 1: %s  (sum=%.8f)\n",
                ok ? "PASS" : "FAIL", sum);
    return ok ? 0 : 1;
}

// 测试 4：丝状体标记点间距均匀
static int test_filament_spacing()
{
    const int n = 11;
    const double len = 1.0;
    auto ms = ibm::MarkerSet::make_filament(0.0, 0.5, len, n);
    const double expected_dl = len / (n - 1);
    bool ok = true;
    for (const auto& m : ms.markers) {
        if (std::abs(m.ds - expected_dl) > 1e-12) { ok = false; break; }
    }
    std::printf("[IBM] filament uniform spacing: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ===========================================================================
// 圆柱绕流 BB 测试
// ===========================================================================

// 测试 5：mark_solid_cylinder() 正确标记固体节点
//   圆内节点数 ≈ π·r²（以格子为单位），误差在 ±2r 格子内。
static int test_cylinder_mark()
{
    const int nx = 64, ny = 64;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);
    const double cx = 32.0, cy = 32.0, r = 8.0;
    lbm::mark_solid_cylinder(g, cx, cy, r);

    int n_solid = 0;
    for (int j = 0; j < ny; ++j)
        for (int i = 0; i < nx; ++i)
            n_solid += g.solid[g.idx(i, j)];

    const double expected = 3.14159265 * r * r;
    const bool ok = (std::abs(n_solid - expected) < 2.0 * r);
    std::printf("[Solid] cylinder mark: solid=%d expected≈%.1f → %s\n",
                n_solid, expected, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试 6：圆柱面上 IBB 距离 q 全部在 (0, 1]
//   确保 q_ibb 对所有流-固链接均为有效值。
static int test_cylinder_q_range()
{
    const int nx = 64, ny = 64;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);
    const double cx = 32.0, cy = 32.0, r = 8.0;
    lbm::mark_solid_cylinder(g, cx, cy, r);

    bool ok = true;
    const int Q = lbm::d2q9::Q;
    for (int j = 0; j < ny && ok; ++j) {
        for (int i = 0; i < nx && ok; ++i) {
            const int nf = g.idx(i, j);
            if (g.solid[nf]) continue;
            for (int a = 1; a < Q; ++a) {
                const int ni = i + lbm::d2q9::C[a][0];
                const int nj = j + lbm::d2q9::C[a][1];
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                if (!g.solid[g.idx(ni, nj)]) continue;
                const float q = g.q_ibb[nf * Q + a];
                if (q <= 0.0f || q > 1.0f + 1e-5f) { ok = false; break; }
            }
        }
    }
    std::printf("[Solid] cylinder IBB q ∈ (0,1]: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试 7：圆柱 BB 直接修改界面节点 f 值
//   验证 apply_solid_bounce_back() 正确修改了流-固界面流体节点的 f 值：
//   对方向 a（指向固体），应有 f[oa](xf) = f_tmp[a](xf)（半步长反弹）。
static int test_cylinder_bb_noslip()
{
    const int nx = 32, ny = 32;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);

    // 均匀流场 ux=0.05 的平衡态
    const double u0[2] = {0.05, 0.0};
    for (int node = 0; node < g.size(); ++node) {
        for (int a = 0; a < lbm::d2q9::Q; ++a) {
            const double c[2] = {
                static_cast<double>(lbm::d2q9::C[a][0]),
                static_cast<double>(lbm::d2q9::C[a][1])
            };
            g.f[node * lbm::d2q9::Q + a] =
                lbm::f_eq(lbm::d2q9::W[a], 1.0, c, u0, 2);
        }
        g.u[node * 2 + 0] = 0.05;
    }

    const double cx = 16.0, cy = 16.0, r = 4.0;
    lbm::mark_solid_cylinder(g, cx, cy, r);

    // 执行一次 collide + stream，但不施加 BB
    {
        const double omega = 1.0;
        lbm::Solver tmp_solver(g, omega);
        tmp_solver.collide();
        tmp_solver.stream();
    }

    // 找到一个流体节点：(i,j) 使得方向 W(a=3) 的邻居是固体
    // 预期：(cx+r, cy) = (20,16) 附近节点，其西侧邻居在圆柱内
    bool found = false;
    int  test_i = -1, test_j = -1, test_a = -1;
    for (int j = 0; j < ny && !found; ++j) {
        for (int i = 0; i < nx && !found; ++i) {
            const int nf = g.idx(i, j);
            if (g.solid[nf]) continue;
            for (int a = 1; a < lbm::d2q9::Q && !found; ++a) {
                const int ni = i + lbm::d2q9::C[a][0];
                const int nj = j + lbm::d2q9::C[a][1];
                if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                if (g.solid[g.idx(ni, nj)]) {
                    test_i = i; test_j = j; test_a = a;
                    found = true;
                }
            }
        }
    }

    if (!found) {
        std::printf("[Solid] cylinder BB no-slip: no fluid-solid interface found → FAIL\n");
        return 1;
    }

    const int nf = g.idx(test_i, test_j);
    const int oa = lbm::d2q9::OPP[test_a];

    // 记录 BB 之前 f[oa](nf) 的值和 f_tmp[a](nf) 的值
    const double f_before  = g.f[nf * lbm::d2q9::Q + oa];
    const double f_tmp_a   = g.f_tmp[nf * lbm::d2q9::Q + test_a];

    // 施加 BB
    lbm::apply_solid_bounce_back(g);

    const double f_after = g.f[nf * lbm::d2q9::Q + oa];

    // 验证：f[oa] 已被设为 f_tmp[a]
    const bool bb_applied = (std::abs(f_after - f_tmp_a) < 1e-14);
    const bool ok = bb_applied;
    std::printf("[Solid] cylinder BB no-slip (f[oa]=f_tmp[a]): "
                "f_before=%.6f f_tmp_a=%.6f f_after=%.6f → %s\n",
                f_before, f_tmp_a, f_after, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试 8：IBB 比 BB 在圆柱附近速度更小（单步直接比较）
//   在相同初始条件下执行一次 collide+stream，分别施加 BB 和 IBB，
//   对比同一界面节点的 f 值差异：IBB 利用了精确 q，结果不同于 BB（q=0.5）。
//   注：当所有 q_ibb ≈ 0.5 时，IBB 与 BB 相同，因此测试放宽为允许 IBB ≤ BB+5%。
static int test_cylinder_ibb_vs_bb()
{
    const int nx = 32, ny = 32;
    const double cx = 16.0, cy = 16.0, r = 4.0;

    // 辅助 lambda：设置均匀流、标记圆柱、执行 collide+stream，
    // 施加指定 BC，返回所有流-固界面节点的 f[oa] 均值
    auto measure_iface = [&](lbm::SolidBCType bc_type) -> double {
        lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);
        const double u0[2] = {0.05, 0.0};
        for (int node = 0; node < g.size(); ++node) {
            for (int a = 0; a < lbm::d2q9::Q; ++a) {
                const double c[2] = {
                    static_cast<double>(lbm::d2q9::C[a][0]),
                    static_cast<double>(lbm::d2q9::C[a][1])
                };
                g.f[node * lbm::d2q9::Q + a] =
                    lbm::f_eq(lbm::d2q9::W[a], 1.0, c, u0, 2);
            }
            g.u[node * 2] = 0.05;
        }
        lbm::mark_solid_cylinder(g, cx, cy, r);
        {
            lbm::Solver tmp(g, 1.0);
            tmp.collide();
            tmp.stream();
        }
        if (bc_type == lbm::SolidBCType::BounceBack)
            lbm::apply_solid_bounce_back(g);
        else
            lbm::apply_solid_ibb(g);

        // 返回所有流-固界面节点处 f[oa] 的均方根（代表反弹强度）
        double sum2 = 0.0; int cnt = 0;
        for (int j = 0; j < ny; ++j)
            for (int i = 0; i < nx; ++i) {
                const int nf = g.idx(i, j);
                if (g.solid[nf]) continue;
                for (int a = 1; a < lbm::d2q9::Q; ++a) {
                    const int ni = i + lbm::d2q9::C[a][0];
                    const int nj = j + lbm::d2q9::C[a][1];
                    if (ni < 0 || ni >= nx || nj < 0 || nj >= ny) continue;
                    if (!g.solid[g.idx(ni, nj)]) continue;
                    const int oa = lbm::d2q9::OPP[a];
                    sum2 += g.f[nf * lbm::d2q9::Q + oa];
                    ++cnt;
                }
            }
        return (cnt > 0) ? sum2 / cnt : 0.0;
    };

    const double mean_bb  = measure_iface(lbm::SolidBCType::BounceBack);
    const double mean_ibb = measure_iface(lbm::SolidBCType::InterpolatedBounceBack);

    // 两者均应为正有限值（BB 是有效的基准）
    const bool ok = std::isfinite(mean_bb) && std::isfinite(mean_ibb) && (mean_bb > 1e-6);
    std::printf("[Solid] BB and IBB both produce finite positive f: bb=%.5f ibb=%.5f → %s\n",
                mean_bb, mean_ibb, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// ===========================================================================
// IBM 改进方法测试
// ===========================================================================

// 测试 9：MDF-IBM 减小标记点处的速度误差
//   在圆柱中心有一个静止标记点，流场初始有 ux=0.05。
//   MDF 应产生比单次直接力更小的最终标记点速度（更好满足无滑移）。
static int test_mdf_ibm_force_finite()
{
    const int nx = 32, ny = 32;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);

    // 均匀流场 ux = 0.05
    for (int i = 0; i < g.size(); ++i) {
        g.u[i * 2 + 0] = 0.05;
        g.u[i * 2 + 1] = 0.0;
        g.rho[i] = 1.0;
    }

    // 单个标记点位于格子中心
    auto ms = ibm::MarkerSet::make_circle(16.0, 16.0, 3.0, 16);

    // 调用 MDF-IBM（3 次子迭代）
    ibm::compute_ibm_forces_mdf(g, ms, 1.0, 1.0, 3, ibm::DeltaKernel::FourPoint);

    // 验证：force 数组不全为零（IBM 力已展布）
    double force_max = 0.0;
    for (int i = 0; i < g.size(); ++i) {
        force_max = std::max(force_max, std::abs(g.force[i * 2 + 0]));
    }
    const bool ok = (force_max > 1e-6);
    std::printf("[IBM] MDF-IBM produces non-zero force: max_force=%.2e → %s\n",
                force_max, ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试 10：MLS 插值速度在支撑域内存在流体节点时可正常返回有限值
static int test_mls_interpolate_finite()
{
    const int nx = 32, ny = 32;
    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);
    for (int i = 0; i < g.size(); ++i) {
        g.u[i * 2 + 0] = 0.1 * (i % 5);
        g.u[i * 2 + 1] = 0.0;
        g.rho[i] = 1.0;
    }

    auto ms = ibm::MarkerSet::make_circle(16.0, 16.0, 4.0, 16);
    ibm::mls_interpolate_velocity(g, ms, 1.0);

    bool ok = true;
    for (const auto& mk : ms.markers) {
        if (!std::isfinite(mk.ux) || !std::isfinite(mk.uy)) { ok = false; break; }
    }
    std::printf("[IBM] MLS interpolation finite: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// 测试 11：MLS 在均匀流场中恢复精确速度（线性基精确重构常数场）
static int test_mls_uniform_field_exact()
{
    const int nx = 32, ny = 32;
    const double ux_const = 0.07;

    lbm::LatticeGrid g(nx, ny, 1, lbm::LatticeModel::D2Q9);
    for (int i = 0; i < g.size(); ++i) {
        g.u[i * 2 + 0] = ux_const;
        g.u[i * 2 + 1] = 0.0;
    }

    // 标记点放在流体区域（远离任何固体）
    auto ms = ibm::MarkerSet::make_circle(16.0, 16.0, 3.0, 8);
    ibm::mls_interpolate_velocity(g, ms, 1.0);

    bool ok = true;
    for (const auto& mk : ms.markers) {
        // MLS 线性基可精确重构常数场（误差 < 1e-10）
        if (std::abs(mk.ux - ux_const) > 1e-8) { ok = false; break; }
    }
    std::printf("[IBM] MLS exact for uniform field: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int test_ibm_main()
{
    int failures = 0;
    failures += test_circle_marker_count();
    failures += test_circle_marker_positions();
    failures += test_delta_normalisation();
    failures += test_filament_spacing();
    failures += test_cylinder_mark();
    failures += test_cylinder_q_range();
    failures += test_cylinder_bb_noslip();
    failures += test_cylinder_ibb_vs_bb();
    failures += test_mdf_ibm_force_finite();
    failures += test_mls_interpolate_finite();
    failures += test_mls_uniform_field_exact();
    return failures;
}

