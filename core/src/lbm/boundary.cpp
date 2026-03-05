#include "lbm/boundary.hpp"
#include "lbm/lattice.hpp"
#include <stdexcept>

#ifdef LBM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace lbm {

// ---------------------------------------------------------------------------
// 流式迁移后各面的未知方向（由周期延拓引入幽灵值，需用反弹条件覆盖）
//
// 流式迁移规则：f[a] 在 (i,j) 处由 (i-C[a][0], j-C[a][1]) 推送而来。
// 若源节点在计算域外（通过取模进入），该分布函数为幽灵值，需由 BC 设置。
//
// D2Q9 方向编号：
//   0:(0,0) 1:(+1,0) 2:(0,+1) 3:(-1,0) 4:(0,-1)
//   5:(+1,+1) 6:(-1,+1) 7:(-1,-1) 8:(+1,-1)
//
// 标准反弹边界条件（halfway bounce-back）物理推导：
//   碰后、迁移前，壁面节点 x_b 处朝向壁面方向 a 的分布函数为 f*_a(x_b, t)。
//   粒子碰壁后原路返回，因此迁移后到达 x_b 的朝流体方向（opp[a]）分布函数应为：
//     f[opp[a]](x_b, t+dt) = f*[a](x_b, t)
//   其中 f*（碰后迁移前的值）在 stream() 完成后存于 g.f_tmp
//   （stream() 末尾执行了 std::swap(g.f, g.f_tmp)）。
//
// 使用 g.f_tmp（碰后迁移前值）而非 g.f（迁移后值）作为反弹源，
// 才符合物理上的时序逻辑，避免引入来自相邻内部节点的错误值。
//
// 各面反弹赋值（f[未知方向] = f_tmp[其反方向]，均取壁面节点本身）：
//   南壁 (j=0)    ：f[2]←f_tmp[4], f[5]←f_tmp[7], f[6]←f_tmp[8]
//   北壁 (j=ny-1) ：f[4]←f_tmp[2], f[7]←f_tmp[5], f[8]←f_tmp[6]
//   西壁 (i=0)    ：f[1]←f_tmp[3], f[5]←f_tmp[7], f[8]←f_tmp[6]
//   东壁 (i=nx-1) ：f[3]←f_tmp[1], f[6]←f_tmp[8], f[7]←f_tmp[5]
// ---------------------------------------------------------------------------
static void apply_bounce_back(LatticeGrid& g, Face face)
{
    if (g.model != LatticeModel::D2Q9) {
        // 三维反弹暂未实现，可仿照 D2Q9 扩展
        return;
    }

    const int nx = g.nx;
    const int ny = g.ny;

    switch (face) {
        case Face::South:
            // 南壁 (j=0)：幽灵方向 N(2), NE(5), NW(6) 由周期延拓自 j=ny-1 而来
            // 使用 f_tmp（碰后迁移前）中壁面节点的反方向分量赋值
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nx; ++i) {
                const int n = g.idx(i, 0);
                double*       f  = &g.f    [n * d2q9::Q];
                const double* fp = &g.f_tmp[n * d2q9::Q];  // 碰后迁移前的值
                f[2] = fp[4];   // N  ← 碰后迁移前的 S
                f[5] = fp[7];   // NE ← 碰后迁移前的 SW
                f[6] = fp[8];   // NW ← 碰后迁移前的 SE
            }
            break;

        case Face::North:
            // 北壁 (j=ny-1)：幽灵方向 S(4), SW(7), SE(8) 由周期延拓自 j=0 而来
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nx; ++i) {
                const int n = g.idx(i, ny - 1);
                double*       f  = &g.f    [n * d2q9::Q];
                const double* fp = &g.f_tmp[n * d2q9::Q];  // 碰后迁移前的值
                f[4] = fp[2];   // S  ← 碰后迁移前的 N
                f[7] = fp[5];   // SW ← 碰后迁移前的 NE
                f[8] = fp[6];   // SE ← 碰后迁移前的 NW
            }
            break;

        case Face::West:
            // 西壁 (i=0)：幽灵方向 E(1), NE(5), SE(8) 由周期延拓自 i=nx-1 而来
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 0; j < ny; ++j) {
                const int n = g.idx(0, j);
                double*       f  = &g.f    [n * d2q9::Q];
                const double* fp = &g.f_tmp[n * d2q9::Q];  // 碰后迁移前的值
                f[1] = fp[3];   // E  ← 碰后迁移前的 W
                f[5] = fp[7];   // NE ← 碰后迁移前的 SW
                f[8] = fp[6];   // SE ← 碰后迁移前的 NW
            }
            break;

        case Face::East:
            // 东壁 (i=nx-1)：幽灵方向 W(3), NW(6), SW(7) 由周期延拓自 i=0 而来
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = 0; j < ny; ++j) {
                const int n = g.idx(nx - 1, j);
                double*       f  = &g.f    [n * d2q9::Q];
                const double* fp = &g.f_tmp[n * d2q9::Q];  // 碰后迁移前的值
                f[3] = fp[1];   // W  ← 碰后迁移前的 E
                f[6] = fp[8];   // NW ← 碰后迁移前的 SE
                f[7] = fp[5];   // SW ← 碰后迁移前的 NE
            }
            break;

        default:
            break;
    }
}

// ---------------------------------------------------------------------------
// Zou-He 速度边界条件（二维，D2Q9）
// 参考文献：Zou & He, Phys. Fluids 9(6), 1997
//
// 流式迁移后各面已知/未知方向分析（f_unknown = 来自域外的幽灵值，需由 BC 设置）：
//   北壁 (j=ny-1)：已知 f[2](N), f[5](NE), f[6](NW)（来自 j=ny-2 内部）
//                  未知 f[4](S),  f[7](SW),  f[8](SE)（来自 j=0 周期延拓）
//   南壁 (j=0)   ：已知 f[4](S),  f[7](SW),  f[8](SE)（来自 j=1  内部）
//                  未知 f[2](N),  f[5](NE),  f[6](NW)（来自 j=ny-1 周期延拓）
//   西壁 (i=0)   ：已知 f[3](W),  f[6](NW),  f[7](SW)（来自 i=1  内部）
//                  未知 f[1](E),  f[5](NE),  f[8](SE)（来自 i=nx-1 周期延拓）
//   东壁 (i=nx-1)：已知 f[1](E),  f[5](NE),  f[8](SE)（来自 i=nx-2 内部）
//                  未知 f[3](W),  f[6](NW),  f[7](SW)（来自 i=0 周期延拓）
// ---------------------------------------------------------------------------
static void apply_zou_he_velocity(LatticeGrid& g, const BoundaryCondition& bc)
{
    if (g.model != LatticeModel::D2Q9) return;

    const int nx = g.nx;
    const int ny = g.ny;

    if (bc.face == Face::North) {
        // 北壁（j=ny-1）：驱动盖板，规定速度 (ux, uy)
        // 已知方向：f[2](N), f[5](NE), f[6](NW) — 来自 j=ny-2 内部节点
        // 未知方向：f[4](S), f[7](SW),  f[8](SE) — 来自 j=0 周期延拓，需由 BC 覆盖
        //
        // 由宏观约束推导：
        //   ρ*(1+uy) = f[0]+f[1]+f[3] + 2*(f[2]+f[5]+f[6])
        //   f[4] = f[2] - (2/3)*ρ*uy
        //   f[7] = f[5] + (1/2)*(f[1]-f[3]) - (1/2)*ρ*ux - (1/6)*ρ*uy
        //   f[8] = f[6] - (1/2)*(f[1]-f[3]) + (1/2)*ρ*ux - (1/6)*ρ*uy
        const double ux = bc.ux;
        const double uy = bc.uy;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < nx; ++i) {
            const int n = g.idx(i, ny - 1);
            double* f = &g.f[n * d2q9::Q];
            // 利用已知的内部方向 f[2], f[5], f[6] 计算壁面密度
            double rho_w = (f[0] + f[1] + f[3]
                          + 2.0 * (f[2] + f[5] + f[6]))
                         / (1.0 + uy);
            // 设置未知的幽灵方向
            f[4] = f[2] - (2.0 / 3.0) * rho_w * uy;
            f[7] = f[5] + 0.5 * (f[1] - f[3])
                        - 0.5 * rho_w * ux
                        - (1.0 / 6.0) * rho_w * uy;
            f[8] = f[6] - 0.5 * (f[1] - f[3])
                        + 0.5 * rho_w * ux
                        - (1.0 / 6.0) * rho_w * uy;
            // 更新宏观量，供下一步碰撞使用
            g.rho[n] = rho_w;
            g.u[n * 2 + 0] = ux;
            g.u[n * 2 + 1] = uy;
        }
    }
    else if (bc.face == Face::South) {
        // 南壁（j=0）：规定速度 (ux, uy)，常用于入口 BC
        // 已知方向：f[4](S), f[7](SW), f[8](SE) — 来自 j=1 内部节点
        // 未知方向：f[2](N), f[5](NE), f[6](NW) — 来自 j=ny-1 周期延拓
        //
        //   ρ*(1-uy) = f[0]+f[1]+f[3] + 2*(f[4]+f[7]+f[8])
        //   f[2] = f[4] + (2/3)*ρ*uy
        //   f[5] = f[7] - (1/2)*(f[1]-f[3]) + (1/2)*ρ*ux + (1/6)*ρ*uy
        //   f[6] = f[8] + (1/2)*(f[1]-f[3]) - (1/2)*ρ*ux + (1/6)*ρ*uy
        const double ux = bc.ux;
        const double uy = bc.uy;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < nx; ++i) {
            const int n = g.idx(i, 0);
            double* f = &g.f[n * d2q9::Q];
            double rho_w = (f[0] + f[1] + f[3]
                          + 2.0 * (f[4] + f[7] + f[8]))
                         / (1.0 - uy);
            f[2] = f[4] + (2.0 / 3.0) * rho_w * uy;
            f[5] = f[7] - 0.5 * (f[1] - f[3])
                        + 0.5 * rho_w * ux
                        + (1.0 / 6.0) * rho_w * uy;
            f[6] = f[8] + 0.5 * (f[1] - f[3])
                        - 0.5 * rho_w * ux
                        + (1.0 / 6.0) * rho_w * uy;
            g.rho[n] = rho_w;
            g.u[n * 2 + 0] = ux;
            g.u[n * 2 + 1] = uy;
        }
    }
    else if (bc.face == Face::West) {
        // 西壁（i=0）：规定速度 (ux, uy)，常用于左侧入口 BC
        // 已知方向：f[3](W), f[6](NW), f[7](SW) — 来自 i=1 内部节点
        // 未知方向：f[1](E), f[5](NE), f[8](SE) — 来自 i=nx-1 周期延拓
        //
        //   ρ*(1-ux) = f[0]+f[2]+f[4] + 2*(f[3]+f[6]+f[7])
        //   f[1] = f[3] + (2/3)*ρ*ux
        //   f[5] = f[7] - (1/2)*(f[2]-f[4]) + (1/2)*ρ*uy + (1/6)*ρ*ux
        //   f[8] = f[6] + (1/2)*(f[2]-f[4]) - (1/2)*ρ*uy + (1/6)*ρ*ux
        const double ux = bc.ux;
        const double uy = bc.uy;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < ny; ++j) {
            const int n = g.idx(0, j);
            double* f = &g.f[n * d2q9::Q];
            double rho_w = (f[0] + f[2] + f[4]
                          + 2.0 * (f[3] + f[6] + f[7]))
                         / (1.0 - ux);
            f[1] = f[3] + (2.0 / 3.0) * rho_w * ux;
            f[5] = f[7] - 0.5 * (f[2] - f[4])
                        + 0.5 * rho_w * uy
                        + (1.0 / 6.0) * rho_w * ux;
            f[8] = f[6] + 0.5 * (f[2] - f[4])
                        - 0.5 * rho_w * uy
                        + (1.0 / 6.0) * rho_w * ux;
            g.rho[n] = rho_w;
            g.u[n * 2 + 0] = ux;
            g.u[n * 2 + 1] = uy;
        }
    }
    else if (bc.face == Face::East) {
        // 东壁（i=nx-1）：规定速度 (ux, uy)，常用于右侧出口/壁面 BC
        // 已知方向：f[1](E), f[5](NE), f[8](SE) — 来自 i=nx-2 内部节点
        // 未知方向：f[3](W), f[6](NW), f[7](SW) — 来自 i=0 周期延拓
        //
        //   ρ*(1+ux) = f[0]+f[2]+f[4] + 2*(f[1]+f[5]+f[8])
        //   f[3] = f[1] - (2/3)*ρ*ux
        //   f[6] = f[8] + (1/2)*(f[2]-f[4]) + (1/2)*ρ*uy - (1/6)*ρ*ux
        //   f[7] = f[5] - (1/2)*(f[2]-f[4]) - (1/2)*ρ*uy - (1/6)*ρ*ux
        const double ux = bc.ux;
        const double uy = bc.uy;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = 0; j < ny; ++j) {
            const int n = g.idx(nx - 1, j);
            double* f = &g.f[n * d2q9::Q];
            double rho_w = (f[0] + f[2] + f[4]
                          + 2.0 * (f[1] + f[5] + f[8]))
                         / (1.0 + ux);
            f[3] = f[1] - (2.0 / 3.0) * rho_w * ux;
            f[6] = f[8] + 0.5 * (f[2] - f[4])
                        + 0.5 * rho_w * uy
                        - (1.0 / 6.0) * rho_w * ux;
            f[7] = f[5] - 0.5 * (f[2] - f[4])
                        - 0.5 * rho_w * uy
                        - (1.0 / 6.0) * rho_w * ux;
            g.rho[n] = rho_w;
            g.u[n * 2 + 0] = ux;
            g.u[n * 2 + 1] = uy;
        }
    }
}

// ---------------------------------------------------------------------------
// 应用所有已注册的边界条件
// ---------------------------------------------------------------------------
void apply_boundary_conditions(LatticeGrid& grid,
                                const std::vector<BoundaryCondition>& bcs)
{
    for (const auto& bc : bcs) {
        switch (bc.type) {
            case BCType::BounceBack:
                apply_bounce_back(grid, bc.face);
                break;
            case BCType::ZouHe_Velocity:
                apply_zou_he_velocity(grid, bc);
                break;
            case BCType::ZouHe_Pressure:
                // TODO: 实现压力边界条件
                break;
            case BCType::Periodic:
                // 周期边界在流式迁移的周期性取模中隐式处理
                break;
        }
    }
}

} // namespace lbm
