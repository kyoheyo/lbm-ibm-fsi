#include "lbm/boundary.hpp"
#include "lbm/lattice.hpp"
#include <stdexcept>

#ifdef LBM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace lbm {

// ===========================================================================
//
// 边界条件实现 — D2Q9 格子 Boltzmann 方法
//
// 方向编号约定（D2Q9）：
//   0:(0,0)  1:(+1,0)  2:(0,+1)  3:(-1,0)  4:(0,-1)
//   5:(+1,+1) 6:(-1,+1) 7:(-1,-1) 8:(+1,-1)
//
// 对立方向查找表 OPP：0↔0, 1↔3, 2↔4, 3↔1, 4↔2, 5↔7, 6↔8, 7↔5, 8↔6
//
// 流式迁移（push 方案）后的数据布局：
//   g.f     — 当前时刻 t+dt 的分布函数
//   g.f_tmp — 上一时刻 t 的碰撞后、迁移前的分布函数
//             （stream() 末尾执行 std::swap(g.f, g.f_tmp) 后保留）
//
// 各面幽灵（未知）方向（由周期延拓引入，需由 BC 覆盖）：
//   南壁 j=0    ：f[2](N), f[5](NE), f[6](NW)
//   北壁 j=ny-1 ：f[4](S), f[7](SW), f[8](SE)
//   西壁 i=0    ：f[1](E), f[5](NE), f[8](SE)
//   东壁 i=nx-1 ：f[3](W), f[6](NW), f[7](SW)
//
// ===========================================================================


// ---------------------------------------------------------------------------
// 1. 半步长反弹（Halfway Bounce-Back，BCType::BounceBack）
//
// 物理模型：壁面位于流体节点与虚固体节点之间的半格处（wall-between-nodes），
//           位置精度为二阶。
//
// 推导：
//   碰后、迁移前，壁面节点 x_w 处朝向壁面方向 a 的分布函数为 f*_a(x_w, t)，
//   保存在 g.f_tmp 中（stream() 后 swap 保留）。
//   粒子在半格处碰壁后原路返回，时序延迟半个时间步自然抵消：
//     f[opp(a)](x_w, t+dt) = f*[a](x_w, t)  ← 读取本节点 f_tmp
//
//   各面赋值：
//     南壁 j=0    ：f[2] ← f_tmp[4], f[5] ← f_tmp[7], f[6] ← f_tmp[8]
//     北壁 j=ny-1 ：f[4] ← f_tmp[2], f[7] ← f_tmp[5], f[8] ← f_tmp[6]
//     西壁 i=0    ：f[1] ← f_tmp[3], f[5] ← f_tmp[7], f[8] ← f_tmp[6]
//     东壁 i=nx-1 ：f[3] ← f_tmp[1], f[6] ← f_tmp[8], f[7] ← f_tmp[5]
// ---------------------------------------------------------------------------
static void apply_bounce_back(LatticeGrid& g, Face face, const PhysicalBounds& pb)
{
    if (g.model != LatticeModel::D2Q9) return;

    const int nx = g.nx;

    switch (face) {
        case Face::South:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nx; ++i) {
                const int n = g.idx(i, pb.j_s);
                double*       f  = &g.f    [n * d2q9::Q];
                const double* fp = &g.f_tmp[n * d2q9::Q];
                f[2] = fp[4];   // N  ← f_tmp[S]
                f[5] = fp[7];   // NE ← f_tmp[SW]
                f[6] = fp[8];   // NW ← f_tmp[SE]
            }
            break;

        case Face::North:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nx; ++i) {
                const int n = g.idx(i, pb.j_n);
                double*       f  = &g.f    [n * d2q9::Q];
                const double* fp = &g.f_tmp[n * d2q9::Q];
                f[4] = fp[2];   // S  ← f_tmp[N]
                f[7] = fp[5];   // SW ← f_tmp[NE]
                f[8] = fp[6];   // SE ← f_tmp[NW]
            }
            break;

        case Face::West:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = pb.j_s; j <= pb.j_n; ++j) {
                const int n = g.idx(pb.i_w, j);
                double*       f  = &g.f    [n * d2q9::Q];
                const double* fp = &g.f_tmp[n * d2q9::Q];
                f[1] = fp[3];   // E  ← f_tmp[W]
                f[5] = fp[7];   // NE ← f_tmp[SW]
                f[8] = fp[6];   // SE ← f_tmp[NW]
            }
            break;

        case Face::East:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = pb.j_s; j <= pb.j_n; ++j) {
                const int n = g.idx(pb.i_e, j);
                double*       f  = &g.f    [n * d2q9::Q];
                const double* fp = &g.f_tmp[n * d2q9::Q];
                f[3] = fp[1];   // W  ← f_tmp[E]
                f[6] = fp[8];   // NW ← f_tmp[SE]
                f[7] = fp[5];   // SW ← f_tmp[NE]
            }
            break;

        default:
            break;
    }
}


// ---------------------------------------------------------------------------
// 2. 全步长反弹（Full-Way / On-Node Bounce-Back，BCType::BounceBackFullWay）
//
// 物理模型：壁面位于格子节点处（wall-at-node），位置精度为一阶。
//
// 推导：
//   迁移后，壁面节点收到来自流体侧（方向 a 的反向 opp(a)）的分布函数，
//   即 g.f[a](x_w) 为由邻居节点推送而来的入射分布。
//   粒子在节点处被立即原路反弹：
//     f[opp(a)](x_w, t+dt) ← f[a](x_w, t+dt)   ← 使用当前迁移后的 g.f
//
//   注意：必须先临时存储所有入射方向，再一次性写入，避免读写混用已修改值。
//
//   各面被覆盖的幽灵方向（利用迁移后 g.f 的已知方向）：
//     南壁 j=0    ：f[2] ← f[4], f[5] ← f[7], f[6] ← f[8]
//     北壁 j=ny-1 ：f[4] ← f[2], f[7] ← f[5], f[8] ← f[6]
//     西壁 i=0    ：f[1] ← f[3], f[5] ← f[7], f[8] ← f[6]
//     东壁 i=nx-1 ：f[3] ← f[1], f[6] ← f[8], f[7] ← f[5]
// ---------------------------------------------------------------------------
static void apply_bounce_back_fullway(LatticeGrid& g, Face face, const PhysicalBounds& pb)
{
    if (g.model != LatticeModel::D2Q9) return;

    const int nx = g.nx;

    switch (face) {
        case Face::South:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nx; ++i) {
                double* f = &g.f[g.idx(i, pb.j_s) * d2q9::Q];
                const double s4 = f[4], s7 = f[7], s8 = f[8];  // 先暂存入射值
                f[2] = s4;
                f[5] = s7;
                f[6] = s8;
            }
            break;

        case Face::North:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nx; ++i) {
                double* f = &g.f[g.idx(i, pb.j_n) * d2q9::Q];
                const double s2 = f[2], s5 = f[5], s6 = f[6];
                f[4] = s2;
                f[7] = s5;
                f[8] = s6;
            }
            break;

        case Face::West:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = pb.j_s; j <= pb.j_n; ++j) {
                double* f = &g.f[g.idx(pb.i_w, j) * d2q9::Q];
                const double s3 = f[3], s7 = f[7], s6 = f[6];
                f[1] = s3;
                f[5] = s7;
                f[8] = s6;
            }
            break;

        case Face::East:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = pb.j_s; j <= pb.j_n; ++j) {
                double* f = &g.f[g.idx(pb.i_e, j) * d2q9::Q];
                const double s1 = f[1], s8 = f[8], s5 = f[5];
                f[3] = s1;
                f[6] = s8;
                f[7] = s5;
            }
            break;

        default:
            break;
    }
}


// ---------------------------------------------------------------------------
// 3. Zou-He 速度边界条件（BCType::ZouHe_Velocity）
//
// 参考文献：Zou Q, He X. "On pressure and velocity boundary conditions for the
//           lattice Boltzmann BGK model." Phys. Fluids 9(6):1591-1598, 1997.
//
// 方法：非平衡反弹格式（Non-equilibrium Bounce-Back, NEBB）
//
// 推导思路（以南壁 j=0, 规定速度 (ux, uy) 为例）：
//   流式迁移后：
//     已知 3 个方向：f[4](S), f[7](SW), f[8](SE) — 来自内部节点 j=1
//     未知 3 个方向：f[2](N), f[5](NE), f[6](NW) — 幽灵值（来自周期延拓）
//     f[0], f[1], f[3] 也已知（来自内部或水平周期）
//   3 个宏观约束：
//     ρ   = Σ_a f[a]  → ρ*(1-uy) = f[0]+f[1]+f[3] + 2*(f[4]+f[7]+f[8])
//     ρ*ux = Σ_a f[a]*cx[a]
//     ρ*uy = Σ_a f[a]*cy[a]
//   附加 NEBB 条件（对角方向的非平衡部分对称）：
//     f[5]-f_eq[5] = f[7]-f_eq[7]  ⟹  f[5]-f[7] = f_eq[5]-f_eq[7]
//     f[6]-f_eq[6] = f[8]-f_eq[8]  ⟹  f[6]-f[8] = f_eq[6]-f_eq[8]
//   联立以上方程组，解出 3 个未知量：
//     ρ  = (f[0]+f[1]+f[3] + 2*(f[4]+f[7]+f[8])) / (1-uy)
//     f[2] = f[4] + (2/3)*ρ*uy
//     f[5] = f[7] - (1/2)*(f[1]-f[3]) + (1/2)*ρ*ux + (1/6)*ρ*uy
//     f[6] = f[8] + (1/2)*(f[1]-f[3]) - (1/2)*ρ*ux + (1/6)*ρ*uy
//
//   其他三面（北/西/东）的推导完全类比，只需交换相应的已知/未知方向。
// ---------------------------------------------------------------------------
static void apply_zou_he_velocity(LatticeGrid& g, const BoundaryCondition& bc,
                                    const PhysicalBounds& pb)
{
    if (g.model != LatticeModel::D2Q9) return;

    const int nx = g.nx;

    if (bc.face == Face::North) {
        const double ux = bc.ux;
        const double uy = bc.uy;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < nx; ++i) {
            const int n = g.idx(i, pb.j_n);
            double* f = &g.f[n * d2q9::Q];
            double rho_w = (f[0] + f[1] + f[3]
                          + 2.0 * (f[2] + f[5] + f[6]))
                         / (1.0 + uy);
            f[4] = f[2] - (2.0 / 3.0) * rho_w * uy;
            f[7] = f[5] + 0.5 * (f[1] - f[3])
                        - (1.0 / 6.0) * rho_w * uy
                        - 0.5 * rho_w * ux;
            f[8] = f[6] - 0.5 * (f[1] - f[3])
                        - (1.0 / 6.0) * rho_w * uy
                        + 0.5 * rho_w * ux;
            g.rho[n]       = rho_w;
            g.u[n * 2 + 0] = ux;
            g.u[n * 2 + 1] = uy;
        }
    }
    else if (bc.face == Face::South) {
        const double ux = bc.ux;
        const double uy = bc.uy;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < nx; ++i) {
            const int n = g.idx(i, pb.j_s);
            double* f = &g.f[n * d2q9::Q];
            double rho_w = (f[0] + f[1] + f[3]
                          + 2.0 * (f[4] + f[7] + f[8]))
                         / (1.0 - uy);
            f[2] = f[4] + (2.0 / 3.0) * rho_w * uy;
            f[5] = f[7] - 0.5 * (f[1] - f[3])
                        + (1.0 / 6.0) * rho_w * uy
                        + 0.5 * rho_w * ux;
            f[6] = f[8] + 0.5 * (f[1] - f[3])
                        + (1.0 / 6.0) * rho_w * uy
                        - 0.5 * rho_w * ux;
            g.rho[n]       = rho_w;
            g.u[n * 2 + 0] = ux;
            g.u[n * 2 + 1] = uy;
        }
    }
    else if (bc.face == Face::West) {
        const double ux = bc.ux;
        const double uy = bc.uy;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = pb.j_s; j <= pb.j_n; ++j) {
            const int n = g.idx(pb.i_w, j);
            double* f = &g.f[n * d2q9::Q];
            double rho_w = (f[0] + f[2] + f[4]
                          + 2.0 * (f[3] + f[6] + f[7]))
                         / (1.0 - ux);
            f[1] = f[3] + (2.0 / 3.0) * rho_w * ux;
            f[5] = f[7] - 0.5 * (f[2] - f[4])
                        + (1.0 / 6.0) * rho_w * ux
                        + 0.5 * rho_w * uy;
            f[8] = f[6] + 0.5 * (f[2] - f[4])
                        + (1.0 / 6.0) * rho_w * ux
                        - 0.5 * rho_w * uy;
            g.rho[n]       = rho_w;
            g.u[n * 2 + 0] = ux;
            g.u[n * 2 + 1] = uy;
        }
    }
    else if (bc.face == Face::East) {
        const double ux = bc.ux;
        const double uy = bc.uy;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = pb.j_s; j <= pb.j_n; ++j) {
            const int n = g.idx(pb.i_e, j);
            double* f = &g.f[n * d2q9::Q];
            double rho_w = (f[0] + f[2] + f[4]
                          + 2.0 * (f[1] + f[5] + f[8]))
                         / (1.0 + ux);
            f[3] = f[1] - (2.0 / 3.0) * rho_w * ux;
            f[6] = f[8] - 0.5 * (f[2] - f[4])
                        - (1.0 / 6.0) * rho_w * ux
                        + 0.5 * rho_w * uy;
            f[7] = f[5] + 0.5 * (f[2] - f[4])
                        - (1.0 / 6.0) * rho_w * ux
                        - 0.5 * rho_w * uy;
            g.rho[n]       = rho_w;
            g.u[n * 2 + 0] = ux;
            g.u[n * 2 + 1] = uy;
        }
    }
}


// ---------------------------------------------------------------------------
// 4. Zou-He 压力边界条件（BCType::ZouHe_Pressure）
//
// 参考文献：同 Zou-He 速度 BC。
//
// 方法：规定边界密度 ρ_b，利用 NEBB 条件推导未知速度分量。
//       通常在入口规定进口压力（密度），出口规定背压（密度）。
//       此处假设切向速度为零（uy=0 对水平壁，ux=0 对垂直壁）。
//       若需非零切向速度，可通过 bc.ux / bc.uy 指定。
//
// 以东壁（i=nx-1，背压出口）为例，规定 ρ_b：
//   已知：f[1](E), f[5](NE), f[8](SE) — 来自 i=nx-2
//   未知：f[3](W), f[6](NW), f[7](SW)
//   由连续方程求 ux：
//     ρ_b*(1+ux) = f[0]+f[2]+f[4] + 2*(f[1]+f[5]+f[8])
//     ux = -1 + (f[0]+f[2]+f[4] + 2*(f[1]+f[5]+f[8])) / ρ_b
//   uy = bc.uy（默认 0）
//   其余同速度 BC 公式：
//     f[3] = f[1] - (2/3)*ρ_b*ux
//     f[6] = f[8] + (1/2)*(f[2]-f[4]) - (1/6)*ρ_b*ux + (1/2)*ρ_b*uy
//     f[7] = f[5] - (1/2)*(f[2]-f[4]) - (1/6)*ρ_b*ux - (1/2)*ρ_b*uy
// ---------------------------------------------------------------------------
static void apply_zou_he_pressure(LatticeGrid& g, const BoundaryCondition& bc,
                                    const PhysicalBounds& pb)
{
    if (g.model != LatticeModel::D2Q9) return;

    const int nx = g.nx;
    const double rho_b = bc.rho;

    if (bc.face == Face::North) {
        const double ux_t = bc.ux;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < nx; ++i) {
            const int n = g.idx(i, pb.j_n);
            double* f = &g.f[n * d2q9::Q];
            const double uy = -1.0 + (f[0] + f[1] + f[3]
                                     + 2.0 * (f[2] + f[5] + f[6])) / rho_b;
            f[4] = f[2] - (2.0 / 3.0) * rho_b * uy;
            f[7] = f[5] + 0.5 * (f[1] - f[3])
                        - (1.0 / 6.0) * rho_b * uy
                        - 0.5 * rho_b * ux_t;
            f[8] = f[6] - 0.5 * (f[1] - f[3])
                        - (1.0 / 6.0) * rho_b * uy
                        + 0.5 * rho_b * ux_t;
            g.rho[n]       = rho_b;
            g.u[n * 2 + 0] = ux_t;
            g.u[n * 2 + 1] = uy;
        }
    }
    else if (bc.face == Face::South) {
        const double ux_t = bc.ux;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int i = 0; i < nx; ++i) {
            const int n = g.idx(i, pb.j_s);
            double* f = &g.f[n * d2q9::Q];
            const double uy = 1.0 - (f[0] + f[1] + f[3]
                                    + 2.0 * (f[4] + f[7] + f[8])) / rho_b;
            f[2] = f[4] + (2.0 / 3.0) * rho_b * uy;
            f[5] = f[7] - 0.5 * (f[1] - f[3])
                        + (1.0 / 6.0) * rho_b * uy
                        + 0.5 * rho_b * ux_t;
            f[6] = f[8] + 0.5 * (f[1] - f[3])
                        + (1.0 / 6.0) * rho_b * uy
                        - 0.5 * rho_b * ux_t;
            g.rho[n]       = rho_b;
            g.u[n * 2 + 0] = ux_t;
            g.u[n * 2 + 1] = uy;
        }
    }
    else if (bc.face == Face::West) {
        const double uy_t = bc.uy;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = pb.j_s; j <= pb.j_n; ++j) {
            const int n = g.idx(pb.i_w, j);
            double* f = &g.f[n * d2q9::Q];
            const double ux = 1.0 - (f[0] + f[2] + f[4]
                                    + 2.0 * (f[3] + f[6] + f[7])) / rho_b;
            f[1] = f[3] + (2.0 / 3.0) * rho_b * ux;
            f[5] = f[7] - 0.5 * (f[2] - f[4])
                        + (1.0 / 6.0) * rho_b * ux
                        + 0.5 * rho_b * uy_t;
            f[8] = f[6] + 0.5 * (f[2] - f[4])
                        + (1.0 / 6.0) * rho_b * ux
                        - 0.5 * rho_b * uy_t;
            g.rho[n]       = rho_b;
            g.u[n * 2 + 0] = ux;
            g.u[n * 2 + 1] = uy_t;
        }
    }
    else if (bc.face == Face::East) {
        const double uy_t = bc.uy;
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (int j = pb.j_s; j <= pb.j_n; ++j) {
            const int n = g.idx(pb.i_e, j);
            double* f = &g.f[n * d2q9::Q];
            const double ux = -1.0 + (f[0] + f[2] + f[4]
                                     + 2.0 * (f[1] + f[5] + f[8])) / rho_b;
            f[3] = f[1] - (2.0 / 3.0) * rho_b * ux;
            f[6] = f[8] - 0.5 * (f[2] - f[4])
                        - (1.0 / 6.0) * rho_b * ux
                        + 0.5 * rho_b * uy_t;
            f[7] = f[5] + 0.5 * (f[2] - f[4])
                        - (1.0 / 6.0) * rho_b * ux
                        - 0.5 * rho_b * uy_t;
            g.rho[n]       = rho_b;
            g.u[n * 2 + 0] = ux;
            g.u[n * 2 + 1] = uy_t;
        }
    }
}


// ---------------------------------------------------------------------------
// 5. 充分发展出口 / 自由出口边界（BCType::FullyDeveloped / FreeOutlet）
// ---------------------------------------------------------------------------
static void apply_fully_developed(LatticeGrid& g, Face face, const PhysicalBounds& pb)
{
    if (g.model != LatticeModel::D2Q9) return;

    const int nx = g.nx;

    switch (face) {
        case Face::South:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nx; ++i) {
                const int n_out = g.idx(i, pb.j_s);
                const int n_in  = g.idx(i, pb.j_s + 1);
                for (int a = 0; a < d2q9::Q; ++a)
                    g.f[n_out * d2q9::Q + a] = g.f[n_in * d2q9::Q + a];
            }
            break;

        case Face::North:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nx; ++i) {
                const int n_out = g.idx(i, pb.j_n);
                const int n_in  = g.idx(i, pb.j_n - 1);
                for (int a = 0; a < d2q9::Q; ++a)
                    g.f[n_out * d2q9::Q + a] = g.f[n_in * d2q9::Q + a];
            }
            break;

        case Face::West:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = pb.j_s; j <= pb.j_n; ++j) {
                const int n_out = g.idx(pb.i_w, j);
                const int n_in  = g.idx(pb.i_w + 1, j);
                for (int a = 0; a < d2q9::Q; ++a)
                    g.f[n_out * d2q9::Q + a] = g.f[n_in * d2q9::Q + a];
            }
            break;

        case Face::East:
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = pb.j_s; j <= pb.j_n; ++j) {
                const int n_out = g.idx(pb.i_e, j);
                const int n_in  = g.idx(pb.i_e - 1, j);
                for (int a = 0; a < d2q9::Q; ++a)
                    g.f[n_out * d2q9::Q + a] = g.f[n_in * d2q9::Q + a];
            }
            break;

        default:
            break;
    }
}


// ---------------------------------------------------------------------------
// 6. 郭照立非平衡外推格式（BCType::Guo_Extrapolation）
//
// 参考文献：Guo Z, Zheng C, Shi B. "Non-equilibrium extrapolation method for
//           velocity and pressure boundary conditions in the lattice Boltzmann
//           method." Chinese Physics 11(4):366-374, 2002.
//
// 方法：将边界节点的分布函数分解为平衡部分与非平衡部分之和：
//   f[a](x_b) = f_eq[a](ρ_b, u_b) + f_neq[a](x_f)
//
// 其中：
//   - ρ_b, u_b 为边界处的宏观量（由边界条件规定）
//   - f_eq[a](ρ_b, u_b) 为用边界宏观量计算的平衡分布
//   - f_neq[a](x_f) = f[a](x_f) - f_eq[a](ρ_f, u_f) 为相邻内部节点的非平衡部分
//   - x_f 为边界节点 x_b 在流体侧的最近邻节点
//
// 对于速度边界（规定 u_b，ρ_b 由内部节点近似）：
//   ρ_b ≈ ρ(x_f)（一阶近似，可用 Zou-He 公式改进，此处取简单近似）
//
// 对于压力边界（规定 ρ_b，u_b 由内部节点外推）：
//   u_b = u(x_f)（一阶近似）
//   如指定非零 bc.ux / bc.uy，则直接使用指定值而非外推。
//
// 本实现：bc.rho > 0 时以 bc.rho 作为 ρ_b（压力 BC）；
//         bc.rho ≤ 0 时用内部节点密度外推（速度 BC）。
//         bc.ux, bc.uy 指定的速度始终作为 u_b（速度 BC）；
//         设为 (0,0) 时则用内部节点速度外推（压力 BC / 无滑移 BC）。
// ---------------------------------------------------------------------------
static void apply_guo_extrapolation(LatticeGrid& g, const BoundaryCondition& bc,
                                      const PhysicalBounds& pb)
{
    if (g.model != LatticeModel::D2Q9) return;

    const int nx = g.nx;

    // 辅助 lambda：计算节点 n 处方向 a 的平衡分布（D2Q9）
    auto feq_node = [&](double rho, double ux, double uy, int a) -> double {
        const double cx = static_cast<double>(d2q9::C[a][0]);
        const double cy = static_cast<double>(d2q9::C[a][1]);
        const double cu = cx * ux + cy * uy;
        const double u2 = ux * ux + uy * uy;
        constexpr double cs2 = 1.0 / 3.0;
        constexpr double cs4 = cs2 * cs2;
        return d2q9::W[a] * rho * (1.0 + cu / cs2 + cu * cu / (2.0 * cs4)
                                          - u2 / (2.0 * cs2));
    };

    switch (bc.face) {
        case Face::South: {
            const bool pressure_mode = (bc.rho > 0.0);
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nx; ++i) {
                const int n_b = g.idx(i, pb.j_s);
                const int n_f = g.idx(i, pb.j_s + 1);
                const double rho_f = g.rho[n_f];
                const double ux_f  = g.u[n_f * 2 + 0];
                const double uy_f  = g.u[n_f * 2 + 1];
                const double rho_b = pressure_mode ? bc.rho : rho_f;
                const double ux_b  = pressure_mode ? ux_f   : bc.ux;
                const double uy_b  = pressure_mode ? uy_f   : bc.uy;
                for (int a = 0; a < d2q9::Q; ++a) {
                    const double feq_b = feq_node(rho_b, ux_b, uy_b, a);
                    const double feq_f = feq_node(rho_f, ux_f,  uy_f, a);
                    g.f[n_b * d2q9::Q + a] = feq_b
                                            + (g.f[n_f * d2q9::Q + a] - feq_f);
                }
                g.rho[n_b]       = rho_b;
                g.u[n_b * 2 + 0] = ux_b;
                g.u[n_b * 2 + 1] = uy_b;
            }
            break;
        }

        case Face::North: {
            const bool pressure_mode = (bc.rho > 0.0);
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int i = 0; i < nx; ++i) {
                const int n_b = g.idx(i, pb.j_n);
                const int n_f = g.idx(i, pb.j_n - 1);
                const double rho_f = g.rho[n_f];
                const double ux_f  = g.u[n_f * 2 + 0];
                const double uy_f  = g.u[n_f * 2 + 1];
                const double rho_b = pressure_mode ? bc.rho : rho_f;
                const double ux_b  = pressure_mode ? ux_f   : bc.ux;
                const double uy_b  = pressure_mode ? uy_f   : bc.uy;
                for (int a = 0; a < d2q9::Q; ++a) {
                    const double feq_b = feq_node(rho_b, ux_b, uy_b, a);
                    const double feq_f = feq_node(rho_f, ux_f,  uy_f, a);
                    g.f[n_b * d2q9::Q + a] = feq_b
                                            + (g.f[n_f * d2q9::Q + a] - feq_f);
                }
                g.rho[n_b]       = rho_b;
                g.u[n_b * 2 + 0] = ux_b;
                g.u[n_b * 2 + 1] = uy_b;
            }
            break;
        }

        case Face::West: {
            const bool pressure_mode = (bc.rho > 0.0);
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = pb.j_s; j <= pb.j_n; ++j) {
                const int n_b = g.idx(pb.i_w, j);
                const int n_f = g.idx(pb.i_w + 1, j);
                const double rho_f = g.rho[n_f];
                const double ux_f  = g.u[n_f * 2 + 0];
                const double uy_f  = g.u[n_f * 2 + 1];
                const double rho_b = pressure_mode ? bc.rho : rho_f;
                const double ux_b  = pressure_mode ? ux_f   : bc.ux;
                const double uy_b  = pressure_mode ? uy_f   : bc.uy;
                for (int a = 0; a < d2q9::Q; ++a) {
                    const double feq_b = feq_node(rho_b, ux_b, uy_b, a);
                    const double feq_f = feq_node(rho_f, ux_f,  uy_f, a);
                    g.f[n_b * d2q9::Q + a] = feq_b
                                            + (g.f[n_f * d2q9::Q + a] - feq_f);
                }
                g.rho[n_b]       = rho_b;
                g.u[n_b * 2 + 0] = ux_b;
                g.u[n_b * 2 + 1] = uy_b;
            }
            break;
        }

        case Face::East: {
            const bool pressure_mode = (bc.rho > 0.0);
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
            for (int j = pb.j_s; j <= pb.j_n; ++j) {
                const int n_b = g.idx(pb.i_e, j);
                const int n_f = g.idx(pb.i_e - 1, j);
                const double rho_f = g.rho[n_f];
                const double ux_f  = g.u[n_f * 2 + 0];
                const double uy_f  = g.u[n_f * 2 + 1];
                const double rho_b = pressure_mode ? bc.rho : rho_f;
                const double ux_b  = pressure_mode ? ux_f   : bc.ux;
                const double uy_b  = pressure_mode ? uy_f   : bc.uy;
                for (int a = 0; a < d2q9::Q; ++a) {
                    const double feq_b = feq_node(rho_b, ux_b, uy_b, a);
                    const double feq_f = feq_node(rho_f, ux_f,  uy_f, a);
                    g.f[n_b * d2q9::Q + a] = feq_b
                                            + (g.f[n_f * d2q9::Q + a] - feq_f);
                }
                g.rho[n_b]       = rho_b;
                g.u[n_b * 2 + 0] = ux_b;
                g.u[n_b * 2 + 1] = uy_b;
            }
            break;
        }

        default:
            break;
    }
}


// ---------------------------------------------------------------------------
// 7. 边角点半步长反弹修正（内部辅助函数，自动在面 BC 后调用）
//
// 修正方案：在所有面 BC 完成后，对满足条件的角节点的全部 5 个幽灵方向
//   施加半步长反弹。使用 g.f_tmp（碰撞后、迁移前的值）覆盖面 BC 近似值。
//
// MPI 兼容性：
//   角节点使用物理坐标（pb.i_w/i_e, pb.j_s/j_n），确保 MPI 模式下不操作幽灵层。
//   条件触发改为"两相邻面均被注册且至少一面为固壁 BC"，防止在无物理壁面的分区
//   边界上错误施加反弹（如 MPI 分块中某 rank 没有 South/North 物理壁）。
//
// 参数：
//   wall_face_bits  — 固壁（BounceBack/BounceBackFullWay）面掩码
//   all_face_bits   — 所有已注册 BC 的面掩码（含 ZouHe/Guo 等非固壁 BC）
//   pb              — 物理边界行/列索引（MPI 模式下须跳过幽灵层）
// ---------------------------------------------------------------------------
static void apply_corner_bounce_back(LatticeGrid& g,
                                      unsigned wall_face_bits,
                                      unsigned all_face_bits,
                                      const PhysicalBounds& pb)
{
    if (g.model != LatticeModel::D2Q9) return;

    // 对立方向查找表（D2Q9）
    // 0↔0, 1↔3, 2↔4, 3↔1, 4↔2, 5↔7, 6↔8, 7↔5, 8↔6
    static const int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

    const bool s_wall  = (wall_face_bits & 0x1u) != 0u;
    const bool n_wall  = (wall_face_bits & 0x2u) != 0u;
    const bool w_wall  = (wall_face_bits & 0x4u) != 0u;
    const bool e_wall  = (wall_face_bits & 0x8u) != 0u;

    // 角点仅在两相邻面均已注册 BC（此 rank 拥有该物理面）且
    // 其中至少一面是固壁 BC 时才施加修正，以避免：
    //   a) 在 MPI 分区边界（无物理壁）处错误施加反弹；
    //   b) 两面均为非固壁 BC 的流体出口角点被强制置 u=0。
    const bool s_owned = (all_face_bits & 0x1u) != 0u;
    const bool n_owned = (all_face_bits & 0x2u) != 0u;
    const bool w_owned = (all_face_bits & 0x4u) != 0u;
    const bool e_owned = (all_face_bits & 0x8u) != 0u;

    // --- SW 角：South + West 均已注册，且至少一面为固壁 ---
    if (s_owned && w_owned && (s_wall || w_wall)) {
        const int n = g.idx(pb.i_w, pb.j_s);
        double*       f  = &g.f    [n * d2q9::Q];
        const double* fp = &g.f_tmp[n * d2q9::Q];
        for (int a : {1, 2, 5, 6, 8}) f[a] = fp[OPP[a]];
    }

    // --- SE 角：South + East 均已注册，且至少一面为固壁 ---
    if (s_owned && e_owned && (s_wall || e_wall)) {
        const int n = g.idx(pb.i_e, pb.j_s);
        double*       f  = &g.f    [n * d2q9::Q];
        const double* fp = &g.f_tmp[n * d2q9::Q];
        for (int a : {2, 3, 5, 6, 7}) f[a] = fp[OPP[a]];
    }

    // --- NW 角：North + West 均已注册，且至少一面为固壁 ---
    if (n_owned && w_owned && (n_wall || w_wall)) {
        const int n = g.idx(pb.i_w, pb.j_n);
        double*       f  = &g.f    [n * d2q9::Q];
        const double* fp = &g.f_tmp[n * d2q9::Q];
        for (int a : {1, 4, 5, 7, 8}) f[a] = fp[OPP[a]];
    }

    // --- NE 角：North + East 均已注册，且至少一面为固壁 ---
    if (n_owned && e_owned && (n_wall || e_wall)) {
        const int n = g.idx(pb.i_e, pb.j_n);
        double*       f  = &g.f    [n * d2q9::Q];
        const double* fp = &g.f_tmp[n * d2q9::Q];
        for (int a : {3, 4, 6, 7, 8}) f[a] = fp[OPP[a]];
    }
}


// ---------------------------------------------------------------------------
// 应用所有已注册的边界条件
// ---------------------------------------------------------------------------

// 辅助：检查本进程是否实际拥有指定面的物理壁（MPI 模式下仅壁面所在进程为 true）
static inline bool is_face_owned(Face f, const PhysicalBounds& pb)
{
    switch (f) {
        case Face::South: return pb.has_south_wall;
        case Face::North: return pb.has_north_wall;
        case Face::West:  return pb.has_west_wall;
        case Face::East:  return pb.has_east_wall;
        default:          return true;  // Bottom/Top：三维暂未并行化，始终拥有
    }
}

void apply_boundary_conditions(LatticeGrid& grid,
                                const std::vector<BoundaryCondition>& bcs,
                                PhysicalBounds pb)
{
    // ---- 第一轮：FullyDeveloped / FreeOutlet ----
    for (const auto& bc : bcs) {
        if (bc.type == BCType::FullyDeveloped || bc.type == BCType::FreeOutlet) {
            // MPI 内部进程不拥有该面物理壁时跳过，防止误覆盖内部物理行
            if (!is_face_owned(bc.face, pb)) continue;
            apply_fully_developed(grid, bc.face, pb);
        }
    }

    // ---- 第二轮：BounceBack / BounceBackFullWay ----
    for (const auto& bc : bcs) {
        // MPI 内部进程不拥有该面物理壁时跳过
        if (!is_face_owned(bc.face, pb)) continue;
        switch (bc.type) {
            case BCType::BounceBack:
                apply_bounce_back(grid, bc.face, pb);
                break;
            case BCType::BounceBackFullWay:
                apply_bounce_back_fullway(grid, bc.face, pb);
                break;
            default:
                break;
        }
    }

    // ---- 第三轮：ZouHe / Guo（最强约束，最后施加） ----
    for (const auto& bc : bcs) {
        // MPI 内部进程不拥有该面物理壁时跳过
        if (!is_face_owned(bc.face, pb)) continue;
        switch (bc.type) {
            case BCType::ZouHe_Velocity:
                apply_zou_he_velocity(grid, bc, pb);
                break;
            case BCType::ZouHe_Pressure:
                apply_zou_he_pressure(grid, bc, pb);
                break;
            case BCType::Guo_Extrapolation:
                apply_guo_extrapolation(grid, bc, pb);
                break;
            case BCType::Periodic:
                // 周期边界在流式迁移的周期性取模中隐式处理
                break;
            default:
                break;
        }
    }

    // 所有面 BC 完成后，对物理角点施加半步长反弹修正。
    // MPI 兼容：使用 pb 中的 has_*_wall 标志，仅在本进程实际拥有该面物理壁时
    // 才计入 owned 掩码，防止在 MPI 内部分区边界（无物理壁）处错误施加。
    if (!bcs.empty()) {
        unsigned wall_face_bits = 0u;
        unsigned all_face_bits  = 0u;
        for (const auto& bc : bcs) {
            // 只统计本进程实际拥有物理壁的面
            if (!is_face_owned(bc.face, pb)) continue;
            unsigned bit = 0u;
            switch (bc.face) {
                case Face::South: bit = 0x1u; break;
                case Face::North: bit = 0x2u; break;
                case Face::West:  bit = 0x4u; break;
                case Face::East:  bit = 0x8u; break;
                default: break;
            }
            all_face_bits |= bit;
            if (bc.type == BCType::BounceBack || bc.type == BCType::BounceBackFullWay)
                wall_face_bits |= bit;
        }
        apply_corner_bounce_back(grid, wall_face_bits, all_face_bits, pb);
    }
}

} // namespace lbm
