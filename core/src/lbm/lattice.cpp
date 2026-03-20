#include "lbm/lattice.hpp"
#include <cmath>
#include <stdexcept>

#ifdef LBM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace lbm {

// ---------------------------------------------------------------------------
// LatticeGrid 构造函数：分配并初始化所有存储
// ---------------------------------------------------------------------------
LatticeGrid::LatticeGrid(int nx_, int ny_, int nz_, LatticeModel model_)
    : nx(nx_), ny(ny_), nz(nz_), model(model_)
{
    q = (model == LatticeModel::D2Q9) ? d2q9::Q : d3q19::Q;
    const int n = size();
    const int d = dim();
    f.assign(n * q, 0.0);          // 分布函数初始化为零（构造函数后立刻覆盖为平衡值）
    f_tmp.assign(n * q, 0.0);      // 流式迁移临时缓冲区
    rho.assign(n, 1.0);            // 密度初始化为 1（格子单位）
    u.assign(n * d, 0.0);          // 速度初始化为零
    force.assign(n * d, 0.0);      // 体力初始化为零
    solid.assign(n, 0);            // 固体标记初始化为全流体
    q_ibb.assign(n * q, 0.5f);     // IBB 距离分数默认 0.5（半步长反弹）
}

// ---------------------------------------------------------------------------
// 由分布函数计算宏观量（密度 ρ 和速度 u）
// ---------------------------------------------------------------------------
void LatticeGrid::compute_macroscopic()
{
    const int n = size();
    const int d = dim();

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < n; ++i) {
        double r = 0.0;
        double ux_loc = 0.0;
        double uy_loc = 0.0;
        double uz_loc = 0.0;

        if (model == LatticeModel::D2Q9) {
            for (int a = 0; a < d2q9::Q; ++a) {
                const double fa = f[i * d2q9::Q + a];
                r      += fa;
                ux_loc += fa * d2q9::C[a][0];
                uy_loc += fa * d2q9::C[a][1];
            }
            rho[i]         = r;
            u[i * d + 0]   = ux_loc / r;
            u[i * d + 1]   = uy_loc / r;
        } else {
            for (int a = 0; a < d3q19::Q; ++a) {
                const double fa = f[i * d3q19::Q + a];
                r      += fa;
                ux_loc += fa * d3q19::C[a][0];
                uy_loc += fa * d3q19::C[a][1];
                uz_loc += fa * d3q19::C[a][2];
            }
            rho[i]         = r;
            u[i * d + 0]   = ux_loc / r;
            u[i * d + 1]   = uy_loc / r;
            u[i * d + 2]   = uz_loc / r;
        }
    }
}

// ---------------------------------------------------------------------------
// Maxwell-Boltzmann 平衡分布函数
// ---------------------------------------------------------------------------
double f_eq(double w, double rho,
            const double* c,
            const double* u_vec,
            int d)
{
    // f_eq = w * rho * [1 + (c·u)/cs² + (c·u)²/(2cs⁴) - u²/(2cs²)]
    // 标准格子声速平方 cs² = 1/3
    constexpr double cs2 = 1.0 / 3.0;
    constexpr double cs4 = cs2 * cs2;

    double cu = 0.0;
    double u2 = 0.0;
    for (int k = 0; k < d; ++k) {
        cu += c[k] * u_vec[k];
        u2 += u_vec[k] * u_vec[k];
    }
    return w * rho * (1.0 + cu / cs2 + cu * cu / (2.0 * cs4) - u2 / (2.0 * cs2));
}

} // namespace lbm
