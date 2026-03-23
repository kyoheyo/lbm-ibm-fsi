// lbm/stretched_grid.cpp — 拉伸（非均匀）坐标网格实现
//
// 本文件实现 stretched_grid.hpp 中声明的 StretchedGrid 工厂函数、
// 网格质量查询工具以及 IS-LBM 流式迁移补充插值修正。

#include "lbm/stretched_grid.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>

#ifdef LBM_ENABLE_OPENMP
#include <omp.h>
#endif

namespace lbm {

// ---------------------------------------------------------------------------
// 内部辅助：从 x_phys/y_phys 计算 dx/dy
// ---------------------------------------------------------------------------
void StretchedGrid::compute_spacings()
{
    const int nx = lattice.nx;
    const int ny = lattice.ny;

    dx.resize(nx);
    for (int i = 0; i < nx - 1; ++i) {
        dx[i] = x_phys[i + 1] - x_phys[i];
    }
    dx[nx - 1] = dx[nx - 2];   // 末端外推

    dy.resize(ny);
    for (int j = 0; j < ny - 1; ++j) {
        dy[j] = y_phys[j + 1] - y_phys[j];
    }
    dy[ny - 1] = dy[ny - 2];   // 末端外推

    // 缓存最小间距（IS-LBM 修正中使用）
    dx_ref = *std::min_element(dx.begin(), dx.end());
    dy_ref = *std::min_element(dy.begin(), dy.end());
}

// ---------------------------------------------------------------------------
// 网格质量查询
// ---------------------------------------------------------------------------

double StretchedGrid::max_stretch_ratio_x() const
{
    if (dx.size() < 2) return 1.0;
    double max_r = 1.0;
    for (int i = 0; i + 1 < static_cast<int>(dx.size()) - 1; ++i) {
        if (dx[i] > 0.0) {
            max_r = std::max(max_r, dx[i + 1] / dx[i]);
        }
    }
    return max_r;
}

double StretchedGrid::max_stretch_ratio_y() const
{
    if (dy.size() < 2) return 1.0;
    double max_r = 1.0;
    for (int j = 0; j + 1 < static_cast<int>(dy.size()) - 1; ++j) {
        if (dy[j] > 0.0) {
            max_r = std::max(max_r, dy[j + 1] / dy[j]);
        }
    }
    return max_r;
}

// ---------------------------------------------------------------------------
// 工厂函数 1：等比拉伸网格
// ---------------------------------------------------------------------------
static std::vector<double> make_geometric_coords(
    int n, double c0, double c1, double ratio)
{
    std::vector<double> coords(n);
    coords[0] = c0;
    if (n == 1) return coords;

    const double L = c1 - c0;
    if (std::abs(ratio - 1.0) < 1e-12) {
        // 均匀（公比 = 1）
        for (int i = 0; i < n; ++i) {
            coords[i] = c0 + L * static_cast<double>(i) / (n - 1);
        }
    } else {
        // 等比数列：h_i = h0 * ratio^i，sum = h0 * (ratio^(n-1) - 1) / (ratio - 1) = L
        const double h0 = L * (ratio - 1.0) / (std::pow(ratio, n - 1) - 1.0);
        for (int i = 1; i < n; ++i) {
            coords[i] = coords[i - 1] + h0 * std::pow(ratio, i - 1);
        }
        coords[n - 1] = c1;   // 强制末端精确（消除浮点累积误差）
    }
    return coords;
}

StretchedGrid StretchedGrid::build_geometric(
    int nx, int ny, int nz,
    double x0, double x1,
    double y0, double y1,
    double sx, double sy,
    LatticeModel model)
{
    if (nx < 2 || ny < 2)
        throw std::invalid_argument("build_geometric: nx, ny must be >= 2");
    if (sx <= 0.0 || sy <= 0.0)
        throw std::invalid_argument("build_geometric: stretch ratios must be > 0");

    StretchedGrid sg;
    sg.lattice  = LatticeGrid(nx, ny, nz, model);
    sg.x_phys   = make_geometric_coords(nx, x0, x1, sx);
    sg.y_phys   = make_geometric_coords(ny, y0, y1, sy);
    sg.compute_spacings();
    return sg;
}

// ---------------------------------------------------------------------------
// 工厂函数 2：双曲正切拉伸网格（两端加密，中央稀疏）
// ---------------------------------------------------------------------------
//
// 坐标公式（x 方向，n 个节点，范围 [c0, c1]，控制参数 delta ∈ [0,1)）：
//
//   若 delta ≈ 0：均匀分布（退化）
//   否则：
//     beta  = atanh(sqrt(1.0 - delta))
//     xi_i  = (double)i / (n - 1)   ∈ [0, 1]
//     x_i   = c0 + (c1-c0) * 0.5 * [1 + tanh(beta * (2*xi_i - 1)) / tanh(beta)]
//
//   当 delta → 1 时，beta → 0⁺，公式退化为均匀（tanh(β(2ξ-1))/tanh(β) → 2ξ-1）
//   当 delta → 0 时，beta → ∞，两端节点密集，中央稀疏（边界层效果显著）
//
// 注意：delta 取反直觉方向——delta 越大（趋近 1），拉伸越轻；delta 越小（趋近 0），
//       拉伸越重（两端越密）。典型配置：delta = 0.8~0.9 提供适度边界层加密。
// ---------------------------------------------------------------------------
static std::vector<double> make_tanh_coords(
    int n, double c0, double c1, double delta)
{
    std::vector<double> coords(n);
    coords[0] = c0;
    if (n == 1) return coords;

    const double L = c1 - c0;

    if (delta <= 1e-10 || delta >= 1.0 - 1e-10) {
        // 退化为均匀
        for (int i = 0; i < n; ++i) {
            coords[i] = c0 + L * static_cast<double>(i) / (n - 1);
        }
    } else {
        const double beta     = std::atanh(std::sqrt(1.0 - delta));
        const double inv_tanh = 1.0 / std::tanh(beta);
        for (int i = 0; i < n; ++i) {
            const double xi = static_cast<double>(i) / (n - 1);
            coords[i] = c0 + L * 0.5 * (1.0 + std::tanh(beta * (2.0 * xi - 1.0)) * inv_tanh);
        }
    }
    coords[n - 1] = c1;  // 强制精确
    return coords;
}

StretchedGrid StretchedGrid::build_tanh(
    int nx, int ny, int nz,
    double x0, double x1,
    double y0, double y1,
    double delta_x, double delta_y,
    LatticeModel model)
{
    if (nx < 2 || ny < 2)
        throw std::invalid_argument("build_tanh: nx, ny must be >= 2");

    StretchedGrid sg;
    sg.lattice  = LatticeGrid(nx, ny, nz, model);
    sg.x_phys   = make_tanh_coords(nx, x0, x1, delta_x);
    sg.y_phys   = make_tanh_coords(ny, y0, y1, delta_y);
    sg.compute_spacings();
    return sg;
}

// ---------------------------------------------------------------------------
// 工厂函数 3：用户自定义坐标
// ---------------------------------------------------------------------------
StretchedGrid StretchedGrid::build_custom(
    int nz,
    std::vector<double> x_coords,
    std::vector<double> y_coords,
    LatticeModel model)
{
    if (x_coords.empty() || y_coords.empty())
        throw std::invalid_argument("build_custom: coordinate arrays must not be empty");

    // 验证单调递增
    for (int i = 0; i + 1 < static_cast<int>(x_coords.size()); ++i) {
        if (x_coords[i + 1] <= x_coords[i])
            throw std::invalid_argument(
                "build_custom: x_coords must be strictly increasing at index " +
                std::to_string(i));
    }
    for (int j = 0; j + 1 < static_cast<int>(y_coords.size()); ++j) {
        if (y_coords[j + 1] <= y_coords[j])
            throw std::invalid_argument(
                "build_custom: y_coords must be strictly increasing at index " +
                std::to_string(j));
    }

    StretchedGrid sg;
    const int nx = static_cast<int>(x_coords.size());
    const int ny = static_cast<int>(y_coords.size());
    sg.lattice  = LatticeGrid(nx, ny, nz, model);
    sg.x_phys   = std::move(x_coords);
    sg.y_phys   = std::move(y_coords);
    sg.compute_spacings();
    return sg;
}

// ---------------------------------------------------------------------------
// IS-LBM 双线性插值修正
// ---------------------------------------------------------------------------
//
// 调用时机：在 Solver::step()（包含 stream()）之后。
// 此时：
//   grid_.f     — 标准整数偏移流式迁移后的分布函数（=我们要修正的值）
//   grid_.f_tmp — 碰撞后、流式迁移前的分布函数（=插值源，pre-streaming values）
//
// 算法（D2Q9，逐节点逐方向）：
//   dx_ref = min(dx), dy_ref = min(dy)  （参考间距，确保出发点 ≤ 1 格内）
//
//   对每个节点 (i,j) 和方向 α = (c_x, c_y)：
//     出发点物理坐标：
//       xd = x_phys[i] - c_x * dx_ref
//       yd = y_phys[j] - c_y * dy_ref
//
//     在 x_phys/y_phys 中搜索出发点所在区间（std::upper_bound, O(log n)）：
//       k_x：最大的下标 k 使得 x_phys[k] ≤ xd（夹紧到 [0, nx-2]）
//       k_y：同理
//
//     双线性权重：
//       ξ_x = (xd - x_phys[k_x]) / dx[k_x]，ξ_y 同理
//
//     双线性插值（从 f_tmp）：
//       f_α(i,j) = (1-ξ_x)(1-ξ_y)*f_tmp_α(k_x,  k_y  )
//                 + ξ_x   (1-ξ_y)*f_tmp_α(k_x+1,k_y  )
//                 + (1-ξ_x) ξ_y  *f_tmp_α(k_x,  k_y+1)
//                 + ξ_x    ξ_y   *f_tmp_α(k_x+1,k_y+1)
//
//   均匀网格时：k_x=i-c_x, ξ_x=0 → 结果 = f_tmp_α(i-c_x, j-c_y) = 标准流式。
//
// ---------------------------------------------------------------------------
void islbm_interpolation_correction(StretchedGrid& sg, double threshold)
{
    // 检查是否需要修正（拉伸比低于阈值时直接返回）
    const double sr_x = sg.max_stretch_ratio_x();
    const double sr_y = sg.max_stretch_ratio_y();
    if (sr_x < threshold && sr_y < threshold) return;

    LatticeGrid& g = sg.lattice;
    if (g.model != LatticeModel::D2Q9) return;  // 仅支持 D2Q9

    const int nx = g.nx;
    const int ny = g.ny;

    const double dx_ref = sg.dx_ref;
    const double dy_ref = sg.dy_ref;

    // 内部辅助：在 coords 数组中查找最大下标 k 使得 coords[k] <= val，
    // 结果夹紧在 [0, n_max]（其中 n_max 通常为 coords.size()-2）。
    auto find_interval = [](const std::vector<double>& coords, double val, int n_max) -> int {
        auto it = std::upper_bound(coords.begin(), coords.end(), val);
        int k = (it == coords.begin()) ? 0 : static_cast<int>(it - coords.begin()) - 1;
        return std::min(k, n_max);
    };

    // 工作缓冲区：存储修正后的 f（避免原地修改干扰后续节点的插值）
    // 大小 = nx * ny * Q
    std::vector<double> f_new(g.f.size());

#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for collapse(2) schedule(static)
#endif
    for (int j = 0; j < ny; ++j) {
        for (int i = 0; i < nx; ++i) {
            const int dst_node = g.idx(i, j);

            for (int a = 0; a < d2q9::Q; ++a) {
                const int    cx = d2q9::C[a][0];
                const int    cy = d2q9::C[a][1];

                // 物理出发点坐标
                const double xd = sg.x_phys[i] - cx * dx_ref;
                const double yd = sg.y_phys[j] - cy * dy_ref;

                // ---- 搜索 x 区间 ----
                const int k_x  = find_interval(sg.x_phys, xd, nx - 2);
                const double xi_x = (sg.dx[k_x] > 0.0)
                    ? std::min(1.0, std::max(0.0, (xd - sg.x_phys[k_x]) / sg.dx[k_x]))
                    : 0.0;

                // ---- 搜索 y 区间 ----
                const int k_y  = find_interval(sg.y_phys, yd, ny - 2);
                const double xi_y = (sg.dy[k_y] > 0.0)
                    ? std::min(1.0, std::max(0.0, (yd - sg.y_phys[k_y]) / sg.dy[k_y]))
                    : 0.0;

                // ---- 双线性插值（从 f_tmp — 碰撞后、流式迁移前的值）----
                const int n00 = g.idx(k_x,     k_y    );
                const int n10 = g.idx(k_x + 1, k_y    );
                const int n01 = g.idx(k_x,     k_y + 1);
                const int n11 = g.idx(k_x + 1, k_y + 1);

                const double w00 = (1.0 - xi_x) * (1.0 - xi_y);
                const double w10 = xi_x          * (1.0 - xi_y);
                const double w01 = (1.0 - xi_x)  * xi_y;
                const double w11 = xi_x           * xi_y;

                f_new[dst_node * d2q9::Q + a] =
                    w00 * g.f_tmp[n00 * d2q9::Q + a]
                  + w10 * g.f_tmp[n10 * d2q9::Q + a]
                  + w01 * g.f_tmp[n01 * d2q9::Q + a]
                  + w11 * g.f_tmp[n11 * d2q9::Q + a];
            }
        }
    }

    // 将修正后的 f 写回 g.f，并更新宏观量
    g.f = std::move(f_new);
    g.compute_macroscopic();
}

// ---------------------------------------------------------------------------
// 局部稳定性检查
// ---------------------------------------------------------------------------
double stretched_grid_max_local_cfl(const StretchedGrid& sg, double nu)
{
    const LatticeGrid& g = sg.lattice;
    const int n    = g.size();
    const int d    = g.dim();
    const int nx   = g.nx;

    double max_cfl = 0.0;
    for (int i = 0; i < n; ++i) {
        double u2 = 0.0;
        for (int k = 0; k < d; ++k) {
            const double uk = g.u[i * d + k];
            u2 += uk * uk;
        }
        const double speed = std::sqrt(u2);
        // 局部 x 坐标用于间距查找
        const int ix = i % nx;
        const double h = sg.dx[ix];    // 局部最小间距方向
        if (h > 0.0) {
            const double cfl = speed * h / (nu + 1e-300);
            if (cfl > max_cfl) max_cfl = cfl;
        }
    }
    return max_cfl;
}

} // namespace lbm
