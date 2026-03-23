#pragma once
// lbm/stretched_grid.hpp — 拉伸（非均匀）坐标网格支持
//
// ============================================================
// 概述
// ============================================================
// 拉伸网格（Stretched Grid）允许在感兴趣的区域（如近壁边界层、障碍物周围）
// 加密节点，同时在远离关注区域处稀疏节点，减少总节点数、提升计算效率。
//
// 本模块提供：
//   1. StretchedGrid 数据结构 — 存储非均匀物理坐标及其与均匀计算坐标的映射
//   2. 常用网格生成工厂函数（等比拉伸、双曲正切拉伸、用户自定义）
//   3. 插值补充 LBM 修正（IS-LBM）— 在流式迁移后补偿非均匀格间距引入的截断误差
//   4. 局部稳定性检查工具
//
// ============================================================
// 物理模型：插值补充 LBM（IS-LBM）
// ============================================================
// 标准 LBM 的流式迁移在均匀网格上是精确的。在拉伸网格上，流式迁移在
// 计算坐标（整数坐标）上仍然精确，但等价于在物理坐标上不同位置的流式迁移，
// 引入截断误差，量级为 O(Δx_max - Δx_min) / Δx_min。
//
// IS-LBM（He & Luo 1997; Mei & Shyy 1998）通过在流式迁移后施加双线性插值
// 修正，将粒子的物理出发点精确地校正到标准化参考间距 dx_ref（= min spacing），
// 使精度恢复到二阶：
//
//   f_α(i,j) ← bilinear_interp(f_post_collision, x_depart(i,α), y_depart(j,α))
//
// 其中出发点在物理坐标中为：
//   x_depart(i,α) = x_phys[i] - c_αx * dx_ref
//   y_depart(j,α) = y_phys[j] - c_αy * dy_ref
//
// 说明：
//   - 均匀网格时 x_depart = x_phys[i-c_αx]，即标准流式（退化）
//   - 拉伸比 ≤ 1.05 时误差在 1% 以内，IS-LBM 修正可省略
//   - 拉伸比 > 1.1 时建议启用修正（通过 islbm_interpolation_correction()）
//
// ============================================================
// 参考文献
// ============================================================
// He X. & Luo L.-S. (1997) "Lattice Boltzmann model for the incompressible
//   Navier-Stokes equation", J. Stat. Phys. 88(3-4):927-944.
// Mei R. & Shyy W. (1998) "On the finite difference-based lattice Boltzmann
//   method in curvilinear coordinates", J. Comput. Phys. 143:426-448.
// He X. et al. (1996) "Some progress in lattice Boltzmann method. Part I.
//   Nonuniform mesh grids", J. Comput. Phys. 129:357-363.

#include "lattice.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace lbm {

// ---------------------------------------------------------------------------
/// 拉伸网格描述符
///
/// StretchedGrid 通过组合（has-a）方式持有标准 LatticeGrid（计算坐标网格），
/// 并额外存储物理坐标数组（x_phys[nx]、y_phys[ny]）和局部格间距（dx[nx]、dy[ny]）。
///
/// 关键约定：
///   - 计算坐标：整数节点 (i, j)，i ∈ [0, nx-1]，j ∈ [0, ny-1]
///   - 物理坐标：x_phys[i]（严格单调递增），y_phys[j]（严格单调递增）
///   - 局部 x 间距：dx[i] = x_phys[i+1] - x_phys[i]，i ∈ [0, nx-2]；
///                  dx[nx-1] = dx[nx-2]（外推，供 IS-LBM 使用）
///   - 局部 y 间距：dy[j]，定义同上
///
/// 典型使用流程：
/// @code
///   // 1. 生成拉伸网格
///   auto sg = StretchedGrid::build_tanh(200, 64, 1, 0.0, 1.0, 0.0, 1.0, 0.0, 0.1);
///
///   // 2. 将 sg.lattice 传给标准 Solver（使用现有 Solver 类）
///   lbm::Solver solver(sg.lattice, omega);
///   solver.add_boundary_condition(bc_west);  // 正常注册 BC
///   solver.add_boundary_condition(bc_east);
///
///   // 3. 每步：碰撞 + 流式迁移（标准），然后可选地施加 IS-LBM 修正
///   for (int t = 0; t < n_steps; ++t) {
///       solver.step();
///       // 拉伸比 > 1.05 时施加修正（在 step() 后调用）
///       islbm_interpolation_correction(sg, 1.05);
///   }
///
///   // 4. 输出时使用物理坐标
///   for (int j = 0; j < sg.lattice.ny; ++j)
///     for (int i = 0; i < sg.lattice.nx; ++i)
///       write(sg.x(i), sg.y(j), sg.lattice.rho[sg.lattice.idx(i,j)], ...);
/// @endcode
// ---------------------------------------------------------------------------
struct StretchedGrid {

    /// 计算坐标上的均匀 LatticeGrid（标准 LBM 在此运行，不受拉伸影响）
    LatticeGrid lattice;

    /// 物理 x 坐标（大小 nx，严格单调递增）
    std::vector<double> x_phys;

    /// 物理 y 坐标（大小 ny，严格单调递增）
    std::vector<double> y_phys;

    /// 物理 x 方向局部间距（大小 nx）
    /// dx[i] = x_phys[i+1] - x_phys[i]，i < nx-1；dx[nx-1] = dx[nx-2]（外推）
    std::vector<double> dx;

    /// 物理 y 方向局部间距（大小 ny）
    /// dy[j] = y_phys[j+1] - y_phys[j]，j < ny-1；dy[ny-1] = dy[ny-2]（外推）
    std::vector<double> dy;

    // ---- 坐标访问器 ----

    /// 节点 (i,j) 的物理 x 坐标
    double x(int i)      const { return x_phys[i]; }
    /// 节点 (i,j) 的物理 y 坐标
    double y(int j)      const { return y_phys[j]; }
    /// 节点 (i,j) 处的物理面积微元（用于面积分、质量流量计算）
    double cell_area(int i, int j) const { return dx[i] * dy[j]; }

    // ---- 网格质量查询 ----

    /// X 方向最大拉伸比（相邻间距比的最大值，均匀时 = 1.0）
    double max_stretch_ratio_x() const;
    /// Y 方向最大拉伸比
    double max_stretch_ratio_y() const;

    /// 最小 x 间距（用于 IS-LBM 的参考间距 dx_ref，compute_spacings() 中缓存）
    double dx_ref = 1.0;
    /// 最小 y 间距（缓存）
    double dy_ref = 1.0;

    /// 最小 x 间距（= dx_ref，保留函数式访问器供旧代码使用）
    double dx_min() const { return dx_ref; }
    /// 最小 y 间距
    double dy_min() const { return dy_ref; }

    // ---- 工厂函数 ----

    /// @brief 等比拉伸网格（Geometric stretch）
    ///
    /// x 方向：相邻格间距以公比 sx 递增（sx > 1 → 向 x1 端间距增大，即 x0 端更密）
    ///         sx = 1.0 → 均匀分布
    /// y 方向：类似，公比 sy。
    ///
    /// 实现：等比数列求和确定总长度与各格间距：
    ///   h_0 = (x1 - x0) * (sx - 1) / (sx^(nx-1) - 1)   (sx ≠ 1)
    ///   h_0 = (x1 - x0) / (nx - 1)                      (sx = 1，均匀)
    ///   h_i = h_0 * sx^i
    ///
    /// @param nx, ny   节点数
    /// @param nz       z 方向节点数（2D 时为 1）
    /// @param x0, x1  物理 x 范围
    /// @param y0, y1  物理 y 范围
    /// @param sx       x 方向公比（> 0；1.0 = 均匀；> 1 = 向 x1 端间距增大）
    /// @param sy       y 方向公比
    /// @param model    格子模型（默认 D2Q9）
    static StretchedGrid build_geometric(
        int nx, int ny, int nz,
        double x0, double x1,
        double y0, double y1,
        double sx, double sy,
        LatticeModel model = LatticeModel::D2Q9);

    /// @brief 双曲正切拉伸网格（Tanh stretch，两端加密）
    ///
    /// 常用于通道流、边界层仿真，在壁面附近自动加密，中央稀疏。
    ///
    /// 坐标公式（x 方向）：
    ///   x_i = x0 + (x1-x0) * 0.5 * [1 + tanh(β*(2*i/(nx-1) - 1)) / tanh(β)]
    ///   β = atanh(sqrt(1 - delta_x))  （beta 控制边界层厚度比）
    ///
    /// @param delta_x  x 方向控制参数 ∈ (0, 1)：
    ///                   → 0 : 均匀分布（tanh(β) → ∞ 时趋近于均匀）
    ///                   → 1 : 极端加密（两端节点极密，中间极稀）
    ///                 典型值：0.8~0.95（边界层网格），0.5~0.7（轻度拉伸）
    ///                 0.0 时自动回退为均匀分布
    /// @param delta_y  y 方向控制参数（含义同 delta_x）
    static StretchedGrid build_tanh(
        int nx, int ny, int nz,
        double x0, double x1,
        double y0, double y1,
        double delta_x,
        double delta_y,
        LatticeModel model = LatticeModel::D2Q9);

    /// @brief 用户自定义坐标网格
    ///
    /// 直接提供物理 x 和 y 坐标数组，由用户完全控制节点分布。
    ///
    /// @param x_coords  物理 x 坐标数组（大小 nx，须严格单调递增）
    /// @param y_coords  物理 y 坐标数组（大小 ny，须严格单调递增）
    /// @param nz        z 方向节点数（2D 为 1）
    /// @throws std::invalid_argument 若数组为空或非单调递增
    static StretchedGrid build_custom(
        int nz,
        std::vector<double> x_coords,
        std::vector<double> y_coords,
        LatticeModel model = LatticeModel::D2Q9);

private:
    /// 从已设定的 x_phys/y_phys 计算并填充 dx/dy 数组
    void compute_spacings();
};

// ---------------------------------------------------------------------------
// IS-LBM 流式迁移补充插值修正
// ---------------------------------------------------------------------------

/// @brief 对拉伸网格施加 IS-LBM 流式迁移补充插值修正。
///
/// 在 Solver::step()（包含 stream()）之后调用，对非均匀格间距引入的截断误差
/// 进行双线性插值修正。修正将 f 值从整数偏移的标准流式来源校正到
/// 物理坐标下精确的出发点位置。
///
/// 算法（D2Q9，以方向 α=(c_x, c_y) 为例）：
///   出发点：xd = x_phys[i] - c_x * dx_ref, yd = y_phys[j] - c_y * dy_ref
///   二分搜索找出 k_x: x_phys[k_x] ≤ xd < x_phys[k_x+1]
///   双线性插值：
///     ξ_x = (xd - x_phys[k_x]) / dx[k_x]
///     ξ_y = (yd - y_phys[k_y]) / dy[k_y]
///     f_α(i,j) = bilinear(f_post_collision, k_x, k_y, ξ_x, ξ_y)
///
/// 退化情形：均匀网格时 k_x=i-c_x, ξ_x=0，与标准流式完全一致（幂等）。
///
/// @param sg         拉伸网格（lattice.f 在 stream() 后已更新；f_tmp 为碰撞后原始值）
/// @param threshold  最大拉伸比阈值（超过此值才施加修正，默认 1.05）；
///                   若 max_stretch_ratio < threshold，修正量极小，自动跳过。
void islbm_interpolation_correction(StretchedGrid& sg, double threshold = 1.05);

/// @brief 计算拉伸网格中的最大局部 CFL 数（用于稳定性监测）
///
/// CFL_local(i,j) = |u(i,j)| * dx_min / nu_effective
/// CFL_max > O(10) 时可能出现数值不稳定（当然 LBM 稳定性实际上更复杂）。
///
/// @param sg  拉伸网格
/// @param nu  流体运动粘度（格子单位）
/// @return    全场最大局部 CFL 数
double stretched_grid_max_local_cfl(const StretchedGrid& sg, double nu);

} // namespace lbm
