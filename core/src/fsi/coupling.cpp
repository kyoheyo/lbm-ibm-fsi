#include "fsi/coupling.hpp"
#include <cstring>

namespace fsi {

// ---------------------------------------------------------------------------
// 分区显式 FSI 耦合步骤
// ---------------------------------------------------------------------------
void fsi_step(lbm::LatticeGrid& fluid,
              ibm::MarkerSet&   markers,
              BeamSolver&       beam,
              const CouplingParams& params)
{
    for (int iter = 0; iter < params.sub_iterations; ++iter) {
        // 1. 在拉格朗日标记点处插值流体速度
        ibm::interpolate_velocity(fluid, markers,
                                  params.dx, params.delta_kernel);

        // 2. 计算 IBM 直接力：
        //    F_IB = rho * (U_IB_target - U_interp) / dt
        //    刚性 IBM：U_IB_target 为结构速度
        //    柔性 IBM：U_IB_target 为梁节点速度
        const auto& beam_dofs = beam.dofs();
        const int n_markers = markers.size();
        for (int m = 0; m < n_markers; ++m) {
            auto& mk = markers.markers[m];
            // 寻找最近梁节点（线性搜索；对大型梁可用 k-d 树优化）
            double best_dist2 = 1e30;
            int    best_node  = 0;
            for (int k = 0; k < beam.n_nodes(); ++k) {
                const double dx = mk.x - beam_dofs[k].x;
                const double dy = mk.y - beam_dofs[k].y;
                const double d2 = dx * dx + dy * dy;
                if (d2 < best_dist2) {
                    best_dist2 = d2;
                    best_node  = k;
                }
            }

            const double target_ux = beam_dofs[best_node].vx;
            const double target_uy = beam_dofs[best_node].vy;

            // 惩罚力（直接力法的刚度系数 κ = 1/dt）
            const double kappa = 1.0 / params.dt;
            mk.fx = kappa * (target_ux - mk.ux);
            mk.fy = kappa * (target_uy - mk.uy);
        }

        // 3. 将 IBM 力展布到欧拉网格
        ibm::spread_force(fluid, markers, params.dx, params.delta_kernel);
    }

    // 4. 将 IBM 反作用力传递到梁（牛顿第三定律）
    auto& beam_dofs = beam.dofs();
    // 先将结构力清零
    for (auto& dof : beam_dofs) {
        dof.fx = 0.0;
        dof.fy = 0.0;
        dof.m  = 0.0;
    }

    const int n_markers = markers.size();
    for (int m = 0; m < n_markers; ++m) {
        const auto& mk = markers.markers[m];
        // 寻找最近梁节点
        double best_dist2 = 1e30;
        int    best_node  = 0;
        for (int k = 0; k < beam.n_nodes(); ++k) {
            const double dx = mk.x - beam_dofs[k].x;
            const double dy = mk.y - beam_dofs[k].y;
            const double d2 = dx * dx + dy * dy;
            if (d2 < best_dist2) {
                best_dist2 = d2;
                best_node  = k;
            }
        }
        // 结构上的反作用力：符号取反，并按弧长元素加权
        beam_dofs[best_node].fx -= mk.fx * mk.ds;
        beam_dofs[best_node].fy -= mk.fy * mk.ds;
    }

    // 5. 推进结构求解器
    beam.step(params.dt);

    // 6. 由梁节点更新标记点位置
    sync_markers_from_beam(markers, beam);
}

// ---------------------------------------------------------------------------
// 将拉格朗日标记点位置与梁节点位置同步
// ---------------------------------------------------------------------------
void sync_markers_from_beam(ibm::MarkerSet& markers,
                             const BeamSolver& beam)
{
    const auto& dofs = beam.dofs();
    const int n_markers = markers.size();
    const int n_nodes   = beam.n_nodes();

    for (int m = 0; m < n_markers; ++m) {
        auto& mk = markers.markers[m];

        // 利用线性形函数从梁节点插值标记点位置
        // 根据参考坐标找到包含该标记点的单元
        const double L = beam.dofs().back().x0 - beam.dofs().front().x0;
        if (L < 1e-15) {
            // 零长度梁 — 无法插值；保持标记点位置不变
            continue;
        }

        // 归一化参考坐标 [0,1]
        const double xi0 = (mk.x0 - dofs[0].x0) / L;
        const double s   = xi0 * (n_nodes - 1);
        const int    e   = std::min(static_cast<int>(s), n_nodes - 2);
        const double t   = s - e;  // 局部坐标，属于 [0,1]

        // 线性插值
        mk.x = (1.0 - t) * dofs[e].x + t * dofs[e + 1].x;
        mk.y = (1.0 - t) * dofs[e].y + t * dofs[e + 1].y;
    }
}

} // namespace fsi
