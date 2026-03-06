#pragma once
#include "structure.hpp"
#include "../ibm/marker.hpp"
#include "../ibm/interpolation.hpp"
#include "../lbm/lattice.hpp"

namespace fsi {

// ---------------------------------------------------------------------------
// FSI 耦合参数
// ---------------------------------------------------------------------------
struct CouplingParams {
    double dx;                    ///< 欧拉网格间距
    double dt;                    ///< 时间步长
    ibm::DeltaKernel delta_kernel; ///< IBM δ 函数核类型
    int    sub_iterations;        ///< 隐式耦合的内迭代次数（1 = 显式）
};

// ---------------------------------------------------------------------------
// 分区（显式）FSI 耦合步骤
//
//  给定流体网格和梁的当前状态，一个 FSI 子循环包含：
//   1. 在 IBM 标记点处插值流体速度（速度耦合）
//   2. 将标记点移动到匹配的插值速度位置
//   3. 计算 IBM 恢复力（惩罚法 / 直接力法）
//   4. 将 IBM 力展布回欧拉网格
//   5. 将结构求解器推进 dt
// ---------------------------------------------------------------------------
void fsi_step(lbm::LatticeGrid& fluid,
              ibm::MarkerSet&   markers,
              BeamSolver&       beam,
              const CouplingParams& params);

// ---------------------------------------------------------------------------
// 将拉格朗日标记点位置与梁节点位置同步
//（结构求解器推进后调用）
// ---------------------------------------------------------------------------
void sync_markers_from_beam(ibm::MarkerSet& markers,
                             const BeamSolver& beam);

} // namespace fsi
