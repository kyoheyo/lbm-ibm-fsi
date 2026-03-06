#pragma once
// lbm/gpu_solver.hpp — 基于 CUDA 的 LBM 求解器接口（D2Q9 BGK）
//
// 设计概述
// --------
// GpuSolver 将分布函数 f、宏观量 ρ/u 存储在 GPU 设备内存中，
// 碰撞（BGK）、流式迁移以及全部六种边界条件均在 GPU 上通过 CUDA 核函数执行。
// 借助 add_boundary_condition() 注册的 BC 将在每次 step() 时自动在 GPU 上施加，
// 无需任何每步 GPU↔CPU 数据传输。
//
// 适用场景
// --------
//   - 单卡（单 GPU）大规模 D2Q9 网格（nx × ny ≳ 512²）
//   - 需要 CUDA 工具链：nvcc + CUDA Toolkit ≥ 11.0
//
// 推荐用法（GPU 端执行所有 BC，无每步传输）
// --------
//   // 1. 在 CPU 侧创建网格并初始化
//   LatticeGrid g(nx, ny, 1, LatticeModel::D2Q9);
//   // 2. 初始化 GPU 求解器（将 g 的数据上传到 GPU）
//   GpuSolver gpu(g, omega);
//   // 3. 注册边界条件（仅记录参数，BC 将在 GPU 上执行）
//   gpu.add_boundary_condition({BCType::ZouHe_Velocity, Face::North, 0.1, 0.0});
//   gpu.add_boundary_condition({BCType::BounceBack, Face::South});
//   // 4. 主循环（全程在 GPU 执行，无每步 CPU↔GPU 传输）
//   for (int step = 0; step < nsteps; ++step) {
//       gpu.step();  // collide + stream + GPU-BC + macroscopic，单次 sync
//   }
//   // 5. 仅在需要输出时下载结果
//   gpu.download(g);
//
// 旧版兼容用法（CPU 端手动应用 BC，每步需传输）
// --------
//   for (int step = 0; step < nsteps; ++step) {
//       gpu.step();                              // 仅 collide+stream+macroscopic
//       gpu.download(g);                         // GPU→CPU（供 BC 使用）
//       apply_boundary_conditions(g, bcs);       // CPU 执行边界条件
//       g.compute_macroscopic();                 // CPU 更新宏观量
//       gpu.upload(g);                           // CPU→GPU（上传修正后的 f）
//   }
//   // 注意：此模式在 512² 网格下每步需传输 ~63 MB（download+upload），
//   // 导致 PCIe 传输成为主要瓶颈（远超 GPU 计算时间）。
//
// 构建说明
// --------
// 需要在 CMakeLists.txt 中 ENABLE_CUDA=ON 并将 gpu_solver.cu 添加到 lbm_core 目标。
// Cargo 侧通过 LBM_ENABLE_CUDA=ON 环境变量触发。

#ifdef LBM_ENABLE_CUDA

#include "lattice.hpp"
#include "boundary.hpp"
#include <cstddef>
#include <vector>

namespace lbm {

// ---------------------------------------------------------------------------
/// 基于 CUDA 的 D2Q9 BGK 求解器
// ---------------------------------------------------------------------------
class GpuSolver {
public:
    /// 构造：将 CPU 网格数据上传到 GPU 设备内存
    explicit GpuSolver(const LatticeGrid& g, double omega);

    /// 析构：释放 GPU 设备内存
    ~GpuSolver();

    // 禁止拷贝（设备内存不可平凡拷贝）
    GpuSolver(const GpuSolver&)            = delete;
    GpuSolver& operator=(const GpuSolver&) = delete;

    /// 注册一个边界条件。注册后每次 step() 将自动在 GPU 上施加该 BC，
    /// 无需任何 CPU↔GPU 数据传输。可多次调用以注册多个 BC（按顺序施加）。
    void add_boundary_condition(const BoundaryCondition& bc);

    /// 单步融合执行：BGK 碰撞 + 流式迁移 + GPU-BC（若有注册）+ 宏观量更新。
    /// 仅发出一次 cudaDeviceSynchronize，最大化 GPU 利用率。
    /// 若已通过 add_boundary_condition() 注册 BC，则本函数全程在 GPU 执行，
    /// 无需任何每步 CPU↔GPU 数据传输。
    void step();

    /// 在 GPU 上执行 BGK 碰撞（原地修改 d_f）
    void collide();

    /// 在 GPU 上执行流式迁移（push 方案，d_f → d_f_tmp，然后交换指针）
    void stream();

    /// 在 GPU 上计算宏观量（d_f → d_rho, d_u）
    void compute_macroscopic();

    /// 将设备端 f、f_tmp、ρ、u 拷贝回 CPU 网格（用于结果输出或 CPU 端 BC）。
    /// 推荐仅在需要写出结果时调用（例如每隔 output_interval 步），而非每步调用。
    void download(LatticeGrid& g) const;

    /// 将 CPU 网格 f 上传到 GPU（手动 CPU-BC 工作流中边界条件修正后调用）
    void upload(const LatticeGrid& g);

    double omega() const { return omega_; }
    void   set_omega(double w) { omega_ = w; }

private:
    int    nx_, ny_;            ///< 网格尺寸
    int    n_;                  ///< 总节点数 = nx * ny
    double omega_;              ///< 松弛频率

    double* d_f      = nullptr; ///< GPU：分布函数 [n * Q]
    double* d_f_tmp  = nullptr; ///< GPU：流式迁移临时缓冲 [n * Q]
    double* d_rho    = nullptr; ///< GPU：密度 [n]
    double* d_u      = nullptr; ///< GPU：速度 [n * 2]

    std::vector<BoundaryCondition> bcs_; ///< 已注册的边界条件（CPU 端保存参数）

    /// 在 GPU 上逐一施加 bcs_ 中的边界条件核函数（在 step() 的 stream 之后调用）
    void apply_boundary_conditions_gpu();
};

} // namespace lbm

#endif // LBM_ENABLE_CUDA
