#pragma once
// lbm/gpu_solver.hpp — 基于 CUDA 的 LBM 求解器接口（D2Q9 BGK）
//
// 设计概述
// --------
// GpuSolver 将分布函数 f、宏观量 ρ/u 存储在 GPU 设备内存中，
// 碰撞（BGK）和流式迁移均在 GPU 上通过 CUDA 核函数执行。
// 边界条件由于逻辑复杂仍在 CPU 执行，每步需要一次 GPU→CPU→GPU 拷贝。
//
// 适用场景
// --------
//   - 单卡（单 GPU）大规模 D2Q9 网格（nx × ny ≳ 512²）
//   - 需要 CUDA 工具链：nvcc + CUDA Toolkit ≥ 11.0
//
// 使用示例
// --------
//   // 1. 在 CPU 侧创建网格并初始化边界条件
//   LatticeGrid g(nx, ny, 1, LatticeModel::D2Q9);
//   // 2. 初始化 GPU 求解器（将 g 的数据上传到 GPU）
//   GpuSolver gpu(g, omega);
//   // 3. 主循环
//   for (int step = 0; step < nsteps; ++step) {
//       gpu.collide();    // GPU 上执行 BGK 碰撞
//       gpu.stream();     // GPU 上执行流式迁移
//       gpu.download(g);  // 将 f 从 GPU 拷贝回 CPU（供 BC 使用）
//       apply_boundary_conditions(g, bcs);   // CPU 执行边界条件
//       g.compute_macroscopic();             // CPU 更新宏观量
//       gpu.upload(g);    // 将修正后的 f 重新上传 GPU
//   }
//   // 4. 获取最终结果
//   gpu.download(g);
//
// 构建说明
// --------
// 需要在 CMakeLists.txt 中 ENABLE_CUDA=ON 并将 gpu_solver.cu 添加到 lbm_core 目标。
// Cargo 侧通过 LBM_ENABLE_CUDA=ON 环境变量触发。

#ifdef LBM_ENABLE_CUDA

#include "lattice.hpp"
#include <cstddef>

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

    /// Single fused step: BGK collision + streaming + macroscopic update,
    /// with one cudaDeviceSynchronize at the end (faster than calling
    /// collide() + stream() + compute_macroscopic() individually).
    /// After returning, d_f / d_f_tmp / d_rho / d_u are all up-to-date.
    void step();

    /// 在 GPU 上执行 BGK 碰撞（原地修改 d_f）
    void collide();

    /// 在 GPU 上执行流式迁移（push 方案，d_f → d_f_tmp，然后交换指针）
    void stream();

    /// 在 GPU 上计算宏观量（d_f → d_rho, d_u）
    void compute_macroscopic();

    /// 将设备端 f、ρ、u 拷贝回 CPU 网格（用于执行 CPU 端边界条件）
    void download(LatticeGrid& g) const;

    /// 将 CPU 网格 f 上传到 GPU（边界条件修正后调用）
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
};

} // namespace lbm

#endif // LBM_ENABLE_CUDA
