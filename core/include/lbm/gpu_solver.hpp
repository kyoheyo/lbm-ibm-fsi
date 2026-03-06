#pragma once
// lbm/gpu_solver.hpp — 基于 CUDA 的 LBM 求解器接口（D2Q9 BGK）
//
// 设计概述
// --------
// GpuSolver 将分布函数 f、宏观量 ρ/u 存储在 GPU 设备内存中，
// 碰撞（BGK）、流式迁移以及全部六种边界条件均在 GPU 上通过 CUDA 核函数执行。
// 借助 add_boundary_condition() 注册的 BC 将在每次 step()/step_async() 时自动在 GPU 上施加，
// 无需任何每步 GPU↔CPU 数据传输。
//
// 输出下载策略与性能分析
// ----------------------
// 对于写文件所需的输出下载，存在两种模式：
//
// 【模式 A：同步下载（简单，但有 I/O 停顿）】
//   gpu.step();
//   gpu.download_rho_u(grid);     // 仅下载 rho+u（6.3 MB @512²），同步
//   write_snapshot(grid);         // 磁盘写入（~20 ms）—— GPU 空闲！
//
// 【模式 B：异步双缓冲（推荐，消除 I/O 停顿）】
//   原理：GPU 计算步 T+1 的同时，CPU 异步拷贝步 T 的数据并写盘。
//
//   // 主循环骨架（详见 §16.4.7 文档）：
//   bool pending = false;  int buf = 0;
//   for step in 0..nsteps:
//       if pending:
//           gpu.sync_async_download();              // 等待拷贝完成（通常已完成）
//           write_snapshot(gpu.h_rho(buf), gpu.h_u(buf), n);  // 写盘（GPU同步计算中）
//           buf ^= 1;
//       gpu.step_async();                           // 非阻塞，在 compute_stream 执行
//       if output_step:
//           gpu.enqueue_async_download_rho_u(buf);  // 非阻塞，在 io_stream 执行
//           pending = true;
//       else: pending = false;
//   gpu.wait_compute();                             // 最终同步
//   if pending: gpu.sync_async_download(); write_snapshot(...);
//
// 性能收益（512² 网格，write_interval=100）：
//   同步模式：每 100 步暂停 ~20 ms（磁盘写入）GPU 利用率约 95%
//   异步模式：20 ms 写盘与 GPU 计算完全重叠，GPU 利用率接近 100%
//
// 适用场景
// --------
//   - 单卡（单 GPU）大规模 D2Q9 网格（nx × ny ≳ 512²）
//   - 需要 CUDA 工具链：nvcc + CUDA Toolkit ≥ 11.0
//
// 构建说明
// --------
// 需要在 CMakeLists.txt 中 ENABLE_CUDA=ON 并将 gpu_solver.cu 添加到 lbm_core 目标。
// Cargo 侧通过 LBM_ENABLE_CUDA=ON 环境变量触发。

#ifdef LBM_ENABLE_CUDA

#include "lattice.hpp"
#include "boundary.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <vector>

namespace lbm {

// ---------------------------------------------------------------------------
/// 基于 CUDA 的 D2Q9 BGK 求解器
// ---------------------------------------------------------------------------
class GpuSolver {
public:
    /// 构造：将 CPU 网格数据上传到 GPU 设备内存；
    /// 同时分配 CUDA 流、事件以及用于异步下载的固定（pinned）主机内存。
    explicit GpuSolver(const LatticeGrid& g, double omega);

    /// 析构：释放 GPU 设备内存、CUDA 流/事件以及 pinned 主机内存。
    ~GpuSolver();

    // 禁止拷贝（设备内存不可平凡拷贝）
    GpuSolver(const GpuSolver&)            = delete;
    GpuSolver& operator=(const GpuSolver&) = delete;

    // ------------------------------------------------------------------
    // 边界条件注册
    // ------------------------------------------------------------------

    /// 注册一个边界条件。注册后每次 step()/step_async() 将自动在 GPU 上施加该 BC，
    /// 无需任何 CPU↔GPU 数据传输。可多次调用以注册多个 BC（按顺序施加）。
    void add_boundary_condition(const BoundaryCondition& bc);

    // ------------------------------------------------------------------
    // 同步主循环 API（适合小网格或不关心 I/O 停顿的场景）
    // ------------------------------------------------------------------

    /// 单步融合执行（同步）：BGK 碰撞 + 流式迁移 + GPU-BC + 宏观量更新。
    /// 使用默认 CUDA 流，结束时发出一次 cudaDeviceSynchronize。
    void step();

    /// 在 GPU 上执行 BGK 碰撞（使用默认流）
    void collide();

    /// 在 GPU 上执行流式迁移（使用默认流）
    void stream();

    /// 在 GPU 上计算宏观量（使用默认流）
    void compute_macroscopic();

    // ------------------------------------------------------------------
    // 异步管线 API（推荐用于有输出 I/O 的大规模仿真）
    // ------------------------------------------------------------------

    /// 异步单步执行：在 compute_stream 上启动全部核函数（BGK + 流式 + GPU-BC + 宏观量），
    /// 末尾记录 compute_done 事件，立即返回（不阻塞 CPU）。
    /// 必须在下一次调用 step_async() 前通过 wait_compute() 或
    /// enqueue_async_download_rho_u() 确保顺序正确。
    void step_async();

    /// 等待 compute_stream 完成（cudaStreamSynchronize）。
    /// 在 step_async() 后、读取 GPU 结果之前调用。
    void wait_compute();

    /// 将 d_rho、d_u 异步拷贝到固定主机缓冲区 buf_idx（0 或 1）的 io_stream 上。
    /// io_stream 会自动等待 compute_done 事件（确保计算完成后才开始拷贝），
    /// 调用本函数后可以立即启动下一步的 step_async()，实现 GPU 计算与数据拷贝的重叠。
    /// buf_idx: 双缓冲索引（0 或 1），调用方在两次输出步之间交替使用。
    void enqueue_async_download_rho_u(int buf_idx = 0);

    /// 等待 io_stream（异步拷贝）完成（cudaStreamSynchronize）。
    /// 完成后可以安全读取 h_rho(buf_idx) / h_u(buf_idx) 中的数据。
    void sync_async_download();

    /// 固定主机内存访问器 — 仅在 sync_async_download() 后调用。
    /// buf_idx 必须与传入 enqueue_async_download_rho_u() 的值一致。
    const double* h_rho(int buf_idx = 0) const;
    const double* h_u(int buf_idx = 0)   const;

    // ------------------------------------------------------------------
    // 数据传输（同步）
    // ------------------------------------------------------------------

    /// 仅下载宏观量（ρ、u）到 CPU 网格（6.3 MB @512²）。
    /// 与 download() 相比减少 7× 数据量，适合仅用于输出的场景。
    void download_rho_u(LatticeGrid& g) const;

    /// 下载完整状态（f、f_tmp、ρ、u）到 CPU 网格（44 MB @512²）。
    /// 用于 CPU 端边界条件修正，或需要完整状态检查点的场景。
    void download(LatticeGrid& g) const;

    /// 将 CPU 网格 f 上传到 GPU（手动 CPU-BC 工作流中边界条件修正后调用）
    void upload(const LatticeGrid& g);

    // ------------------------------------------------------------------
    // 辅助查询
    // ------------------------------------------------------------------

    int    n()     const { return n_; }   ///< 总节点数
    double omega() const { return omega_; }
    void   set_omega(double w) { omega_ = w; }

private:
    int    nx_, ny_;            ///< 网格尺寸
    int    n_;                  ///< 总节点数 = nx * ny
    double omega_;              ///< 松弛频率

    // GPU 设备内存
    double* d_f      = nullptr; ///< 分布函数 [n * Q]
    double* d_f_tmp  = nullptr; ///< 流式迁移临时缓冲 [n * Q]
    double* d_rho    = nullptr; ///< 密度 [n]
    double* d_u      = nullptr; ///< 速度 [n * 2]

    // CUDA 流与事件
    cudaStream_t compute_stream_ = nullptr; ///< 计算流（step_async 使用）
    cudaStream_t io_stream_      = nullptr; ///< I/O 流（异步拷贝使用）
    cudaEvent_t  compute_done_   = nullptr; ///< 计算完成事件（io_stream 等待此事件后才开始拷贝）

    // 固定（pinned）主机缓冲区（双缓冲，用于异步输出下载）
    double* h_rho_[2] = {nullptr, nullptr}; ///< 固定 rho 缓冲区 [n] × 2
    double* h_u_[2]   = {nullptr, nullptr}; ///< 固定 u 缓冲区 [n * 2] × 2

    // 已注册的边界条件（CPU 端保存参数）
    std::vector<BoundaryCondition> bcs_;

    /// 在指定 CUDA 流上施加所有已注册的边界条件核函数
    void apply_boundary_conditions_on_stream(cudaStream_t s);

    /// 在默认流上施加所有已注册的边界条件核函数（兼容旧接口）
    void apply_boundary_conditions_gpu();
};

} // namespace lbm

#endif // LBM_ENABLE_CUDA

// ---------------------------------------------------------------------------
// GPU 拉伸网格求解器接口设计（API 文档）
// ---------------------------------------------------------------------------
//
// 以下为 StretchedGpuSolver 的接口设计，用于在 GPU 上加速拉伸网格 LBM 仿真。
// 实际 CUDA 实现需要 LBM_ENABLE_CUDA=ON 编译环境；本注释块作为 API 规范。
//
// 设计原则（与 GpuSolver 保持一致）：
//   - 数据常驻 GPU 设备内存：f、f_tmp、rho、u、x_phys_d、y_phys_d、dx_d、dy_d
//   - 碰撞（BGK/MRT）与标准 GpuSolver 完全相同（局部操作，无需物理坐标）
//   - 流式迁移（stream_kernel）使用标准整数偏移（同 GpuSolver）
//   - IS-LBM 修正（islbm_correction_kernel）：在 stream_kernel 后以额外 CUDA 核函数施加
//     * 每线程负责一个节点 (i,j) 的所有方向 α
//     * 使用 GPU 端 x_phys_d/y_phys_d 数组以及预计算的 dx_ref/dy_ref
//     * 插值使用 CUDA texture 内存加速双线性查找（可选优化）
//
// 典型用法：
//   StretchedGpuSolver gpu(sg, omega);
//   gpu.add_boundary_condition(bc_west);
//   gpu.add_boundary_condition(bc_east);
//   for (int t = 0; t < n_steps; ++t) {
//       gpu.step(/*islbm=*/true);   // 包含 IS-LBM 修正
//   }
//   gpu.download_rho_u(sg.lattice); // 下载宏观量到 CPU
//
// 参数：
//   islbm_threshold  float — 拉伸比阈值（同 CPU 端 islbm_interpolation_correction）
//                    默认 1.05；低于此值自动跳过 IS-LBM 核函数
//
// 性能预期（A100 GPU, 1024×512 网格）：
//   标准流式步：~0.5 ms
//   IS-LBM 修正步：~0.8 ms（额外开销约 60%，轻度拉伸时可关闭修正）
//   每步总计：~1.5 ms（含 BC 施加）
//
// 实现状态：
//   [ ] 设备内存分配（x_phys_d/y_phys_d/dx_d/dy_d）
//   [ ] islbm_correction_kernel（D2Q9，双线性插值）
//   [ ] 支持 GpuSolver 所有 BC 类型（BounceBack、ZouHe、Guo）
//   [ ] 异步双缓冲输出管线（同 GpuSolver）
//
#ifdef LBM_ENABLE_CUDA

namespace lbm {

// Forward declaration of StretchedGrid (include stretched_grid.hpp to use)
struct StretchedGrid;

// ---------------------------------------------------------------------------
/// GPU 加速拉伸网格求解器（IS-LBM，D2Q9 BGK）
///
/// 继承 GpuSolver 的所有功能，额外支持拉伸网格 IS-LBM 修正。
///
/// @note 构造时需传入 StretchedGrid（包含物理坐标）；内部将坐标数组
///       一并上传到 GPU 设备内存供 IS-LBM 核函数使用。
// ---------------------------------------------------------------------------
class StretchedGpuSolver : public GpuSolver {
public:
    /// 构造：上传 CPU 拉伸网格数据（含物理坐标）到 GPU。
    explicit StretchedGpuSolver(const StretchedGrid& sg, double omega);

    ~StretchedGpuSolver();

    // 禁止拷贝
    StretchedGpuSolver(const StretchedGpuSolver&)            = delete;
    StretchedGpuSolver& operator=(const StretchedGpuSolver&) = delete;

    /// 单步：碰撞 + 流式迁移 + IS-LBM 修正（可选）+ GPU-BC + 宏观量更新
    /// @param apply_islbm  是否施加 IS-LBM 修正（默认 true；均匀网格可关闭）
    void step(bool apply_islbm = true);

    /// 仅施加 IS-LBM 流式迁移补充插值修正核函数（在 stream() 后调用）
    void islbm_correct();

    /// 更新拉伸比阈值（低于此值时 islbm_correct() 自动跳过）
    void set_islbm_threshold(double t) { islbm_threshold_ = t; }

private:
    // GPU 端物理坐标和间距数组
    double* d_x_phys  = nullptr;  ///< 物理 x 坐标 [nx]
    double* d_y_phys  = nullptr;  ///< 物理 y 坐标 [ny]
    double* d_dx      = nullptr;  ///< x 方向局部间距 [nx]
    double* d_dy      = nullptr;  ///< y 方向局部间距 [ny]
    double  dx_ref_   = 1.0;      ///< 参考 x 间距（min dx）
    double  dy_ref_   = 1.0;      ///< 参考 y 间距（min dy）
    double  islbm_threshold_ = 1.05;  ///< IS-LBM 修正拉伸比阈值
    double  max_sr_x_ = 1.0;      ///< 最大 x 拉伸比（构造时计算）
    double  max_sr_y_ = 1.0;      ///< 最大 y 拉伸比（构造时计算）
};

} // namespace lbm

#endif // LBM_ENABLE_CUDA (second block)
