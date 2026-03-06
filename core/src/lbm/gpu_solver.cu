// core/src/lbm/gpu_solver.cu — CUDA D2Q9 BGK 求解器实现
//
// 编译要求：nvcc + CUDA Toolkit >= 11.0
// 构建方式：ENABLE_CUDA=ON（CMake）或 LBM_ENABLE_CUDA=ON（cargo build）

#ifdef LBM_ENABLE_CUDA

#include "lbm/gpu_solver.hpp"
#include "lbm/lattice.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cstring>   // memcpy

namespace lbm {

// ---------------------------------------------------------------------------
// D2Q9 常量（设备端只读内存）
// ---------------------------------------------------------------------------
// 方向编号 a:  0   1   2   3   4   5   6   7   8
//              rest E   N   W   S  NE  NW  SW  SE
__constant__ int   d_Cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
__constant__ int   d_Cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};
__constant__ double d_W[9]  = {
    4.0/9.0,                              // 0 静止
    1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, // 1-4 轴向
    1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0  // 5-8 对角
};

// ---------------------------------------------------------------------------
// 辅助宏：CUDA 错误检查
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(std::string("CUDA error: ")               \
                                     + cudaGetErrorString(_e));                 \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// 平衡分布函数（设备端内联）
// ---------------------------------------------------------------------------
__device__ __forceinline__
double feq_device(double rho, double ux, double uy, int a)
{
    constexpr double cs2  = 1.0 / 3.0;
    constexpr double cs4  = cs2 * cs2;
    const double cu = d_Cx[a] * ux + d_Cy[a] * uy;
    const double u2 = ux * ux + uy * uy;
    return d_W[a] * rho * (1.0 + cu / cs2 + cu * cu / (2.0 * cs4) - u2 / (2.0 * cs2));
}

// ---------------------------------------------------------------------------
// BGK 碰撞核函数
//
//   f_a* = f_a - ω (f_a - f_eq)
//
// 每个线程负责一个节点（i = blockIdx.x * blockDim.x + threadIdx.x）。
// 访问模式：f[node * Q + a]，Q=9。AoS 布局对每个节点的 9 个方向是连续的，
// 单节点内部数据局部性好；对跨节点的同方向访问（如 a=1 对所有节点）则不连续。
// 若需进一步优化可切换到 SoA 布局（f[a * n + node]），此处以简洁为优先。
// ---------------------------------------------------------------------------
__global__
void collide_bgk_kernel(double* __restrict__ f,
                         const double* __restrict__ rho,
                         const double* __restrict__ u,
                         double omega,
                         int n)
{
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) return;

    const double ri = rho[node];
    const double ux = u[node * 2 + 0];
    const double uy = u[node * 2 + 1];

    for (int a = 0; a < 9; ++a) {
        const double fa  = f[node * 9 + a];
        const double feq = feq_device(ri, ux, uy, a);
        f[node * 9 + a]  = fa - omega * (fa - feq);
    }
}

// ---------------------------------------------------------------------------
// 流式迁移核函数（push 方案）
//
// 每个线程处理节点 (i, j) 的全部 9 个方向。
// 使用周期性边界条件处理域边界节点。
//
// 性能优化：由于 D2Q9 的每个方向速度分量仅为 {-1, 0, +1}，
// 周期性回绕只会越界至多一格，因此用条件加减代替整数取模，
// 避免硬件代价较高的除法指令。
// f_src[node*Q+a] → f_dst[dest_node*Q+a]
// ---------------------------------------------------------------------------
__global__
void stream_kernel(const double* __restrict__ f_src,
                   double*       __restrict__ f_dst,
                   int nx, int ny)
{
    const int total = nx * ny;
    const int node  = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= total) return;

    const int ii = node % nx;
    const int jj = node / nx;

    for (int a = 0; a < 9; ++a) {
        // Periodic wrap using conditional arithmetic — avoids costly integer
        // division that `% nx` / `% ny` would produce on GPU.
        // Each lattice velocity component is in {-1, 0, +1}, so at most one
        // conditional branch is taken.
        int di = ii + d_Cx[a];
        if (di < 0) di += nx; else if (di >= nx) di -= nx;
        int dj = jj + d_Cy[a];
        if (dj < 0) dj += ny; else if (dj >= ny) dj -= ny;
        f_dst[(dj * nx + di) * 9 + a] = f_src[node * 9 + a];
    }
}

// ---------------------------------------------------------------------------
// 宏观量核函数
//
// ρ = Σ_a f_a,   u = (Σ_a c_a f_a) / ρ
//
// 稳定性：当 ρ ≤ 0（所有 f 为零或数值发散）时，令 u = 0 而非产生 NaN/Inf，
// 防止后续碰撞步因非有限速度导致整个模拟崩溃。
// ---------------------------------------------------------------------------
__global__
void macroscopic_kernel(const double* __restrict__ f,
                         double* __restrict__ rho,
                         double* __restrict__ u,
                         int n)
{
    const int node = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (node >= n) return;

    double r = 0.0, cx_sum = 0.0, cy_sum = 0.0;
    for (int a = 0; a < 9; ++a) {
        const double fa = f[node * 9 + a];
        r      += fa;
        cx_sum += fa * d_Cx[a];
        cy_sum += fa * d_Cy[a];
    }
    rho[node] = r;
    // Guard against divide-by-zero: if density is non-positive (degenerate
    // state), set velocity to zero rather than producing NaN/Inf.
    if (r > 0.0) {
        u[node * 2 + 0] = cx_sum / r;
        u[node * 2 + 1] = cy_sum / r;
    } else {
        u[node * 2 + 0] = 0.0;
        u[node * 2 + 1] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// GpuSolver 实现
// ---------------------------------------------------------------------------

GpuSolver::GpuSolver(const LatticeGrid& g, double omega)
    : nx_(g.nx), ny_(g.ny), n_(g.size()), omega_(omega)
{
    const std::size_t f_bytes   = static_cast<std::size_t>(n_) * 9 * sizeof(double);
    const std::size_t rho_bytes = static_cast<std::size_t>(n_)     * sizeof(double);
    const std::size_t u_bytes   = static_cast<std::size_t>(n_) * 2 * sizeof(double);

    CUDA_CHECK(cudaMalloc(&d_f,     f_bytes));
    CUDA_CHECK(cudaMalloc(&d_f_tmp, f_bytes));
    CUDA_CHECK(cudaMalloc(&d_rho,   rho_bytes));
    CUDA_CHECK(cudaMalloc(&d_u,     u_bytes));

    // 上传初始数据
    CUDA_CHECK(cudaMemcpy(d_f,   g.f.data(),   f_bytes,   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rho, g.rho.data(), rho_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_u,   g.u.data(),   u_bytes,   cudaMemcpyHostToDevice));
    // f_tmp 初始化为与 f 相同（用于 BounceBack BC 读取碰后值）
    CUDA_CHECK(cudaMemcpy(d_f_tmp, g.f.data(), f_bytes, cudaMemcpyHostToDevice));
}

GpuSolver::~GpuSolver()
{
    if (d_f)     cudaFree(d_f);
    if (d_f_tmp) cudaFree(d_f_tmp);
    if (d_rho)   cudaFree(d_rho);
    if (d_u)     cudaFree(d_u);
}

void GpuSolver::collide()
{
    constexpr int BLOCK = 256;
    const int grid = (n_ + BLOCK - 1) / BLOCK;
    collide_bgk_kernel<<<grid, BLOCK>>>(d_f, d_rho, d_u, omega_, n_);
    // Check for kernel launch errors; execution errors are caught at the next sync
    CUDA_CHECK(cudaGetLastError());
    // No cudaDeviceSynchronize here: kernels in the default stream execute in
    // order, so stream_kernel (or the caller's download/sync) will naturally
    // wait for this kernel to finish first.
}

void GpuSolver::stream()
{
    // Push-streaming: every cell in d_f_tmp is overwritten by exactly one
    // source cell, so no prior backup copy of d_f into d_f_tmp is needed.
    // After the kernel: d_f_tmp holds the streamed distribution.
    // After the pointer swap below: d_f = streamed result,
    //   d_f_tmp = pre-streaming (post-collision) values — kept for
    //   CPU-side half-way bounce-back BC (download() copies it to g.f_tmp).
    constexpr int BLOCK = 256;
    const int grid = (n_ + BLOCK - 1) / BLOCK;
    stream_kernel<<<grid, BLOCK>>>(d_f, d_f_tmp, nx_, ny_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    // Pointer swap (no GPU work): d_f ↔ d_f_tmp
    double* tmp = d_f;
    d_f     = d_f_tmp;
    d_f_tmp = tmp;
}

void GpuSolver::compute_macroscopic()
{
    constexpr int BLOCK = 256;
    const int grid = (n_ + BLOCK - 1) / BLOCK;
    macroscopic_kernel<<<grid, BLOCK>>>(d_f, d_rho, d_u, n_);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void GpuSolver::download(LatticeGrid& g) const
{
    const std::size_t f_bytes   = static_cast<std::size_t>(n_) * 9 * sizeof(double);
    const std::size_t rho_bytes = static_cast<std::size_t>(n_)     * sizeof(double);
    const std::size_t u_bytes   = static_cast<std::size_t>(n_) * 2 * sizeof(double);

    CUDA_CHECK(cudaMemcpy(g.f.data(),   d_f,   f_bytes,   cudaMemcpyDeviceToHost));
    // d_f_tmp 存碰后值（供 BounceBack BC 读取 g.f_tmp）
    CUDA_CHECK(cudaMemcpy(g.f_tmp.data(), d_f_tmp, f_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g.rho.data(), d_rho, rho_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g.u.data(),   d_u,   u_bytes,   cudaMemcpyDeviceToHost));
}

void GpuSolver::upload(const LatticeGrid& g)
{
    const std::size_t f_bytes = static_cast<std::size_t>(n_) * 9 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_f, g.f.data(), f_bytes, cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// GpuSolver::step() — single-synchronize fused step
//
// Launches three kernels (collide → stream → macroscopic) back-to-back in
// the default CUDA stream (which serialises them automatically) and issues
// ONE cudaDeviceSynchronize at the end.  This avoids the two extra host-GPU
// round-trips that the individual collide() / stream() methods would incur.
//
// After this call:
//   d_f     — post-stream distribution function (ready for download / BC)
//   d_f_tmp — post-collision, pre-stream distribution (for half-way BB BC)
//   d_rho, d_u — updated macroscopic fields
// ---------------------------------------------------------------------------
void GpuSolver::step()
{
    constexpr int BLOCK = 256;
    const int g = (n_ + BLOCK - 1) / BLOCK;

    // 1. BGK collision: relax d_f toward equilibrium in-place
    collide_bgk_kernel<<<g, BLOCK>>>(d_f, d_rho, d_u, omega_, n_);
    CUDA_CHECK(cudaGetLastError());

    // 2. Push streaming: propagate d_f into d_f_tmp; the stream kernel
    //    overwrites every element of d_f_tmp, so no prior backup copy needed.
    stream_kernel<<<g, BLOCK>>>(d_f, d_f_tmp, nx_, ny_);
    CUDA_CHECK(cudaGetLastError());
    // Pointer swap (host-side only, zero GPU cost):
    //   d_f     = streamed result (old d_f_tmp)
    //   d_f_tmp = pre-stream / post-collision values (old d_f)
    double* tmp = d_f;
    d_f     = d_f_tmp;
    d_f_tmp = tmp;

    // 3. Compute macroscopic fields (ρ, u) from the updated d_f
    macroscopic_kernel<<<g, BLOCK>>>(d_f, d_rho, d_u, n_);
    CUDA_CHECK(cudaGetLastError());

    // Single synchronise — catches execution errors from all three kernels
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace lbm

#endif // LBM_ENABLE_CUDA
