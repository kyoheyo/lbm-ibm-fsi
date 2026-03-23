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

// ===========================================================================
// GPU 边界条件核函数
// ===========================================================================
//
// 约定：流式迁移（push 方案）+ 指针交换后：
//   f      = d_f     — 流式迁移后的分布函数（待施加 BC）
//   f_pre  = d_f_tmp — 碰撞后、迁移前的分布函数（供半步长反弹 BC 使用）
//   rho / u           — 碰撞前已有的宏观量（在 apply 后由 macroscopic_kernel 刷新）
//
// 各面在 D2Q9 中的幽灵（未知）方向：
//   南壁 j=0    : f[2](N), f[5](NE), f[6](NW)
//   北壁 j=ny-1 : f[4](S), f[7](SW), f[8](SE)
//   西壁 i=0    : f[1](E), f[5](NE), f[8](SE)
//   东壁 i=nx-1 : f[3](W), f[6](NW), f[7](SW)
//
// 线程粒度：每个线程处理一个边界面节点（S/N 面：i∈[0,nx)；E/W 面：j∈[0,ny)）
// ===========================================================================

// ---------------------------------------------------------------------------
// 1. 半步长反弹（BounceBack）— 使用碰后迁移前的分布函数 f_pre
// ---------------------------------------------------------------------------
__global__ void bc_bounce_back_south(double* f, const double* f_pre, int nx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int n = i;  // j=0: idx(i,0) = i
    f[n*9+2] = f_pre[n*9+4];
    f[n*9+5] = f_pre[n*9+7];
    f[n*9+6] = f_pre[n*9+8];
}

__global__ void bc_bounce_back_north(double* f, const double* f_pre, int nx, int ny)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int n = (ny-1)*nx + i;  // j=ny-1
    f[n*9+4] = f_pre[n*9+2];
    f[n*9+7] = f_pre[n*9+5];
    f[n*9+8] = f_pre[n*9+6];
}

__global__ void bc_bounce_back_west(double* f, const double* f_pre, int nx, int ny)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    const int n = j*nx;  // i=0: idx(0,j) = j*nx
    f[n*9+1] = f_pre[n*9+3];
    f[n*9+5] = f_pre[n*9+7];
    f[n*9+8] = f_pre[n*9+6];
}

__global__ void bc_bounce_back_east(double* f, const double* f_pre, int nx, int ny)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    const int n = j*nx + (nx-1);  // i=nx-1
    f[n*9+3] = f_pre[n*9+1];
    f[n*9+6] = f_pre[n*9+8];
    f[n*9+7] = f_pre[n*9+5];
}

// ---------------------------------------------------------------------------
// 2. 全步长反弹（BounceBackFullWay）— 使用迁移后的分布函数 f（就地）
// ---------------------------------------------------------------------------
__global__ void bc_bounce_back_fw_south(double* f, int nx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int n = i;
    const double s4=f[n*9+4], s7=f[n*9+7], s8=f[n*9+8];
    f[n*9+2]=s4; f[n*9+5]=s7; f[n*9+6]=s8;
}

__global__ void bc_bounce_back_fw_north(double* f, int nx, int ny)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int n = (ny-1)*nx + i;
    const double s2=f[n*9+2], s5=f[n*9+5], s6=f[n*9+6];
    f[n*9+4]=s2; f[n*9+7]=s5; f[n*9+8]=s6;
}

__global__ void bc_bounce_back_fw_west(double* f, int nx, int ny)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    const int n = j*nx;
    const double s3=f[n*9+3], s7=f[n*9+7], s6=f[n*9+6];
    f[n*9+1]=s3; f[n*9+5]=s7; f[n*9+8]=s6;
}

__global__ void bc_bounce_back_fw_east(double* f, int nx, int ny)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    const int n = j*nx + (nx-1);
    const double s1=f[n*9+1], s8=f[n*9+8], s5=f[n*9+5];
    f[n*9+3]=s1; f[n*9+6]=s8; f[n*9+7]=s5;
}

// ---------------------------------------------------------------------------
// 3. Zou-He 速度边界条件（非平衡反弹格式）
// ---------------------------------------------------------------------------
__global__ void bc_zou_he_vel_north(double* f, double* rho, double* u,
                                     int nx, int ny, double bc_ux, double bc_uy)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int n = (ny-1)*nx + i;
    double* fp = &f[n*9];
    const double rw = (fp[0]+fp[1]+fp[3] + 2.0*(fp[2]+fp[5]+fp[6])) / (1.0+bc_uy);
    fp[4] = fp[2] - (2.0/3.0)*rw*bc_uy;
    fp[7] = fp[5] + 0.5*(fp[1]-fp[3]) - (1.0/6.0)*rw*bc_uy - 0.5*rw*bc_ux;
    fp[8] = fp[6] - 0.5*(fp[1]-fp[3]) - (1.0/6.0)*rw*bc_uy + 0.5*rw*bc_ux;
    rho[n]=rw; u[n*2+0]=bc_ux; u[n*2+1]=bc_uy;
}

__global__ void bc_zou_he_vel_south(double* f, double* rho, double* u,
                                     int nx, double bc_ux, double bc_uy)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int n = i;
    double* fp = &f[n*9];
    const double rw = (fp[0]+fp[1]+fp[3] + 2.0*(fp[4]+fp[7]+fp[8])) / (1.0-bc_uy);
    fp[2] = fp[4] + (2.0/3.0)*rw*bc_uy;
    fp[5] = fp[7] - 0.5*(fp[1]-fp[3]) + (1.0/6.0)*rw*bc_uy + 0.5*rw*bc_ux;
    fp[6] = fp[8] + 0.5*(fp[1]-fp[3]) + (1.0/6.0)*rw*bc_uy - 0.5*rw*bc_ux;
    rho[n]=rw; u[n*2+0]=bc_ux; u[n*2+1]=bc_uy;
}

__global__ void bc_zou_he_vel_west(double* f, double* rho, double* u,
                                    int nx, int ny, double bc_ux, double bc_uy)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    const int n = j*nx;
    double* fp = &f[n*9];
    const double rw = (fp[0]+fp[2]+fp[4] + 2.0*(fp[3]+fp[6]+fp[7])) / (1.0-bc_ux);
    fp[1] = fp[3] + (2.0/3.0)*rw*bc_ux;
    fp[5] = fp[7] - 0.5*(fp[2]-fp[4]) + (1.0/6.0)*rw*bc_ux + 0.5*rw*bc_uy;
    fp[8] = fp[6] + 0.5*(fp[2]-fp[4]) + (1.0/6.0)*rw*bc_ux - 0.5*rw*bc_uy;
    rho[n]=rw; u[n*2+0]=bc_ux; u[n*2+1]=bc_uy;
}

__global__ void bc_zou_he_vel_east(double* f, double* rho, double* u,
                                    int nx, int ny, double bc_ux, double bc_uy)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    const int n = j*nx + (nx-1);
    double* fp = &f[n*9];
    const double rw = (fp[0]+fp[2]+fp[4] + 2.0*(fp[1]+fp[5]+fp[8])) / (1.0+bc_ux);
    fp[3] = fp[1] - (2.0/3.0)*rw*bc_ux;
    fp[6] = fp[8] - 0.5*(fp[2]-fp[4]) - (1.0/6.0)*rw*bc_ux + 0.5*rw*bc_uy;
    fp[7] = fp[5] + 0.5*(fp[2]-fp[4]) - (1.0/6.0)*rw*bc_ux - 0.5*rw*bc_uy;
    rho[n]=rw; u[n*2+0]=bc_ux; u[n*2+1]=bc_uy;
}

// ---------------------------------------------------------------------------
// 4. Zou-He 压力边界条件
// ---------------------------------------------------------------------------
__global__ void bc_zou_he_pres_north(double* f, double* rho, double* u,
                                      int nx, int ny, double bc_rho, double bc_ux_t)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int n = (ny-1)*nx + i;
    double* fp = &f[n*9];
    const double uy = -1.0 + (fp[0]+fp[1]+fp[3] + 2.0*(fp[2]+fp[5]+fp[6])) / bc_rho;
    fp[4] = fp[2] - (2.0/3.0)*bc_rho*uy;
    fp[7] = fp[5] + 0.5*(fp[1]-fp[3]) - (1.0/6.0)*bc_rho*uy - 0.5*bc_rho*bc_ux_t;
    fp[8] = fp[6] - 0.5*(fp[1]-fp[3]) - (1.0/6.0)*bc_rho*uy + 0.5*bc_rho*bc_ux_t;
    rho[n]=bc_rho; u[n*2+0]=bc_ux_t; u[n*2+1]=uy;
}

__global__ void bc_zou_he_pres_south(double* f, double* rho, double* u,
                                      int nx, double bc_rho, double bc_ux_t)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int n = i;
    double* fp = &f[n*9];
    const double uy = 1.0 - (fp[0]+fp[1]+fp[3] + 2.0*(fp[4]+fp[7]+fp[8])) / bc_rho;
    fp[2] = fp[4] + (2.0/3.0)*bc_rho*uy;
    fp[5] = fp[7] - 0.5*(fp[1]-fp[3]) + (1.0/6.0)*bc_rho*uy + 0.5*bc_rho*bc_ux_t;
    fp[6] = fp[8] + 0.5*(fp[1]-fp[3]) + (1.0/6.0)*bc_rho*uy - 0.5*bc_rho*bc_ux_t;
    rho[n]=bc_rho; u[n*2+0]=bc_ux_t; u[n*2+1]=uy;
}

__global__ void bc_zou_he_pres_west(double* f, double* rho, double* u,
                                     int nx, int ny, double bc_rho, double bc_uy_t)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    const int n = j*nx;
    double* fp = &f[n*9];
    const double ux = 1.0 - (fp[0]+fp[2]+fp[4] + 2.0*(fp[3]+fp[6]+fp[7])) / bc_rho;
    fp[1] = fp[3] + (2.0/3.0)*bc_rho*ux;
    fp[5] = fp[7] - 0.5*(fp[2]-fp[4]) + (1.0/6.0)*bc_rho*ux + 0.5*bc_rho*bc_uy_t;
    fp[8] = fp[6] + 0.5*(fp[2]-fp[4]) + (1.0/6.0)*bc_rho*ux - 0.5*bc_rho*bc_uy_t;
    rho[n]=bc_rho; u[n*2+0]=ux; u[n*2+1]=bc_uy_t;
}

__global__ void bc_zou_he_pres_east(double* f, double* rho, double* u,
                                     int nx, int ny, double bc_rho, double bc_uy_t)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    const int n = j*nx + (nx-1);
    double* fp = &f[n*9];
    const double ux = -1.0 + (fp[0]+fp[2]+fp[4] + 2.0*(fp[1]+fp[5]+fp[8])) / bc_rho;
    fp[3] = fp[1] - (2.0/3.0)*bc_rho*ux;
    fp[6] = fp[8] - 0.5*(fp[2]-fp[4]) - (1.0/6.0)*bc_rho*ux + 0.5*bc_rho*bc_uy_t;
    fp[7] = fp[5] + 0.5*(fp[2]-fp[4]) - (1.0/6.0)*bc_rho*ux - 0.5*bc_rho*bc_uy_t;
    rho[n]=bc_rho; u[n*2+0]=ux; u[n*2+1]=bc_uy_t;
}

// ---------------------------------------------------------------------------
// 5. 充分发展出口（FullyDeveloped）— 复制上游相邻节点分布函数
// ---------------------------------------------------------------------------
__global__ void bc_fully_developed_south(double* f, int nx)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int n_out = i;
    const int n_in  = nx + i;  // j=1
    for (int a = 0; a < 9; ++a) f[n_out*9+a] = f[n_in*9+a];
}

__global__ void bc_fully_developed_north(double* f, int nx, int ny)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int n_out = (ny-1)*nx + i;
    const int n_in  = (ny-2)*nx + i;
    for (int a = 0; a < 9; ++a) f[n_out*9+a] = f[n_in*9+a];
}

__global__ void bc_fully_developed_west(double* f, int nx, int ny)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    const int n_out = j*nx;
    const int n_in  = j*nx + 1;
    for (int a = 0; a < 9; ++a) f[n_out*9+a] = f[n_in*9+a];
}

__global__ void bc_fully_developed_east(double* f, int nx, int ny)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    const int n_out = j*nx + (nx-1);
    const int n_in  = j*nx + (nx-2);
    for (int a = 0; a < 9; ++a) f[n_out*9+a] = f[n_in*9+a];
}

// ---------------------------------------------------------------------------
// 6. 郭照立非平衡外推格式（Guo_Extrapolation）
//    f[a](x_b) = f_eq[a](ρ_b, u_b) + (f[a](x_f) - f_eq[a](ρ_f, u_f))
// ---------------------------------------------------------------------------
__global__ void bc_guo_extrap_south(double* f, const double* rho, const double* u,
                                     int nx, double bc_ux, double bc_uy, double bc_rho)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int n_b = i;          // j=0
    const int n_f = nx + i;     // j=1
    const double rho_f = rho[n_f];
    const double ux_f  = u[n_f*2+0];
    const double uy_f  = u[n_f*2+1];
    const double rho_b = (bc_rho > 0.0) ? bc_rho : rho_f;
    const double ux_b  = (bc_rho > 0.0) ? ux_f   : bc_ux;
    const double uy_b  = (bc_rho > 0.0) ? uy_f   : bc_uy;
    for (int a = 0; a < 9; ++a)
        f[n_b*9+a] = feq_device(rho_b,ux_b,uy_b,a)
                   + (f[n_f*9+a] - feq_device(rho_f,ux_f,uy_f,a));
}

__global__ void bc_guo_extrap_north(double* f, const double* rho, const double* u,
                                     int nx, int ny, double bc_ux, double bc_uy, double bc_rho)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nx) return;
    const int n_b = (ny-1)*nx + i;
    const int n_f = (ny-2)*nx + i;
    const double rho_f = rho[n_f];
    const double ux_f  = u[n_f*2+0];
    const double uy_f  = u[n_f*2+1];
    const double rho_b = (bc_rho > 0.0) ? bc_rho : rho_f;
    const double ux_b  = (bc_rho > 0.0) ? ux_f   : bc_ux;
    const double uy_b  = (bc_rho > 0.0) ? uy_f   : bc_uy;
    for (int a = 0; a < 9; ++a)
        f[n_b*9+a] = feq_device(rho_b,ux_b,uy_b,a)
                   + (f[n_f*9+a] - feq_device(rho_f,ux_f,uy_f,a));
}

__global__ void bc_guo_extrap_west(double* f, const double* rho, const double* u,
                                    int nx, int ny, double bc_ux, double bc_uy, double bc_rho)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    const int n_b = j*nx;
    const int n_f = j*nx + 1;
    const double rho_f = rho[n_f];
    const double ux_f  = u[n_f*2+0];
    const double uy_f  = u[n_f*2+1];
    const double rho_b = (bc_rho > 0.0) ? bc_rho : rho_f;
    const double ux_b  = (bc_rho > 0.0) ? ux_f   : bc_ux;
    const double uy_b  = (bc_rho > 0.0) ? uy_f   : bc_uy;
    for (int a = 0; a < 9; ++a)
        f[n_b*9+a] = feq_device(rho_b,ux_b,uy_b,a)
                   + (f[n_f*9+a] - feq_device(rho_f,ux_f,uy_f,a));
}

__global__ void bc_guo_extrap_east(double* f, const double* rho, const double* u,
                                    int nx, int ny, double bc_ux, double bc_uy, double bc_rho)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= ny) return;
    const int n_b = j*nx + (nx-1);
    const int n_f = j*nx + (nx-2);
    const double rho_f = rho[n_f];
    const double ux_f  = u[n_f*2+0];
    const double uy_f  = u[n_f*2+1];
    const double rho_b = (bc_rho > 0.0) ? bc_rho : rho_f;
    const double ux_b  = (bc_rho > 0.0) ? ux_f   : bc_ux;
    const double uy_b  = (bc_rho > 0.0) ? uy_f   : bc_uy;
    for (int a = 0; a < 9; ++a)
        f[n_b*9+a] = feq_device(rho_b,ux_b,uy_b,a)
                   + (f[n_f*9+a] - feq_device(rho_f,ux_f,uy_f,a));
}

// ---------------------------------------------------------------------------
// 7. 边角点半步长反弹修正（一次性处理全部 4 个角节点）
//
// 与 CPU 端 apply_corner_bounce_back() 逻辑完全对应（见 boundary.cpp §7）：
//   在所有面 BC 核函数执行完毕后，对满足条件的角节点的幽灵方向施加 f[a]=f_pre[opp(a)]。
//
// corner_mask 位掩码（由调用方计算，仅对置位的角节点施加修正）：
//   bit 0（tid=0）: SW(0,0)       幽灵方向：1,2,5,6,8
//   bit 1（tid=1）: SE(nx-1,0)    幽灵方向：2,3,5,6,7
//   bit 2（tid=2）: NW(0,ny-1)    幽灵方向：1,4,5,7,8
//   bit 3（tid=3）: NE(nx-1,ny-1) 幽灵方向：3,4,6,7,8
//
// 线程映射：1 个 block，4 个线程，每线程处理 1 个角节点。
// ---------------------------------------------------------------------------
__global__ void bc_corner_bounce_back_all(double* f, const double* f_pre,
                                           int nx, int ny, unsigned corner_mask)
{
    // 对立方向表：opp[a]
    const int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // 该线程对应的角节点不需要修正时直接返回
    if (!((corner_mask >> tid) & 1u)) return;

    int n;
    // 各角节点的幽灵方向（须由 BC 覆盖的 5 个方向）
    if (tid == 0) {
        // SW (0, 0)
        n = 0;
        const int ghosts[5] = {1, 2, 5, 6, 8};
        for (int k = 0; k < 5; ++k) {
            const int a = ghosts[k];
            f[n*9 + a] = f_pre[n*9 + OPP[a]];
        }
    } else if (tid == 1) {
        // SE (nx-1, 0)
        n = nx - 1;
        const int ghosts[5] = {2, 3, 5, 6, 7};
        for (int k = 0; k < 5; ++k) {
            const int a = ghosts[k];
            f[n*9 + a] = f_pre[n*9 + OPP[a]];
        }
    } else if (tid == 2) {
        // NW (0, ny-1)
        n = (ny - 1) * nx;
        const int ghosts[5] = {1, 4, 5, 7, 8};
        for (int k = 0; k < 5; ++k) {
            const int a = ghosts[k];
            f[n*9 + a] = f_pre[n*9 + OPP[a]];
        }
    } else if (tid == 3) {
        // NE (nx-1, ny-1)
        n = (ny - 1) * nx + (nx - 1);
        const int ghosts[5] = {3, 4, 6, 7, 8};
        for (int k = 0; k < 5; ++k) {
            const int a = ghosts[k];
            f[n*9 + a] = f_pre[n*9 + OPP[a]];
        }
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

    // 设备内存
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

    // CUDA 流与事件（用于异步输出下载管线）
    CUDA_CHECK(cudaStreamCreate(&compute_stream_));
    CUDA_CHECK(cudaStreamCreate(&io_stream_));
    // cudaEventDisableTiming：不记录时间，降低同步开销
    CUDA_CHECK(cudaEventCreateWithFlags(&compute_done_, cudaEventDisableTiming));

    // 固定（pinned）主机缓冲区（双缓冲，rho/u 各两份）
    CUDA_CHECK(cudaMallocHost(&h_rho_[0], rho_bytes));
    CUDA_CHECK(cudaMallocHost(&h_rho_[1], rho_bytes));
    CUDA_CHECK(cudaMallocHost(&h_u_[0],   u_bytes));
    CUDA_CHECK(cudaMallocHost(&h_u_[1],   u_bytes));
}

GpuSolver::~GpuSolver()
{
    // 设备内存
    if (d_f)     cudaFree(d_f);
    if (d_f_tmp) cudaFree(d_f_tmp);
    if (d_rho)   cudaFree(d_rho);
    if (d_u)     cudaFree(d_u);

    // CUDA 流与事件
    if (compute_stream_ != nullptr) cudaStreamDestroy(compute_stream_);
    if (io_stream_      != nullptr) cudaStreamDestroy(io_stream_);
    if (compute_done_   != nullptr) cudaEventDestroy(compute_done_);

    // 固定主机缓冲区
    if (h_rho_[0] != nullptr) cudaFreeHost(h_rho_[0]);
    if (h_rho_[1] != nullptr) cudaFreeHost(h_rho_[1]);
    if (h_u_[0]   != nullptr) cudaFreeHost(h_u_[0]);
    if (h_u_[1]   != nullptr) cudaFreeHost(h_u_[1]);
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

void GpuSolver::add_boundary_condition(const BoundaryCondition& bc)
{
    bcs_.push_back(bc);
}

void GpuSolver::apply_boundary_conditions_gpu()
{
    apply_boundary_conditions_on_stream(nullptr);  // default (legacy) stream
}

void GpuSolver::apply_boundary_conditions_on_stream(cudaStream_t s)
{
    // 边界核函数的线程块大小。边界面节点数通常为 nx 或 ny（O(√n)），
    // BLOCK_BC=128 对 nx/ny ≤ 128 的情况也能正常工作（不足 1 个 block）。
    constexpr int BLOCK_BC = 128;

    for (const auto& bc : bcs_) {
        // 根据面的方向确定本次 kernel 覆盖的节点数
        const int count = (bc.face == Face::South || bc.face == Face::North) ? nx_ : ny_;
        const int grid_bc = (count + BLOCK_BC - 1) / BLOCK_BC;

        switch (bc.type) {
        case BCType::BounceBack:
            switch (bc.face) {
            case Face::South: bc_bounce_back_south<<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_f_tmp, nx_);            break;
            case Face::North: bc_bounce_back_north<<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_f_tmp, nx_, ny_);       break;
            case Face::West:  bc_bounce_back_west <<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_f_tmp, nx_, ny_);       break;
            case Face::East:  bc_bounce_back_east <<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_f_tmp, nx_, ny_);       break;
            default: break;
            }
            break;
        case BCType::BounceBackFullWay:
            switch (bc.face) {
            case Face::South: bc_bounce_back_fw_south<<<grid_bc, BLOCK_BC, 0, s>>>(d_f, nx_);         break;
            case Face::North: bc_bounce_back_fw_north<<<grid_bc, BLOCK_BC, 0, s>>>(d_f, nx_, ny_);    break;
            case Face::West:  bc_bounce_back_fw_west <<<grid_bc, BLOCK_BC, 0, s>>>(d_f, nx_, ny_);    break;
            case Face::East:  bc_bounce_back_fw_east <<<grid_bc, BLOCK_BC, 0, s>>>(d_f, nx_, ny_);    break;
            default: break;
            }
            break;
        case BCType::ZouHe_Velocity:
            switch (bc.face) {
            case Face::North: bc_zou_he_vel_north<<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_rho, d_u, nx_, ny_, bc.ux, bc.uy); break;
            case Face::South: bc_zou_he_vel_south<<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_rho, d_u, nx_, bc.ux, bc.uy);      break;
            case Face::West:  bc_zou_he_vel_west <<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_rho, d_u, nx_, ny_, bc.ux, bc.uy); break;
            case Face::East:  bc_zou_he_vel_east <<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_rho, d_u, nx_, ny_, bc.ux, bc.uy); break;
            default: break;
            }
            break;
        case BCType::ZouHe_Pressure:
            switch (bc.face) {
            case Face::North: bc_zou_he_pres_north<<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_rho, d_u, nx_, ny_, bc.rho, bc.ux); break;
            case Face::South: bc_zou_he_pres_south<<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_rho, d_u, nx_, bc.rho, bc.ux);      break;
            case Face::West:  bc_zou_he_pres_west <<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_rho, d_u, nx_, ny_, bc.rho, bc.uy); break;
            case Face::East:  bc_zou_he_pres_east <<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_rho, d_u, nx_, ny_, bc.rho, bc.uy); break;
            default: break;
            }
            break;
        case BCType::FullyDeveloped:
            switch (bc.face) {
            case Face::South: bc_fully_developed_south<<<grid_bc, BLOCK_BC, 0, s>>>(d_f, nx_);        break;
            case Face::North: bc_fully_developed_north<<<grid_bc, BLOCK_BC, 0, s>>>(d_f, nx_, ny_);   break;
            case Face::West:  bc_fully_developed_west <<<grid_bc, BLOCK_BC, 0, s>>>(d_f, nx_, ny_);   break;
            case Face::East:  bc_fully_developed_east <<<grid_bc, BLOCK_BC, 0, s>>>(d_f, nx_, ny_);   break;
            default: break;
            }
            break;
        case BCType::Guo_Extrapolation:
            switch (bc.face) {
            case Face::South: bc_guo_extrap_south<<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_rho, d_u, nx_, bc.ux, bc.uy, bc.rho);          break;
            case Face::North: bc_guo_extrap_north<<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_rho, d_u, nx_, ny_, bc.ux, bc.uy, bc.rho);     break;
            case Face::West:  bc_guo_extrap_west <<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_rho, d_u, nx_, ny_, bc.ux, bc.uy, bc.rho);     break;
            case Face::East:  bc_guo_extrap_east <<<grid_bc, BLOCK_BC, 0, s>>>(d_f, d_rho, d_u, nx_, ny_, bc.ux, bc.uy, bc.rho);     break;
            default: break;
            }
            break;
        default:
            break;
        }
        CUDA_CHECK(cudaGetLastError());
    }

    // 所有面 BC 核函数完成后，对满足条件的角节点施加半步长反弹修正（对应 CPU 端逻辑）。
    // 仅对两相邻面中至少有一面是固壁 BC 的角节点施加修正。
    if (!bcs_.empty()) {
        // 计算固壁面掩码：bit0=South, bit1=North, bit2=West, bit3=East
        unsigned wall_face_bits = 0u;
        for (const auto& bc : bcs_) {
            if (bc.type == BCType::BounceBack || bc.type == BCType::BounceBackFullWay) {
                switch (bc.face) {
                    case Face::South: wall_face_bits |= 0x1u; break;
                    case Face::North: wall_face_bits |= 0x2u; break;
                    case Face::West:  wall_face_bits |= 0x4u; break;
                    case Face::East:  wall_face_bits |= 0x8u; break;
                    default: break;
                }
            }
        }
        // corner_mask：bit0=SW, bit1=SE, bit2=NW, bit3=NE
        const bool south = (wall_face_bits & 0x1u) != 0u;
        const bool north = (wall_face_bits & 0x2u) != 0u;
        const bool west  = (wall_face_bits & 0x4u) != 0u;
        const bool east  = (wall_face_bits & 0x8u) != 0u;
        const unsigned corner_mask = ((south || west) ? 0x1u : 0u)
                                   | ((south || east) ? 0x2u : 0u)
                                   | ((north || west) ? 0x4u : 0u)
                                   | ((north || east) ? 0x8u : 0u);
        if (corner_mask != 0u) {
            bc_corner_bounce_back_all<<<1, 4, 0, s>>>(d_f, d_f_tmp, nx_, ny_, corner_mask);
            CUDA_CHECK(cudaGetLastError());
        }
    }
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

void GpuSolver::download_rho_u(LatticeGrid& g) const
{
    // 仅下载宏观量（ρ、u），用于输出快照。
    // 数据量：n×8 + n×16 = 24n 字节；512² 时约 6.3 MB（比 download() 少 7×）。
    const std::size_t rho_bytes = static_cast<std::size_t>(n_)     * sizeof(double);
    const std::size_t u_bytes   = static_cast<std::size_t>(n_) * 2 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(g.rho.data(), d_rho, rho_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(g.u.data(),   d_u,   u_bytes,   cudaMemcpyDeviceToHost));
}

void GpuSolver::upload(const LatticeGrid& g)
{
    const std::size_t f_bytes = static_cast<std::size_t>(n_) * 9 * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d_f, g.f.data(), f_bytes, cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// GpuSolver::step() — 同步单步（默认流 + cudaDeviceSynchronize）
//
// 标准同步 API，适合大多数场景：
//   - 小网格调试
//   - 无输出（纯计算）的大批量仿真
//   - output_interval 极大（输出间隔 >> 磁盘写盘时间），I/O 开销可忽略
//
// 当 output_interval 较小时，每次 download_rho_u/download + 写盘会让 GPU 空闲
// 数十毫秒。若这成为瓶颈，改用 step_async() + enqueue_async_download_rho_u()
// 实现 GPU 计算与磁盘 I/O 的重叠（见 §16.4.7）。
// ---------------------------------------------------------------------------
void GpuSolver::step()
{
    constexpr int BLOCK = 256;
    const int g = (n_ + BLOCK - 1) / BLOCK;

    // 1. BGK collision: relax d_f toward equilibrium in-place
    collide_bgk_kernel<<<g, BLOCK>>>(d_f, d_rho, d_u, omega_, n_);
    CUDA_CHECK(cudaGetLastError());

    // 2. Push streaming
    stream_kernel<<<g, BLOCK>>>(d_f, d_f_tmp, nx_, ny_);
    CUDA_CHECK(cudaGetLastError());
    double* tmp = d_f; d_f = d_f_tmp; d_f_tmp = tmp;

    // 3. GPU-native boundary conditions
    if (!bcs_.empty()) {
        apply_boundary_conditions_gpu();
    }

    // 4. Macroscopic fields
    macroscopic_kernel<<<g, BLOCK>>>(d_f, d_rho, d_u, n_);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
}

// ---------------------------------------------------------------------------
// GpuSolver::step_async() — 异步单步（compute_stream，无阻塞同步）
//
// 在 compute_stream_ 上提交全部核函数，末尾记录 compute_done_ 事件。
// 立即返回，不阻塞 CPU。
//
// 典型用法（双缓冲输出管线，参见 §16.4.7 文档）：
//   step_async()                          // 启动计算，立即返回
//   enqueue_async_download_rho_u(buf_idx) // 异步拷贝前一步结果到 pinned 内存
//   write_file(...)                        // CPU 写盘（与 GPU 计算并行）
//   wait_compute()                         // 等待当前步计算完成
// ---------------------------------------------------------------------------
void GpuSolver::step_async()
{
    constexpr int BLOCK = 256;
    const int g = (n_ + BLOCK - 1) / BLOCK;

    // 全部核函数提交到 compute_stream_
    collide_bgk_kernel<<<g, BLOCK, 0, compute_stream_>>>(d_f, d_rho, d_u, omega_, n_);
    CUDA_CHECK(cudaGetLastError());

    stream_kernel<<<g, BLOCK, 0, compute_stream_>>>(d_f, d_f_tmp, nx_, ny_);
    CUDA_CHECK(cudaGetLastError());
    double* tmp = d_f; d_f = d_f_tmp; d_f_tmp = tmp;

    if (!bcs_.empty()) {
        apply_boundary_conditions_on_stream(compute_stream_);
    }

    macroscopic_kernel<<<g, BLOCK, 0, compute_stream_>>>(d_f, d_rho, d_u, n_);
    CUDA_CHECK(cudaGetLastError());

    // 记录计算完成事件（io_stream_ 的异步拷贝将等待此事件）
    CUDA_CHECK(cudaEventRecord(compute_done_, compute_stream_));
}

void GpuSolver::wait_compute()
{
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
}

// ---------------------------------------------------------------------------
// GpuSolver::enqueue_async_download_rho_u() — 异步拷贝 ρ/u 到固定主机缓冲区
//
// io_stream_ 等待 compute_done_ 事件（确保 GPU 计算完成后才开始拷贝），
// 然后发起两个 cudaMemcpyAsync（device→pinned host），立即返回。
//
// buf_idx: 双缓冲索引（0 或 1）。调用方在连续两次输出步之间交替使用，
//          以避免 CPU 正在读取上一次缓冲区时被覆盖。
// ---------------------------------------------------------------------------
void GpuSolver::enqueue_async_download_rho_u(int buf_idx)
{
    const std::size_t rho_bytes = static_cast<std::size_t>(n_)     * sizeof(double);
    const std::size_t u_bytes   = static_cast<std::size_t>(n_) * 2 * sizeof(double);

    // io_stream_ 等待 compute_done_ 事件（GPU 计算完成后再拷贝）
    CUDA_CHECK(cudaStreamWaitEvent(io_stream_, compute_done_, 0));

    const int b = buf_idx & 1;  // 限制为 0 或 1；调用方应保证 buf_idx ∈ {0, 1}
    CUDA_CHECK(cudaMemcpyAsync(h_rho_[b], d_rho, rho_bytes, cudaMemcpyDeviceToHost, io_stream_));
    CUDA_CHECK(cudaMemcpyAsync(h_u_[b],   d_u,   u_bytes,   cudaMemcpyDeviceToHost, io_stream_));
}

void GpuSolver::sync_async_download()
{
    CUDA_CHECK(cudaStreamSynchronize(io_stream_));
}

const double* GpuSolver::h_rho(int buf_idx) const { return h_rho_[buf_idx & 1]; }
const double* GpuSolver::h_u(int buf_idx)   const { return h_u_  [buf_idx & 1]; }

} // namespace lbm

#endif // LBM_ENABLE_CUDA
