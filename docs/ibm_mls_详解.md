# 浸入边界法 IBM 完整技术详解

> **参考论文**：  
> Wu Z. & Fu X., *"An implicit moving-least-squares immersed boundary method for high fidelity fluid–structure interaction simulations"*, Journal of Computational Physics, 2025.  
> （以下简称 **2025 JCP**）

**源文件对应关系**

| 头文件 | 实现文件 |
|--------|---------|
| `core/include/ibm/interpolation.hpp` | `core/src/ibm/interpolation.cpp` |
| `core/include/ibm/marker.hpp` | `core/src/ibm/marker.cpp` |

---

## 目录

1. [数据结构](#1-数据结构)
2. [Peskin δ 函数核](#2-peskin-δ-函数核)
3. [标准 Peskin 速度插值（J 算子）](#3-标准-peskin-速度插值j-算子)
4. [标准 Peskin 力展布（J^T 算子）](#4-标准-peskin-力展布jt-算子)
5. [多重直接力法 MDF-IBM（Algorithm 0）](#5-多重直接力法-mdf-ibmalgorithm-0)
6. [MLS 速度插值（MLS-J 算子）](#6-mls-速度插值mls-j-算子)
   - 6.1 MLS 矩阵构建
   - 6.2 solve3x3 对称矩阵 Cramér 法则
   - 6.3 在标记点处求值
7. [MLS 力展布（MLS-J^T 算子）](#7-mls-力展布mls-jt-算子)
8. [原始 MLS-IBM（Algorithm 1）](#8-原始-mls-ibmalgorithm-1)
9. [显式 MLS-IBM（Algorithm 2）](#9-显式-mls-ibmalgorithm-2)
10. [隐式 MLS-IBM（Algorithm 3）核心辅助函数](#10-隐式-mls-ibmalgorithm-3核心辅助函数)
    - 10.1 build_mls_shape_functions
    - 10.2 build_correlation_matrix（Eq.28）
    - 10.3 gmres_dense_jacobi（GMRES 求解器）
    - 10.4 lu_factor_dense / lu_solve_dense（LU 分解）
11. [隐式 MLS-IBM Scheme II：GMRES（Algorithm 3 C2 路径）](#11-隐式-mls-ibm-scheme-iigmres)
12. [隐式 MLS-IBM Scheme I：固定物体直接矩阵求逆](#12-隐式-mls-ibm-scheme-i固定物体)
13. [罚函数 IBM（Penalty-IBM）](#13-罚函数-ibmpenalty-ibm)
14. [IBM 合力统计](#14-ibm-合力统计)
15. [MPI 分区适配](#15-mpi-分区适配)
16. [MPI 幽灵层交换](#16-mpi-幽灵层交换)
17. [方案对比与选用建议](#17-方案对比与选用建议)
18. [代码审查发现的问题与注意事项](#18-代码审查发现的问题与注意事项)

---

## 1. 数据结构

### 1.1 Marker（拉格朗日标记点）

```cpp
// core/include/ibm/marker.hpp
struct Marker {
    double x, y, z;      // 当前位置（格子单位；对固体 = 物理位置 / dx）
    double x0, y0, z0;   // 参考/未变形位置
    double fx, fy, fz;   // IBM 力密度（格子单位）：F_k^b（论文 Eq.18）
    double ux, uy, uz;   // MLS/δ 插值得到的流体速度 U_k*（Eq.15/20）
    double ds;           // 弧长元素 ΔS_k（2D）或面积元素（3D）
};
```

**物理含义**：
- `ds`：对应论文中每个 Lagrangian 标记点的 **弧长元素 ΔS_k**（2D）。  
  圆形标记点：`ds = 2π·R / N_l`（均匀分布）。  
  在力展布中作为离散积分权重：`f_j += φ_j^k · F_k^b · ΔS_k`。

### 1.2 MarkerSet（标记点集合）

```cpp
struct MarkerSet {
    std::vector<Marker> markers;
    int owner_i_lo = 0, owner_i_hi = INT_MAX;  // MPI 归属范围（本地坐标）
    int owner_j_lo = 0, owner_j_hi = INT_MAX;
};
```

`owner_*` 字段由 `ibm_marker_set_adapt_to_partition()` 设置，用于 MPI 多分区归属过滤（防止同一标记点被多个进程重复计算）。

---

## 2. Peskin δ 函数核

**论文依据**：Peskin (2002) *Acta Numerica*，以及标准 LBM-IBM 文献。

### 2.1 数学定义

**二点（线性）核**（支撑宽度 2h）：

$$\phi(r) = \frac{1}{h}\hat{\phi}\left(\frac{r}{h}\right), \quad \hat{\phi}(s) = \begin{cases} 1 - |s| & |s| < 1 \\ 0 & \text{其他} \end{cases}$$

**四点 Peskin 余弦核**（支撑宽度 4h，默认核）：

$$\hat{\phi}(s) = \begin{cases} \dfrac{1 + \cos\!\left(\dfrac{\pi s}{2}\right)}{4} & |s| < 2 \\ 0 & \text{其他} \end{cases}$$

### 2.2 代码对应

```cpp
// interpolation.cpp:22–38
double delta_phi(double r, double h, DeltaKernel kernel)
{
    const double roh = r / h;   // 无量纲距离 r/h
    if (kernel == DeltaKernel::TwoPoint) {
        const double absr = std::abs(roh);
        if (absr < 1.0) return (1.0 - absr) / h;   // φ = (1 - |r/h|) / h
        return 0.0;
    } else {
        // FourPoint Peskin 核
        const double absr = std::abs(roh);
        if (absr < 2.0)
            return (1.0 + std::cos(PI * roh / 2.0)) / (4.0 * h);
        return 0.0;
    }
}
```

**二维 δ 函数**（可分离）：

$$\delta(\mathbf{x} - \mathbf{X}) = \phi(x - X) \cdot \phi(y - Y)$$

对应代码：
```cpp
const double phi = delta_phi(mk.x - ii*dx, dx, kernel)
                 * delta_phi(mk.y - jj*dx, dx, kernel)
                 * dx * dx;   // 乘以 Δx² = dx²（体积积分因子）
```

---

## 3. 标准 Peskin 速度插值（J 算子）

**连续形式（2025 JCP Eq.6）**：

$$\mathbf{U}_k^* = \int_\Omega \mathbf{u}^*(\mathbf{x})\, \delta(\mathbf{x} - \mathbf{X}_k)\, d\mathbf{x}$$

**离散形式**：

$$\mathbf{U}_k^* = \sum_{j \in \mathcal{S}(k)} \mathbf{u}_j^* \cdot \phi(x_j - X_k) \cdot \phi(y_j - Y_k) \cdot \Delta x^2$$

其中 $\mathcal{S}(k)$ 为标记点 $\mathbf{X}_k$ 的支撑域内的 Euler 节点集合。

### 3.1 代码对应

```cpp
// interpolation.cpp:43–102
void interpolate_velocity(const lbm::LatticeGrid& grid,
                          MarkerSet& ms, double dx, DeltaKernel kernel)
{
    const int support = (kernel == DeltaKernel::TwoPoint) ? 1 : 2;

    for (int m = 0; m < ms.size(); ++m) {
        const double xm = mk.x / dx;          // 标记点格子坐标
        const int i0 = std::floor(xm);         // 左邻格点（非对称支撑域）
        // 支撑范围：[i0 - support, i0 + support + 1]
        // 原因：floor 使标记点位于 [i0, i0+1)，Peskin 核左右各延伸 support 格
        for (int dj = -support; dj <= support + 1; ++dj) {
            for (int di = -support; di <= support + 1; ++di) {
                const double phi = delta_phi(mk.x - ii*dx, dx, kernel)
                                 * delta_phi(mk.y - jj*dx, dx, kernel)
                                 * dx * dx;   // ← 对应 δ(x-X)·Δx²
                ux_sum += grid.u[node*2] * phi;   // U_k^x = Σ u_j φ_j
            }
        }
    }
}
```

**关键设计说明**：
- 使用 `std::floor` 而非 `std::round`：Peskin δ 核是非对称支撑（基于 $\lfloor x/\Delta x \rfloor$），MLS 则使用 `std::round`（对称矩形支撑域，见第 6 节）。
- 固体节点过滤：`if (grid.solid[node]) continue;`——固体内部速度无效，跳过以防污染插值。

---

## 4. 标准 Peskin 力展布（J^T 算子）

**连续形式（2025 JCP Eq.7）**：

$$\mathbf{f}(\mathbf{x}) = \int_\Gamma \mathbf{F}^b(\mathbf{X})\, \delta(\mathbf{x} - \mathbf{X})\, d\Gamma(\mathbf{X})$$

**离散形式**：

$$f_j = \sum_{k=1}^{N_l} F_k^b \cdot \phi(x_j - X_k) \cdot \phi(y_j - Y_k) \cdot \Delta S_k$$

### 4.1 代码对应

```cpp
// interpolation.cpp:107–159
void spread_force(lbm::LatticeGrid& grid, const MarkerSet& ms,
                  double dx, DeltaKernel kernel)
{
    std::fill(grid.force.begin(), grid.force.end(), 0.0);   // 先清零

    for (int m = 0; m < ms.size(); ++m) {
        const double phi = delta_phi(mk.x - ii*dx, dx, kernel)
                         * delta_phi(mk.y - jj*dx, dx, kernel)
                         * mk.ds;   // ← 展布权重 = φ·ΔS（弧长元素）
#ifdef LBM_ENABLE_OPENMP
#pragma omp atomic
#endif
        grid.force[node*2]   += mk.fx * phi;   // f_j += F_k · φ_j^k · ΔS_k
        grid.force[node*2+1] += mk.fy * phi;
    }
}
```

**展布与插值的量纲一致性**：
- 插值：`φ·Δx²`（面积元素，对应体积积分 $\int d\mathbf{x}$）
- 展布：`φ·ΔS`（弧长元素，对应曲线积分 $\int d\Gamma$）

---

## 5. 多重直接力法 MDF-IBM（Algorithm 0）

**参考**：Luo K. et al. (2007) *J. Comput. Phys.* 227:454–483.

### 5.1 算法

MDF 通过多次子迭代逐步消除无滑移误差：

$$
\begin{aligned}
&\text{初始化：} \mathbf{F}_{\text{total}} = 0,\quad \mathbf{u}^{(0)} = \mathbf{u}^* \\
&\text{对 } \ell = 0, 1, \ldots, N_{\text{iter}}-1: \\
&\quad (1)\; \mathbf{U}_k^{(\ell)} = \sum_j \mathbf{u}_j^{(\ell)} \phi_j^k \Delta x^2 \\
&\quad (2)\; \delta\mathbf{F}_k^{(\ell)} = \frac{\mathbf{0} - \mathbf{U}_k^{(\ell)}}{\Delta t} \\
&\quad (3)\; \delta \mathbf{f}_j^{(\ell)} = \sum_k \delta\mathbf{F}_k^{(\ell)} \phi_j^k \Delta S_k \\
&\quad (4)\; \mathbf{u}_j^{(\ell+1)} = \mathbf{u}_j^{(\ell)} + \Delta t \cdot \delta\mathbf{f}_j^{(\ell)} \\
&\quad (5)\; \mathbf{F}_{\text{total}} \mathrel{+}= \delta\mathbf{f}^{(\ell)}
\end{aligned}
$$

### 5.2 代码对应

```cpp
// interpolation.cpp:164–230
void compute_ibm_forces_mdf(fluid, ms, dx, dt, n_iter, kernel)
{
    std::vector<double> u_work = fluid.u;          // u^(0) = u*
    std::vector<double> F_total(n*d, 0.0);
    std::vector<double> dF_euler(n*d, 0.0);

    for (int iter = 0; iter < n_iter; ++iter) {
        // Step (1): 插值 u^(ℓ) 到 Lagrangian 点
        std::swap(fluid.u, u_work);
        interpolate_velocity(fluid, ms, dx, kernel);   // 写入 mk.ux/uy
        std::swap(fluid.u, u_work);

        // Step (2): δF_k = (0 - U_k^(ℓ)) / dt
        for (auto& mk : ms.markers) {
            mk.fx = (0.0 - mk.ux) / dt;
            mk.fy = (0.0 - mk.uy) / dt;
        }

        // Step (3): 展布 δF 到 Euler 网格
        spread_force(fluid, ms, dx, kernel);    // 写入 fluid.force
        std::swap(fluid.force, dF_euler);       // dF_euler = δf^(ℓ)

        // Step (4): u^(ℓ+1) = u^(ℓ) + dt · δf^(ℓ)
        for (int i = 0; i < n; ++i) {
            u_work[i*d+0] += dt * dF_euler[i*d+0];
            u_work[i*d+1] += dt * dF_euler[i*d+1];
        }

        // Step (5): 累积
        for (int i = 0; i < n*d; ++i) F_total[i] += dF_euler[i];
    }

    fluid.force = F_total;   // 最终总力写入 fluid.force（供 Guo 体力格式）
}
```

**关键实现技巧**：使用 `std::swap(fluid.u, u_work)` 临时替换速度场，避免额外的数据拷贝。

---

## 6. MLS 速度插值（MLS-J 算子）

**参考**：Liu et al. (1997) *Int. J. Numer. Meth. Fluids*；  
Vanella & Balaras (2009) *J. Comput. Phys.* 228:2366–2391；  
2025 JCP Wu & Fu §3.1，Eq.(10)–(14)。

### 6.1 MLS 方法概述

MLS（Moving Least Squares，移动最小二乘）用**局部多项式**代替 δ 函数来插值速度。对标记点 $\mathbf{X}_k$，在其支撑域 $\mathcal{S}(k)$ 内求解加权最小二乘问题：

$$\min_{\mathbf{a}} \sum_{j \in \mathcal{S}(k)} w(\mathbf{x}_j - \mathbf{X}_k) \left[ p(\mathbf{x}_j - \mathbf{X}_k) \cdot \mathbf{a} - u(\mathbf{x}_j) \right]^2$$

**基函数**（线性基，3 个自由度）：

$$p(\Delta\mathbf{x}) = \left[1,\; \frac{\Delta x}{\Delta x_{\text{ref}}},\; \frac{\Delta y}{\Delta x_{\text{ref}}}\right]^T$$

其中 $\Delta x_{\text{ref}} = \Delta x$（格子间距），归一化以改善条件数。

**Gaussian 权函数**（2025 JCP Eq.14）：

$$w(\mathbf{x}_j - \mathbf{X}_k) = \exp\!\left(-\frac{|\mathbf{x}_j - \mathbf{X}_k|^2}{(H_k \cdot \varepsilon)^2}\right)$$

$$H_k = 1.5\,\Delta x \quad(\text{支撑域半宽}),\quad \varepsilon = 0.3 \quad(\text{Gaussian 集中参数})$$

因此有效高斯宽度 $h_{\text{eff}} = H_k \cdot \varepsilon = 0.45\,\Delta x$。

**矩形支撑域**（2025 JCP Fig.2）：

$$|\Delta x| \leq H_k \;\text{且}\; |\Delta y| \leq H_k \quad \Rightarrow \quad (2H_k/\Delta x + 1)^2 = 4^2 = 16\text{ 个格点（最多）}$$

### 6.2 MLS 矩阵构建

$$M = \sum_{j \in \mathcal{S}(k)} w_j\, p_j \otimes p_j^T \quad (3\times3 \text{ 正定对称矩阵})$$

$$\mathbf{b}_u = \sum_{j \in \mathcal{S}(k)} w_j\, u_j^x\, p_j, \quad \mathbf{b}_v = \sum_{j \in \mathcal{S}(k)} w_j\, u_j^y\, p_j$$

**代码对应**（`mls_interpolate_velocity`，行 293–388）：

```cpp
// interpolation.cpp:293–301（参数设置，对应 2025 JCP Eq.14）
const double H_k   = 1.5 * dx;          // 支撑域半宽
const double eps   = 0.3;               // Gaussian 集中参数 ε
const double h_eff = H_k * eps;         // = 0.45·dx
const double h2    = h_eff * h_eff;     // = (0.45·dx)²（Gaussian 分母）
const int    iR    = std::ceil(H_k/dx); // = ceil(1.5) = 2（覆盖半径格数）

// 最近格点（对称支撑域，使用 round 而非 floor）
const int i0 = std::round(mk.x / dx);
const int j0 = std::round(mk.y / dx);

for (int dj = -iR; dj <= iR; ++dj) {      // 遍历 [-2, +2] × [-2, +2]
    for (int di = -iR; di <= iR; ++di) {
        const double ddx = ii*dx - mk.x;   // Δx = x_j - X_k
        const double ddy = jj*dx - mk.y;   // Δy = y_j - Y_k

        if (std::abs(ddx) > H_k || std::abs(ddy) > H_k) continue;  // 矩形域过滤

        const double r2  = ddx*ddx + ddy*ddy;
        const double w = std::exp(-r2 / h2);    // w_j = exp(-r²/(H_k·ε)²)

        const double p[3] = {1.0, ddx/dx, ddy/dx};   // 归一化基函数 p_j

        // M += w · p ⊗ p（3×3 对称矩阵）
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                M[r][c] += w * p[r] * p[c];

        // b_u += w · u_j · p，b_v += w · v_j · p
        for (int r = 0; r < 3; ++r) {
            bu[r] += w * ux_i * p[r];
            bv[r] += w * uy_i * p[r];
        }
    }
}
```

**为何用 `round` 而非 `floor`**：MLS 支撑域是以标记点为中心的对称矩形 `[-H_k, H_k]²`，使用 `round` 找最近格点作为中心，确保支撑域在标记点两侧对称。Peskin δ 核则使用 `floor` 因为其公式是基于 `⌊x/Δx⌋` 定义的非对称区间。

### 6.3 solve3x3 对称矩阵 Cramér 法则

求解 $M \mathbf{a} = \mathbf{b}$（3×3 线性系统）。

**数学原理**：Cramér 法则 $a_k = \det(M_k) / \det(M)$，其中 $M_k$ 是将第 $k$ 列替换为 $\mathbf{b}$ 的矩阵。

**代码实现技巧**：利用 $\det(M) = \det(M^T)$，将"替换列"等价转换为"替换行"（对称矩阵转置等于原矩阵），使三个分量的计算结构统一（均沿第一行展开）：

```cpp
// interpolation.cpp:261–280
// x[0]：替换第0列 → det([[b[0],A[0][1],A[0][2]], [b[1],A[1][1],A[1][2]], [b[2],A[2][1],A[2][2]]])
x[0] = inv_det * (b[0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
                - b[1]*(A[0][1]*A[2][2]-A[0][2]*A[2][1])
                + b[2]*(A[0][1]*A[1][2]-A[0][2]*A[1][1]));

// x[1]：替换第1列 → 等价计算转置行列式 det([[A[0][0],A[0][1],A[0][2]], [b[0],b[1],b[2]], [A[2][0],A[2][1],A[2][2]]])
// 对称矩阵下等价于正确的 Cramér 公式（见代码注释）
x[1] = inv_det * (A[0][0]*(b[1]*A[2][2]-b[2]*A[2][1])
                - A[0][1]*(b[0]*A[2][2]-b[2]*A[2][0])
                + A[0][2]*(b[0]*A[2][1]-b[1]*A[2][0]));

// x[2]：替换第2列 → 等价计算 det([[A[0][0],A[0][1],A[0][2]], [A[1][0],A[1][1],A[1][2]], [b[0],b[1],b[2]]])
x[2] = inv_det * (A[0][0]*(A[1][1]*b[2]-A[1][2]*b[1])
                - A[0][1]*(A[1][0]*b[2]-A[1][2]*b[0])
                + A[0][2]*(A[1][0]*b[1]-A[1][1]*b[0]));
```

> ⚠️ **重要约束**：`solve3x3` 仅对**对称矩阵**正确。MLS 矩阵 $M = \sum w_j p_j p_j^T$ 始终是对称半正定矩阵，满足此条件。请勿将此函数用于非对称矩阵。

### 6.4 在标记点处求值

解 $M \mathbf{a} = \mathbf{b}_u$ 得到系数向量 $\mathbf{a} = [a_0, a_1, a_2]^T$。插值速度为：

$$U_k^x = \mathbf{p}(\mathbf{0}) \cdot \mathbf{a} = a_0 \quad (\text{因为 } p(\mathbf{0}) = [1, 0, 0]^T)$$

```cpp
mk.ux = au[0];   // 因为 p(0) = [1, 0, 0]^T，所以插值值 = 常数项系数 a_0
mk.uy = av[0];
```

---

## 7. MLS 力展布（MLS-J^T 算子）

MLS 展布是 MLS 插值算子的**伴随（转置）算子**，保证离散恒等式：

$$\sum_k F_k \cdot (J \mathbf{u})_k = \sum_j u_j \cdot (J^T \mathbf{F})_j$$

### 7.1 MLS 形状函数

展布使用 MLS **形状函数** $\phi_j^k$：

$$\phi_j^k = w(\mathbf{x}_j - \mathbf{X}_k) \cdot \mathbf{c}^T \cdot p(\mathbf{x}_j - \mathbf{X}_k)$$

其中 $\mathbf{c}$ 是方程 $M \mathbf{c} = \mathbf{e}_0$（$\mathbf{e}_0 = [1,0,0]^T$）的解，即 MLS 矩阵第一行的系数。

**力展布公式（2025 JCP Eq.16 + Eq.18）**：

$$f_j = \sum_{k=1}^{N_l} c_k \cdot \phi_j^k \cdot F_k^b, \quad c_k = \Delta S_k \quad (\text{Eq.18，uniform lattice 下 } \Delta V_j = \Delta x^2 = 1)$$

### 7.2 代码对应

```cpp
// interpolation.cpp:401–497（mls_spread_force）
// 第一遍：构建 MLS 矩阵 M，求解 M·c = e_0
double e0[3] = {1.0, 0.0, 0.0};
double c[3];
if (!solve3x3(M, e0, c)) continue;   // c = M^{-1} e_0

// 第二遍：φ_j^k = w_j · (c^T · p_j)，展布力
const double phi = w * (c[0]*p[0] + c[1]*p[1] + c[2]*p[2]);
grid.force[node*2]   += phi * mk.fx * mk.ds;   // f_j += φ_j^k · F_k^b · ΔS_k
grid.force[node*2+1] += phi * mk.fy * mk.ds;
```

**为什么要两遍循环**：第一遍需要先知道所有支撑节点的 M 矩阵才能求解 c，然后第二遍才能计算每个节点的 φ_j^k 并展布力。

---

## 8. 原始 MLS-IBM（Algorithm 1）

**参考**：2025 JCP §3.1；Vanella & Balaras (2009)。

### 8.1 算法（单步直接力法）

1. **MLS 插值**：$U_k^* = J \mathbf{u}^*$（调用 `mls_interpolate_velocity`）
2. **直接力**：$F_k^b = \dfrac{\mathbf{u}_{\text{target}} - U_k^*}{\Delta t}$
3. **MLS 展布**（含守恒因子 $c_k = \Delta S_k$）：$f_j = J^T \mathbf{F}^b$（调用 `mls_spread_force`）

**无滑移误差分析**（2025 JCP Fig.3a）：原始 MLS 的插值算子 $J$ 与展布算子 $J^T$ 并非精确互为伴随（因为展布后的力场再经 LBM 速度更新后再插值不能精确还原目标速度），因此无滑移残差约 O(1e-3)。

### 8.2 代码对应

```cpp
// interpolation.cpp:446–459（compute_ibm_forces_mls_original）
void compute_ibm_forces_mls_original(fluid, ms, dx, dt, kernel,
                                      u_target_x, u_target_y)
{
    mls_interpolate_velocity(fluid, ms, dx);    // Step 1: U_k* = J·u*

    for (auto& mk : ms.markers) {
        mk.fx = (u_target_x - mk.ux) / dt;     // Step 2: F_k^b
        mk.fy = (u_target_y - mk.uy) / dt;
    }

    mls_spread_force(fluid, ms, dx);            // Step 3: f = J^T·F^b（含 ds）
}
```

---

## 9. 显式 MLS-IBM（Algorithm 2）

**参考**：2025 JCP §3.2；Chen et al. (2022) *Phys. Rev. E* 106:015307。

### 9.1 算法（全局 Z 修正）

1. $U_k^* = J \mathbf{u}^*$
2. $F_k^b = (\mathbf{u}_{\text{target}} - U_k^*) / \Delta t$
3. $\mathbf{f} = J^T \mathbf{F}^b$（第一次展布）
4. $g_k = J \mathbf{f}$（重插值：将展布后的力场插值回 Lagrangian 点）
5. **全局修正因子**（2025 JCP Eq.21，最小二乘解）：

$$Z = \frac{\sum_k F_k \cdot g_k}{\sum_k |g_k|^2}$$

6. $\mathbf{f} \leftarrow Z \cdot \mathbf{f}$（修正展布力场）

**局限性**（2025 JCP Table 1）：Z 因子是单个全局标量，无法为不同位置的标记点独立调整，破坏力和力矩守恒。无滑移误差仍约 O(1e-2)（Fig.3b）。

### 9.2 代码对应

```cpp
// interpolation.cpp:1087–1134（compute_ibm_forces_mls_explicit）

// Step 4：重插值技巧 — 将力场临时替换为"伪速度场"
std::swap(fluid.u, fluid.force);   // fluid.u = force field
mls_interpolate_velocity(fluid, ms, dx);   // g_k = J·f（写入 mk.ux/uy）
std::swap(fluid.u, fluid.force);   // 恢复 fluid.u

// Step 5：Z = Σ(F_k·g_k) / Σ|g_k|²
double num = 0.0, den = 0.0;
for (int m = 0; m < ms.size(); ++m) {
    // mk.fx/fy = F_k^b（步骤2结果，尚未被覆盖）
    // mk.ux/uy = g_k  （步骤4的重插值结果）
    num += mk.fx * mk.ux + mk.fy * mk.uy;   // Σ F_k · g_k
    den += mk.ux * mk.ux + mk.uy * mk.uy;   // Σ |g_k|²
}
const double Z = (den > 1e-30) ? num / den : 1.0;

// Step 6：f → Z·f
for (auto& f : fluid.force) f *= Z;
```

**swap 技巧**：利用 `std::swap(fluid.u, fluid.force)` 将力场临时赋给 u 字段，再调用插值函数读取，避免新增临时数组。

---

## 10. 隐式 MLS-IBM（Algorithm 3）核心辅助函数

### 10.1 build_mls_shape_functions

**作用**：预计算每个 Lagrangian 标记点的 MLS 形状函数支撑集，供后续插值、矩阵构建、力展布共用，消除三处重复计算。

```cpp
// interpolation.cpp:502–583
struct MlsSupportSet {
    std::vector<int>    idx;   // Euler 节点全局索引（排列顺序与循环一致）
    std::vector<double> phi;   // 对应 MLS 形状函数值 φ_j^k
};

static void build_mls_shape_functions(grid, ms, dx, phi_data[Nl])
```

对每个 Lagrangian 点 $k$：
1. 第一遍（构建 M）：累积 $M = \sum w_j p_j p_j^T$
2. 求解 $M \mathbf{c} = \mathbf{e}_0$
3. 第二遍（计算并存储 $\phi_j^k$）：$\phi_j^k = w_j (\mathbf{c}^T p_j)$

**结果**：`phi_data[k].idx[s]` 和 `phi_data[k].phi[s]` 分别是第 $k$ 个标记点第 $s$ 个支撑 Euler 节点的索引和形状函数值。

### 10.2 build_correlation_matrix（对应 Eq.28）

**论文 Eq.28**：相关矩阵 $A \in \mathbb{R}^{N_l \times N_l}$，

$$A_{ki} = \sum_{j \in \mathcal{S}(k) \cap \mathcal{S}(i)} \phi_j^k \cdot c_i \cdot \phi_j^i, \quad c_i = \Delta S_i$$

$A$ 的矩阵元 $A_{ki}$ 反映了标记点 $k$ 和 $i$ 通过共享 Euler 支撑节点的耦合强度。

**高效构建策略**：使用**反向索引** `euler_to_lag[j]` 枚举所有共享同一 Euler 节点 $j$ 的 $(k, i)$ 对：

```cpp
// interpolation.cpp:813–841
static void build_correlation_matrix(phi_data, ms, grid_size, A_mat)
{
    // 1. 构建反向索引：euler_to_lag[j] = {(k, φ_j^k), …}
    std::vector<std::vector<std::pair<int,double>>> euler_to_lag(grid_size);
    for (int k = 0; k < Nl; ++k)
        for (auto& [j, phi_jk] : zip(phi_data[k].idx, phi_data[k].phi))
            euler_to_lag[j].emplace_back(k, phi_jk);

    // 2. 对每个 (k, Euler节点j, Lag点i共享j)，累积 A_{ki}
    for (int k = 0; k < Nl; ++k) {
        for (auto& [euler_j, phi_jk] : phi_data[k]) {
            for (auto& [i, phi_ji] : euler_to_lag[euler_j]) {
                // A_{ki} += φ_j^k · ΔS_i · φ_j^i  ←  Eq.28
                A_mat[k*Nl + i] += phi_jk * ms.markers[i].ds * phi_ji;
            }
        }
    }
}
```

**复杂度分析**：
- 暴力枚举：$O(N_l^2 \cdot N_e)$（对每对 $(k,i)$ 求交集）
- 反向索引方法：$O(N_l \cdot N_e \cdot \bar{N}_i)$，其中 $N_e \approx 9$（矩形支撑域），$\bar{N}_i \approx N_l \cdot N_e / N_{\text{Euler}}$（每个 Euler 节点被多少 Lagrangian 点共享）。对稀疏情况（$N_l \ll N_{\text{Euler}}$），此方法远优于暴力枚举。

### 10.3 gmres_dense_jacobi（GMRES 求解器）

**求解**：$A \mathbf{x} = \mathbf{b}$（$N_l \times N_l$ 稠密系统）

**算法**：无重启 Arnoldi + Givens 旋转 + 对角 Jacobi 预处理

#### 预处理

对角 Jacobi 预处理子 $M_D = \text{diag}(A)$，求解预处理系统 $M_D^{-1} A \mathbf{x} = M_D^{-1} \mathbf{b}$：

```cpp
// D_inv[i] = 1/A[i][i]（对角线倒数）
std::vector<double> D_inv(N, 1.0);
for (int i = 0; i < N; ++i)
    if (std::abs(A[i*N+i]) > 1e-30) D_inv[i] = 1.0 / A[i*N+i];

// 初始残差 r = D^{-1}·(b - A·x)
for (int i = 0; i < N; ++i) {
    double ax = 0; for (int j = 0; j < N; ++j) ax += A[i*N+j]*x[j];
    r[i] = D_inv[i] * (b[i] - ax);
}
```

#### Arnoldi 过程

对第 $j$ 步 Krylov 迭代：

$$w = M_D^{-1} A v_j \quad (\text{矩阵向量乘积，代码行 690–695})$$

$$h_{ij} = \langle w, v_i \rangle,\quad w \leftarrow w - h_{ij} v_i \quad (i = 0,\ldots,j) \quad (\text{改进 Gram-Schmidt，行 698–703})$$

$$h_{j+1,j} = \|w\|,\quad v_{j+1} = w / h_{j+1,j} \quad (\text{行 704–709})$$

#### Givens 旋转消元

对新 Hessenberg 列应用已有旋转，再计算新旋转 $(c_j, s_j)$：

$$c_j = \frac{h_{jj}}{\sqrt{h_{jj}^2 + h_{j+1,j}^2}},\quad s_j = \frac{h_{j+1,j}}{\sqrt{h_{jj}^2 + h_{j+1,j}^2}}$$

```cpp
const double denom = std::hypot(H[j][j], H[j+1][j]);
cs[j] = H[j][j]   / denom;
sn[j] = H[j+1][j] / denom;
H[j][j]   = cs[j]*H[j][j] + sn[j]*H[j+1][j];
H[j+1][j] = 0.0;
```

#### 残差监控与收敛判据

$$|g_{j+1}| < \text{tol} \cdot \beta_0 \quad (\text{相对收敛判据，tol} = 10^{-14})$$

```cpp
if (std::abs(g[j+1]) < tol * beta) {   // 相对残差已收敛
    j_done = j + 1; break;
}
```

#### 回代求解

```cpp
// 求解上三角系统 H·y = g（后代）
for (int i = js-1; i >= 0; --i) {
    y[i] = g[i];
    for (int k = i+1; k < js; ++k) y[i] -= H[i][k] * y[k];
    y[i] /= H[i][i];
}
// 更新解：x += Σ_k y[k] · v_k
for (int k = 0; k < js; ++k)
    for (int i = 0; i < N; ++i)
        x[i] += V[k][i] * y[k];
```

### 10.4 lu_factor_dense / lu_solve_dense（LU 分解）

对固定物体（Scheme I）缓存 $A^{-1}$ 的实现：行主元 LU 分解。

**LU 分解**（原地，行主元选取）：

```cpp
// interpolation.cpp:586–638（LU 分解与代换）
for (int k = 0; k < N; ++k) {
    // 寻找最大主元
    int p = k; double max_val = |A[k][k]|;
    for (int i = k+1; i < N; ++i)
        if (|A[i][k]| > max_val) { max_val = |A[i][k]|; p = i; }
    piv[k] = p;
    if (p != k) swap row k and row p;   // 行交换

    // Gaussian 消元
    const double inv_akk = 1.0 / A[k][k];
    for (int i = k+1; i < N; ++i) {
        A[i][k] *= inv_akk;                         // L 因子（下三角）
        for (int j = k+1; j < N; ++j)
            A[i][j] -= A[i][k] * A[k][j];           // U 因子（上三角）
    }
}
```

**LU 代换**（lu_solve_dense）：

```cpp
// 1. 行置换 P·b
for (int k = 0; k < N; ++k)
    if (piv[k] != k) swap(x[k], x[piv[k]]);

// 2. 前代 L·y = Pb（对角元为1，隐含）
for (int i = 1; i < N; ++i)
    for (int j = 0; j < i; ++j)
        x[i] -= A_lu[i*N+j] * x[j];

// 3. 后代 U·x = y
for (int i = N-1; i >= 0; --i) {
    for (int j = i+1; j < N; ++j)
        x[i] -= A_lu[i*N+j] * x[j];
    x[i] /= A_lu[i*N+i];
}
```

---

## 11. 隐式 MLS-IBM Scheme II：GMRES

**参考**：2025 JCP §4，Algorithm 3，Scheme II

### 11.1 完整算法对应

算法步骤与代码行号：

| 算法步骤 | 公式 | 代码位置 |
|---------|------|---------|
| A1：构建传递算子 Φ | $\phi_j^k = w_j \mathbf{c}^T p_j$（Eq.10–14） | `build_mls_shape_functions()`（行 878） |
| A2：重建 Lagrangian 速度 | $\mathbf{U}^* = J \mathbf{u}^*$（Eq.15） | `interpolate_with_phi()`（行 881） |
| 构建 B | $B_k = (u_{\text{target}} - U_k^*) / \Delta t$（Eq.24c） | 行 883–896 |
| C2：构建 A | $A_{ki} = \sum_j \phi_j^k c_i \phi_j^i$（Eq.28） | `build_correlation_matrix()`（行 900） |
| C2：GMRES 求解 | $A \mathbf{F}^x = \mathbf{B}^x,\; A \mathbf{F}^y = \mathbf{B}^y$ | `gmres_dense_jacobi()`（行 907–908） |
| A4：力展布 | $f_j = \sum_k c_k \phi_j^k F_k^b$（Eq.16） | `spread_with_phi()`（行 917） |
| A5：速度更新 | $\mathbf{u}^{n+1} = \mathbf{u}^* + \Delta t \mathbf{f}$（由调用方执行） | solver.step() |

### 11.2 完整代码

```cpp
// interpolation.cpp:861–919
void compute_ibm_forces_mls_implicit(fluid, ms, dx, dt,
                                      gmres_max_iter,  // 原 n_iter 参数
                                      u_target_x, u_target_y)
{
    const int Nl = ms.size();

    // A1：构建 MLS 形状函数（一次构建，后续 A2/A4 共用）
    std::vector<MlsSupportSet> phi_data;
    build_mls_shape_functions(fluid, ms, dx, phi_data);

    // A2：U* = J·u*（MLS 插值）
    interpolate_with_phi(fluid, ms, phi_data);    // 写入 mk.ux/uy

    // 构建右端向量 B（Eq.24c）
    std::vector<double> Bx(Nl, 0.0), By(Nl, 0.0);
    for (int k = 0; k < Nl; ++k) {
        Bx[k] = (u_target_x - mk.ux) / dt;
        By[k] = (u_target_y - mk.uy) / dt;
    }

    // C2：构建相关矩阵 A（Eq.28）
    std::vector<double> A_mat;
    build_correlation_matrix(phi_data, ms, fluid.nx*fluid.ny, A_mat);

    // C2：GMRES 求解 A·Fx = Bx，A·Fy = By（Scheme II）
    std::vector<double> Fx(Nl, 0.0), Fy(Nl, 0.0);   // 初始猜测 x₀ = 0
    gmres_dense_jacobi(A_mat, Bx, Fx, Nl, 1e-14, gmres_max_iter);
    gmres_dense_jacobi(A_mat, By, Fy, Nl, 1e-14, gmres_max_iter);

    // 写入 Lagrangian 力（供 FSI 反作用力计算）
    for (int k = 0; k < Nl; ++k) {
        ms.markers[k].fx = Fx[k];
        ms.markers[k].fy = Fy[k];
    }

    // A4：f = Σ_k c_k φ_j^k F_k^b（Eq.16，spread_with_phi 包含 dk=mk.ds）
    spread_with_phi(fluid, ms, phi_data);
}
```

**参数说明**：
- `gmres_max_iter`（原 `n_iter` 位置，后向兼容）：GMRES 最大迭代次数。理论上 $N_l$ 步内必然收敛（Krylov 维数上界）。默认值 3 保留后向兼容；机器精度需设 $\geq N_l$。
- GMRES 相对收敛判据 `tol = 1e-14`：对 $N_l = 32$ 标记点，实测无滑移残差达 `max_res = 1.08e-16`（≈ double ε = 2.2e-16）。

### 11.3 无滑移误差分析

设向量 $\mathbf{U}^* = J \mathbf{u}^*$（Lagrangian 速度），$\mathbf{u}^{n+1} = \mathbf{u}^* + \Delta t (J^T C \mathbf{F}^b)$（速度更新），则：

$$J \mathbf{u}^{n+1} = J \mathbf{u}^* + \Delta t J J^T C \mathbf{F}^b = \mathbf{U}^* + \Delta t A \mathbf{F}^b$$

无滑移条件要求 $J \mathbf{u}^{n+1} = \mathbf{u}_{\text{target}}$，即：

$$A \mathbf{F}^b = \frac{\mathbf{u}_{\text{target}} - \mathbf{U}^*}{\Delta t} = \mathbf{B} \quad \Rightarrow \quad \mathbf{F}^b = A^{-1} \mathbf{B}$$

其中 $A = J J^T C$（$J$ 为 MLS 插值算子，$C = \text{diag}(\Delta S_k)$）即为 Eq.28 中的相关矩阵。GMRES 精确求解此系统，因此无滑移误差达机器精度。

---

## 12. 隐式 MLS-IBM Scheme I：固定物体直接矩阵求逆

**参考**：2025 JCP §4，Algorithm 3，Scheme I

对于几何固定的物体（$\mathbf{X}_k$ 不随时间变化），传递算子 $\Phi$ 和相关矩阵 $A$ 不变。Scheme I 在第一次调用时完成 LU 分解并缓存，后续步骤仅执行 $O(N_l^2)$ 的 LU 代换。

### 12.1 完整代码与逻辑

```cpp
// interpolation.cpp:939–1001
void compute_ibm_forces_mls_implicit_stationary(
    fluid, ms, dx, dt,
    A_lu_cache,   // LU 因子缓存（调用方持久化；空=尚未初始化）
    piv_cache,    // 主元缓存
    u_target_x, u_target_y)
{
    // A1（每步均需执行，因为 phi_data 用于 A2 插值和 A4 展布）
    std::vector<MlsSupportSet> phi_data;
    build_mls_shape_functions(fluid, ms, dx, phi_data);

    // C1：仅在首次调用时构建 A 并完成 LU 分解（Scheme I 的核心优化）
    if (A_lu_cache.empty()) {
        build_correlation_matrix(phi_data, ms, fluid.nx*fluid.ny, A_lu_cache);
        if (!lu_factor_dense(A_lu_cache, piv_cache, Nl)) {
            // 奇异矩阵（极少情况）：回退到 GMRES
            A_lu_cache.clear(); piv_cache.clear();
            compute_ibm_forces_mls_implicit(fluid, ms, dx, dt, Nl, ...);
            return;
        }
        // A_lu_cache 现在存储 LU 分解结果（L\U 原地覆盖 A）
    }

    // A2：U* = J·u*（每步必须，因为流体速度每步变化）
    interpolate_with_phi(fluid, ms, phi_data);

    // 构建 B（每步必须）
    std::vector<double> Bx(Nl), By(Nl);
    for (int k = 0; k < Nl; ++k) {
        Bx[k] = (u_target_x - mk.ux) / dt;
        By[k] = (u_target_y - mk.uy) / dt;
    }

    // C2 Scheme I：X = A^{-1}·B（LU 代换，O(N_l²)）
    lu_solve_dense(A_lu_cache, piv_cache, Bx.data(), Nl);   // Bx → Fx
    lu_solve_dense(A_lu_cache, piv_cache, By.data(), Nl);   // By → Fy

    // 写入 Lagrangian 力
    for (int k = 0; k < Nl; ++k) {
        ms.markers[k].fx = Bx[k];
        ms.markers[k].fy = By[k];
    }

    // A4：展布力到 Eulerian 网格
    spread_with_phi(fluid, ms, phi_data);
}
```

### 12.2 使用示例

```cpp
std::vector<double> A_lu_cache;   // 持久化 LU 缓存（时间循环外）
std::vector<int>    piv_cache;

for (int step = 0; step < n_steps; ++step) {
    solver.step();
    // 第 1 步：build_mls_shape_functions + build_correlation_matrix + LU
    // 第 2..N 步：build_mls_shape_functions + LU 代换（复用缓存）
    compute_ibm_forces_mls_implicit_stationary(
        fluid, ms, dx, dt, A_lu_cache, piv_cache);
}
```

**性能对比**（2025 JCP Table 2）：

| 步骤 | Scheme II（GMRES，每步） | Scheme I（LU 缓存，第 2..N 步） |
|------|------------------------|-------------------------------|
| build_mls_shape_functions | $O(N_l \cdot N_e^2)$ | $O(N_l \cdot N_e^2)$ |
| build_correlation_matrix | $O(N_l \cdot N_e \cdot \bar{N}_i)$ | 仅第 1 步 |
| LU 分解 / GMRES | $O(N_l^2 \cdot \text{iter})$ | 仅第 1 步 |
| LU 代换 | — | $O(N_l^2)$ |

> ⚠️ **已知性能局限（Scheme I）**：当前实现每步仍调用 `build_mls_shape_functions`，因为 `phi_data` 需用于 A2 插值和 A4 展布。对静止物体，`phi_data` 不随时间变化，可进一步缓存以消除此重复开销——这是一个已知的优化机会，需将 `MlsSupportSet` 暴露到公共头文件中。

---

## 13. 罚函数 IBM（Penalty-IBM）

**参考**：Goldstein D. et al. (1993) *J. Comput. Phys.* 105:354–366。

### 13.1 原理

$$F(t) = \alpha \cdot e(t) + \beta \cdot \int_0^t e(\tau)\,d\tau, \quad e(t) = \mathbf{u}_{\text{target}} - \mathbf{u}_{\text{IB}}(t)$$

其中 $\alpha$（比例增益）和 $\beta$（积分增益）为大正数（可调），无滑移精度取决于增益大小。

### 13.2 代码对应

```cpp
// interpolation.cpp:1125–1177（compute_ibm_forces_penalty）

// 1. 插值：u_IB = J·u（Peskin δ 函数）
interpolate_velocity(fluid, ms, dx, kernel);

for (int m = 0; m < nm; ++m) {
    const double ex = u_target_x - mk.ux;   // e = u_target - u_IB
    const double ey = u_target_y - mk.uy;

    // 3. 积分更新（简单 Euler 积分 + 抗饱和限幅）
    integral_x[m] += dt * ex;
    integral_y[m] += dt * ey;
    // 限幅：|integral| ≤ 10/β（抗积分饱和）
    if (integral_x[m] > max_integral) integral_x[m] = max_integral;
    // ...

    // 4. F = α·e + β·integral
    mk.fx = alpha * ex + beta * integral_x[m];
    mk.fy = alpha * ey + beta * integral_y[m];
}

// 5. Peskin 力展布
spread_force(fluid, ms, dx, kernel);
```

**参数推荐**：
- $\alpha = O(1/\Delta t^2)$：比例增益，对 $\Delta t = 1$（格子单位）推荐 $\alpha \in [2, 10]$
- $\beta = 0$ 或 $\beta \approx \alpha/100$：积分增益，设 0 等同于纯比例控制

---

## 14. IBM 合力统计

**物理意义**：IBM 体力合力等于固体所受流体作用力（牛顿第三定律）。

$$F_x = \sum_k F_k^{bx} \cdot \Delta S_k, \quad F_y = \sum_k F_k^{by} \cdot \Delta S_k$$

```cpp
// interpolation.cpp:1195–1211
void compute_ibm_body_force(const MarkerSet& ms, double& out_fx, double& out_fy)
{
#ifdef LBM_ENABLE_OPENMP
#pragma omp parallel for reduction(+:fx,fy)
#endif
    for (int m = 0; m < ms.size(); ++m) {
        fx += mk.fx * mk.ds;   // Σ F_k^b · ΔS_k
        fy += mk.fy * mk.ds;
    }
    out_fx = fx; out_fy = fy;
}
```

**符号约定**：此函数返回 IBM 力作用到**流体**上的方向（正值 = 流体加速方向）。固体受到的合力为其**负值**（牛顿第三定律）。

---

## 15. MPI 分区适配

`ibm_marker_set_adapt_to_partition()` 将标记点坐标从全局格子坐标系转换到本地坐标系：

$$\text{offset}_x = \text{phys\_x0} - \text{x\_start}, \quad \text{offset}_y = \text{phys\_y0} - \text{y\_start}$$

$$mk.x \mathrel{+}= \text{offset}_x, \quad mk.y \mathrel{+}= \text{offset}_x$$

并设置 `owner_i_lo/hi = [phys_x0, phys_x0 + local_nx)`，用于 MPI 多进程下的标记点归属过滤。

---

## 16. MPI 幽灵层交换

### 16.1 ibm_halo_exchange_u_2d（IBM 插值前）

LBM PUSH 流式迁移后，幽灵行的 $u$ 是从本地物理行**外推**的（非邻居真实速度）。在 IBM 插值前必须调用此函数更新幽灵行 $u$：

```cpp
void ibm_halo_exchange_u_2d(grid, decomp) {
    lbm::halo_exchange_u_2d(grid, decomp);   // 代理到底层通用实现
}
```

### 16.2 ibm_halo_reduce_force_2d（IBM 展布后）

IBM 力展布可能向幽灵行写入贡献（属于邻居进程物理行的一部分），需通过 `MPI_Sendrecv` 归并回邻居物理行：

- **S/N 方向**：连续内存，直接 `MPI_Sendrecv` 整行
- **W/E 方向**：列不连续，需打包（pack）到连续 buffer 后传输

支持 `n_ghost` 层幽灵交换（由 `MpiDecomp2D::n_ghost` 控制）：TwoPoint 核需 1 层，FourPoint 核需 2 层（需创建时指定 `n_ghost=2`）。

---

## 17. 方案对比与选用建议

| 方案 | 函数 | 无滑移误差 | 力守恒 | 力矩守恒 | 每步开销 | 适用场景 |
|------|------|-----------|-------|---------|---------|---------|
| Peskin δ + 直接力 | `interpolate_velocity` + 手动力 | O(1e-2) | ✗ | ✗ | O(N_l·N_e) | 测试/原型 |
| MDF-IBM | `compute_ibm_forces_mdf` | O(1e-3)×iter | ✓ | ✓ | O(iter·N_l·N_e) | 稳健，适中精度 |
| 原始 MLS（Alg.1） | `compute_ibm_forces_mls_original` | O(1e-3) | ✗ | ✗ | O(N_l·N_e²) | 高阶插值，低精度展布 |
| 显式 MLS（Alg.2） | `compute_ibm_forces_mls_explicit` | O(1e-2) | ✗ | ✗ | O(N_l·N_e²) | 不推荐（Z 因子破坏守恒） |
| **隐式 MLS Scheme II** | **`compute_ibm_forces_mls_implicit`** | **机器精度** | **✓** | **✓** | O(N_l²·iter + N_l·N_e²) | **移动物体，推荐** |
| **隐式 MLS Scheme I** | **`compute_ibm_forces_mls_implicit_stationary`** | **机器精度** | **✓** | **✓** | O(N_l²) 复用LU | **固定物体，最快** |
| 罚函数 | `compute_ibm_forces_penalty` | O(1/α) | ✗ | ✗ | O(N_l·N_e) | 大刚度系数近似无滑移 |

---

## 18. 代码审查发现的问题与注意事项

### 18.1 ✅ solve3x3 仅对对称矩阵正确（设计选择，非 bug）

**现象**：`solve3x3` 的 `x[1]` 和 `x[2]` 公式利用了对称矩阵的性质 $A[i][j] = A[j][i]$（等价于 $\det(M) = \det(M^T)$），因此**仅对对称矩阵正确**。

**为何无害**：MLS 矩阵 $M = \sum w_i p_i p_i^T$ 始终是对称半正定矩阵（乘积和形式保证对称性），所有调用点均满足此条件。

**修复**：代码已添加注释说明此约束（`interpolation.cpp:247–261`），防止误用于非对称矩阵。

### 18.2 ⚠️ Scheme I 每步仍重建 phi_data（已知性能局限）

**现象**：`compute_ibm_forces_mls_implicit_stationary` 在每步调用 `build_mls_shape_functions`，即便对静止物体 `phi_data` 不随时间变化。

**原因**：`phi_data` 需用于每步的 A2（速度插值）和 A4（力展布），当前 API 未缓存它；缓存需将 `MlsSupportSet` 暴露到公共头文件（API 变更）。

**影响**：Scheme I 的 `build_mls_shape_functions` 开销 $O(N_l \cdot N_e^2)$ 与 `build_correlation_matrix` $O(N_l \cdot N_e^2)$ 同量级，实际加速比约 2× 而非理论上的无限加速比（第一步之后）。

**建议（未来优化）**：创建 `IbmStationaryCache` 结构体，同时缓存 `phi_data` + LU 分解。

### 18.3 ⚠️ 隐式 MLS 在 MPI 多进程模式下不完整

**现象**：`build_correlation_matrix` 仅使用本进程 `phi_data`（仅包含本进程拥有的标记点贡献）。不同进程上的标记点可能共享 Euler 支撑节点，但跨进程的 $A_{ki}$ 贡献未通过 `MPI_Allreduce` 汇聚。

**影响**：在 MPI 多进程模式下，每进程的 $A$ 矩阵是全局矩阵的子块，GMRES 求解结果仅对本进程拥有的标记点正确。

**当前状态**：单进程（非 MPI）模式下完全正确，已通过所有测试（测试均在 `ENABLE_MPI=OFF` 下运行）。MPI 多进程下的隐式 MLS 需额外实现 `MPI_Allreduce` 归约 $A$ 矩阵。

### 18.4 ✅ Peskin 与 MLS 使用不同的最近格点策略（设计合理）

| 方法 | 最近格点 | 原因 |
|------|---------|------|
| Peskin（`interpolate_velocity`） | `std::floor(x/dx)` | Peskin 核基于 $\lfloor x/\Delta x \rfloor$ 定义，支撑 $[i_0 - s, i_0 + s + 1]$（非对称） |
| MLS（`mls_interpolate_velocity` 等） | `std::round(x/dx)` | MLS 支撑域 $[-H_k, H_k]$ 以标记点为中心，对称分布 |

### 18.5 ✅ MLS 插值回退策略的不一致性（已知，可接受）

- `mls_interpolate_velocity`（奇异）：回退到加权平均 `bu[0]/M[0][0]`
- `mls_spread_force`（奇异）：直接跳过此标记点（`continue`）

两种回退策略不同但都是降级处理，实际情况下（正确参数 + 足够支撑节点）不会触发。

### 18.6 ✅ GMRES 收敛判据（设计合理）

初始绝对判据 `if (beta < tol)` 防止零右端向量情况。后续迭代内使用相对判据 `|g[j+1]| < tol * beta`。对 `tol = 1e-14`，实测结果见 `test_mls_implicit_machine_precision`：32 次迭代后 `max_res = 1.08e-16` ≈ double ε。

---

## 附录 A：符号速查表

| 论文符号 | 代码变量 | 含义 |
|---------|---------|------|
| $N_l$ | `ms.size()` / `Nl` | Lagrangian 标记点数 |
| $N_e$ | `phi_data[k].idx.size()` | 每标记点支撑 Euler 节点数（≈9） |
| $\mathbf{X}_k$ | `(mk.x, mk.y)` | 第 $k$ 个标记点坐标 |
| $\mathbf{x}_j$ | `(ii*dx, jj*dx)` | 第 $j$ 个 Euler 节点坐标 |
| $\Delta S_k$ | `mk.ds` | 弧长/面积元素 |
| $\phi_j^k$ | `phi_data[k].phi[s]` | MLS 形状函数 |
| $w_j^k$ | `w = exp(-r²/h²)` | Gaussian 权函数 |
| $H_k$ | `H_k = 1.5*dx` | 支撑域半宽 |
| $\varepsilon$ | `eps = 0.3` | Gaussian 集中参数 |
| $M$ | `M[3][3]` | MLS 矩阵 $\sum w_j p_j p_j^T$ |
| $A$ | `A_mat` | 相关矩阵 $N_l \times N_l$ |
| $J$ | `interpolate_with_phi()` | MLS 插值算子 |
| $J^T C$ | `spread_with_phi()` | MLS 展布算子（含 $C=\text{diag}(\Delta S_k)$） |
| $\mathbf{F}^b$ | `(mk.fx, mk.fy)` | Lagrangian IBM 力密度 |
| $\mathbf{f}$ | `fluid.force` | Eulerian 体力场 |
| $\mathbf{u}^*$ | `fluid.u` | LBM 碰撞-流式后速度 |
| $\mathbf{U}^*$ | `(mk.ux, mk.uy)` | MLS 插值到 Lagrangian 的速度 |

---

## 附录 B：调用顺序（单进程）

```
每个时间步 n：
  1. solver.step()               ← LBM 碰撞 + 流式 + 宏观量计算
  2. [可选] ibm_halo_exchange_u_2d()   ← MPI 幽灵层 u 场同步
  3. compute_ibm_forces_mls_implicit()  ← Algorithm 3
     ├── build_mls_shape_functions      (A1)
     ├── interpolate_with_phi           (A2：U* = J·u*)
     ├── 构建 B = (u_target - U*)/dt    (Eq.24c)
     ├── build_correlation_matrix       (C2：A = JJ^TC)
     ├── gmres_dense_jacobi             (C2：A·F = B)
     └── spread_with_phi                (A4：f = J^T C F)
  4. [可选] ibm_halo_reduce_force_2d() ← MPI 幽灵层力归并
  5. 下一步 solver.step() 的碰撞步使用 fluid.force（Guo 体力格式）
```

---

*文档生成日期：2026-03-23；对应代码版本：commit `70453e3`（interpolation.cpp 行 1–1397）*
