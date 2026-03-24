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
   - 5.1 问题背景：单次直接力法的不足
   - 5.2 完整算法推导（对应 2011 Suzuki & Inamuro §3.2）
   - 5.3 代码逐行对应
   - 5.4 完整时间步流程（MDF-IBM + LBM 全耦合）
   - 5.5 δ 函数实现细节
   - 5.6 `interpolate_velocity`：Lagrangian 点插值
   - 5.7 `spread_force`：力展布
   - 5.8 FSI 合力计算
   - 5.9 C API 与 Rust 绑定封装
   - 5.10 数值参数建议与收敛性
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
    - 18.1 solve3x3 对称矩阵约束
    - 18.2 Scheme I phi_data 缓存
    - 18.3 隐式 MLS 在 MPI 多进程模式下已完整
    - 18.4 Peskin 与 MLS 最近格点策略差异
    - 18.5 MLS 插值回退策略的不一致性
    - 18.6 GMRES 收敛判据
    - 18.7 ✅ MDF Lagrangian 力累加修正（已修复）

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

**参考论文**：
- Wang Z., Fan J., Luo K. (2008) *"Combined multi-direct forcing and immersed boundary method for simulating flows with moving particles"*，Int. J. Multiphase Flow **34**:283–302。（**原始 MDF 方法**）
- Suzuki K., Inamuro T. (2011) *"Effect of internal mass in the simulation of a moving body by the immersed boundary method"*，Comput. Fluids **49**:173–187。（**MDF 应用于 LBM 的版本**，本项目的直接参考）

**本节符号约定**（与代码变量对照）

| 论文符号 | 含义 | 代码变量 |
|---------|------|---------|
| $\mathbf{u}^* = \mathbf{u}^n$ | 时间步 $n$ 结束后的欧拉速度场（stream后） | `fluid.u`（传入时的值）|
| $\mathbf{u}^{(\ell)}$ | 第 $\ell$ 次子迭代的工作速度场 | `u_work` |
| $\mathbf{U}_k$ | 第 $k$ 个 Lagrangian 点的目标（固体壁面）速度 | 静止时 = **0** |
| $\Delta \mathbf{g}^{(\ell)}_k$ | 第 $\ell$ 次迭代在 Lagrangian 点上的力增量 | `mk.fx`, `mk.fy`（每次迭代临时）|
| $\mathbf{g}^{(L)}_k$ | 最终 Lagrangian 总体力 $= \sum_\ell \Delta\mathbf{g}^{(\ell)}_k$ | `mk.fx`, `mk.fy`（函数返回后）|
| $\delta\mathbf{f}^{(\ell)}(\mathbf{x})$ | 第 $\ell$ 次迭代展布到欧拉网格的增量体力场 | `dF_euler` |
| $\mathbf{g}^{(L)}(\mathbf{x})$ | 最终欧拉总体力场 $= \sum_\ell \delta\mathbf{f}^{(\ell)}$ | `fluid.force`（返回后）|
| $W(\mathbf{x} - \mathbf{X}_k)$ | 离散 δ 函数（Peskin 4 点余弦核） | `delta_phi()` |
| $\Delta V_k = \Delta S_k$ | Lagrangian 点体积元/弧长元素 | `mk.ds` |
| $N_F$ | 子迭代次数 | `n_iter` |
| $\Delta x$ | 格子间距 | `dx` |
| $\Delta t$ | 时间步长 | `dt` |

---

### 5.1 问题背景：单次直接力法的不足

**直接力法**（Fadlun et al. 2000，Eq. (10) in 2008 paper）要求在每个时间步 $n \to n+1$ 中，对第 $k$ 个 Lagrangian 点施加力：

$$F_k(\mathbf{x}_k) = \frac{\mathbf{U}_k - \hat{\mathbf{u}}_k}{\Delta t}  \tag{Wang2008 Eq.10}$$

其中 $\hat{\mathbf{u}}_k$ 是该 Lagrangian 点处由欧拉场插值得到的"预测速度"（即 $\hat{u}_k = \sum_j \hat{u}_j W_{jk} \Delta x^2$），$\mathbf{U}_k$ 是固体壁面目标速度。

> **问题**：展布力后，相邻 Lagrangian 点会相互干扰，导致实际插值速度 $\hat{\mathbf{u}}^1_k \neq \mathbf{U}_k$，无滑移条件 $O(\Delta x)$ 量级上不满足（2008 paper §2.2）。

**多重直接力法（MDF）** 通过 $N_F$ 次子迭代（每次相当于额外施加一次直接力修正）逐步消除残余误差，使 Lagrangian 点速度收敛到目标速度：

$$\text{误差} \sim O(\Delta x^2) \text{ 对 NF=1}，\text{~以} -2 \text{ 斜率下降（log-log图中，2008 Fig.1)}$$

---

### 5.2 完整算法推导（对应 2011 Suzuki & Inamuro §3.2）

**背景**（2011 paper Eq. 16–17）：LBM 每时间步分两步更新分布函数：

$$f_i^*(\mathbf{x}, t+\Delta t) = f_i(\mathbf{x}, t) - \frac{1}{s}\left[f_i - f_i^{\rm eq}\right]  \quad \text{（无体力碰撞）} \tag{2011 Eq.16}$$

$$f_i(\mathbf{x}, t+\Delta t) = f_i^*(\mathbf{x}+\mathbf{c}_i\Delta x, t+\Delta t) + 3\Delta x\, E_i\, \mathbf{c}_i \cdot \mathbf{g}(\mathbf{x}, t+\Delta t)  \tag{2011 Eq.17}$$

其中 $\mathbf{g}(\mathbf{x}, t+\Delta t)$ 是欧拉体力，$E_i$ 是权重。这意味着欧拉速度场通过体力被修正为：

$$\mathbf{u}^*(\mathbf{x}) = \mathbf{u}^n(\mathbf{x}) + \Delta t \cdot \mathbf{g}(\mathbf{x}) \tag{等价于}$$

**MDFM 迭代过程**（2011 paper §3.2，Step 0–4）：

**Step 0**（初始化，2011 Eq.21）：计算第一次 Lagrangian 力增量，

$$\Delta\mathbf{g}^{(0)}_k = \frac{\mathbf{U}_k - \mathbf{u}^*(\mathbf{X}_k)}{\Delta t}  \tag{2011 Eq.21}$$

其中 $\mathbf{u}^*(\mathbf{X}_k) = \sum_j \mathbf{u}^*(\mathbf{x}_j) W(\mathbf{x}_j - \mathbf{X}_k) \Delta x^2$ 是从 $\mathbf{u}^*$ 插值得到的。

**迭代 $\ell = 0, 1, \ldots, N_F - 1$**：

**Step 1**（展布，2011 Eq.22）：将 Lagrangian 增量力展布到欧拉网格，

$$\delta\mathbf{f}^{(\ell)}(\mathbf{x}) = \sum_{k=1}^{N} \Delta\mathbf{g}^{(\ell)}_k \cdot W(\mathbf{x} - \mathbf{X}_k) \cdot \Delta V_k  \tag{2011 Eq.22}$$

其中 $\Delta V_k = S/N \cdot \Delta x \approx \Delta S_k$（弧长元素），即代码中的 `mk.ds`。

**Step 2**（修正欧拉速度，2011 Eq.23）：

$$\mathbf{u}^{(\ell+1)}(\mathbf{x}) = \mathbf{u}^{(\ell)}(\mathbf{x}) + \frac{\Delta t}{\text{Sh}} \cdot \delta\mathbf{f}^{(\ell)}(\mathbf{x})  \tag{2011 Eq.23}$$

在格子单位中 $\text{Sh}/\Delta t = 1/\Delta x = 1$（$\text{Sh} = \Delta t$），故化简为：

$$\mathbf{u}^{(\ell+1)}(\mathbf{x}) = \mathbf{u}^{(\ell)}(\mathbf{x}) + \Delta t \cdot \delta\mathbf{f}^{(\ell)}(\mathbf{x})  \tag{格子单位}$$

**Step 3**（插值，2011 Eq.24）：将修正后的欧拉速度插值回 Lagrangian 点，

$$\mathbf{u}^{(\ell)}(\mathbf{X}_k) = \sum_{\mathbf{x}} \mathbf{u}^{(\ell+1)}(\mathbf{x}) \cdot W(\mathbf{x} - \mathbf{X}_k) \cdot \Delta x^2  \tag{2011 Eq.24}$$

**Step 4**（更新 Lagrangian 力，2011 Eq.25）：

$$\Delta\mathbf{g}^{(\ell+1)}_k = \Delta\mathbf{g}^{(\ell)}_k + \frac{\mathbf{U}_k - \mathbf{u}^{(\ell)}(\mathbf{X}_k)}{\Delta t}  \tag{2011 Eq.25（等价形式）}$$

> **注意**：2011 paper 中 Step 4 表述为新的体力 = 旧体力 + 残余修正。等价地，也可写成每次增量直接计算：$\Delta\mathbf{g}^{(\ell)}_k = (\mathbf{U}_k - \mathbf{u}^{(\ell-1)}(\mathbf{X}_k)) / \Delta t$，即每次迭代用当前工作速度重新计算增量。两者对最终总力 $\mathbf{g}^{(L)}_k = \sum_\ell \Delta\mathbf{g}^{(\ell)}_k$ 完全等价（累加相同）。

**最终结果**：

- **欧拉体力**：$\mathbf{g}^{(L)}(\mathbf{x}) = \sum_{\ell=0}^{N_F-1} \delta\mathbf{f}^{(\ell)}(\mathbf{x})$，写入 `fluid.force`，供 Guo 体力格式（§5.4）使用。
- **Lagrangian 总体力**：$\mathbf{g}^{(L)}_k = \sum_{\ell=0}^{N_F-1} \Delta\mathbf{g}^{(\ell)}_k$，写入 `mk.fx`/`mk.fy`，供 FSI 合力计算（§14）使用。

收敛性（2008 paper §3.1 & Fig.1）：$\ell_2$ 范数误差以 $-2$ 斜率在 log-log 图上随 $N_F$ 减小，**$N_F = 5$ 通常足够精确**（2011 paper Appendix D）。

---

### 5.3 代码逐行对应（`core/src/ibm/interpolation.cpp`）

```cpp
// ====================================================================
// core/src/ibm/interpolation.cpp: compute_ibm_forces_mdf()
// 实现参考：Wang 2008（原始 MDF）；Suzuki & Inamuro 2011（LBM 版本）
// ====================================================================
void compute_ibm_forces_mdf(lbm::LatticeGrid& fluid,
                             MarkerSet& ms,
                             double dx,   // 格子间距（通常 = 1.0）
                             double dt,   // 时间步长（通常 = 1.0）
                             int    n_iter,
                             DeltaKernel kernel)
{
    const int n  = fluid.size();   // 欧拉节点总数
    const int d  = fluid.dim();    // = 2（D2Q9）
    const int nm = ms.size();      // Lagrangian 标记点总数 N

    // ── 工作速度场 u_work ──────────────────────────────────────────
    // u_work 初始化为 u*（Step 0 前的欧拉速度，即 LBM stream 后）
    // 对应：u^(0) = u*  （2011 §3.2 初始化）
    std::vector<double> u_work = fluid.u;

    // ── 欧拉总体力累加器 F_total ───────────────────────────────────
    // g^(L)(x) = Σ_ℓ δf^(ℓ)(x)，最终写入 fluid.force
    std::vector<double> F_total(n * d, 0.0);

    // ── Lagrangian 总体力累加器 ────────────────────────────────────
    // g^(L)_k = Σ_ℓ Δg^(ℓ)_k，最终写入 mk.fx/fy
    // 必须单独累加，不能只保留末次迭代值（末次迭代增量→0）
    std::vector<double> total_lag_fx(nm, 0.0);
    std::vector<double> total_lag_fy(nm, 0.0);

    // ── 当前迭代增量力场 dF_euler ──────────────────────────────────
    // δf^(ℓ)(x) = spread(Δg^(ℓ))
    std::vector<double> dF_euler(n * d, 0.0);

    // ================================================================
    // 主迭代循环：ℓ = 0, 1, …, n_iter-1
    // ================================================================
    for (int iter = 0; iter < n_iter; ++iter) {

        // ── Step 0/3（插值）：u^(ℓ)(x) → U^(ℓ)(X_k) ─────────────
        // 对应：2011 Eq.21（ℓ=0）或 Eq.24（ℓ>0）
        //   U^(ℓ)(X_k) = Σ_x u^(ℓ)(x) · W(x - X_k) · Δx²
        //
        // 技巧：将 u_work 临时换入 fluid.u，再调用通用插值函数，
        //       以复用 interpolate_velocity 中的 MPI 归属过滤逻辑。
        std::swap(fluid.u, u_work);
        interpolate_velocity(fluid, ms, dx, kernel);   // 写入 mk.ux, mk.uy
        std::swap(fluid.u, u_work);

        // ── Step 0/4（计算增量力）：Δg^(ℓ)_k ───────────────────────
        // 对应：2011 Eq.21（ℓ=0），Eq.25 等价形式（ℓ>0）
        //   Δg^(ℓ)_k = (U_k - U^(ℓ)(X_k)) / Δt
        //   静止固体：U_k = 0，故 Δg^(ℓ)_k = -U^(ℓ)(X_k) / Δt
        for (int m = 0; m < nm; ++m) {
            auto& mk = ms.markers[m];
            const double dFx = (0.0 - mk.ux) / dt;   // ρ=1 格子单位
            const double dFy = (0.0 - mk.uy) / dt;
            mk.fx = dFx;          // 暂存增量，供 spread_force() 使用
            mk.fy = dFy;
            total_lag_fx[m] += dFx;   // 累加到 Lagrangian 总力
            total_lag_fy[m] += dFy;
        }

        // ── Step 1（展布）：spread(Δg^(ℓ)) → δf^(ℓ)(x) ─────────────
        // 对应：2011 Eq.22
        //   δf^(ℓ)(x) = Σ_k Δg^(ℓ)_k · W(x - X_k) · ΔS_k
        //
        // spread_force() 读取 mk.fx/fy（增量力）和 mk.ds（弧长元素 ΔS_k），
        // 将结果写入 fluid.force，然后 swap 到 dF_euler 保存。
        std::fill(dF_euler.begin(), dF_euler.end(), 0.0);
        spread_force(fluid, ms, dx, kernel);   // 写入 fluid.force
        std::swap(fluid.force, dF_euler);      // dF_euler ← δf^(ℓ)(x)

        // ── Step 2（修正欧拉速度）：u^(ℓ+1) = u^(ℓ) + Δt·δf^(ℓ) ────
        // 对应：2011 Eq.23（格子单位 Sh/Δt=1）
        for (int i = 0; i < n; ++i) {
            u_work[i * d + 0] += dt * dF_euler[i * d + 0];
            u_work[i * d + 1] += dt * dF_euler[i * d + 1];
        }

        // ── 累积欧拉总力：g^(L)(x) = Σ_ℓ δf^(ℓ)(x) ─────────────────
        for (int i = 0; i < n * d; ++i) {
            F_total[i] += dF_euler[i];
        }
    }   // end of iter loop

    // ================================================================
    // 写出最终结果
    // ================================================================

    // 欧拉总体力 → fluid.force（供下一步 collide 中 Guo 格式使用）
    fluid.force = F_total;

    // Lagrangian 总体力 → mk.fx/fy（供 FSI 合力统计 §14 使用）
    // 重要：这里写的是各迭代增量之和，而非末次迭代增量。
    // 末次迭代增量趋近于零（已收敛），仅用其会严重低估合力。
    for (int m = 0; m < nm; ++m) {
        ms.markers[m].fx = total_lag_fx[m];
        ms.markers[m].fy = total_lag_fy[m];
    }
}
```

---

### 5.4 完整时间步流程（MDF-IBM + LBM 全耦合）

以下是一个完整时间步的执行顺序，结合代码调用关系说明各部分如何衔接。

```
时间步 n → n+1（main.rs: run_loop）
─────────────────────────────────────────────────────────────────────
① solver.step(grid)
   ├── collide()                     ← 使用 fluid.force（来自上步 IBM）
   │     BGK: f_a* = f_a - ω(f_a - f_eq)
   │     Guo: f_a* += w_a(1-ω/2) [(c_a-u)/cs² + (c_a·u)c_a/cs⁴]·F  [见§5.4.1]
   ├── stream()                      ← 传播 f*，计算 u^{n+1} = Σ f_a c_a / ρ
   └── apply_BC() + compute_macroscopic()

   → fluid.u 更新为 u^* = u^{n+1}（stream 后、IBM 力修正前）

② grid.zero_force()                  ← 清零 fluid.force

③ ms.step_mdf(grid, dx, dt, n_iter)  ← 本节实现
   = lbm_ibm_compute_mdf(g, ms, dx, dt, n_iter)
   → 读取 fluid.u（= u^*），执行 N_F 次子迭代
   → 写入 fluid.force = g^(L)(x)（总欧拉体力）
   → 写入 mk.fx/fy = g^(L)_k（总 Lagrangian 力）

④ （MPI）ibm_halo_reduce_force_2d()  ← 归并幽灵行力贡献

   → 此时 fluid.force 包含下一步 collide 所需的 IBM 体力

⑤ 时间步循环返回 ①
```

**关键耦合点**：IBM 力 `fluid.force` 在 `step()` 的 `collide()` 中**同步消费**（Guo 格式），而不是在 stream 后修正 `fluid.u`。这与 2011 paper 的 Eq.17 完全吻合：LBM 的体力直接修正分布函数 $f_i$，宏观速度通过 $\mathbf{u} = \sum f_i \mathbf{c}_i / \rho + \Delta t \mathbf{g} / (2\rho)$ 隐含修正。

#### 5.4.1 Guo 体力格式（`solver.cpp: apply_guo_forcing`）

2011 paper Eq.17 的 LBM 体力修正项 $3\Delta x E_i \mathbf{c}_i \cdot \mathbf{g}$ 对应标准 Guo et al. (2002) 格式（格子单位 $\Delta x = \Delta t = 1$，$c_s^2 = 1/3$）：

$$F_\alpha = w_\alpha \left(1 - \frac{\omega}{2}\right) \left[\frac{\mathbf{c}_\alpha - \mathbf{u}}{c_s^2} + \frac{(\mathbf{c}_\alpha \cdot \mathbf{u})\mathbf{c}_\alpha}{c_s^4}\right] \cdot \mathbf{F}  \tag{Guo 2002}$$

代码（`solver.cpp:385–403`）：

```cpp
// apply_guo_forcing(node, w_a, c_a, F, &f_a)
constexpr double cs2 = 1.0/3.0, cs4 = 1.0/9.0;
double cu = Σ c_a[α] * u[α];                                // c_a · u
double term = Σ ((c_a[α]-u[α])/cs2 + cu*c_a[α]/cs4) * F[α]; // 矢量内积
*f_a += w_a * (1.0 - 0.5*ω) * term;                         // Guo 修正
```

---

### 5.5 δ 函数实现细节（`delta_phi`，对应论文中的 $W$）

**2008 paper Eq.(14)–(15)** 采用 Griffith & Peskin (2005) 的四点核：

$$\phi_h(r) = \begin{cases}
\dfrac{1}{8}\!\left(3 - 2|r| + \sqrt{1 + 4|r| - 4r^2}\right) & 0 \leq |r| < 1 \\[6pt]
\dfrac{1}{8}\!\left(5 - 2|r| - \sqrt{-7 + 12|r| - 4r^2}\right) & 1 \leq |r| < 2 \\[4pt]
0 & |r| \geq 2
\end{cases}$$

本项目使用的是数学上等价但形式不同的 **Peskin 余弦核**（`DeltaKernel::FourPoint`），也是四点支撑宽度的最常用写法：

$$\phi_h(r) = \frac{1}{4h}\left(1 + \cos\!\frac{\pi r}{2h}\right), \quad |r| \leq 2h  \tag{代码 interpolation.cpp:31–36}$$

两者均为 $C^1$ 连续，**满足矩条件**（归一化、保守性）：$\sum_j \phi_h(x_j - X_k) \Delta x = 1$。

**二维 δ 函数（乘积形式）**（2008 paper Eq.14，2011 paper Eq.19）：

$$W(\mathbf{x} - \mathbf{X}_k) = \phi_h(x - X_k^x) \cdot \phi_h(y - X_k^y)  \tag{张量积}$$

代码（`interpolation.cpp:88–91` 插值，`141–146` 展布）：

```cpp
// 插值中的权重（×Δx²，对应论文中 ΔV = Δx² 的积分元素）
const double phi_x = delta_phi(mk.x - ii*dx, dx, kernel);
const double phi_y = delta_phi(mk.y - jj*dx, dx, kernel);
const double phi   = phi_x * phi_y * dx * dx;   // W·Δx²

// 展布中的权重（×ΔS_k，对应论文中 ΔV_k = mk.ds）
const double phi = phi_x * phi_y * mk.ds;        // W·ΔS_k
```

> **注意插值与展布的权重不同**：
> - 插值（J 算子）：权重 = $W \cdot \Delta x^2$（欧拉积分元素）
> - 展布（J^T 算子）：权重 = $W \cdot \Delta S_k$（Lagrangian 弧长元素）
>
> 这是 IBM 中 J 与 J^T 的**伴随关系**（adjoint property），保证了动量守恒：
> $\sum_j f_j^{\rm IBM} \Delta x^2 = \sum_k F_k \Delta S_k$（2008 paper §2.1）。

**支撑范围循环**（`interpolation.cpp:77–96`）：

```cpp
const int support = (kernel == DeltaKernel::FourPoint) ? 2 : 1;
// i0, j0 = floor(mk.x/dx), floor(mk.y/dx)（左下角格点）
for (int dj = -support; dj <= support + 1; ++dj)    // 4 点：dj ∈ {-2,-1,0,1,2,3}?
    for (int di = -support; di <= support + 1; ++di)  //    注：实为 [-2,+2) 共4格
```

> **循环范围说明**：`dj ∈ {-support, …, support+1}` 即 `{-2,-1,0,1}`（FourPoint 时），覆盖以 `floor(xm)` 为左边界的 4 格支撑区间 $[i_0-1, i_0+2]$，对应 Peskin 核的 $[-2h, +2h]$ 支撑。

---

### 5.6 `interpolate_velocity`：Lagrangian 点插值（J 算子）

对应 2011 paper Eq.(24)，2008 paper Eq.(17)（`interpolation.cpp:43–102`）：

$$\mathbf{U}^{(\ell)}_k = \mathbf{u}^{(\ell)}(\mathbf{X}_k) = \sum_{\mathbf{x}_j \in \mathcal{S}(k)} \mathbf{u}^{(\ell)}(\mathbf{x}_j) \cdot W(\mathbf{x}_j - \mathbf{X}_k) \cdot \Delta x^2  \tag{J 算子}$$

```cpp
// 伪代码摘要：
for (int m = 0; m < ms.size(); ++m) {
    MPI归属过滤（owner_i/j_lo/hi）;           // 仅本进程负责的标记点
    for di, dj in support_range:              // 4×4 = 16 个邻近格点
        phi = delta_phi(Δx)*delta_phi(Δy)*dx²;
        ux_sum += fluid.u[node*2+0] * phi;   // Σ u_j · W_jk · Δx²
        uy_sum += fluid.u[node*2+1] * phi;
    mk.ux = ux_sum;  mk.uy = uy_sum;        // 写入 U^(ℓ)_k
}
```

**注意**：固体节点（`grid.solid[node] == true`）被跳过，避免固体内部的无效速度参与插值。

---

### 5.7 `spread_force`：力展布（J^T 算子）

对应 2011 paper Eq.(22)，2008 paper Eq.(19)（`interpolation.cpp:107–159`）：

$$\delta\mathbf{f}^{(\ell)}(\mathbf{x}_j) = \sum_{k=1}^{N} \Delta\mathbf{g}^{(\ell)}_k \cdot W(\mathbf{x}_j - \mathbf{X}_k) \cdot \Delta S_k  \tag{J^T 算子}$$

```cpp
// 伪代码摘要（散射操作）：
std::fill(grid.force, 0.0);                 // 清零
for (int m = 0; m < ms.size(); ++m) {
    MPI归属过滤;
    for di, dj in support_range:
        phi = delta_phi(Δx)*delta_phi(Δy)*mk.ds;  // W_jk · ΔS_k
        grid.force[node*2+0] += mk.fx * phi;      // Σ_k Δg_k · W · ΔS
        grid.force[node*2+1] += mk.fy * phi;
}
```

**OpenMP 原子操作**：展布是散射操作（多个标记点写同一欧拉节点），并行时需加 `#pragma omp atomic`。

---

### 5.8 FSI 合力计算（`compute_ibm_body_force`，§14）

根据 2011 paper Eq.(5)（符号相反）：

$$\mathbf{F}_{\rm solid} = -\int_{\Omega} \mathbf{g}(\mathbf{x}) \, d\mathbf{x} \approx -\sum_{k} \mathbf{g}^{(L)}_k \cdot \Delta S_k$$

代码（`interpolation.cpp:1227–1244`）：

```cpp
// compute_ibm_body_force(ms, out_fx, out_fy)
for (int m = 0; m < ms.size(); ++m)
    fx += mk.fx * mk.ds;    // Σ g^(L)_k · ΔS_k（施加到流体上的合力）
    fy += mk.fy * mk.ds;
// 固体所受合力 = (-fx, -fy)（牛顿第三定律，调用方取反）
```

> **为何必须累加各迭代 Lagrangian 力**（而非仅取末次迭代值）：  
> 末次迭代增量 $\Delta\mathbf{g}^{(N_F-1)}_k \to 0$（已收敛），若只用末次值则合力≈0。  
> 应使用累积总力 $\mathbf{g}^{(L)}_k = \sum_\ell \Delta\mathbf{g}^{(\ell)}_k$，这在 `compute_ibm_forces_mdf` 中已通过 `total_lag_fx/fy` 正确实现。

---

### 5.9 C API 与 Rust 绑定封装

**C API**（`lbm_capi.cpp:1248–1268`）：

```cpp
void lbm_ibm_compute_mdf(LatticeGrid* g, IbmMarkerSetHandle* ms,
                          double dx, double dt, int n_iter)
{
    // 多体支持：先保存前序 IBM 体的力，计算后累加
    std::vector<double> pre_force = g->force;
    ibm::compute_ibm_forces_mdf(*g, *marker_set, dx, dt, n_iter);
    for (size_t i = 0; i < g->force.size(); ++i)
        g->force[i] += pre_force[i];    // 多体合并
}
```

**Rust 封装**（`bindings/src/lib.rs:1277–1280`）：

```rust
pub fn step_mdf(&mut self, grid: &mut LbmGrid, dx: f64, dt: f64, n_iter: i32) {
    unsafe { ffi::lbm_ibm_compute_mdf(grid.ptr, self.ptr, dx, dt, n_iter) }
}
```

**典型调用**（`main.rs: step_ibm`）：

```rust
fn step_ibm(cfg, grid, ibm_entries) {
    grid.zero_force();                                     // 清零上步残留
    for entry in ibm_entries {
        entry.ms.step_mdf(grid, 1.0, cfg.dt, entry.n_iter); // N_F 次子迭代
    }
}
// 调用时机：solver.step(grid) 之后，下次 solver.step 的 collide() 之前
```

---

### 5.10 数值参数建议与收敛性

| 参数 | 推荐值 | 说明 |
|------|-------|------|
| `n_iter`（$N_F$）| 3–5 | 2008 Fig.1 显示 $N_F=5$ 时 $\ell_2$ 误差降至 $\sim10^{-6}$；$N_F=20$ 达 $\sim10^{-6}$ |
| `kernel` | `FourPoint` | 4 点 Peskin 核，$C^1$ 连续，满足矩条件 |
| `dt` | $\leq$ Ma·$c_s$ 稳定范围 | 格子单位下通常 = 1.0 |
| `dx` | 1.0 | 格子单位 |
| Lagrangian 点间距 | $\approx \Delta x$（圆周上） | $\Delta S_k = 2\pi R / N \approx \Delta x$，防止"漏洞" |

**时间步内力的生效顺序**：IBM 力在时间步 $n$ 末尾写入 `fluid.force`，在时间步 $n+1$ 的 `collide()` 中通过 Guo 格式生效——即 **IBM 力超前一步**施加，这是显式耦合方案（explicit coupling）的固有特性。

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
| A1：构建传递算子 Φ | $\phi_j^k = w_j \mathbf{c}^T p_j$（Eq.10–14） | `build_mls_shape_functions()`（行 881） |
| A2：重建 Lagrangian 速度 | $\mathbf{U}^* = J \mathbf{u}^*$（Eq.15） | `interpolate_with_phi()`（行 884） |
| 构建 B | $B_k = (u_{\text{target}} - U_k^*) / \Delta t$（Eq.24c） | 行 886–899 |
| C2：构建 A | $A_{ki} = \sum_j \phi_j^k c_i \phi_j^i$（Eq.28） | `build_correlation_matrix()`（行 903） |
| C2：GMRES 求解 | $A \mathbf{F}^x = \mathbf{B}^x,\; A \mathbf{F}^y = \mathbf{B}^y$ | `gmres_dense_jacobi()`（行 917–918） |
| A4：力展布 | $f_j = \sum_k c_k \phi_j^k F_k^b$（Eq.16） | `spread_with_phi()`（行 927） |
| A5：速度更新 | $\mathbf{u}^{n+1} = \mathbf{u}^* + \Delta t \mathbf{f}$（由调用方执行） | solver.step() |

### 11.2 完整代码

```cpp
// interpolation.cpp:864–929
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
// interpolation.cpp:949–1025
void compute_ibm_forces_mls_implicit_stationary(
    fluid, ms, dx, dt,
    A_lu_cache,   // LU 因子缓存（调用方持久化；空=尚未初始化）
    piv_cache,    // 主元缓存
    phi_cache,    // MLS 形状函数缓存（调用方持久化；空=尚未初始化）
    u_target_x, u_target_y)
{
    // A1：只在首次调用时构建 phi_cache（静止物体几何不变）
    if (phi_cache.empty()) {
        build_mls_shape_functions(fluid, ms, dx, phi_cache);
    }

    // C1：仅在首次调用时构建 A 并完成 LU 分解（Scheme I 的核心优化）
    if (A_lu_cache.empty()) {
        build_correlation_matrix(phi_cache, ms, fluid.nx*fluid.ny, A_lu_cache);
        if (!lu_factor_dense(A_lu_cache, piv_cache, Nl)) {
            // 奇异矩阵（极少情况）：回退到 GMRES
            A_lu_cache.clear(); piv_cache.clear(); phi_cache.clear();
            compute_ibm_forces_mls_implicit(fluid, ms, dx, dt, Nl, ...);
            return;
        }
        // A_lu_cache 现在存储 LU 分解结果（L\U 原地覆盖 A）
    }

    // A2：U* = J·u*（每步必须，因为流体速度每步变化；使用缓存的 phi_cache）
    interpolate_with_phi(fluid, ms, phi_cache);

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

    // A4：展布力到 Eulerian 网格（使用缓存的 phi_cache）
    spread_with_phi(fluid, ms, phi_cache);
}
```

### 12.2 使用示例

```cpp
std::vector<double>        A_lu_cache;   // 持久化 LU 缓存（时间循环外）
std::vector<int>           piv_cache;
std::vector<MlsSupportSet> phi_cache;    // 持久化 MLS 形状函数缓存

for (int step = 0; step < n_steps; ++step) {
    solver.step();
    // 第 1 步：build_mls_shape_functions + build_correlation_matrix + LU
    // 第 2..N 步：全部复用缓存（仅执行 interpolate + LU代换 + spread）
    compute_ibm_forces_mls_implicit_stationary(
        fluid, ms, dx, dt, A_lu_cache, piv_cache, phi_cache);
}
```

**性能对比**（2025 JCP Table 2）：

| 步骤 | Scheme II（GMRES，每步） | Scheme I（LU 缓存，第 2..N 步） |
|------|------------------------|-------------------------------|
| build_mls_shape_functions | $O(N_l \cdot N_e^2)$ | **仅第 1 步**（phi_cache 已缓存） |
| build_correlation_matrix | $O(N_l \cdot N_e \cdot \bar{N}_i)$ | 仅第 1 步 |
| LU 分解 / GMRES | $O(N_l^2 \cdot \text{iter})$ | 仅第 1 步 |
| LU 代换 | — | $O(N_l^2)$ |

> **Scheme I 性能说明**：当前实现同时缓存 `phi_cache`（MLS 形状函数）和 `A_lu_cache`（LU 分解），因此对静止物体从第 2 步起每步开销仅为 $O(N_l^2)$ 的 LU 代换，无需重建 phi_data 或 A 矩阵。

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

$$mk.x \mathrel{+}= \text{offset}_x, \quad mk.y \mathrel{+}= \text{offset}_y$$

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

### 18.2 ✅ Scheme I phi_data 缓存已实现

**现象**：`compute_ibm_forces_mls_implicit_stationary` 新增 `phi_cache` 参数（`std::vector<MlsSupportSet>&`），在首次调用时构建并缓存，后续步骤直接复用，避免重复 `build_mls_shape_functions`。

**实现**：`phi_cache`、`A_lu_cache`、`piv_cache` 三个缓存均由调用方在时间循环外持久化，传入函数。首次调用时依次填充，后续步骤利用缓存零开销跳过构建阶段（仅执行 A2 插值 + LU 代换 + A4 展布）。

**状态**：✅ 已完成（见 `interpolation.cpp:949–1025`，`interpolation.hpp:310–330`）。

### 18.3 ✅ 隐式 MLS 在 MPI 多进程模式下已完整

**已修复项目**：

1. **Scheme I（固定物体）**：`build_correlation_matrix` 后对 `A_mat` 执行 `MPI_Allreduce(SUM)`，汇聚所有进程的局部贡献得到全局相关矩阵；构建 B 向量后同样 `MPI_Allreduce`（见 `interpolation.cpp:974–1011`）。

2. **Scheme II（GMRES）**：同样对 `A_mat` 和 `B` 向量执行 `MPI_Allreduce`，确保每进程求解相同的全局系统（见 `interpolation.cpp:903–918`）。

3. **显式 MLS Z 因子**：`compute_ibm_forces_mls_explicit` 中 Z 因子分子/分母 `num`/`den` 执行 `MPI_Allreduce`，保证 Z 值在所有进程上一致（见 `interpolation.cpp:1122–1132`）。

**当前状态**：✅ 三处均已修复，单进程和 MPI 多进程模式下均正确。

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

### 18.7 ✅ MDF：Lagrangian 标记点力累加修正（已修复）

**问题**（已修复）：原实现中，每次子迭代直接用 `mk.fx = dFx` 覆盖标记点力，导致 `mk.fx/fy` 在 $N_F > 1$ 时仅保存末次迭代增量 $\Delta\mathbf{g}^{(N_F-1)}_k \to 0$（收敛后趋近于零）。`compute_ibm_body_force()` 随即返回近零 FSI 合力，严重低估阻力/升力。

**根因**：$\mathbf{g}^{(L)}_k = \sum_\ell \Delta\mathbf{g}^{(\ell)}_k$（各迭代增量之和）才是正确的 Lagrangian 总体力（2008 paper Eq.27 及 2011 paper §3.2），而非末次增量。

**修复方案**（`interpolation.cpp: compute_ibm_forces_mdf`）：
- 新增 `total_lag_fx[m]`/`total_lag_fy[m]` 累加器，每次迭代同步累加。
- `mk.fx/fy` 在迭代内临时保存增量（供 `spread_force()` 使用），迭代结束后将累加总力写回 `mk.fx/fy`。

**欧拉力场 `fluid.force` 不受影响**（始终正确，为各增量展布之和）。

**验证**：新增测试 `test_mdf_marker_force_accumulates`：3 次迭代后 `rms(mk.fx)` 与 1 次迭代量级相当（均非零），而修复前 3 次迭代后值趋近于 0。

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

*文档生成日期：2026-03-24；对应代码版本：commit `3a23166`（interpolation.cpp 行 1–1440）*
