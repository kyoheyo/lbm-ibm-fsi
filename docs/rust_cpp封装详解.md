# Rust 如何封装 C++ 的 LBM 核心 — 深入浅出详解

> **适合读者**：了解基本 Rust 语法，但对 FFI（外部函数接口）或 Rust-C++ 互操作尚不熟悉的开发者。  
> **阅读目标**：读完本文后，能够独立理解并修改 `bindings/` 与 `core/src/capi/` 两层的任意代码。

---

## 目录

1. [为什么要用 Rust 封装 C++ 核心？](#1-为什么要用-rust-封装-c-核心)
2. [五层架构总览](#2-五层架构总览)
3. [第一层：C++ 计算核心](#3-第一层c-计算核心)
4. [第二层：C ABI 桥接层](#4-第二层c-abi-桥接层)
5. [第三层：Rust FFI 声明（`mod ffi`）](#5-第三层rust-ffi-声明mod-ffi)
6. [第四层：安全的 Rust 包装类型](#6-第四层安全的-rust-包装类型)
7. [第五层：主控驱动（orchestrator）](#7-第五层主控驱动orchestrator)
8. [构建系统：build.rs + CMake 协作](#8-构建系统buildrs--cmake-协作)
9. [插件系统：C 函数指针 ↔ C++ 虚函数](#9-插件系统c-函数指针--c-虚函数)
10. [内存安全深度分析](#10-内存安全深度分析)
11. [常见构建问题与解决方案](#11-常见构建问题与解决方案)
12. [完整调用链一览](#12-完整调用链一览)
13. [Rust 语言特性速查——读懂 `bindings/src/lib.rs`](#13-rust-语言特性速查读懂-bindingssrclibrs)
14. [C++ 语言特性速查——读懂 `lbm_capi.cpp` 与 `solver.cpp`](#14-c-语言特性速查读懂-lbm_capicpp-与-solvercpp)

---

## 1 为什么要用 Rust 封装 C++ 核心？

### 1.1 各语言的优势与分工

| 语言 | 适合做什么 | 本项目中的角色 |
|------|-----------|---------------|
| **C++17** | 极致性能、SIMD/OpenMP/MPI 并行、丰富的科学计算生态 | LBM 碰撞/迁移、IBM 耦合、FSI 结构求解（数值内核） |
| **Rust** | 内存安全、无 GC、所有权系统、优秀的工具链 | 安全驱动层、配置管理、文件输出、主控流程 |
| **Python** | 快速原型、科学可视化、丰富的数据处理库 | 预处理（网格、配置生成）、后处理（绘图、分析） |

核心矛盾是：**C++ 代码快，但不安全**（悬垂指针、双重释放、缓冲区溢出随时可能发生）；**Rust 代码安全，但不能直接调用 C++ 代码**（C++ 名称修饰（name mangling）使符号名不稳定）。

解决方案就是本文的主题：**C ABI（Application Binary Interface）桥**。

### 1.2 ABI 兼容性问题

C++ 编译器会对函数名进行"名称修饰"，例如：

```cpp
// C++ 源码
namespace lbm { class Solver { void step(); }; }

// 编译后的符号名（GCC）
_ZN3lbm6Solver4stepEv   // ← Rust 根本不知道这个名字
```

C 语言没有名称空间和重载，编译后符号就是原始函数名：

```c
// C 源码
void lbm_solver_step(Solver* s, LatticeGrid* g);

// 编译后的符号名
lbm_solver_step          // ← Rust 能直接声明并调用
```

因此，**中间必须有一层 `extern "C"` 的 C ABI 桥**，将 C++ 接口"翻译"成稳定的 C 接口，才能被 Rust 调用。

---

## 2 五层架构总览

```
┌──────────────────────────────────────────────────────────────────────┐
│  第五层：orchestrator/src/main.rs（Rust 主控驱动）                    │
│  · 解析 TOML 配置                                                     │
│  · 调用第四层的安全类型驱动仿真循环                                    │
│  · 写出 NPZ 快照 / CSV 监控日志                                        │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ 调用 LbmGrid / LbmSolver / register_plugins
┌────────────────────────▼─────────────────────────────────────────────┐
│  第四层：bindings/src/lib.rs（安全 Rust 包装）                        │
│  · LbmGrid        → 持有不透明 C++ 指针，实现 Drop / Send             │
│  · LbmSolver      → 同上                                              │
│  · PluginCallbacks → 函数指针容器 + register_plugins()                │
│  · 所有 pub 接口均为 safe（无 unsafe 关键字）                          │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ unsafe extern "C" 调用
┌────────────────────────▼─────────────────────────────────────────────┐
│  第三层：bindings/src/lib.rs 内的 mod ffi（Rust FFI 声明）            │
│  · 声明 extern "C" 块                                                  │
│  · 用枚举模拟不透明 C++ 类的句柄（LatticeGridHandle / SolverHandle）   │
│  · 所有函数声明均使用 C 兼容类型（c_int, f64, *mut, *const …）        │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ 静态链接 liblbm_core.a
┌────────────────────────▼─────────────────────────────────────────────┐
│  第二层：core/src/capi/lbm_capi.cpp（C ABI 桥接层）                   │
│  · extern "C" { … } 内定义所有 lbm_* 函数                             │
│  · 将 C 参数（int, double, void*）转换为 C++ 对象并调用               │
│  · 负责 new / delete — C++ 堆对象的生命周期在此层管理                  │
└────────────────────────┬─────────────────────────────────────────────┘
                         │ 直接调用 C++ 类方法
┌────────────────────────▼─────────────────────────────────────────────┐
│  第一层：core/include + core/src（C++ 计算核心）                       │
│  · lbm::LatticeGrid — 存储分布函数 f[节点 * Q + 方向]                  │
│  · lbm::Solver       — BGK/MRT 碰撞 + 周期性流式迁移                   │
│  · lbm::PluginRegistry — 单例，管理四种扩展插件                        │
│  · ibm::MarkerSet / fsi::Structure — IBM + FSI 物理模型                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 3 第一层：C++ 计算核心

### 3.1 LatticeGrid — 格子数据结构

文件：`core/include/lbm/lattice.hpp` / `core/src/lbm/lattice.cpp`

```cpp
struct LatticeGrid {
    int nx, ny, nz;   // 格子尺寸
    int q;            // 离散速度数（D2Q9=9，D3Q19=19）
    LatticeModel model;

    std::vector<double> f;      // 分布函数：f[节点索引 * q + 方向]
    std::vector<double> f_tmp;  // 流式迁移临时缓冲区
    std::vector<double> rho;    // 宏观密度：rho[节点索引]
    std::vector<double> u;      // 宏观速度：u[节点索引 * dim + 分量]
    std::vector<double> force;  // 体力：force[节点索引 * dim + 分量]
};
```

**关键点**：所有数据用 `std::vector<double>` 连续存储，方便 SIMD 向量化和 OpenMP 并行。

### 3.2 Solver — BGK/MRT 碰撞求解器

文件：`core/src/lbm/solver.cpp`

每个时间步执行三步（碰撞 + 流式迁移 + 边界条件）：

```
step() = collide() + stream() + apply_boundary_conditions() + compute_macroscopic()
         ↓             ↓                  ↓                          ↓
   碰撞（原地）    流式迁移           边界条件修正幽灵方向       重算边界节点 ρ/u
```

**BGK 碰撞公式**（每个格点、每个方向）：

```
f_α* = f_α - ω × (f_α - f_α^eq)
```

- `ω`：松弛频率，由粘度 `ν` 决定：`ω = 1 / (3ν + 0.5)`
- `f_α^eq`：Maxwell-Boltzmann 平衡分布（只依赖局部密度 ρ 和速度 **u**）

**流式迁移**：将碰撞后的 `f_α*` 沿 `c_α` 方向传播到相邻格点（周期性边界）。
对于边界节点，周期性取模会从对侧引入**幽灵值**，由后续边界条件覆盖修正：

```cpp
int di = (i + C[a][0] + nx) % nx;  // 周期性 x（边界节点产生幽灵值）
int dj = (j + C[a][1] + ny) % ny;  // 周期性 y
f_tmp[dst * Q + a] = f[src * Q + a];
// stream() 末尾第一次 compute_macroscopic()——边界节点 ρ/u 含幽灵值
```

### 3.3 PluginRegistry — 单例插件注册中心

文件：`core/include/plugins/plugin_registry.hpp`

这是一个经典的**单例（Singleton）+ 策略模式**：

```cpp
class PluginRegistry {
public:
    static PluginRegistry& instance() {
        static PluginRegistry reg;  // C++11 保证线程安全的局部静态初始化
        return reg;
    }
    void set_boundary_plugin(IBoundaryPlugin* p) { boundary_ = p; }
    // …同理有 set_mesh_plugin / set_motion_plugin / set_flexible_plugin
private:
    IBoundaryPlugin*       boundary_ = nullptr;
    IMeshPlugin*           mesh_     = nullptr;
    IMotionPlugin*         motion_   = nullptr;
    IFlexibleSolverPlugin* flexible_ = nullptr;
};
```

在每个时间步内，`lbm_solver_step()` 按固定顺序调用各插件（如果已注册）：

```
① update_motion()   — 碰撞前更新固体位置
② Solver::step()    — 碰撞 + 流式迁移 + 内置 BC + 宏观量重算（详见下方）
③ apply_boundary()  — 自定义边界条件插件扩展点（流式迁移后）
④ adapt_mesh()      — 网格自适应（流式迁移后）
⑤ step_flexible()   — 柔性体推进（流式迁移后）
```

**`Solver::step()` 的内部执行顺序（v0.1.1 修复后）：**

```
Solver::step():
  ├─ collide()                         — BGK/MRT 碰撞，使用上一步末尾的 ρ/u
  ├─ stream()                          — 周期性迁移，末尾第一次 compute_macroscopic()
  │                                      （边界节点 ρ/u 含幽灵值）
  ├─ apply_boundary_conditions()       — 以物理正确的 f 值覆盖幽灵方向
  └─ compute_macroscopic()（第二次）   — 边界节点 ρ/u 得到正确值供下步碰撞使用
```

---

## 4 第二层：C ABI 桥接层

文件：`core/src/capi/lbm_capi.cpp`

### 4.1 为什么需要这一层？

直接原因：Rust 的 FFI 只能调用 C 函数（没有 C++ name mangling，没有虚函数表，没有构造/析构语义）。

**C++ `class` 不能直接跨 ABI 边界传递**，因为：
- 编译器可能在不同编译单元之间使用不同的虚函数表布局；
- C++ 对象的构造/析构语义（RAII）在 C ABI 中无对应概念；
- 名称修饰规则未标准化（GCC、Clang、MSVC 各不同）。

### 4.2 解决方案：指针 + `extern "C"`

C ABI 桥的核心技巧是：**让 C++ 在堆上分配对象，只向外暴露原始指针**。

```cpp
extern "C" {

// 1. 在堆上创建 C++ 对象，返回指针
lbm::LatticeGrid* lbm_grid_new(int nx, int ny, int nz, int model_id)
{
    if (model_id < 0 || model_id > 2) return nullptr;  // 参数校验
    auto model = static_cast<lbm::LatticeModel>(model_id);
    return new (std::nothrow) lbm::LatticeGrid(nx, ny, nz, model);
    //         ^^^^^^^^^^^^^ 分配失败返回 nullptr，而非抛异常
}

// 2. 释放堆上的 C++ 对象
void lbm_grid_free(lbm::LatticeGrid* g) { delete g; }

// 3. 读取字段（带空指针和越界检查）
double lbm_grid_rho(const lbm::LatticeGrid* g, int idx)
{
    if (!g || idx < 0 || idx >= g->size()) return 0.0;
    return g->rho[idx];  // 直接读取 C++ vector 元素
}

} // extern "C"
```

**关键点**：
- `extern "C"` 告诉 C++ 编译器：**不要做名称修饰**，按原始名导出符号。
- `new (std::nothrow)` 代替普通 `new`，防止异常穿越 C ABI 边界（C ABI 没有异常处理机制）。
- 枚举值通过 `int` 传递，在 C++ 侧显式 `static_cast`，并先做范围检查防止 UB。

### 4.3 C ABI 函数签名对照表

**CPU 求解器（`lbm_solver_*`）**

| C ABI 函数 | C++ 内部操作 | 返回 |
|-----------|-------------|------|
| `lbm_grid_new(nx, ny, nz, model_id)` | `new LatticeGrid(…)` | `LatticeGrid*`（nullptr 表示失败） |
| `lbm_grid_free(g)` | `delete g` | void |
| `lbm_grid_nx/ny/nz(g)` | `g->nx / g->ny / g->nz` | int |
| `lbm_grid_rho(g, idx)` | `g->rho[idx]` | double |
| `lbm_grid_ux(g, idx)` | `g->u[idx * dim + 0]` | double |
| `lbm_grid_uy(g, idx)` | `g->u[idx * dim + 1]` | double |
| `lbm_solver_new(g, omega, cm_id)` | `new Solver(*g, omega, cm)` | `Solver*` |
| `lbm_solver_free(s)` | `delete s` | void |
| `lbm_solver_step(s, g)` | 调用插件 + `s->step()` | void |
| `lbm_solver_step_n(s, g, step, dt)` | 带时间信息版本 | void |
| `lbm_set_plugins(…8 个参数…)` | 更新 PluginRegistry 单例 | void |

**GPU 求解器（`lbm_gpu_*`，需 `ENABLE_CUDA=ON` 编译）**

| C ABI 函数 | C++ 内部操作 | 返回 |
|-----------|-------------|------|
| `lbm_gpu_new(g, omega)` | `new GpuSolver(*g, omega)` | `GpuSolver*` |
| `lbm_gpu_free(s)` | `delete s` | void |
| `lbm_gpu_add_bc(s, type, face, ux, uy, rho)` | `s->add_boundary_condition(bc)` | void |
| `lbm_gpu_step(s)` | `s->step()` 同步单步 | void |
| `lbm_gpu_download(s, g)` | `s->download(*g)`（f+rho+u 全量 D→H） | void |
| `lbm_gpu_download_rho_u(s, g)` | `s->download_rho_u(*g)`（仅 rho+u，~6.3 MB） | void |
| `lbm_gpu_upload(s, g)` | `s->upload(*g)`（f H→D） | void |
| `lbm_gpu_step_async(s)` | `s->step_async()`（在 `compute_stream_` 上异步） | void |
| `lbm_gpu_wait_compute(s)` | `s->wait_compute()`（等待 compute 流完成）| void |
| `lbm_gpu_enqueue_async_download_rho_u(s, buf)` | `s->enqueue_async_download_rho_u(buf)`（在 io 流上排队 D→H） | void |
| `lbm_gpu_sync_async_download(s)` | `s->sync_async_download()`（等待 io 流完成）| void |
| `lbm_gpu_pinned_rho(s, buf)` | `s->h_rho(buf)`（返回固定内存 rho 缓冲区指针）| `const double*` |
| `lbm_gpu_pinned_u(s, buf)` | `s->h_u(buf)`（返回固定内存 u 缓冲区指针）| `const double*` |
| `lbm_gpu_n(s)` | `s->n_`（总节点数）| int |


---

## 5 第三层：Rust FFI 声明（`mod ffi`）

文件：`bindings/src/lib.rs`，顶部 `mod ffi { … }` 块

### 5.1 不透明句柄（Opaque Handle）模式

Rust 侧不知道 C++ `LatticeGrid` 的内部布局，因此用**空枚举**模拟不透明指针：

```rust
mod ffi {
    // 空枚举 = 大小为零的类型，永远无法被实例化
    // 只用于持有 *mut LatticeGridHandle — 保证 Rust 不会误解引用其内容
    pub enum LatticeGridHandle {}
    pub enum SolverHandle {}
}
```

为什么不用 `struct LatticeGridHandle;`（单元结构体）？  
因为单元结构体可以被实例化，而空枚举不可以，这从类型层面强制"只能通过指针持有，不能直接创建"。

### 5.2 C 枚举映射

C++ 中 `LatticeModel::D2Q9 = 0` 等枚举值在 Rust 侧必须用 `#[repr(C)]` 标注，保证内存布局与 C 侧一致：

```rust
#[repr(C)]  // ← 告诉 Rust 编译器：按 C 的方式布局，从 0 开始连续编号
pub enum LatticeModelC {
    D2Q9  = 0,
    D3Q19 = 1,
    D3Q27 = 2,
}
```

**没有 `#[repr(C)]` 会怎样？**  
Rust 编译器可能选择任意整数值或重排枚举变体，导致传给 C++ 的值与预期不符，产生难以调试的运行时 bug。

### 5.3 extern "C" 声明块

```rust
extern "C" {
    // 声明 C 函数的存在，告诉 Rust 链接器去 liblbm_core.a 里找这些符号
    pub fn lbm_grid_new(nx: c_int, ny: c_int, nz: c_int,
                        model: LatticeModelC) -> *mut LatticeGridHandle;
    pub fn lbm_grid_free(g: *mut LatticeGridHandle);
    pub fn lbm_grid_rho(g: *const LatticeGridHandle, idx: c_int) -> f64;
    // … 其余函数类似
}
```

**这段代码只是"声明"，不是"实现"**。Rust 编译器看到这段声明后，会在链接阶段去 `liblbm_core.a` 里查找对应符号。如果找不到，链接报错（`error: undefined reference to 'lbm_grid_new'`）。

**类型对应关系**：

| C/C++ 类型 | Rust FFI 类型 |
|-----------|--------------|
| `int` | `std::ffi::c_int`（= `i32` on 所有主流平台） |
| `double` | `f64` |
| `void*` | `*mut std::ffi::c_void` |
| `const void*` | `*const std::ffi::c_void` |
| `T*`（返回值，可为 null） | `*mut T`（Rust 中裸指针可为 null） |

---

## 6 第四层：安全的 Rust 包装类型

这是整个封装的核心价值所在：**用 Rust 类型系统把 `unsafe` 局限在最小范围内**。

### 6.1 LbmGrid — 格子网格的安全包装

```rust
pub struct LbmGrid {
    ptr: *mut ffi::LatticeGridHandle,  // 唯一持有 C++ 堆对象的字段
}
```

**构造：**

```rust
pub fn new(nx: i32, ny: i32, nz: i32, model: LatticeModel) -> Self {
    let ptr = unsafe {
        ffi::lbm_grid_new(nx, ny, nz, model.into())  // 唯一的 unsafe 调用
    };
    assert!(!ptr.is_null(), "lbm_grid_new returned null");  // 提前捕获失败
    LbmGrid { ptr }
}
```

**析构（RAII）：**

```rust
impl Drop for LbmGrid {
    fn drop(&mut self) {
        unsafe { ffi::lbm_grid_free(self.ptr) };
        // drop 完成后 ptr 成为悬垂指针，但 Rust 保证 drop 后不再访问 self
    }
}
```

> **RAII（Resource Acquisition Is Initialization）**：资源（这里是 C++ 堆内存）在 `LbmGrid` 构造时获取，在 `LbmGrid` 析构时释放。只要 `LbmGrid` 在 Rust 作用域内，内存就是有效的；离开作用域时自动释放，不可能忘记。

**读取字段：**

```rust
pub fn rho(&self, idx: i32) -> f64 {
    unsafe { ffi::lbm_grid_rho(self.ptr, idx) }
    // 越界由 C++ 侧负责返回 0.0，Rust 侧无需额外检查
}
```

### 6.2 LbmSolver — 求解器的安全包装

```rust
pub fn new(grid: &mut LbmGrid, omega: f64, cm: CollisionModel) -> Self {
    let ptr = unsafe {
        ffi::lbm_solver_new(grid.as_mut_ptr(), omega, cm.into())
    };
    assert!(!ptr.is_null(), "lbm_solver_new returned null");
    LbmSolver { ptr }
}
```

**注意** `grid: &mut LbmGrid` 参数：

- Rust 的借用规则保证：在调用 `step()` 时，`grid` 没有其他可变借用者，
  因此 C++ 侧可以安全地修改 `LatticeGrid` 的内容，不会有数据竞争。
- `&mut` 是编译期的"独占访问锁"——与操作系统锁不同，它的开销为零。

### 6.3 From trait — 安全的枚举转换

```rust
impl From<LatticeModel> for ffi::LatticeModelC {
    fn from(m: LatticeModel) -> Self {
        match m {
            LatticeModel::D2Q9  => ffi::LatticeModelC::D2Q9,
            LatticeModel::D3Q19 => ffi::LatticeModelC::D3Q19,
            LatticeModel::D3Q27 => ffi::LatticeModelC::D3Q27,
        }
    }
}
```

这样 `model.into()` 就能自动将公开的 `LatticeModel` 转为内部的 `ffi::LatticeModelC`，调用者不需要接触任何 `ffi` 模块内的类型。

### 6.4 unsafe impl Send

```rust
unsafe impl Send for LbmGrid {}
unsafe impl Send for LbmSolver {}
```

Rust 默认不允许含有裸指针的类型跨线程传递（`*mut T` 没有实现 `Send`）。  
这里手动标注 `Send` 的**含义**是：我们（作为封装作者）承诺——C++ 的 `LatticeGrid` 在堆上分配，指针不会被多个线程同时修改（调用方必须保证同步）。

---

## 7 第五层：主控驱动（orchestrator）

文件：`orchestrator/src/main.rs`

主控层只使用第四层的安全类型，**完全没有 `unsafe` 代码**：

```rust
use lbm_bindings::{LatticeModel, CollisionModel, LbmGrid, LbmSolver};

fn main() -> Result<()> {
    let cfg = Config::from_file(&args.config)?;

    // 创建格子网格（C++ LatticeGrid 在堆上分配）
    let mut grid = LbmGrid::new(
        cfg.fluid.nx as i32, cfg.fluid.ny as i32, cfg.fluid.nz as i32,
        model,
    );

    // 创建求解器（C++ Solver 在堆上分配，持有 &mut LatticeGrid）
    let mut solver = LbmSolver::new(&mut grid, cfg.omega(), cm);

    // 仿真主循环
    for step in 0..cfg.simulation.n_steps {
        solver.step(&mut grid);   // 调用 C++ Solver::step()，完全安全
        // … 写快照、写 CSV、调 Python …
    }
    // main() 返回时，solver 和 grid 的 Drop 自动释放 C++ 堆内存
}
```

**整个 `main.rs` 的安全性保证**：
- 不写 `unsafe`，不操作裸指针；
- 借用检查器保证 `grid` 和 `solver` 的访问不会冲突；
- `Drop` 保证无论正常退出还是 `?` 早返，C++ 内存都会被释放。

---

## 8 构建系统：build.rs + CMake 协作

文件：`bindings/build.rs`

Rust 项目通常只使用 Cargo 构建。但当需要编译 C/C++ 代码时，Cargo 提供了"构建脚本（build script）"机制：在 crate 根目录放一个 `build.rs`，Cargo 会在编译 crate 之前先运行它。

### 8.1 构建脚本的执行时机

```
cargo build
    │
    ├─ 1. 检测依赖变化（cargo:rerun-if-changed 指令）
    │
    ├─ 2. 运行 bindings/build.rs
    │       └─ cmake::Config::new(repo_root)
    │               └─ cmake configure → cmake --build → cmake --install
    │                  生成 <out_dir>/lib/liblbm_core.a
    │
    └─ 3. 编译 bindings/src/lib.rs
            └─ rustc 链接 liblbm_core.a（由 build.rs 的 cargo: 指令指定）
```

### 8.2 build.rs 的关键步骤

```rust
fn main() {
    // 步骤 1：找到仓库根目录（Cargo.toml 所在位置的上一级）
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.parent().unwrap();
    // manifest_dir = /path/to/repo/bindings/
    // repo_root    = /path/to/repo/

    // 步骤 2：调用 CMake 构建 liblbm_core.a
    let dst = cmake::Config::new(repo_root)
        .define("BUILD_TESTS", "OFF")   // 不需要编译 C++ 测试可执行文件
        .define("ENABLE_MPI", std::env::var("LBM_ENABLE_MPI")
                               .unwrap_or_else(|_| "OFF".into()))
        .define("CMAKE_BUILD_TYPE", cmake_build_type)  // Debug 或 Release
        .build();
    // dst = <cargo_target_dir>/build/lbm-bindings-xxxxx/out/
    // 生成物：dst/lib/liblbm_core.a

    // 步骤 3：告诉 rustc 去哪里找 .a 文件
    println!("cargo:rustc-link-search=native={}/lib", dst.display());

    // 步骤 4：告诉 rustc 链接 liblbm_core.a
    println!("cargo:rustc-link-lib=static=lbm_core");

    // 步骤 5：链接 C++ 标准库（平台相关）
    match std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default().as_str() {
        "macos" | "ios" => println!("cargo:rustc-link-lib=c++"),
        "windows"       => {} // MSVC 自动链接，无需声明
        _               => println!("cargo:rustc-link-lib=stdc++"),
    }

    // 步骤 6：增量构建守卫（只有这些文件变化才重新编译）
    println!("cargo:rerun-if-changed=../core/src");
    println!("cargo:rerun-if-changed=../core/include");
}
```

### 8.3 cargo: 协议指令说明

| 指令 | 含义 |
|------|------|
| `cargo:rustc-link-search=native=<路径>` | 告诉 `rustc` 在此目录搜索 `.a` / `.lib` 文件 |
| `cargo:rustc-link-lib=static=lbm_core` | 静态链接 `liblbm_core.a`（Unix）或 `lbm_core.lib`（Windows） |
| `cargo:rustc-link-lib=stdc++` | 动态链接 `libstdc++.so`（C++ 运行时） |
| `cargo:rerun-if-changed=<路径>` | 指定路径变化时才重新运行 build.rs |
| `cargo:rerun-if-env-changed=<VAR>` | 指定环境变量变化时才重新运行 build.rs |

### 8.4 增量构建工作原理

首次 `cargo build` 后，Cargo 会记录 `cargo:rerun-if-changed` 中所有路径的哈希值。  
再次 `cargo build` 时：

- **没有文件变化** → 跳过 build.rs，直接复用缓存的 `liblbm_core.a`，编译时间极短；
- **C++ 源文件变化** → 重新运行 build.rs，触发 CMake 增量编译（只重编变化的 `.cpp` 文件），然后重新链接。

---

## 9 插件系统：C 函数指针 ↔ C++ 虚函数

这是整个封装中最复杂的部分，涉及三层翻译。

### 9.1 问题描述

C++ 插件接口使用虚函数：

```cpp
class IBoundaryPlugin {
public:
    virtual void apply(LatticeGrid& grid, int step) = 0;
    virtual const char* name() const = 0;
};
```

Rust 无法直接提供一个实现 C++ 虚函数表的对象。需要将 Rust（或 C）的函数指针"包装"成 C++ 虚函数的实现。

### 9.2 三层翻译流程

```
Rust 侧（类型安全）
    PluginCallbacks { boundary_fn: Some(my_rust_fn), … }
            │  fn register_plugins(cbs: PluginCallbacks)
            ▼
C ABI 层（lbm_bindings::register_plugins）
    unsafe { ffi::lbm_set_plugins(cbs.boundary_fn, cbs.boundary_data, …) }
            │  调用 lbm_set_plugins（8 个参数）
            ▼
C++ 适配器层（plugin_registry.cpp）
    g_boundary_adapter.fn   = boundary_fn;   // 保存 C 函数指针
    reg.set_boundary_plugin(&g_boundary_adapter);  // 注册适配器对象
            │  PluginRegistry 调用 boundary_->apply(grid, step)
            ▼
C++ 适配器类（BoundaryAdapter）
    void apply(LatticeGrid& grid, int step) override {
        if (fn) fn(static_cast<void*>(&grid), step, data);
        //       ↑ 将 C++ 引用转为 void*，调用 C 函数指针
    }
```

### 9.3 适配器类详解

文件：`core/src/plugins/plugin_registry.cpp`

```cpp
// 适配器：将 C 回调包装为 IBoundaryPlugin
class BoundaryAdapter final : public lbm::IBoundaryPlugin {
public:
    lbm_boundary_fn fn   = nullptr;  // C 函数指针
    void*           data = nullptr;  // 用户自定义数据（不透明指针）

    void apply(lbm::LatticeGrid& grid, int step) override {
        if (fn) fn(static_cast<void*>(&grid), step, data);
        // C++ 引用 → void* → 传给 C 函数
    }
    const char* name() const override { return "c_callback_boundary"; }
};

// 静态实例：生命周期 = 程序运行期
static BoundaryAdapter g_boundary_adapter;
```

**静态实例的意义**：C ABI 不能动态 `new` 插件对象（那样需要手动管理生命周期），直接用静态全局变量，生命周期覆盖整个进程，不会提前析构。

### 9.4 Rust 侧如何注册自定义边界条件

```rust
// 定义 C 兼容函数
unsafe extern "C" fn my_boundary(
    grid_ptr: *mut std::ffi::c_void,
    step: i32,
    _data: *mut std::ffi::c_void,
) {
    // 将 void* 重新解释为 C++ LatticeGrid 指针
    // 注意：这里需要知道 LatticeGrid 的内存布局，通常通过额外的 C ABI 函数读取
    let _ = (grid_ptr, step);
    println!("自定义边界条件：step = {step}");
}

// 注册
let mut cbs = lbm_bindings::PluginCallbacks::default();
cbs.boundary_fn = Some(my_boundary);
lbm_bindings::register_plugins(cbs);
```

### 9.5 函数指针类型签名对照

| 插件类型 | C++ 虚函数签名 | C ABI 函数指针签名 | Rust 函数指针类型 |
|---------|--------------|-------------------|-----------------|
| 边界条件 | `void apply(LatticeGrid&, int)` | `void(*)(void*, int, void*)` | `unsafe extern "C" fn(*mut c_void, c_int, *mut c_void)` |
| 网格自适应 | `void adapt(LatticeGrid&, int)` | `void(*)(void*, int, void*)` | 同上 |
| 运动更新 | `void update(LatticeGrid&, MarkerSet*, double, int)` | `void(*)(void*, void*, double, int, void*)` | `unsafe extern "C" fn(*mut c_void, *mut c_void, f64, c_int, *mut c_void)` |
| 柔性体步进 | `void step(MarkerSet&, double, int)` | `void(*)(void*, double, int, void*)` | `unsafe extern "C" fn(*mut c_void, f64, c_int, *mut c_void)` |

---

## 10 内存安全深度分析

### 10.1 所有权唯一性保证

```rust
let mut grid = LbmGrid::new(64, 64, 1, LatticeModel::D2Q9);
//  ^^^^  所有权在此变量，全局唯一

let mut solver = LbmSolver::new(&mut grid, 1.0, CollisionModel::Bgk);
//                               ^^^^ 可变借用：grid 暂时"借给" solver::new

// solver::new 调用结束后，可变借用归还，grid 可以再次使用
solver.step(&mut grid);  // 再次可变借用 grid
```

Rust 的借用规则保证：**任意时刻，一个对象只有一个可变借用**，因此 C++ 侧不会看到同一 `LatticeGrid` 被多个调用者并发修改。

### 10.2 RAII 与双重释放防护

```
grid 离开作用域
    → Rust 调用 <LbmGrid as Drop>::drop(&mut self)
        → unsafe { ffi::lbm_grid_free(self.ptr) }
            → C++ delete g
                → ~LatticeGrid() 析构，std::vector 自动释放堆内存
```

Rust 的所有权规则保证 `drop` 只调用一次（不会双重释放），且在 `drop` 之后不能再访问 `self.ptr`（移动语义）。

### 10.3 空指针检查策略

在 C ABI 层（C++侧）做检查，而不是 Rust 侧：

```cpp
double lbm_grid_rho(const lbm::LatticeGrid* g, int idx) {
    if (!g || idx < 0 || idx >= g->size()) return 0.0;  // 防御性检查
    return g->rho[idx];
}
```

在 Rust 侧，`LbmGrid::new()` 已经用 `assert!(!ptr.is_null(), …)` 在构造时捕获空指针，之后 `self.ptr` 必然非空，读取操作不需要重复检查。

### 10.4 `unsafe` 的最小化原则

本项目将 `unsafe` 严格限制在以下三处：

1. **`mod ffi` 中的 `extern "C"` 块**：声明外部 C 函数（必须 unsafe）；
2. **`LbmGrid` / `LbmSolver` 的构造和 `drop` 方法**：调用 C ABI 函数（`new` / `free`）；
3. **`unsafe impl Send`**：手动断言类型的线程安全性。

**整个 `orchestrator/src/main.rs`（主控程序）没有任何 `unsafe`**，这是封装设计正确的标志。

---

## 11 常见构建问题与解决方案

### 11.1 LNK1181 / undefined reference：找不到 `stdc++`（Windows）

**错误信息**：
```
error: linking with `link.exe` failed
LINK1181: cannot open input file 'stdc++.lib'
```

**原因**：`build.rs` 中对 Windows 也写了 `cargo:rustc-link-lib=stdc++`，而 MSVC 不存在 `stdc++.lib`。

**修复**（已在本项目中实现）：

```rust
let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
match target_os.as_str() {
    "macos" | "ios" => println!("cargo:rustc-link-lib=c++"),
    "windows"       => {} // MSVC 自动链接 C++ 运行时，不需要显式声明
    _               => println!("cargo:rustc-link-lib=stdc++"),
}
```

### 11.2 LNK1120：运行时库冲突（Windows MSVC）

**错误信息**：
```
LINK1120: 1 unresolved externals
(mixing /MT and /MD runtime libraries)
```

**原因**：Rust 默认使用 `/MD`（MultiThreadedDLL），但 CMake 默认 Debug 配置使用 `/MDd`。两者不兼容。

**修复**（已在本项目 `CMakeLists.txt` 中实现）：

```cmake
# 强制所有配置使用 /MD（与 Rust 一致）
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" CACHE STRING "" FORCE)
```

同时在 `build.rs` 中将 `CMAKE_BUILD_TYPE` 与 Cargo 配置对齐：

```rust
let cmake_build_type = if std::env::var("OPT_LEVEL").unwrap_or_default() == "0" {
    "Debug"    // cargo build
} else {
    "Release"  // cargo build --release
};
.define("CMAKE_BUILD_TYPE", cmake_build_type)
```

### 11.3 `lbm_grid_new` 返回 nullptr

**原因**：
- 传入的 `model_id` 超出范围（0–2）；
- 系统内存不足（`new (std::nothrow)` 返回 nullptr）。

**诊断**：

```rust
// LbmGrid::new 内部已有 assert
assert!(!ptr.is_null(), "lbm_grid_new returned null");
// 触发此 panic 时，检查 nx/ny/nz 是否合理，或系统内存是否充足
```

### 11.4 仿真结果全为零

**常见原因**：忘记调用 `solver.step()` 后才读取 `grid.rho()`；或者步数为 0。

**诊断**：

```rust
println!("rho[0] = {}", grid.rho(0));   // 初始值应为 1.0
solver.step(&mut grid);
println!("rho[0] = {}", grid.rho(0));   // 碰撞/迁移后应有轻微变化
```

---

## 12 完整调用链一览

以下是一次 `solver.step(&mut grid)` 调用从 Rust 到 C++ 再返回的完整路径：

```
Rust: solver.step(&mut grid)
  │
  │ [Rust 安全层，bindings/src/lib.rs]
  ├─ LbmSolver::step(&mut self, grid: &mut LbmGrid)
  │       unsafe { ffi::lbm_solver_step(self.ptr, grid.as_mut_ptr()) }
  │
  │ [ABI 边界：Rust 调用 extern "C" 函数]
  │
  ├─ lbm_solver_step(s: *mut Solver, g: *mut LatticeGrid)
  │   位置：core/src/capi/lbm_capi.cpp
  │
  │ [C++ 侧，调用 PluginRegistry 和 Solver]
  │
  ├─ reg.update_motion(*g, nullptr, 1.0, 0)
  │   → if (motion_) motion_->update(grid, nullptr, dt, step)
  │     → MotionAdapter::update() → 调用 Rust 注册的 C 函数指针
  │
  ├─ s->step()
  │   位置：core/src/lbm/solver.cpp
  │   → collide()  [BGK 或 MRT 碰撞，OpenMP 并行]
  │   → stream()   [周期性迁移，交换 f/f_tmp，第一次 compute_macroscopic()]
  │   → apply_boundary_conditions()  [覆盖幽灵方向：反弹/Zou-He]
  │   → compute_macroscopic()（第二次，边界节点得到正确 ρ/u）
  │
  ├─ reg.apply_boundary(*g, 0)
  │   → if (boundary_) boundary_->apply(grid, step)
  │     → BoundaryAdapter::apply() → 调用 Rust 注册的 C 函数指针
  │
  ├─ reg.adapt_mesh(*g, 0)
  │   → if (mesh_) mesh_->adapt(grid, step)
  │
  └─ reg.step_flexible(nullptr, 1.0, 0)
      → 跳过（markers == nullptr）

  [ABI 边界：C++ 函数返回]

Rust: solver.step 返回，grid 中的 rho/u 已更新
```

整个调用过程中：
- **所有 C++ 堆内存**（`LatticeGrid`、`Solver`）由 C++ 侧的 `new`/`delete` 管理；
- **Rust 持有指向这些对象的指针**，通过 `Drop` 触发 `delete`；
- **C++ 不知道 Rust 的存在**，只看到普通的 C 函数调用；
- **Rust 不知道 C++ 类的内存布局**，只通过不透明句柄访问。

这就是"C ABI 桥"的全部精髓：**两种语言只在 C 函数调用这个最小公共接口上相遇，其余细节各自隐藏**。

---

## 13 Rust 语言特性速查——读懂 `bindings/src/lib.rs`

> 本节面向对 Rust 语法尚不熟悉的读者，逐条解释 `lib.rs` 中出现的关键 Rust 特性。

### 13.1 `#[repr(C)]` — 内存布局控制

```rust
#[repr(C)]
pub enum LatticeModelC {
    D2Q9  = 0,
    D3Q19 = 1,
    D3Q27 = 2,
}
```

**默认情况下**，Rust 编译器可以任意调整枚举的内存表示（大小、变体编号、对齐方式），以便优化。这在纯 Rust 程序中没问题，但如果要把枚举值传给 C/C++ 函数，双方必须对"这个 `int` 代表哪个值"达成一致。

`#[repr(C)]` 告诉 Rust 编译器：**按 C 的规则布局这个类型**，即变体从 0 开始连续编号，大小与 C 的 `int` 一致。这样 Rust 的 `LatticeModelC::D2Q9` 与 C++ 的 `lbm::LatticeModel::D2Q9`（值为 0）就完全对应，跨 ABI 边界传递时不会产生误解。

**结论**：凡是要跨 FFI 边界传递的枚举，都必须加 `#[repr(C)]`。

### 13.2 空枚举 `{}` — 不透明类型的惯用法

```rust
pub enum LatticeGridHandle {}  // 没有任何变体
```

**为什么不用 `struct LatticeGridHandle;`（单元结构体）？**

- 单元结构体可以被实例化：`let _ = LatticeGridHandle;` 合法。
- 空枚举**永远无法被实例化**（没有任何变体可以构造），Rust 编译器在编译期就会阻止任何试图创建它的代码。

这等价于 C/C++ 中的"不完整类型前向声明"：

```c
// C 侧：只声明结构存在，不给出定义，外部代码只能持有指针
struct LatticeGrid;
typedef struct LatticeGrid LatticeGrid;
```

Rust 侧通过空枚举达到同样效果：**只能持有 `*mut LatticeGridHandle` 指针，不能创建或解引用**。

### 13.3 裸指针 `*mut T` vs 引用 `&mut T`

| 特性 | 裸指针 `*mut T` | 可变引用 `&mut T` |
|------|----------------|-----------------|
| 可为 null | ✓（需手动检查） | ✗（引用永远非空） |
| 借用检查 | 无（`unsafe` 中使用） | 编译器强制执行 |
| 生命周期追踪 | 无 | 编译器追踪 |
| 典型用途 | FFI 边界、底层内存操作 | 普通 Rust 代码 |

在 `LbmGrid` 中，`ptr: *mut ffi::LatticeGridHandle` 是裸指针，因为：
1. C++ 返回 `LatticeGrid*`，可能为 null（分配失败）；
2. 裸指针的使用必须在 `unsafe` 块内，提醒开发者这里需要额外注意安全性；
3. Rust 的借用检查器不知道 C++ 对象的生命周期，必须由封装代码（`Drop`）手动管理。

### 13.4 `impl From<A> for B` 与 `.into()` 语法糖

```rust
impl From<LatticeModel> for ffi::LatticeModelC {
    fn from(m: LatticeModel) -> Self { … }
}

// 使用时：
ffi::lbm_grid_new(nx, ny, nz, model.into())
//                              ^^^^^^^^^^
//                              编译器自动推断：需要 ffi::LatticeModelC，
//                              LatticeModel 实现了 From<LatticeModel>，
//                              因此调用 ffi::LatticeModelC::from(model)
```

`Into<T>` trait 是 `From<T>` 的镜像：只要实现了 `From<A> for B`，Rust 就会自动实现 `Into<B> for A`，所以 `model.into()` 等价于 `ffi::LatticeModelC::from(model)`。

好处：调用者（第七层 orchestrator）不需要写任何转换代码，也不需要 `use` 引入 `ffi` 模块的内部类型。

### 13.5 `impl Drop` — Rust 的 RAII

```rust
impl Drop for LbmGrid {
    fn drop(&mut self) {
        unsafe { ffi::lbm_grid_free(self.ptr) };
    }
}
```

`Drop` trait 是 Rust 实现 RAII（资源获取即初始化）的机制：**当变量离开作用域时，Rust 编译器自动调用 `drop()`**。

与 C++ 的析构函数完全对应：

```cpp
// C++ 等价
~LatticeGrid() { /* 自动析构，释放 vector 内存 */ }
```

Rust 保证：
- `drop` 在变量生命周期结束时**恰好调用一次**（不会遗漏，也不会双重释放）；
- `drop` 之后，Rust 的移动语义确保不再能访问已释放的 `ptr`（与 C++ 不同，C++ 析构后指针仍在作用域内）。

### 13.6 `unsafe impl Send` / `unsafe impl Sync`

```rust
unsafe impl Send for LbmGrid {}
```

Rust 默认规定：**含有裸指针的类型不能跨线程传递**（`*mut T` 没有实现 `Send`）。这是因为编译器无法自动证明裸指针跨线程安全。

`unsafe impl Send` 是封装作者对编译器的承诺：_"我已手动验证，这个类型跨线程传递是安全的（前提是调用方保证同步）"_。

- `Send`：类型的**所有权**可以转移到其他线程；
- `Sync`：类型的**不可变引用**可以被多个线程同时持有。

加了 `unsafe` 的 `impl` 会绕过编译器的自动检查，因此必须由开发者负责正确性。

### 13.7 `pub(crate)` 可见性

```rust
pub(crate) fn as_mut_ptr(&mut self) -> *mut ffi::LatticeGridHandle { … }
```

Rust 的可见性修饰符从窄到宽：

| 修饰符 | 可见范围 |
|--------|---------|
| （无修饰） | 当前模块及其子模块 |
| `pub(crate)` | 当前 crate 内所有模块 |
| `pub(super)` | 父模块 |
| `pub` | 任意外部 crate |

`as_mut_ptr` 返回裸指针，只应在 `LbmSolver::step()` 内部调用，不应暴露给外部使用者。`pub(crate)` 精确表达了这一意图，同时允许同一 crate 内的 `LbmSolver` 调用它。

### 13.8 `#[derive(Default)]` — 自动派生默认值

```rust
#[derive(Default, Clone, Copy)]
pub struct PluginCallbacks { … }
```

`#[derive(Default)]` 让 Rust 自动生成 `PluginCallbacks::default()` 方法：
- `Option<…>` 字段 → `None`
- `*mut c_void` 字段 → `std::ptr::null_mut()`（空指针）

这样调用方可以：

```rust
let mut cbs = PluginCallbacks::default();  // 所有插件均为"未注册"状态
cbs.boundary_fn = Some(my_bc);             // 只激活边界条件插件
register_plugins(cbs);
```

而不需要手动初始化每一个字段。

---

## 14 C++ 语言特性速查——读懂 `lbm_capi.cpp` 与 `solver.cpp`

> 本节面向对 C++ 尚不熟悉的读者，逐条解释核心 C++ 代码中的关键特性。

### 14.1 `extern "C"` — 禁止名称修饰

```cpp
extern "C" {
    lbm::LatticeGrid* lbm_grid_new(int nx, int ny, int nz, int model_id);
}
```

C++ 编译器默认对函数名进行"名称修饰"（name mangling），在符号表中写入包含参数类型的复杂名称，例如：

```
_ZN3lbm6Solver4stepEv   // ← GCC 对 lbm::Solver::step() 的修饰名
```

`extern "C"` 告诉编译器：**这个函数按 C 规则导出，符号名就是函数名本身**：

```
lbm_grid_new            // ← 稳定的 C 符号名，Rust 能直接声明
```

没有 `extern "C"` 时，每次重新编译或换编译器都可能产生不同的修饰名，Rust 链接器就找不到这些符号。

### 14.2 `new (std::nothrow)` — 异常安全的堆分配

```cpp
return new (std::nothrow) lbm::LatticeGrid(nx, ny, nz, model);
//         ^^^^^^^^^^^^^ "nothrow placement new"
```

普通 `new T(…)` 在内存分配失败时会**抛出 `std::bad_alloc` 异常**。问题是：C ABI 边界没有异常处理机制——如果异常从 C++ 函数传播到 Rust 代码，行为是**未定义的**（通常直接崩溃）。

`new (std::nothrow)` 是 C++ 标准库提供的"不抛异常版本"：内存不足时返回 `nullptr` 而非抛出异常。调用方（Rust 侧）通过检查返回值是否为 `nullptr` 来处理失败。

**规则**：凡是在 `extern "C"` 函数内分配内存，必须用 `new (std::nothrow)` 或 `try/catch` 捕获所有异常，防止异常逃逸到 C ABI 边界。

### 14.3 `static_cast<T>` — C++ 安全类型转换

```cpp
auto model = static_cast<lbm::LatticeModel>(model_id);
```

C++ 提供四种类型转换运算符（比 C 风格的 `(T)x` 更安全）：

| 运算符 | 用途 |
|--------|------|
| `static_cast<T>` | 编译期已知的合法转换（`int → enum`，`double → float` 等） |
| `dynamic_cast<T>` | 运行时多态向下转型（需要虚函数表） |
| `reinterpret_cast<T>` | 位级重新解释（最危险，慎用） |
| `const_cast<T>` | 去除/添加 `const` 修饰 |

`static_cast<lbm::LatticeModel>(model_id)` 将 `int` 转为枚举，编译器会检查转换是否合法。注意：C++ 不检查枚举值是否在有效范围内，因此代码在 `static_cast` 之前先做了范围检查：

```cpp
if (model_id < 0 || model_id > 2) return nullptr;  // 先验证，再转换
auto model = static_cast<lbm::LatticeModel>(model_id);
```

### 14.4 `const T*` vs `T*` — const 正确性

```cpp
int lbm_grid_nx(const lbm::LatticeGrid* g) { return g ? g->nx : 0; }
//              ^^^^^
double lbm_grid_rho(const lbm::LatticeGrid* g, int idx) { … }
```

- `const lbm::LatticeGrid* g`：指针 `g` 指向的对象是只读的，函数不能通过 `g` 修改 `LatticeGrid` 的内容。
- `lbm::LatticeGrid* g`：函数可能修改 `LatticeGrid` 的内容（例如 `lbm_solver_step`）。

在 Rust 侧，这对应：
- `*const LatticeGridHandle` → `const T*`（只读）
- `*mut LatticeGridHandle` → `T*`（可读写）

遵循 const 正确性的好处：编译器能在编译期发现"不该修改却修改了"的错误，而不是等到运行时。

### 14.5 局部静态单例模式

```cpp
static PluginRegistry& instance() {
    static PluginRegistry reg;  // 局部静态变量
    return reg;
}
```

这是 C++11 引入的**线程安全局部静态初始化**（也称 Meyers Singleton）：

1. **首次调用** `instance()` 时，`reg` 被构造，并一直存活到程序结束；
2. C++11 标准保证：如果多个线程同时第一次调用 `instance()`，构造过程是线程安全的（编译器生成必要的互斥锁）；
3. **后续调用**直接返回已构造的引用，无额外开销。

与普通全局变量相比，局部静态单例的优势是**初始化顺序有保证**：只有在第一次调用时才初始化，不存在"全局对象初始化顺序不确定"（Static Initialization Order Fiasco）的问题。

### 14.6 `[[nodiscard]]` — 强制检查返回值

```cpp
[[nodiscard]] int size() const { return nx * ny * nz; }
```

C++17 的 `[[nodiscard]]` 属性告诉编译器：**调用方必须使用这个函数的返回值**。如果调用方写了：

```cpp
grid.size();  // ← 返回值被丢弃，没有用于任何计算
```

编译器会产生警告（`warning: ignoring return value`）。这防止了常见错误，例如忘记检查 `lbm_grid_new` 的返回值是否为 `nullptr`。

---

*文档最后更新：2026-03-04*  
*对应源代码版本：参见 `Cargo.toml` / `CMakeLists.txt` 顶部的版本号*
