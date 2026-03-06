use serde::Deserialize;
use std::path::Path;
use anyhow::{Context, Result};

// ---------------------------------------------------------------------------
// 顶层仿真配置（从 TOML 文件加载）
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub simulation: SimulationConfig,
    pub fluid: FluidConfig,
    pub structure: Option<StructureConfig>,
    pub ibm: Option<IbmConfig>,
    pub output: OutputConfig,
    /// 可选 Python 集成（子进程脚本 + FFI 绘图）
    #[serde(default)]
    pub python: PythonConfig,
    /// 可选插件选择（边界条件 / 网格 / 运动 / 柔性体）
    #[serde(default)]
    pub plugins: PluginsConfig,
    /// 可选并行配置（OpenMP 线程数等）
    #[serde(default)]
    pub parallel: ParallelConfig,
    /// 可选 MPI 并行配置（域分解模式、块数等）
    #[serde(default)]
    pub mpi: MpiRunConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SimulationConfig {
    /// 总时间步数
    pub n_steps: u64,
    /// 物理时间步长
    pub dt: f64,
    /// 格子模型：`"D2Q9"`、`"D3Q19"` 或 `"D3Q27"`
    #[serde(default = "default_lattice_model")]
    pub lattice_model: String,
    /// 碰撞模型：`"BGK"` 或 `"MRT"`
    #[serde(default = "default_collision_model")]
    pub collision_model: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FluidConfig {
    /// 网格尺寸
    pub nx: u32,
    pub ny: u32,
    #[serde(default = "default_nz")]
    pub nz: u32,
    /// 运动粘度（格子单位）
    pub nu: f64,
    /// 初始均匀密度
    #[serde(default = "default_rho")]
    pub rho0: f64,
    /// 边界条件列表
    #[serde(default)]
    pub boundary_conditions: Vec<BoundaryConditionConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BoundaryConditionConfig {
    /// `"bounce_back"`、`"zou_he_velocity"` 或 `"zou_he_pressure"`
    pub bc_type: String,
    /// `"west"`、`"east"`、`"south"`、`"north"`、`"bottom"` 或 `"top"`
    pub face: String,
    #[serde(default)]
    pub ux: f64,
    #[serde(default)]
    pub uy: f64,
    #[serde(default)]
    pub uz: f64,
    #[serde(default = "default_rho")]
    pub rho: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct StructureConfig {
    /// 杨氏模量
    pub young_modulus: f64,
    /// 截面惯性矩
    pub second_moment: f64,
    /// 结构密度
    pub density: f64,
    /// 截面面积
    pub area: f64,
    /// 静止长度
    pub length: f64,
    /// 有限元单元数
    pub n_elements: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct IbmConfig {
    /// `"circle"` 或 `"filament"`
    pub geometry: String,
    /// 中心 x 坐标（圆形）或起点 x 坐标（丝状体）
    pub x0: f64,
    /// 中心 y 坐标（圆形）或起点 y 坐标（丝状体）
    pub y0: f64,
    /// 半径（圆形）或长度（丝状体）
    pub size: f64,
    /// 拉格朗日标记点数量
    pub n_markers: u32,
    /// Delta 核函数：`"two_point"` 或 `"four_point"`
    #[serde(default = "default_delta_kernel")]
    pub delta_kernel: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OutputConfig {
    /// 每隔多少时间步写出一个快照文件。
    /// 这是**高频**原生 Rust 输出路径。
    pub write_interval: u64,
    /// 快照和图像的输出目录
    #[serde(default = "default_output_dir")]
    pub directory: String,
    /// 输出格式选择：
    ///
    /// | 值 | 格式 | 文件扩展名 | 说明 |
    /// |----|------|------------|------|
    /// | `"npz"` | NumPy `.npz` 压缩归档（**默认**）| `.npz` | 可被 `lbm_post.NpzReader` 读取 |
    /// | `"tecplot_asc"` | ASCII Tecplot POINT 格式 | `.dat` | 可用 Tecplot/ParaView 直接打开，人类可读 |
    /// | `"tecplot_bin"` | 二进制 Tecplot PLT（TDV112）| `.plt` | 可用 Tecplot 打开，体积最小 |
    ///
    /// 可在 TOML 中通过 `output.format` 选择：
    /// ```toml
    /// [output]
    /// write_interval = 500
    /// directory      = "output/lid_cavity"
    /// format         = "tecplot_asc"   # 或 "npz"（默认）或 "tecplot_bin"
    /// ```
    #[serde(default = "default_output_format")]
    pub format: String,
    /// 每隔多少步调用 Python FFI 绘图器生成等值线图（步数）。
    ///
    /// ## 启用 FFI 绘图的完整步骤
    ///
    /// 1. 在 TOML 中设置 `plot_interval = <N>` 和 `python.pythonpath = "python"`。
    /// 2. 用 `python-ffi` 特性编译 Rust 求解器：
    ///    ```bash
    ///    cargo build --release --features python-ffi
    ///    ```
    /// 3. 运行求解器，每隔 N 步 Rust 会在进程内直接调用
    ///    `lbm_post.bridge.plot_field_raw()`，将速度幅值与涡量云图
    ///    保存为 `<directory>/<field>_<NNNNNN>.png`，**不创建中间 .npz 文件**。
    ///
    /// `None`（或 TOML 中缺失）表示禁用进程内绘图（默认）。
    /// 这是**低频** Python FFI 输出路径。
    ///
    /// # 注意
    ///
    /// 若未以 `--features python-ffi` 编译，本字段被忽略，
    /// 不会报错，只是不生成 FFI 云图。
    #[serde(default)]
    pub plot_interval: Option<u64>,
    /// 为 `true` 时每步向 CSV 文件追加一行监控量数据。
    /// 提供**逐步轻量级**时间序列输出。
    #[serde(default)]
    pub enable_csv_monitor: bool,
}

/// Python 集成配置。
///
/// 支持两种互补模式：
///
/// 1. **子进程模式**（`pre_script` / `post_script`）：Rust 主控程序
///    在仿真循环前/后启动独立的 Python 子进程。
///    适用于可以接受进程启动开销的重型预/后处理任务（网格生成、
///    完整后处理流水线）。
///
/// 2. **FFI 模式**（`[output]` 中的 `plot_interval`）：pyo3 在进程内
///    直接调用 Python 函数，内存中传递数据——流场数组以 Python list
///    形式传递，不创建临时文件。需要 `python-ffi` Cargo 特性。
#[derive(Debug, Deserialize, Clone)]
pub struct PythonConfig {
    /// 子进程调用使用的 Python 解释器（默认：`"python3"`）
    #[serde(default = "default_python_interpreter")]
    pub interpreter: String,
    /// 仿真循环前执行的预处理脚本。
    /// 脚本接受配置文件路径作为第一个位置参数。
    pub pre_script: Option<String>,
    /// 仿真循环后执行的后处理脚本。
    /// 脚本接受输出目录作为第一个位置参数。
    pub post_script: Option<String>,
    /// 使用 FFI 模式时预置到 `sys.path` 的额外目录，
    /// 使 `lbm_pre` / `lbm_post` 无需 pip 安装即可导入。
    /// 通常设为仓库根目录下的 `python/` 子目录。
    pub pythonpath: Option<String>,
}

impl Default for PythonConfig {
    fn default() -> Self {
        PythonConfig {
            interpreter: default_python_interpreter(),
            pre_script: None,
            post_script: None,
            pythonpath: None,
        }
    }
}

// ---------------------------------------------------------------------------
/// 插件选择配置。
///
/// 每个字段命名一个内置插件（或自定义外部插件）以激活对应扩展点。
/// 空字符串或缺失字段表示"使用默认值 / 不使用插件"。
///
/// ## 可用扩展点
///
/// | 字段 | 接口 | 调用时机 | 使用场景示例 |
/// |------|------|----------|------------|
/// | `boundary` | `IBoundaryPlugin` | 每步标准边界条件后 | 对流出口、开放压力、NRBC |
/// | `mesh` | `IMeshPlugin` | 每步流式迁移后 | 自适应细化、拉伸网格 |
/// | `motion` | `IMotionPlugin` | 每步碰撞前 | 运动壁面、6 自由度刚体、ALE |
/// | `flexible` | `IFlexibleSolverPlugin` | 每步流式迁移后 | Kirchhoff 板、协转梁 |
///
/// ## TOML 示例
///
/// ```toml
/// [plugins]
/// boundary = "convective_outlet"  # 启动日志中打印的名称
/// mesh     = ""                   # 无自适应网格
/// motion   = "prescribed_sine"
/// flexible = ""                   # 使用内置 BeamSolver
/// ```
///
/// 名称非空的插件会在启动时打印日志。实际插件逻辑在仿真循环前通过
/// Rust 代码注册；名称字段仅供提示，不自动加载共享库。
#[derive(Debug, Deserialize, Clone, Default)]
pub struct PluginsConfig {
    /// 自定义边界条件插件名称（空 = 不使用）。
    #[serde(default)]
    pub boundary: String,
    /// 网格处理 / 自适应细化插件名称（空 = 不使用）。
    #[serde(default)]
    pub mesh: String,
    /// 运动网格 / 运动固体插件名称（空 = 不使用）。
    #[serde(default)]
    pub motion: String,
    /// 替代柔性体求解器插件名称（空 = 不使用）。
    #[serde(default)]
    pub flexible: String,
}

impl PluginsConfig {
    /// 若至少一个插件名称已配置则返回 true。
    pub fn any_active(&self) -> bool {
        !self.boundary.is_empty()
            || !self.mesh.is_empty()
            || !self.motion.is_empty()
            || !self.flexible.is_empty()
    }
}

// ---------------------------------------------------------------------------
// 默认值函数
// ---------------------------------------------------------------------------
fn default_lattice_model()      -> String { "D2Q9".to_string() }
fn default_collision_model()    -> String { "BGK".to_string() }
fn default_nz()                 -> u32    { 1 }
fn default_rho()                -> f64    { 1.0 }
fn default_delta_kernel()       -> String { "four_point".to_string() }
fn default_output_dir()         -> String { "output".to_string() }
fn default_output_format()      -> String { "npz".to_string() }
fn default_python_interpreter() -> String { "python3".to_string() }
fn default_mpi_mode()           -> String { "block".to_string() }

// ---------------------------------------------------------------------------
/// 并行计算配置（OpenMP 线程数等运行期设置）
///
/// ## TOML 示例
///
/// ```toml
/// [parallel]
/// omp_num_threads = 8   # 设置 OpenMP 线程数；0 或缺省 = 使用全部 CPU 核心
/// ```
///
/// `omp_num_threads` 的优先级高于 `OMP_NUM_THREADS` 环境变量（在进程内通过
/// `omp_set_num_threads()` 即时生效）。若同时设置了两者，TOML 中的值优先。
#[derive(Debug, Deserialize, Clone, Default)]
pub struct ParallelConfig {
    /// OpenMP 线程数。
    /// - `0`（默认）：使用 OpenMP 运行时默认值（通常是全部 CPU 核心数，
    ///   也可由 `OMP_NUM_THREADS` 环境变量覆盖）。
    /// - 正整数 N：强制使用 N 个线程（等价于 `OMP_NUM_THREADS=N`，但优先级更高）。
    #[serde(default)]
    pub omp_num_threads: u32,
}

// ---------------------------------------------------------------------------
/// MPI 并行运行配置
///
/// 控制域分解模式和块数。支持三种模式：
///
/// | `mode` | 别名 | 说明 |
/// |--------|------|------|
/// | `"block"` | `"2d_xy"`、`"1d_y"`、`"1d_x"` | XY 块分解（1D 为特例） |
/// | `"independent"` | `"multi_grid"` | 每进程独立仿真，无通信 |
/// | `"multigrid"` | — | 嵌套式多重网格（框架预留，AMR 风格） |
///
/// ## 模式一：XY 块分解（`"block"`，默认）
///
/// 将网格切分为 `nx_blocks × ny_blocks` 块（须满足 `nx_blocks * ny_blocks == nprocs`）。
///
/// - **一维 Y 切片**（`nx_blocks=1, ny_blocks=nprocs`）：沿 Y 方向均匀切片，
///   通信仅在南北方向，适合 nx << ny 的细长网格。
///   ```toml
///   [mpi]
///   mode = "block"
///   nx_blocks = 1    # 省略时默认 1
///   ny_blocks = 4    # Y 方向切 4 块（需 mpirun -n 4）
///   ```
/// - **一维 X 切片**（`nx_blocks=nprocs, ny_blocks=1`）：沿 X 方向切片，
///   通信仅在东西方向，适合 ny << nx 的扁平网格。
///   ```toml
///   [mpi]
///   mode = "block"
///   nx_blocks = 4    # X 方向切 4 块
///   ny_blocks = 1
///   ```
/// - **二维 XY 块**（`nx_blocks=px, ny_blocks=py`，px*py==nprocs）：
///   均匀切分两个方向，适合接近方形的网格（通信面积最小）。
///   ```toml
///   [mpi]
///   mode      = "block"
///   nx_blocks = 4
///   ny_blocks = 2    # 需 mpirun -n 8
///   ```
///
/// **旧名称兼容**：`mode = "1d_y"` / `mode = "2d_xy"` 均等价于 `mode = "block"`
/// 并会打印弃用提示。
///
/// ## 模式二：多进程独立仿真（`"independent"`）
///
/// 每个 MPI 进程运行**完全独立**的仿真，无任何进程间通信。
/// 适用于参数扫描（多套 Re、多套网格尺寸）。
/// 输出写入 `<output.directory>/rank_<N>/`。
/// ```toml
/// [mpi]
/// mode = "independent"   # 或旧名 "multi_grid"
/// ```
///
/// ## 模式三：嵌套式多重网格（`"multigrid"`，框架预留）
///
/// 细网格嵌套在粗网格上，支持任意嵌套层数（N 叉树数据结构 `MgTree`）。
/// 可与块分解组合（每层细化区域可独立并行分解）。
/// 当前状态：框架已就绪（`lbm::MgTree` / `lbm::MgNode` / `LbmMgTree` Rust 封装），
/// 实际 LBM 插值/限制算子（restriction/prolongation）待未来实现。
/// ```toml
/// [mpi]
/// mode = "multigrid"
/// # 嵌套关系通过代码（lbm_bindings::LbmMgTree）配置
/// ```
///
/// ## 三维预留（`nz_blocks`）
///
/// ```toml
/// [mpi]
/// mode      = "block"
/// nx_blocks = 2
/// ny_blocks = 2
/// nz_blocks = 2    # 3D 扩展预留（D3Q19/D3Q27），需配合 3D 求解器
/// ```
#[derive(Debug, Deserialize, Clone)]
pub struct MpiRunConfig {
    /// MPI 域分解模式：
    /// - `"block"`（默认）：XY 块分解（1D X/Y 切片是 nx_blocks=1 或 ny_blocks=1 的特例）
    /// - `"independent"`：每进程独立仿真，无通信
    /// - `"multigrid"`：嵌套式多重网格（框架预留）
    /// - `"1d_y"` / `"2d_xy"` / `"multi_grid"`：旧名称，向后兼容
    #[serde(default = "default_mpi_mode")]
    pub mode: String,
    /// X 方向进程块数（mode="block" 时有效；0=自动（ny_blocks=nprocs，1D Y 切片）；默认 1）
    #[serde(default = "default_one")]
    pub nx_blocks: u32,
    /// Y 方向进程块数（mode="block" 时有效；nx_blocks*ny_blocks 须等于 mpirun -n N；默认 nprocs）
    #[serde(default = "default_zero")]
    pub ny_blocks: u32,
    /// Z 方向进程块数（3D 扩展预留；当前 2D 仿真时设 0 或 1；默认 1）
    #[serde(default = "default_one")]
    pub nz_blocks: u32,
}

fn default_one()  -> u32 { 1 }
fn default_zero() -> u32 { 0 }

impl Default for MpiRunConfig {
    fn default() -> Self {
        MpiRunConfig {
            mode:      default_mpi_mode(),
            nx_blocks: 1,
            ny_blocks: 0,   // 0 = 自动：由 nprocs 决定
            nz_blocks: 1,
        }
    }
}

impl MpiRunConfig {
    /// 将旧模式名称归一化为新名称，并返回是否进行了降级（用于打印警告）。
    ///
    /// | 输入 | 输出 |
    /// |------|------|
    /// | `"1d_y"` | `"block"` |
    /// | `"1d_x"` | `"block"` |
    /// | `"2d_xy"` | `"block"` |
    /// | `"multi_grid"` | `"independent"` |
    /// | 其他 | 原值 |
    pub fn normalized_mode(&self) -> (&str, bool) {
        match self.mode.as_str() {
            "1d_y" | "1d_x" | "2d_xy" => ("block", true),
            "multi_grid"               => ("independent", true),
            m                          => (m, false),
        }
    }

    /// 计算有效的 (nx_blocks, ny_blocks)：
    /// - `ny_blocks == 0` 表示自动：nx_blocks=1, ny_blocks=nprocs（1D Y 切片）
    pub fn effective_blocks(&self, nprocs: i32) -> (u32, u32) {
        let px = self.nx_blocks.max(1);
        let py = if self.ny_blocks == 0 {
            (nprocs as u32).max(1) / px
        } else {
            self.ny_blocks.max(1)
        };
        (px, py)
    }
}

// ---------------------------------------------------------------------------
impl Config {
    /// 从 TOML 文件加载配置
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        toml::from_str(&text)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))
    }

    /// 由运动粘度 ν 计算 LBM 松弛频率 ω
    ///   ν = cs² (1/ω - 1/2) = (1/3)(1/ω - 1/2)
    ///   ⟹ ω = 1 / (3ν + 0.5)
    pub fn omega(&self) -> f64 {
        1.0 / (3.0 * self.fluid.nu + 0.5)
    }
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_omega_calculation() {
        // ν = 1/6 → τ = 1 → ω = 1
        let nu = 1.0 / 6.0;
        let omega = 1.0 / (3.0 * nu + 0.5);
        assert!((omega - 1.0_f64).abs() < 1e-12);
    }

    #[test]
    fn test_config_parse() {
        let toml_str = r#"
            [simulation]
            n_steps = 1000
            dt = 1.0

            [fluid]
            nx = 64
            ny = 64
            nu = 0.1

            [output]
            write_interval = 100
        "#;

        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.simulation.n_steps, 1000);
        assert_eq!(cfg.fluid.nx, 64);
        assert!((cfg.fluid.rho0 - 1.0_f64).abs() < 1e-12);
        assert_eq!(cfg.simulation.lattice_model, "D2Q9");
        // [output] 中 format 缺省 → "npz"
        assert_eq!(cfg.output.format, "npz");
        // [python] 段可选；默认值应生效
        assert_eq!(cfg.python.interpreter, "python3");
        assert!(cfg.python.pre_script.is_none());
        assert!(cfg.python.post_script.is_none());
        assert!(cfg.output.plot_interval.is_none());
        assert!(!cfg.output.enable_csv_monitor);
        // [plugins] 段缺失 → 所有名称默认为空字符串
        assert!(!cfg.plugins.any_active());
    }

    #[test]
    fn test_output_format_tecplot() {
        // 验证 output.format 能正确解析 tecplot_asc 和 tecplot_bin
        let toml_str = r#"
            [simulation]
            n_steps = 100
            dt = 1.0

            [fluid]
            nx = 16
            ny = 16
            nu = 0.1

            [output]
            write_interval = 10
            format = "tecplot_asc"
        "#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.output.format, "tecplot_asc");

        let toml_str2 = r#"
            [simulation]
            n_steps = 100
            dt = 1.0

            [fluid]
            nx = 16
            ny = 16
            nu = 0.1

            [output]
            write_interval = 10
            format = "tecplot_bin"
        "#;
        let cfg2: Config = toml::from_str(toml_str2).unwrap();
        assert_eq!(cfg2.output.format, "tecplot_bin");
    }

    #[test]
    fn test_python_config_parse() {
        let toml_str = r#"
            [simulation]
            n_steps = 100
            dt = 1.0

            [fluid]
            nx = 16
            ny = 16
            nu = 0.1

            [output]
            write_interval = 10
            plot_interval  = 50
            enable_csv_monitor = true

            [python]
            interpreter = "python3.11"
            pre_script  = "scripts/pre.py"
            post_script = "scripts/post.py"
            pythonpath  = "python"
        "#;

        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.python.interpreter, "python3.11");
        assert_eq!(cfg.python.pre_script.as_deref(), Some("scripts/pre.py"));
        assert_eq!(cfg.python.post_script.as_deref(), Some("scripts/post.py"));
        assert_eq!(cfg.python.pythonpath.as_deref(), Some("python"));
        assert_eq!(cfg.output.plot_interval, Some(50));
        assert!(cfg.output.enable_csv_monitor);
    }

    #[test]
    fn test_plugins_config_parse() {
        let toml_str = r#"
            [simulation]
            n_steps = 100
            dt = 1.0

            [fluid]
            nx = 16
            ny = 16
            nu = 0.1

            [output]
            write_interval = 10

            [plugins]
            boundary = "convective_outlet"
            mesh     = "adaptive_uniform"
            motion   = "prescribed_sine"
            flexible = "kirchhoff_love_plate"
        "#;

        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.plugins.boundary, "convective_outlet");
        assert_eq!(cfg.plugins.mesh,     "adaptive_uniform");
        assert_eq!(cfg.plugins.motion,   "prescribed_sine");
        assert_eq!(cfg.plugins.flexible, "kirchhoff_love_plate");
        assert!(cfg.plugins.any_active());
    }

    #[test]
    fn test_plugins_config_defaults() {
        // [plugins] 段缺失时，所有名称应默认为空字符串。
        let toml_str = r#"
            [simulation]
            n_steps = 10
            dt = 1.0

            [fluid]
            nx = 4
            ny = 4
            nu = 0.1

            [output]
            write_interval = 5
        "#;
        let cfg: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.plugins.boundary, "");
        assert_eq!(cfg.plugins.mesh,     "");
        assert_eq!(cfg.plugins.motion,   "");
        assert_eq!(cfg.plugins.flexible, "");
        assert!(!cfg.plugins.any_active());
    }
}
