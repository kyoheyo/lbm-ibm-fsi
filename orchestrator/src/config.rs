use serde::Deserialize;
use std::path::Path;
use anyhow::{Context, Result};

// ---------------------------------------------------------------------------
// Top-level simulation configuration (loaded from a TOML file)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub simulation: SimulationConfig,
    pub fluid: FluidConfig,
    pub structure: Option<StructureConfig>,
    pub ibm: Option<IbmConfig>,
    pub output: OutputConfig,
    /// Optional Python integration (subprocess scripts + FFI plotting)
    #[serde(default)]
    pub python: PythonConfig,
    /// Optional plugin selection (boundary / mesh / motion / flexible)
    #[serde(default)]
    pub plugins: PluginsConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SimulationConfig {
    /// Total number of time steps
    pub n_steps: u64,
    /// Physical time step size
    pub dt: f64,
    /// Lattice model: "D2Q9", "D3Q19", or "D3Q27"
    #[serde(default = "default_lattice_model")]
    pub lattice_model: String,
    /// Collision model: "BGK" or "MRT"
    #[serde(default = "default_collision_model")]
    pub collision_model: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct FluidConfig {
    /// Grid dimensions
    pub nx: u32,
    pub ny: u32,
    #[serde(default = "default_nz")]
    pub nz: u32,
    /// Kinematic viscosity (lattice units)
    pub nu: f64,
    /// Initial uniform density
    #[serde(default = "default_rho")]
    pub rho0: f64,
    /// Boundary conditions
    #[serde(default)]
    pub boundary_conditions: Vec<BoundaryConditionConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct BoundaryConditionConfig {
    /// "bounce_back", "zou_he_velocity", "zou_he_pressure"
    pub bc_type: String,
    /// "west", "east", "south", "north", "bottom", "top"
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
    /// Young's modulus
    pub young_modulus: f64,
    /// Second moment of area
    pub second_moment: f64,
    /// Structural density
    pub density: f64,
    /// Cross-sectional area
    pub area: f64,
    /// Rest length
    pub length: f64,
    /// Number of finite elements
    pub n_elements: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct IbmConfig {
    /// "circle" or "filament"
    pub geometry: String,
    /// Centre x (circle) or start x (filament)
    pub x0: f64,
    /// Centre y (circle) or start y (filament)
    pub y0: f64,
    /// Radius (circle) or length (filament)
    pub size: f64,
    /// Number of Lagrangian markers
    pub n_markers: u32,
    /// Delta kernel: "two_point" or "four_point"
    #[serde(default = "default_delta_kernel")]
    pub delta_kernel: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OutputConfig {
    /// Write a NumPy `.npz` snapshot every this many time steps.
    /// This is the **high-frequency** native-Rust output path.
    pub write_interval: u64,
    /// Output directory for snapshots and plots
    #[serde(default = "default_output_dir")]
    pub directory: String,
    /// How often (in steps) to invoke the Python FFI plotter for contour images.
    /// `None` (or absent from TOML) disables in-process plotting.
    /// This is the **low-frequency** Python-FFI output path.
    #[serde(default)]
    pub plot_interval: Option<u64>,
    /// Append a CSV row of monitor-point values every step when `true`.
    /// Provides a **per-step lightweight** time-series output.
    #[serde(default)]
    pub enable_csv_monitor: bool,
}

/// Python integration configuration.
///
/// Supports two complementary modes:
///
/// 1. **Subprocess mode** (`pre_script` / `post_script`): the Rust orchestrator
///    launches a separate Python process before/after the simulation loop.
///    Suitable for heavy pre/post-processing tasks (mesh generation, full
///    post-processing pipelines) that can tolerate process-spawn overhead.
///
/// 2. **FFI mode** (`plot_interval` in `[output]`): pyo3 calls Python functions
///    in-process with direct memory transfer — field arrays are passed as
///    Python lists without creating temporary files.  Requires the
///    `python-ffi` Cargo feature.
#[derive(Debug, Deserialize, Clone)]
pub struct PythonConfig {
    /// Python interpreter used for subprocess calls (default: "python3")
    #[serde(default = "default_python_interpreter")]
    pub interpreter: String,
    /// Pre-processing script run before the simulation loop.
    /// Receives the config file path as its first positional argument.
    pub pre_script: Option<String>,
    /// Post-processing script run after the simulation loop.
    /// Receives the output directory as its first positional argument.
    pub post_script: Option<String>,
    /// Extra directory prepended to `sys.path` when using the FFI mode
    /// so that `lbm_pre` / `lbm_post` are importable without a pip install.
    /// Typically set to the `python/` sub-directory of the repository root.
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
/// Plugin selection configuration.
///
/// Each field names a built-in plugin (or custom external plugin) to activate
/// for the corresponding extension point.  An empty string or absent field
/// means "use the default / no plugin".
///
/// ## Available extension points
///
/// | Field | Interface | When called | Use-case examples |
/// |-------|-----------|-------------|-------------------|
/// | `boundary` | `IBoundaryPlugin` | After standard BCs each step | Convective outlet, open pressure, NRBC |
/// | `mesh` | `IMeshPlugin` | After streaming each step | Adaptive refinement, stretched grid |
/// | `motion` | `IMotionPlugin` | Before collision each step | Moving wall, 6-DOF rigid body, ALE |
/// | `flexible` | `IFlexibleSolverPlugin` | After streaming each step | Kirchhoff plate, co-rotational beam |
///
/// ## TOML example
///
/// ```toml
/// [plugins]
/// boundary = "convective_outlet"  # name printed in startup log
/// mesh     = ""                   # no adaptive mesh
/// motion   = "prescribed_sine"
/// flexible = ""                   # use built-in BeamSolver
/// ```
///
/// Plugins with non-empty names are logged at startup.  Actual plugin logic
/// is registered via Rust code before the simulation loop; the name field is
/// informational only and does not auto-load shared libraries.
#[derive(Debug, Deserialize, Clone, Default)]
pub struct PluginsConfig {
    /// Name of the custom boundary-condition plugin (empty = none).
    #[serde(default)]
    pub boundary: String,
    /// Name of the mesh-handling / adaptive-refinement plugin (empty = none).
    #[serde(default)]
    pub mesh: String,
    /// Name of the moving-mesh / moving-body plugin (empty = none).
    #[serde(default)]
    pub motion: String,
    /// Name of the alternative flexible-body solver plugin (empty = none).
    #[serde(default)]
    pub flexible: String,
}

impl PluginsConfig {
    /// Returns true if at least one plugin name is configured.
    pub fn any_active(&self) -> bool {
        !self.boundary.is_empty()
            || !self.mesh.is_empty()
            || !self.motion.is_empty()
            || !self.flexible.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------
fn default_lattice_model()      -> String { "D2Q9".to_string() }
fn default_collision_model()    -> String { "BGK".to_string() }
fn default_nz()                 -> u32    { 1 }
fn default_rho()                -> f64    { 1.0 }
fn default_delta_kernel()       -> String { "four_point".to_string() }
fn default_output_dir()         -> String { "output".to_string() }
fn default_python_interpreter() -> String { "python3".to_string() }

// ---------------------------------------------------------------------------
impl Config {
    /// Load configuration from a TOML file
    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        toml::from_str(&text)
            .with_context(|| format!("Failed to parse config file: {}", path.display()))
    }

    /// Compute the LBM relaxation frequency ω from kinematic viscosity ν
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
        // [python] section is optional; defaults should apply
        assert_eq!(cfg.python.interpreter, "python3");
        assert!(cfg.python.pre_script.is_none());
        assert!(cfg.python.post_script.is_none());
        assert!(cfg.output.plot_interval.is_none());
        assert!(!cfg.output.enable_csv_monitor);
        // [plugins] section absent → all names default to empty
        assert!(!cfg.plugins.any_active());
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
        // When [plugins] section is absent, all names should be empty.
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
