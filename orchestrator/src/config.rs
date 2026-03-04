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
    /// How often (in time steps) to write VTK output
    pub write_interval: u64,
    /// Output directory
    #[serde(default = "default_output_dir")]
    pub directory: String,
}

// ---------------------------------------------------------------------------
// Defaults
// ---------------------------------------------------------------------------
fn default_lattice_model()   -> String { "D2Q9".to_string() }
fn default_collision_model() -> String { "BGK".to_string() }
fn default_nz()              -> u32    { 1 }
fn default_rho()             -> f64    { 1.0 }
fn default_delta_kernel()    -> String { "four_point".to_string() }
fn default_output_dir()      -> String { "output".to_string() }

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
    }
}
