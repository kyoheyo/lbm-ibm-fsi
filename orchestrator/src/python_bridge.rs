//! Python FFI bridge — calls `lbm_pre` / `lbm_post` Python functions
//! in-process using pyo3, without creating any intermediate files.
//!
//! This module is compiled only when the `python-ffi` Cargo feature is
//! enabled (`cargo build --features python-ffi`).
//!
//! # Architecture
//!
//! ```text
//! Rust orchestrator (main.rs)
//!     │
//!     ├─ markers_from_geometry()  ──pyo3──▶  lbm_pre.bridge.geometry_markers_raw()
//!     │        returns (Vec<f64>, Vec<f64>, Vec<f64>) directly in memory
//!     │
//!     └─ plot_field()  ────────────pyo3──▶  lbm_post.bridge.plot_field_raw()
//!              passes &[f64] slices; Python reshapes into numpy arrays and
//!              saves PNG — no .npz file written by Rust
//! ```
//!
//! # Python path
//!
//! The `python/` directory of the repository must be on `sys.path` for the
//! imports to succeed.  Call [`add_python_path`] once at start-up, or set
//! `pythonpath` in the `[python]` TOML section.

use anyhow::Result;

/// Prepend `extra_path` to `sys.path` so that `lbm_pre` and `lbm_post` are
/// importable when they have not been installed via `pip`.
///
/// This is a no-op (succeeds silently) when the `python-ffi` feature is
/// disabled.
pub fn add_python_path(extra_path: &str) -> Result<()> {
    #[cfg(feature = "python-ffi")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyList;
        Python::with_gil(|py| -> PyResult<()> {
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            let path = path.downcast::<PyList>()?;
            path.insert(0, extra_path)?;
            Ok(())
        })?;
    }
    #[cfg(not(feature = "python-ffi"))]
    {
        let _ = extra_path;
    }
    Ok(())
}

/// Call `lbm_pre.bridge.geometry_markers_raw()` and return the IBM Lagrangian
/// marker positions and arc-length elements as Rust `Vec<f64>`.
///
/// No CSV file is written — the data is transferred directly through the
/// Python C API.
///
/// # Arguments
///
/// * `geometry`  – `"circle"` or `"filament"`
/// * `x0`, `y0`  – centre (circle) or start point (filament) in lattice units
/// * `size`      – radius (circle) or length (filament) in lattice units
/// * `n_markers` – number of Lagrangian markers
///
/// # Returns
///
/// `(x, y, ds)` where each `Vec<f64>` has length `n_markers`.
///
/// Returns an error when the `python-ffi` feature is not active.
pub fn markers_from_geometry(
    geometry: &str,
    x0: f64,
    y0: f64,
    size: f64,
    n_markers: u32,
) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>)> {
    #[cfg(feature = "python-ffi")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyTuple;
        let result = Python::with_gil(|py| -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
            let bridge = py.import("lbm_pre.bridge")?;
            let ret = bridge.call_method1(
                "geometry_markers_raw",
                (geometry, x0, y0, size, n_markers as usize),
            )?;
            let tuple = ret.downcast::<PyTuple>()?;
            let x:  Vec<f64> = tuple.get_item(0)?.extract()?;
            let y:  Vec<f64> = tuple.get_item(1)?.extract()?;
            let ds: Vec<f64> = tuple.get_item(2)?.extract()?;
            Ok((x, y, ds))
        })?;
        Ok(result)
    }
    #[cfg(not(feature = "python-ffi"))]
    {
        let _ = (geometry, x0, y0, size, n_markers);
        anyhow::bail!(
            "markers_from_geometry requires the `python-ffi` Cargo feature. \
             Rebuild with: cargo build --features python-ffi"
        )
    }
}

/// Call `lbm_post.bridge.plot_field_raw()` to save a contour PNG.
///
/// Field arrays (`rho`, `ux`, `uy`) are passed as flat row-major slices of
/// length `nx * ny` — no `.npz` file is created on the Rust side.  Python
/// reshapes them into 2-D NumPy arrays and saves the figure to
/// `<out_dir>/<field>_<NNNNNN>.png`.
///
/// # Arguments
///
/// * `rho`, `ux`, `uy` – flat (row-major) field slices, length `nx * ny`
/// * `nx`, `ny`        – grid dimensions
/// * `step`            – time-step index (used in the file name)
/// * `time`            – physical time (used as an axis label)
/// * `out_dir`         – output directory
/// * `field`           – which field to plot:
///   `"velocity_magnitude"` | `"vorticity"` | `"pressure"` | `"streamlines"`
///
/// Returns an error when the `python-ffi` feature is not active.
pub fn plot_field(
    rho: &[f64],
    ux: &[f64],
    uy: &[f64],
    nx: usize,
    ny: usize,
    step: u64,
    time: f64,
    out_dir: &str,
    field: &str,
) -> Result<()> {
    #[cfg(feature = "python-ffi")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyList;
        Python::with_gil(|py| -> PyResult<()> {
            let bridge = py.import("lbm_post.bridge")?;
            let rho_list = PyList::new(py, rho)?;
            let ux_list  = PyList::new(py, ux)?;
            let uy_list  = PyList::new(py, uy)?;
            bridge.call_method1(
                "plot_field_raw",
                (rho_list, ux_list, uy_list, nx, ny, step, time, out_dir, field),
            )?;
            Ok(())
        })?;
        Ok(())
    }
    #[cfg(not(feature = "python-ffi"))]
    {
        let _ = (rho, ux, uy, nx, ny, step, time, out_dir, field);
        anyhow::bail!(
            "plot_field requires the `python-ffi` Cargo feature. \
             Rebuild with: cargo build --features python-ffi"
        )
    }
}

