//! Python FFI 桥接模块 — 使用 pyo3 在进程内调用 `lbm_pre` / `lbm_post` Python 函数，
//! 无需创建任何中间文件。
//!
//! 本模块仅在启用了 `python-ffi` Cargo 特性时编译
//! （`cargo build --features python-ffi`）。
//!
//! # 架构
//!
//! ```text
//! Rust 主控程序（main.rs）
//!     │
//!     ├─ markers_from_geometry()  ──pyo3──▶  lbm_pre.bridge.geometry_markers_raw()
//!     │        直接在内存中返回 (Vec<f64>, Vec<f64>, Vec<f64>)
//!     │
//!     └─ plot_field()  ────────────pyo3──▶  lbm_post.bridge.plot_field_raw()
//!              传递 &[f64] 切片；Python 将其重塑为二维 NumPy 数组并保存 PNG
//!              —— Rust 侧不生成 .npz 文件
//! ```
//!
//! # Python 路径
//!
//! 仓库根目录下的 `python/` 目录必须在 `sys.path` 中，导入才能成功。
//! 可在启动时调用 [`add_python_path`]，或在 TOML 的 `[python]` 段设置 `pythonpath`。

use anyhow::{Context, Result};

/// 以子进程方式运行 Python 脚本并等待其结束。
///
/// 将 `interpreter script args...` 作为子进程启动，等待其完成，
/// 并在返回非零退出码时返回 `Err`。
pub fn run_subprocess(interpreter: &str, script: &str, args: &[&str]) -> Result<()> {
    println!("  [python subprocess] {} {} {}", interpreter, script, args.join(" "));
    let status = std::process::Command::new(interpreter)
        .arg(script)
        .args(args)
        .status()
        .with_context(|| format!("Failed to launch Python script: {script}"))?;
    if !status.success() {
        anyhow::bail!("Python script `{script}` exited with status: {status}");
    }
    Ok(())
}

/// 将 `extra_path` 前置到 `sys.path`，使 `lbm_pre` 和 `lbm_post`
/// 在未通过 `pip` 安装时仍可导入。
///
/// 未启用 `python-ffi` 特性时此函数为无操作（静默成功）。
pub fn add_python_path(extra_path: &str) -> Result<()> {
    #[cfg(feature = "python-ffi")]
    {
        use pyo3::prelude::*;
        use pyo3::types::PyList;
        Python::attach(|py| -> PyResult<()> {
            let sys = py.import("sys")?;
            let path = sys.getattr("path")?;
            let path = path.cast::<PyList>()?;
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

/// 调用 `lbm_pre.bridge.geometry_markers_raw()` 并将 IBM 拉格朗日标记点
/// 的位置和弧长元素以 Rust `Vec<f64>` 形式返回。
///
/// 不写出 CSV 文件 — 数据直接通过 Python C API 传递。
///
/// # 参数
///
/// * `geometry`  — `"circle"` 或 `"filament"`
/// * `x0`, `y0`  — 圆心（圆形）或起点（丝状体），单位为格子长度
/// * `size`      — 半径（圆形）或长度（丝状体），单位为格子长度
/// * `n_markers` — 拉格朗日标记点数量
///
/// # 返回值
///
/// `(x, y, ds)` — 每个 `Vec<f64>` 长度均为 `n_markers`。
///
/// 未启用 `python-ffi` 特性时返回错误。
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
        let result = Python::attach(|py| -> PyResult<(Vec<f64>, Vec<f64>, Vec<f64>)> {
            let bridge = py.import("lbm_pre.bridge")?;
            let ret = bridge.call_method1(
                "geometry_markers_raw",
                (geometry, x0, y0, size, n_markers as usize),
            )?;
            let tuple = ret.cast::<PyTuple>()?;
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

/// 调用 `lbm_post.bridge.plot_field_raw()` 保存等值线 PNG 图。
///
/// 流场数组（`rho`、`ux`、`uy`）以长度为 `nx * ny` 的平坦行主序切片传递
/// —— Rust 侧不创建 `.npz` 文件。Python 将其重塑为二维 NumPy 数组
/// 并将图像保存到 `<out_dir>/<field>_<NNNNNN>.png`。
///
/// # 参数
///
/// * `rho`, `ux`, `uy` — 平坦（行主序）场切片，长度为 `nx * ny`
/// * `nx`, `ny`        — 网格尺寸
/// * `step`            — 时间步索引（用于文件命名）
/// * `time`            — 物理时间（用作坐标轴标签）
/// * `out_dir`         — 输出目录
/// * `field`           — 绘制哪个场：
///   `"velocity_magnitude"` | `"vorticity"` | `"pressure"` | `"streamlines"`
///
/// 未启用 `python-ffi` 特性时返回错误。
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
        Python::attach(|py| -> PyResult<()> {
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
