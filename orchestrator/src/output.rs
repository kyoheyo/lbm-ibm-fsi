//! Native Rust output routines for the LBM orchestrator.
//!
//! Two complementary output strategies are implemented:
//!
//! ## High-frequency: NPZ snapshot files
//!
//! [`write_snapshot_npz`] writes the complete Eulerian field (ρ, ux, uy) to a
//! NumPy-compatible `.npz` archive every `write_interval` time steps.  The
//! format is readable by `lbm_post.vtk_reader.NpzReader` without any
//! additional tools.
//!
//! ## Per-step: CSV monitor log
//!
//! [`append_monitor_csv`] appends a single row of scalar quantities to a
//! lightweight CSV file every time step (or at whatever frequency the caller
//! chooses).  This is suitable for time-series data such as drag coefficient,
//! lift coefficient, bulk kinetic energy, etc., which need to be recorded at
//! every step but consume negligible storage.

use std::io::Write as IoWrite;
use std::path::Path;
use anyhow::{Context, Result};

use lbm_bindings::LbmGrid;

// ---------------------------------------------------------------------------
// NPZ snapshot writer
// ---------------------------------------------------------------------------

/// Write one Eulerian field snapshot as a NumPy `.npz` archive.
///
/// The archive contains the arrays expected by `lbm_post.vtk_reader.NpzReader`:
///
/// | Key      | Shape       | Dtype   | Description           |
/// |----------|-------------|---------|-----------------------|
/// | `rho`    | `(ny, nx)`  | float64 | density               |
/// | `ux`     | `(ny, nx)`  | float64 | x-velocity            |
/// | `uy`     | `(ny, nx)`  | float64 | y-velocity            |
/// | `step`   | scalar      | int64   | time-step index       |
/// | `time`   | scalar      | float64 | physical time         |
///
/// File name: `<directory>/fluid_<NNNNNN>.npz`
pub fn write_snapshot_npz(
    grid: &LbmGrid,
    step: u64,
    time: f64,
    directory: &str,
) -> Result<()> {
    let nx = grid.nx() as usize;
    let ny = grid.ny() as usize;
    let n  = nx * ny;

    // Collect field arrays from the C++ grid (row-major: index = j*nx + i)
    let mut rho = Vec::with_capacity(n);
    let mut ux  = Vec::with_capacity(n);
    let mut uy  = Vec::with_capacity(n);
    for idx in 0..n {
        rho.push(grid.rho(idx as i32));
        ux .push(grid.ux (idx as i32));
        uy .push(grid.uy (idx as i32));
    }

    let path = format!("{}/fluid_{:06}.npz", directory, step);
    let file = std::fs::File::create(&path)
        .with_context(|| format!("Cannot create snapshot file: {path}"))?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("rho.npy", options)?;
    zip.write_all(&npy_f64(&rho, &[ny, nx]))?;

    zip.start_file("ux.npy", options)?;
    zip.write_all(&npy_f64(&ux, &[ny, nx]))?;

    zip.start_file("uy.npy", options)?;
    zip.write_all(&npy_f64(&uy, &[ny, nx]))?;

    zip.start_file("step.npy", options)?;
    zip.write_all(&npy_i64(&[step as i64], &[]))?;

    zip.start_file("time.npy", options)?;
    zip.write_all(&npy_f64(&[time], &[]))?;

    zip.finish()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// CSV monitor log writer
// ---------------------------------------------------------------------------

/// Append one row to a CSV monitor log.
///
/// If the file does not yet exist it is created and a header row is written
/// first.  Otherwise the row is simply appended.
///
/// # Arguments
///
/// * `path`   – path to the CSV file (e.g. `"output/monitor.csv"`)
/// * `step`   – current time-step index
/// * `time`   – current physical time
/// * `fields` – slice of `(name, value)` pairs written as extra columns
pub fn append_monitor_csv(
    path: &str,
    step: u64,
    time: f64,
    fields: &[(&str, f64)],
) -> Result<()> {
    let p = Path::new(path);
    let needs_header = !p.exists();

    // Open in append mode (create if absent)
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(p)
        .with_context(|| format!("Cannot open monitor CSV: {path}"))?;

    if needs_header {
        let mut header = "step,time".to_string();
        for (name, _) in fields {
            header.push(',');
            header.push_str(name);
        }
        header.push('\n');
        file.write_all(header.as_bytes())?;
    }

    let mut row = format!("{step},{time:.6}");
    for (_, value) in fields {
        row.push_str(&format!(",{value:.10e}"));
    }
    row.push('\n');
    file.write_all(row.as_bytes())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// NumPy .npy encoding helpers
// ---------------------------------------------------------------------------
// The .npy format (v1.0) is:
//   magic (6 B)  +  version (2 B)  +  HEADER_LEN (u16 LE)  +  header  +  data
// The preamble is 10 bytes; the total of preamble + header must be padded to
// a multiple of 64 bytes.

/// Encode a `f64` slice as a `.npy` byte buffer with the given shape.
pub fn npy_f64(data: &[f64], shape: &[usize]) -> Vec<u8> {
    npy_encode(data, "<f8", shape, |v: &f64, buf: &mut Vec<u8>| {
        buf.extend_from_slice(&v.to_le_bytes());
    })
}

/// Encode an `i64` slice as a `.npy` byte buffer with the given shape.
pub fn npy_i64(data: &[i64], shape: &[usize]) -> Vec<u8> {
    npy_encode(data, "<i8", shape, |v: &i64, buf: &mut Vec<u8>| {
        buf.extend_from_slice(&v.to_le_bytes());
    })
}

fn npy_encode<T>(
    data: &[T],
    dtype_str: &str,
    shape: &[usize],
    write_elem: impl Fn(&T, &mut Vec<u8>),
) -> Vec<u8> {
    let shape_str = if shape.is_empty() {
        "()".to_string()
    } else if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        let inner: Vec<String> = shape.iter().map(|s| s.to_string()).collect();
        format!("({})", inner.join(", "))
    };

    let raw_header = format!(
        "{{'descr': '{dtype_str}', 'fortran_order': False, 'shape': {shape_str}, }}"
    );

    // Pad so that 10 (preamble) + header_len is a multiple of 64
    let raw_len = raw_header.len() + 1; // +1 for trailing '\n'
    let total_pre = 10_usize + raw_len;
    let pad = if total_pre % 64 == 0 { 0 } else { 64 - (total_pre % 64) };
    let header_len = raw_len + pad;

    let mut header_bytes = raw_header.into_bytes();
    header_bytes.resize(header_len - 1, b' ');
    header_bytes.push(b'\n');

    let data_bytes = data.len() * std::mem::size_of::<T>();
    let mut buf = Vec::with_capacity(10 + header_len + data_bytes);

    buf.extend_from_slice(b"\x93NUMPY");
    buf.push(0x01); // version major
    buf.push(0x00); // version minor
    let hlen = header_len as u16;
    buf.extend_from_slice(&hlen.to_le_bytes());
    buf.extend_from_slice(&header_bytes);
    for elem in data {
        write_elem(elem, &mut buf);
    }
    buf
}

// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the .npy header produced by `npy_f64` can be parsed by
    /// checking the magic bytes, version, and total alignment.
    #[test]
    fn test_npy_magic_and_alignment() {
        let buf = npy_f64(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]);
        // Magic
        assert_eq!(&buf[0..6], b"\x93NUMPY");
        // Version 1.0
        assert_eq!(buf[6], 1);
        assert_eq!(buf[7], 0);
        // Total header region (preamble 10 B + header_len) must be % 64 == 0
        let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        assert_eq!((10 + hlen) % 64, 0);
        // Data region must contain 4 x 8 = 32 bytes
        assert_eq!(buf.len(), 10 + hlen + 32);
    }

    #[test]
    fn test_npy_scalar_shape() {
        // A 0-D array (scalar) uses shape `()`
        let buf = npy_i64(&[42_i64], &[]);
        // Should contain the 8-byte LE value of 42
        let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        let data = &buf[10 + hlen..];
        assert_eq!(data.len(), 8);
        assert_eq!(i64::from_le_bytes(data.try_into().unwrap()), 42);
    }

    #[test]
    fn test_append_monitor_csv_creates_header() {
        let dir = std::env::temp_dir().join(format!(
            "lbm_monitor_test_{}", std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let csv_path = dir.join("monitor.csv");
        let p = csv_path.to_str().unwrap();

        append_monitor_csv(p, 0, 0.0, &[("ke", 0.001), ("div_max", 1e-5)]).unwrap();
        let text = std::fs::read_to_string(&csv_path).unwrap();
        assert!(text.starts_with("step,time,ke,div_max\n"));
        assert!(text.contains("0,0.000000,"));
    }

    #[test]
    fn test_append_monitor_csv_appends() {
        let dir = std::env::temp_dir().join(format!(
            "lbm_monitor_append_{}", std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let csv_path = dir.join("monitor_append.csv");
        let p = csv_path.to_str().unwrap();

        append_monitor_csv(p, 1, 1.0, &[("ke", 0.1)]).unwrap();
        append_monitor_csv(p, 2, 2.0, &[("ke", 0.2)]).unwrap();
        let text = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        // 1 header + 2 data rows
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("step,time"));
    }
}
