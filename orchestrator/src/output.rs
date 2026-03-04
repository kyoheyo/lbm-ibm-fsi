//! LBM 主控程序的原生 Rust 输出模块。
//!
//! 实现了两种互补的输出策略：
//!
//! ## 高频：NPZ 快照文件
//!
//! [`write_snapshot_npz`] 每隔 `write_interval` 步将完整欧拉场（ρ、ux、uy）
//! 写入兼容 NumPy 格式的 `.npz` 压缩归档文件。
//! 该格式可直接被 `lbm_post.vtk_reader.NpzReader` 读取，无需额外工具。
//!
//! ## 逐步：CSV 监控日志
//!
//! [`append_monitor_csv`] 每个时间步（或调用方选定的频率）向轻量级 CSV
//! 文件追加一行标量数据，适用于需要逐步记录但存储量极小的时间序列，
//! 例如阻力系数、升力系数、体积平均动能等。

use std::io::Write as IoWrite;
use std::path::Path;
use anyhow::{Context, Result};

use lbm_bindings::LbmGrid;

// ---------------------------------------------------------------------------
// NPZ 快照写出器
// ---------------------------------------------------------------------------

/// 将一个欧拉场快照写为 NumPy `.npz` 压缩归档文件。
///
/// 归档中包含 `lbm_post.vtk_reader.NpzReader` 所期望的数组：
///
/// | 键名     | 形状        | 数据类型  | 说明           |
/// |----------|-------------|-----------|----------------|
/// | `rho`    | `(ny, nx)`  | float64   | 密度           |
/// | `ux`     | `(ny, nx)`  | float64   | x 方向速度     |
/// | `uy`     | `(ny, nx)`  | float64   | y 方向速度     |
/// | `step`   | 标量        | int64     | 时间步索引     |
/// | `time`   | 标量        | float64   | 物理时间       |
///
/// 文件名格式：`<directory>/fluid_<NNNNNN>.npz`
pub fn write_snapshot_npz(
    grid: &LbmGrid,
    step: u64,
    time: f64,
    directory: &str,
) -> Result<()> {
    let nx = grid.nx() as usize;
    let ny = grid.ny() as usize;
    let n  = nx * ny;

    // 从 C++ 格子网格收集场数组（行主序：索引 = j*nx + i）
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
// CSV 监控日志写出器
// ---------------------------------------------------------------------------

/// 向 CSV 监控日志追加一行数据。
///
/// 若文件不存在则创建并先写入标题行；否则直接追加数据行。
///
/// # 参数
///
/// * `path`   — CSV 文件路径（例如 `"output/monitor.csv"`）
/// * `step`   — 当前时间步索引
/// * `time`   — 当前物理时间
/// * `fields` — `(列名, 值)` 元组切片，写为额外列
pub fn append_monitor_csv(
    path: &str,
    step: u64,
    time: f64,
    fields: &[(&str, f64)],
) -> Result<()> {
    let p = Path::new(path);
    let needs_header = !p.exists();

    // 以追加模式打开（不存在则创建）
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
// NumPy .npy 编码辅助函数
// ---------------------------------------------------------------------------
// .npy 格式（v1.0）：
//   魔数（6 字节）+ 版本号（2 字节）+ HEADER_LEN（u16 小端）+ 头部 + 数据
// 前导部分共 10 字节；前导 + 头部总长度必须是 64 字节的倍数。

/// 将 `f64` 切片编码为指定形状的 `.npy` 字节缓冲区。
pub fn npy_f64(data: &[f64], shape: &[usize]) -> Vec<u8> {
    npy_encode(data, "<f8", shape, |v: &f64, buf: &mut Vec<u8>| {
        buf.extend_from_slice(&v.to_le_bytes());
    })
}

/// 将 `i64` 切片编码为指定形状的 `.npy` 字节缓冲区。
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

    // 填充使得 10（前导）+ header_len 是 64 的倍数
    let raw_len = raw_header.len() + 1; // +1 为末尾 '\n'
    let total_pre = 10_usize + raw_len;
    let pad = if total_pre % 64 == 0 { 0 } else { 64 - (total_pre % 64) };
    let header_len = raw_len + pad;

    let mut header_bytes = raw_header.into_bytes();
    header_bytes.resize(header_len - 1, b' ');
    header_bytes.push(b'\n');

    let data_bytes = data.len() * std::mem::size_of::<T>();
    let mut buf = Vec::with_capacity(10 + header_len + data_bytes);

    buf.extend_from_slice(b"\x93NUMPY");
    buf.push(0x01); // 版本号主版本
    buf.push(0x00); // 版本号次版本
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

    /// 验证 `npy_f64` 生成的 .npy 头部中魔数、版本号和对齐均正确。
    #[test]
    fn test_npy_magic_and_alignment() {
        let buf = npy_f64(&[1.0_f64, 2.0, 3.0, 4.0], &[2, 2]);
        // 魔数
        assert_eq!(&buf[0..6], b"\x93NUMPY");
        // 版本 1.0
        assert_eq!(buf[6], 1);
        assert_eq!(buf[7], 0);
        // 前导（10 字节）+ header_len 必须是 64 的倍数
        let hlen = u16::from_le_bytes([buf[8], buf[9]]) as usize;
        assert_eq!((10 + hlen) % 64, 0);
        // 数据区必须包含 4 × 8 = 32 字节
        assert_eq!(buf.len(), 10 + hlen + 32);
    }

    #[test]
    fn test_npy_scalar_shape() {
        // 0 维数组（标量）使用形状 `()`
        let buf = npy_i64(&[42_i64], &[]);
        // 应包含 42 的 8 字节小端表示
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
        // 1 行标题 + 2 行数据
        assert_eq!(lines.len(), 3);
        assert!(lines[0].starts_with("step,time"));
    }
}
