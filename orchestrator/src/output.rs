//! LBM 主控程序的原生 Rust 输出模块。
//!
//! 实现了三种快照格式和一个轻量级时序日志：
//!
//! ## 高频：NPZ 快照文件（默认）
//!
//! [`write_snapshot_npz`] 每隔 `write_interval` 步将完整欧拉场（ρ、ux、uy）
//! 写入兼容 NumPy 格式的 `.npz` 压缩归档文件。
//! 该格式可直接被 `lbm_post.vtk_reader.NpzReader` 读取，无需额外工具。
//!
//! ## 高频：ASCII Tecplot 格式（`.dat`）
//!
//! [`write_snapshot_tecplot_asc`] 将同一欧拉场写为 Tecplot ASCII POINT
//! 格式文件，可用 Tecplot、ParaView 或本项目 Python 后处理包直接读取。
//! 文件名格式：`<directory>/fluid_<NNNNNN>.dat`
//!
//! ## 高频：二进制 Tecplot 格式（`.plt`，TDV112）
//!
//! [`write_snapshot_tecplot_bin`] 将同一欧拉场写为 Tecplot 二进制 PLT 格式
//! （TDV112 版本），可直接在 Tecplot 软件中打开，文件体积约为 ASCII 版本的 1/3。
//! 文件名格式：`<directory>/fluid_<NNNNNN>.plt`
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
// MPI 分区信息（用于多进程块分解模式输出）
// ---------------------------------------------------------------------------

/// MPI 块分解模式下本进程的物理分区信息。
///
/// 当使用 MPI 块分解（`mode = "block"`）时，每个进程持有全局域的一个分区，
/// 本地网格含幽灵行/列。此结构体记录物理分区的精确范围，供写出器：
/// - 跳过幽灵行/列（仅写出物理节点数据）
/// - 在输出文件中嵌入分区元数据，便于后处理将各进程分区拼合为全局场
///
/// ## 字段说明
///
/// | 字段         | 含义                                           |
/// |-------------|------------------------------------------------|
/// | `phys_x0`   | 物理区域在本地网格中的 X 偏移（0 或 1）         |
/// | `phys_y0`   | 物理区域在本地网格中的 Y 偏移（0 或 1）         |
/// | `local_nx`  | 物理列数（不含幽灵列）                          |
/// | `local_ny`  | 物理行数（不含幽灵行）                          |
/// | `x_start`   | 物理区域在全局坐标系中的 X 起始坐标             |
/// | `y_start`   | 物理区域在全局坐标系中的 Y 起始坐标             |
/// | `global_nx` | 全局域 X 总节点数                              |
/// | `global_ny` | 全局域 Y 总节点数                              |
#[derive(Clone, Copy, Debug)]
pub struct PartitionInfo {
    /// 物理区域在本地网格中的 X 偏移（0 = 无西幽灵列；1 = 有西幽灵列）
    pub phys_x0:   usize,
    /// 物理区域在本地网格中的 Y 偏移（0 = 无南幽灵行；1 = 有南幽灵行）
    pub phys_y0:   usize,
    /// 物理列数（不含幽灵列）
    pub local_nx:  usize,
    /// 物理行数（不含幽灵行）
    pub local_ny:  usize,
    /// 物理区域在全局坐标系中的 X 起始坐标
    pub x_start:   usize,
    /// 物理区域在全局坐标系中的 Y 起始坐标
    pub y_start:   usize,
    /// 全局域 X 总节点数
    pub global_nx: usize,
    /// 全局域 Y 总节点数
    pub global_ny: usize,
}

// ---------------------------------------------------------------------------
// 物理场提取辅助函数
// ---------------------------------------------------------------------------

/// 从 LBM 网格中提取物理节点的流场数据（ρ、ux、uy）。
///
/// 当 `partition` 为 `Some` 时，仅提取物理区域（跳过幽灵行/列）；
/// 否则提取所有节点。
///
/// # 返回值
///
/// `(rho, ux, uy, nx, ny)` — 三个长度为 `nx × ny` 的行主序向量及网格尺寸。
pub fn extract_physical_fields(
    grid: &LbmGrid,
    partition: Option<PartitionInfo>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, usize, usize) {
    let (px0, py0, pnx, pny, gnx) = if let Some(p) = partition {
        (p.phys_x0, p.phys_y0, p.local_nx, p.local_ny, grid.nx() as usize)
    } else {
        (0, 0, grid.nx() as usize, grid.ny() as usize, grid.nx() as usize)
    };
    let n = pnx * pny;
    let mut rho = Vec::with_capacity(n);
    let mut ux  = Vec::with_capacity(n);
    let mut uy  = Vec::with_capacity(n);
    for j in py0..(py0 + pny) {
        for i in px0..(px0 + pnx) {
            let idx = (j * gnx + i) as i32;
            rho.push(grid.rho(idx));
            ux .push(grid.ux (idx));
            uy .push(grid.uy (idx));
        }
    }
    (rho, ux, uy, pnx, pny)
}

// ---------------------------------------------------------------------------
// NPZ 快照写出器
// ---------------------------------------------------------------------------

/// 将一个欧拉场快照写为 NumPy `.npz` 压缩归档文件。
///
/// 归档中包含 `lbm_post.vtk_reader.NpzReader` 所期望的数组：
///
/// | 键名        | 形状           | 数据类型  | 说明                         |
/// |------------|----------------|-----------|------------------------------|
/// | `rho`      | `(ny, nx)`     | float64   | 密度（仅物理节点）             |
/// | `ux`       | `(ny, nx)`     | float64   | x 方向速度（仅物理节点）       |
/// | `uy`       | `(ny, nx)`     | float64   | y 方向速度（仅物理节点）       |
/// | `step`     | 标量           | int64     | 时间步索引                    |
/// | `time`     | 标量           | float64   | 物理时间                      |
/// | `x_start`  | 标量（可选）   | int64     | 分区全局 X 起始（MPI 块模式） |
/// | `y_start`  | 标量（可选）   | int64     | 分区全局 Y 起始（MPI 块模式） |
/// | `global_nx`| 标量（可选）   | int64     | 全局域 X 节点数（MPI 块模式）|
/// | `global_ny`| 标量（可选）   | int64     | 全局域 Y 节点数（MPI 块模式）|
///
/// 当 `part` 为 `Some(info)` 时，仅写出物理节点（跳过幽灵行/列），
/// 并在归档中附加分区元数据，供后处理程序拼合全局场。
///
/// 文件名格式：`<directory>/fluid_<NNNNNN>.npz`
pub fn write_snapshot_npz(
    grid: &LbmGrid,
    step: u64,
    time: f64,
    directory: &str,
    part: Option<PartitionInfo>,
) -> Result<()> {
    // 确定本次写出的物理节点范围
    let (grid_nx, phys_x0, phys_y0, out_nx, out_ny) = if let Some(p) = part {
        (grid.nx() as usize, p.phys_x0, p.phys_y0, p.local_nx, p.local_ny)
    } else {
        (grid.nx() as usize, 0, 0, grid.nx() as usize, grid.ny() as usize)
    };
    let n = out_nx * out_ny;

    // 从 C++ 格子网格收集物理节点场数组（行主序：索引 = j*grid_nx + i）
    let mut rho = Vec::with_capacity(n);
    let mut ux  = Vec::with_capacity(n);
    let mut uy  = Vec::with_capacity(n);
    for j in phys_y0..(phys_y0 + out_ny) {
        for i in phys_x0..(phys_x0 + out_nx) {
            let idx = (j * grid_nx + i) as i32;
            rho.push(grid.rho(idx));
            ux .push(grid.ux (idx));
            uy .push(grid.uy (idx));
        }
    }

    let path = format!("{}/fluid_{:06}.npz", directory, step);
    let file = std::fs::File::create(&path)
        .with_context(|| format!("Cannot create snapshot file: {path}"))?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("rho.npy", options)?;
    zip.write_all(&npy_f64(&rho, &[out_ny, out_nx]))?;

    zip.start_file("ux.npy", options)?;
    zip.write_all(&npy_f64(&ux, &[out_ny, out_nx]))?;

    zip.start_file("uy.npy", options)?;
    zip.write_all(&npy_f64(&uy, &[out_ny, out_nx]))?;

    zip.start_file("step.npy", options)?;
    zip.write_all(&npy_i64(&[step as i64], &[]))?;

    zip.start_file("time.npy", options)?;
    zip.write_all(&npy_f64(&[time], &[]))?;

    // MPI 块分解时嵌入分区元数据，便于后处理拼合全局场
    if let Some(p) = part {
        zip.start_file("x_start.npy", options)?;
        zip.write_all(&npy_i64(&[p.x_start as i64], &[]))?;

        zip.start_file("y_start.npy", options)?;
        zip.write_all(&npy_i64(&[p.y_start as i64], &[]))?;

        zip.start_file("global_nx.npy", options)?;
        zip.write_all(&npy_i64(&[p.global_nx as i64], &[]))?;

        zip.start_file("global_ny.npy", options)?;
        zip.write_all(&npy_i64(&[p.global_ny as i64], &[]))?;
    }

    zip.finish()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// ASCII Tecplot 快照写出器
// ---------------------------------------------------------------------------

/// 将一个欧拉场快照写为 ASCII Tecplot POINT 格式文件（`.dat`）。
///
/// 生成的文件遵循 Tecplot ASCII 格式规范，可直接被 Tecplot、ParaView
/// 以及本项目的 `lbm_post.vtk_reader.TecplotReader` 读取。
///
/// ## 文件格式示例
///
/// ```text
/// TITLE = "LBM Flow Field step=000100 time=100.000"
/// VARIABLES = "X" "Y" "RHO" "UX" "UY"
/// ZONE T="fluid", I=100, J=100, K=1, DATAPACKING=POINT, SOLUTIONTIME=100.0
/// 0.5 0.5 1.0001 0.0012 0.0003
/// 1.5 0.5 1.0002 0.0015 0.0004
/// ...
/// ```
///
/// 文件名格式：`<directory>/fluid_<NNNNNN>.dat`
pub fn write_snapshot_tecplot_asc(
    grid: &LbmGrid,
    step: u64,
    time: f64,
    directory: &str,
    part: Option<PartitionInfo>,
) -> Result<()> {
    let (grid_nx, phys_x0, phys_y0, out_nx, out_ny, x0_global, y0_global) =
        if let Some(p) = part {
            (grid.nx() as usize, p.phys_x0, p.phys_y0,
             p.local_nx, p.local_ny, p.x_start, p.y_start)
        } else {
            let nx = grid.nx() as usize;
            let ny = grid.ny() as usize;
            (nx, 0, 0, nx, ny, 0, 0)
        };

    let path = format!("{}/fluid_{:06}.dat", directory, step);
    let mut file = std::fs::File::create(&path)
        .with_context(|| format!("failed to create Tecplot ASCII file: {path}"))?;

    // 写文件头：标题、变量名、Zone 描述
    writeln!(file, "TITLE = \"LBM Flow Field step={step:06} time={time:.3}\"")?;
    writeln!(file, "VARIABLES = \"X\" \"Y\" \"RHO\" \"UX\" \"UY\"")?;
    writeln!(
        file,
        "ZONE T=\"fluid\", I={out_nx}, J={out_ny}, K=1, DATAPACKING=POINT, SOLUTIONTIME={time}"
    )?;

    // 逐节点写出物理节点数据（行主序：j 为外循环，i 为内循环）
    // 坐标为全局格子中心点：x = x_global + 0.5，y = y_global + 0.5
    for j in phys_y0..(phys_y0 + out_ny) {
        let local_j = j - phys_y0;   // 物理行偏移（loop 保证 j >= phys_y0）
        for i in phys_x0..(phys_x0 + out_nx) {
            let local_i = i - phys_x0; // 物理列偏移（loop 保证 i >= phys_x0）
            let idx = (j * grid_nx + i) as i32;
            let x   = (x0_global + local_i) as f64 + 0.5;
            let y   = (y0_global + local_j) as f64 + 0.5;
            let rho = grid.rho(idx);
            let ux  = grid.ux(idx);
            let uy  = grid.uy(idx);
            writeln!(file, "{x:.4} {y:.4} {rho:.8e} {ux:.8e} {uy:.8e}")?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// 二进制 Tecplot PLT 快照写出器（TDV112 格式）
// ---------------------------------------------------------------------------

/// 将一个欧拉场快照写为二进制 Tecplot PLT 格式文件（TDV112 版本）。
///
/// 生成的文件符合 Tecplot TDV112 二进制格式规范，可直接在 Tecplot 软件中打开，
/// 也可被本项目的 `lbm_post.vtk_reader.TecplotBinReader` 读取。
///
/// ## TDV112 格式结构（有序矩形网格）
///
/// ```text
/// 1. 魔数   "#!TDV112" + 0x01（小端 int = 1 表示字节序）
/// 2. 文件类型  0 = 全场（Grid + Solution）
/// 3. 标题字符串（空字符终止）
/// 4. 变量数量 N（本实现 N=5：X, Y, RHO, UX, UY）
/// 5. 变量名字符串（各自空字符终止）
/// 6. Zone 头（标记 = 299.0f32）
/// 7. EOH 标记（= 357.0f32）
/// 8. Zone 数据块（变量格式 = double = 2）
/// ```
///
/// 文件名格式：`<directory>/fluid_<NNNNNN>.plt`
pub fn write_snapshot_tecplot_bin(
    grid: &LbmGrid,
    step: u64,
    time: f64,
    directory: &str,
    part: Option<PartitionInfo>,
) -> Result<()> {
    use std::io::Write as _;

    let (grid_nx, phys_x0, phys_y0, out_nx, out_ny, x0_global, y0_global) =
        if let Some(p) = part {
            (grid.nx() as usize, p.phys_x0, p.phys_y0,
             p.local_nx, p.local_ny, p.x_start, p.y_start)
        } else {
            let nx = grid.nx() as usize;
            let ny = grid.ny() as usize;
            (nx, 0, 0, nx, ny, 0, 0)
        };
    let n = out_nx * out_ny;

    // 从 C++ 格子网格收集物理节点场数组（行主序）
    let mut rho_vec = Vec::with_capacity(n);
    let mut ux_vec  = Vec::with_capacity(n);
    let mut uy_vec  = Vec::with_capacity(n);
    for j in phys_y0..(phys_y0 + out_ny) {
        for i in phys_x0..(phys_x0 + out_nx) {
            let idx = (j * grid_nx + i) as i32;
            rho_vec.push(grid.rho(idx));
            ux_vec .push(grid.ux(idx));
            uy_vec .push(grid.uy(idx));
        }
    }

    let path = format!("{}/fluid_{:06}.plt", directory, step);
    let mut file = std::fs::File::create(&path)
        .with_context(|| format!("failed to create Tecplot binary file: {path}"))?;

    // -----------------------------------------------------------------------
    // 1. 魔数（8 字节 ASCII + 空终止）+ 字节序标志
    // -----------------------------------------------------------------------
    file.write_all(b"#!TDV112")?;
    // 字节序标志：1 = 小端（Intel 字节序）
    file.write_all(&1_i32.to_le_bytes())?;

    // -----------------------------------------------------------------------
    // 2. 文件类型（0 = 全场）
    // -----------------------------------------------------------------------
    file.write_all(&0_i32.to_le_bytes())?;

    // -----------------------------------------------------------------------
    // 3. 数据集标题（空终止字符串）
    // -----------------------------------------------------------------------
    let title = format!("LBM Flow Field step={step:06} time={time:.3}");
    write_tec_string(&mut file, &title)?;

    // -----------------------------------------------------------------------
    // 4. 变量数量
    // -----------------------------------------------------------------------
    let n_vars: i32 = 5; // X, Y, RHO, UX, UY
    file.write_all(&n_vars.to_le_bytes())?;

    // -----------------------------------------------------------------------
    // 5. 变量名（各自空终止字符串）
    // -----------------------------------------------------------------------
    for name in &["X", "Y", "RHO", "UX", "UY"] {
        write_tec_string(&mut file, name)?;
    }

    // -----------------------------------------------------------------------
    // 6. Zone 头（Zone 标记 299.0 + Zone 元信息）
    // -----------------------------------------------------------------------
    // Zone 标记
    file.write_all(&299.0_f32.to_le_bytes())?;
    // Zone 名称
    write_tec_string(&mut file, "fluid")?;
    // 父 Zone 索引（-1 = 无）
    file.write_all(&(-1_i32).to_le_bytes())?;
    // Strand ID（-1 = 自动）
    file.write_all(&(-1_i32).to_le_bytes())?;
    // 求解时间
    file.write_all(&time.to_le_bytes())?;
    // Zone 颜色（-1 = 自动）
    file.write_all(&(-1_i32).to_le_bytes())?;
    // Zone 类型（0 = Ordered 有序网格）
    file.write_all(&0_i32.to_le_bytes())?;
    // 变量位置（0 = 节点中心）
    file.write_all(&0_i32.to_le_bytes())?;
    // 是否提供原始邻居（0 = 否）
    file.write_all(&0_i32.to_le_bytes())?;
    // 用户定义面邻居连接数（0 = 无）
    file.write_all(&0_i32.to_le_bytes())?;
    // I-max, J-max, K-max（有序网格尺寸，仅物理节点）
    file.write_all(&(out_nx as i32).to_le_bytes())?;
    file.write_all(&(out_ny as i32).to_le_bytes())?;
    file.write_all(&1_i32.to_le_bytes())?;
    // 辅助数据对数（0 = 无）
    file.write_all(&0_i32.to_le_bytes())?;

    // -----------------------------------------------------------------------
    // 7. EOH（Header 结束标志 357.0）
    // -----------------------------------------------------------------------
    file.write_all(&357.0_f32.to_le_bytes())?;

    // -----------------------------------------------------------------------
    // 8. 数据区域（Zone 数据标记 + 各变量数据）
    // -----------------------------------------------------------------------
    // 区域数据标记
    file.write_all(&299.0_f32.to_le_bytes())?;

    // 各变量的数据格式（2 = float64 double，对应所有 5 个变量）
    for _ in 0..n_vars {
        file.write_all(&2_i32.to_le_bytes())?; // 2 = IEEE双精度浮点
    }

    // 是否有被动变量（0 = 无）
    file.write_all(&0_i32.to_le_bytes())?;
    // 是否有变量共享（0 = 无）
    file.write_all(&0_i32.to_le_bytes())?;
    // 共享连接的 Zone 编号（-1 = 不共享）
    file.write_all(&(-1_i32).to_le_bytes())?;

    // 变量 1：X 坐标（全局格子中心：x_global + 0.5）
    for _j in 0..out_ny {
        for i in 0..out_nx {
            let x = (x0_global + i) as f64 + 0.5;
            file.write_all(&x.to_le_bytes())?;
        }
    }
    // 变量 2：Y 坐标（全局格子中心：y_global + 0.5）
    for j in 0..out_ny {
        let y = (y0_global + j) as f64 + 0.5;
        for _ in 0..out_nx {
            file.write_all(&y.to_le_bytes())?;
        }
    }
    // 变量 3：密度 ρ
    for v in &rho_vec {
        file.write_all(&v.to_le_bytes())?;
    }
    // 变量 4：x 方向速度 ux
    for v in &ux_vec {
        file.write_all(&v.to_le_bytes())?;
    }
    // 变量 5：y 方向速度 uy
    for v in &uy_vec {
        file.write_all(&v.to_le_bytes())?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tecplot 字符串写出辅助函数
// ---------------------------------------------------------------------------

/// 将 UTF-8 字符串按 Tecplot TDV112 格式写出：
/// 逐字符以 i32 小端写出 ASCII 码，末尾追加 0（空字符终止符）。
fn write_tec_string(file: &mut std::fs::File, s: &str) -> Result<()> {
    use std::io::Write as _;
    for ch in s.chars() {
        file.write_all(&(ch as i32).to_le_bytes())?;
    }
    // 空字符终止
    file.write_all(&0_i32.to_le_bytes())?;
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
// CSV 监控日志内存缓冲区
// ---------------------------------------------------------------------------

/// In-memory write buffer for the CSV monitor log.
///
/// `append_monitor_csv` opens, writes, and closes the output file on every
/// call.  When called once per time step (typical for kinetic-energy
/// monitoring), this produces O(n_steps) file-system syscalls per MPI rank,
/// causing measurable CPU-utilization spikes between compute phases.
///
/// `CsvBuffer` accumulates data rows in memory and writes them all in a
/// single bulk operation when `flush` is called.  The caller should flush
/// at snapshot intervals (`write_interval`) and at the end of the simulation
/// to keep memory consumption bounded.
#[allow(dead_code)]
pub struct CsvBuffer {
    path:   String,
    /// Lazily-built header line (set on first `push_row` call).
    header: Option<String>,
    /// Accumulated formatted data rows (no trailing newline).
    rows:   Vec<String>,
}

#[allow(dead_code)]
impl CsvBuffer {
    /// Create a new, empty buffer targeting `path`.
    pub fn new(path: &str) -> Self {
        Self { path: path.to_owned(), header: None, rows: Vec::new() }
    }

    /// Append one data row to the in-memory buffer (no I/O).
    ///
    /// The header is built lazily from `fields` column names the first time
    /// this method is called — subsequent calls must supply the same columns
    /// in the same order.
    pub fn push_row(&mut self, step: u64, time: f64, fields: &[(&str, f64)]) {
        if self.header.is_none() {
            let mut h = "step,time".to_string();
            for (name, _) in fields {
                h.push(',');
                h.push_str(name);
            }
            self.header = Some(h);
        }
        let mut row = format!("{step},{time:.6}");
        for (_, v) in fields {
            row.push_str(&format!(",{v:.10e}"));
        }
        self.rows.push(row);
    }

    /// Write all buffered rows to the target CSV file in one operation.
    ///
    /// * If the file does not yet exist the header line is prepended.
    /// * The internal buffer is cleared on success.
    /// * Returns immediately (no-op) when the buffer is empty.
    pub fn flush(&mut self) -> Result<()> {
        if self.rows.is_empty() {
            return Ok(());
        }
        let p = Path::new(&self.path);
        let needs_header = !p.exists();
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(p)
            .with_context(|| format!("Cannot open monitor CSV: {}", self.path))?;

        if needs_header {
            if let Some(ref h) = self.header {
                file.write_all(h.as_bytes())?;
                file.write_all(b"\n")?;
            }
        }
        // Write all buffered rows in one call to minimise syscall count.
        let mut content = String::with_capacity(self.rows.len() * 64);
        for row in &self.rows {
            content.push_str(row);
            content.push('\n');
        }
        file.write_all(content.as_bytes())?;
        self.rows.clear();
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// MPI 块合并 → 全局场 gather 工具
// ---------------------------------------------------------------------------

/// 将 MPI 各进程的物理分区数据 gather 到 rank-0，返回全局场数组（行主序）。
///
/// 若 `nprocs == 1`（无 MPI）则直接克隆本进程数据并返回。
///
/// # 参数
///
/// * `local_data`   — 本进程的物理场数组（行主序，长度 = `local_nx * local_ny`）
/// * `part`         — 本进程的分区信息（含 global_nx/ny 及 x_start/y_start）
/// * `root`         — gather 目标进程编号（通常为 0）
///
/// # 返回
///
/// * `Some((rho, global_nx, global_ny))` — root 进程返回全局拼合数组及尺寸
/// * `None`                              — 非 root 进程返回 None
pub fn gather_field_to_root(
    local_data: &[f64],
    part: &PartitionInfo,
    root: i32,
) -> Option<(Vec<f64>, usize, usize)> {
    let rank   = lbm_bindings::mpi_rank();
    let nprocs = lbm_bindings::mpi_size();

    let gnx    = part.global_nx;
    let gny    = part.global_ny;
    let local_count = local_data.len() as i32;

    if nprocs == 1 {
        // 单进程：直接构造全局数组（x_start/y_start 可能非零，但单进程时应为 0）
        return Some((local_data.to_vec(), gnx, gny));
    }

    // Step 1：每进程将本地 count 和 x_start/y_start/local_nx/local_ny gather 到 root
    let counts = lbm_bindings::mpi_gather_int(local_count, root);
    let x_starts  = lbm_bindings::mpi_gather_int(part.x_start  as i32, root);
    let y_starts  = lbm_bindings::mpi_gather_int(part.y_start  as i32, root);
    let local_nxs = lbm_bindings::mpi_gather_int(part.local_nx as i32, root);
    let local_nys = lbm_bindings::mpi_gather_int(part.local_ny as i32, root);

    if rank == root {
        let nprocs_u = nprocs as usize;
        let displs: Vec<i32> = counts.iter()
            .scan(0i32, |acc, &c| { let d = *acc; *acc += c; Some(d) })
            .collect();

        // Step 2：MPI_Gatherv 收集所有分区数据
        let flat = lbm_bindings::mpi_gatherv_f64(local_data, &counts, &displs, root);

        // Step 3：将各分区数据按 (x_start, y_start) 放置到全局数组中
        // 全局数组布局：行主序，索引 = j * gnx + i
        let mut global = vec![0.0f64; gnx * gny];
        let mut offset = 0usize;
        for r in 0..nprocs_u {
            let nx_r = local_nxs[r] as usize;
            let ny_r = local_nys[r] as usize;
            let xs   = x_starts[r] as usize;
            let ys   = y_starts[r] as usize;
            let cnt  = (counts[r] as usize).min(nx_r * ny_r);
            for row in 0..ny_r {
                for col in 0..nx_r {
                    let local_idx  = row * nx_r + col;
                    let global_idx = (ys + row) * gnx + (xs + col);
                    if local_idx + offset < flat.len() && global_idx < global.len() {
                        global[global_idx] = flat[offset + local_idx];
                    }
                }
            }
            offset += cnt;
        }

        Some((global, gnx, gny))
    } else {
        // 非 root 进程：仍需调用 gatherv（参与通信），但不返回数据
        lbm_bindings::mpi_gatherv_f64(local_data, &[], &[], root);
        None
    }
}

// ---------------------------------------------------------------------------
// 全局合并快照写出器（供 combine_blocks=true 时由 rank-0 调用）
// ---------------------------------------------------------------------------

/// 将预先 gather 好的全局场写出为 NPZ 格式。
///
/// 与 [`write_snapshot_npz`] 不同，此函数直接接受全局场数组，不再从 LbmGrid 读取。
/// 文件写到 `<directory>/fluid_<NNNNNN>.npz`（不含 rank 子目录）。
pub fn write_global_snapshot_npz(
    global_rho: &[f64],
    global_ux:  &[f64],
    global_uy:  &[f64],
    global_nx:  usize,
    global_ny:  usize,
    step: u64,
    time: f64,
    directory: &str,
) -> Result<()> {
    let path = format!("{}/fluid_{:06}.npz", directory, step);
    let file = std::fs::File::create(&path)
        .with_context(|| format!("Cannot create combined snapshot file: {path}"))?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);

    zip.start_file("rho.npy", options)?;
    zip.write_all(&npy_f64(global_rho, &[global_ny, global_nx]))?;

    zip.start_file("ux.npy", options)?;
    zip.write_all(&npy_f64(global_ux, &[global_ny, global_nx]))?;

    zip.start_file("uy.npy", options)?;
    zip.write_all(&npy_f64(global_uy, &[global_ny, global_nx]))?;

    zip.start_file("step.npy", options)?;
    zip.write_all(&npy_i64(&[step as i64], &[]))?;

    zip.start_file("time.npy", options)?;
    zip.write_all(&npy_f64(&[time], &[]))?;

    zip.finish()?;
    Ok(())
}

/// 将预先 gather 好的全局场写出为 ASCII Tecplot (.dat) 格式。
pub fn write_global_snapshot_tecplot_asc(
    global_rho: &[f64],
    global_ux:  &[f64],
    global_uy:  &[f64],
    global_nx:  usize,
    global_ny:  usize,
    step: u64,
    time: f64,
    directory: &str,
) -> Result<()> {
    let path = format!("{}/fluid_{:06}.dat", directory, step);
    let mut file = std::fs::File::create(&path)
        .with_context(|| format!("failed to create combined Tecplot ASCII file: {path}"))?;

    writeln!(file, "TITLE = \"LBM Flow Field step={step:06} time={time:.3}\"")?;
    writeln!(file, "VARIABLES = \"X\" \"Y\" \"RHO\" \"UX\" \"UY\"")?;
    writeln!(
        file,
        "ZONE T=\"fluid\", I={global_nx}, J={global_ny}, K=1, DATAPACKING=POINT, SOLUTIONTIME={time}"
    )?;

    for j in 0..global_ny {
        for i in 0..global_nx {
            let idx = j * global_nx + i;
            let x   = i as f64 + 0.5;
            let y   = j as f64 + 0.5;
            writeln!(
                file,
                "{x:.4} {y:.4} {rho:.8e} {ux:.8e} {uy:.8e}",
                rho = global_rho[idx],
                ux  = global_ux[idx],
                uy  = global_uy[idx],
            )?;
        }
    }
    Ok(())
}

/// 将预先 gather 好的全局场写出为二进制 Tecplot PLT（TDV112）格式。
pub fn write_global_snapshot_tecplot_bin(
    global_rho: &[f64],
    global_ux:  &[f64],
    global_uy:  &[f64],
    global_nx:  usize,
    global_ny:  usize,
    step: u64,
    time: f64,
    directory: &str,
) -> Result<()> {
    use std::io::Write as _;

    let path = format!("{}/fluid_{:06}.plt", directory, step);
    let mut file = std::fs::File::create(&path)
        .with_context(|| format!("failed to create combined Tecplot binary file: {path}"))?;

    // 魔数 + 字节序标志
    file.write_all(b"#!TDV112")?;
    file.write_all(&1_i32.to_le_bytes())?;
    // 文件类型
    file.write_all(&0_i32.to_le_bytes())?;
    // 数据集标题
    let title = format!("LBM Flow Field step={step:06} time={time:.3}");
    write_tec_string(&mut file, &title)?;
    // 变量数量（5：X Y RHO UX UY）
    file.write_all(&5_i32.to_le_bytes())?;
    for name in &["X", "Y", "RHO", "UX", "UY"] {
        write_tec_string(&mut file, name)?;
    }
    // Zone 头
    file.write_all(&299.0_f32.to_le_bytes())?;
    write_tec_string(&mut file, "fluid")?;
    file.write_all(&(-1_i32).to_le_bytes())?;
    file.write_all(&(-1_i32).to_le_bytes())?;
    file.write_all(&time.to_le_bytes())?;
    file.write_all(&(-1_i32).to_le_bytes())?;
    file.write_all(&0_i32.to_le_bytes())?;
    file.write_all(&0_i32.to_le_bytes())?;
    file.write_all(&0_i32.to_le_bytes())?;
    file.write_all(&0_i32.to_le_bytes())?;
    file.write_all(&(global_nx as i32).to_le_bytes())?;
    file.write_all(&(global_ny as i32).to_le_bytes())?;
    file.write_all(&1_i32.to_le_bytes())?;
    file.write_all(&0_i32.to_le_bytes())?;
    // EOH
    file.write_all(&357.0_f32.to_le_bytes())?;
    // 数据区域
    file.write_all(&299.0_f32.to_le_bytes())?;
    for _ in 0..5 { file.write_all(&2_i32.to_le_bytes())?; }
    file.write_all(&0_i32.to_le_bytes())?;
    file.write_all(&0_i32.to_le_bytes())?;
    file.write_all(&(-1_i32).to_le_bytes())?;
    // X 坐标
    for _j in 0..global_ny {
        for i in 0..global_nx {
            file.write_all(&(i as f64 + 0.5).to_le_bytes())?;
        }
    }
    // Y 坐标
    for j in 0..global_ny {
        let y = j as f64 + 0.5;
        for _ in 0..global_nx { file.write_all(&y.to_le_bytes())?; }
    }
    // RHO
    for v in global_rho { file.write_all(&v.to_le_bytes())?; }
    // UX
    for v in global_ux  { file.write_all(&v.to_le_bytes())?; }
    // UY
    for v in global_uy  { file.write_all(&v.to_le_bytes())?; }

    Ok(())
}

// ---------------------------------------------------------------------------
// 基于预提取数组的快照写出器（异步模式用，无需访问 LbmGrid）
// ---------------------------------------------------------------------------

/// 从预先提取的物理场数组写出 NPZ 格式快照（不访问 LbmGrid）。
pub fn write_snapshot_npz_raw(
    rho: &[f64], ux: &[f64], uy: &[f64],
    nx: usize, ny: usize,
    step: u64, time: f64, directory: &str,
    part: Option<PartitionInfo>,
) -> Result<()> {
    let path = format!("{}/fluid_{:06}.npz", directory, step);
    let file = std::fs::File::create(&path)
        .with_context(|| format!("Cannot create snapshot file: {path}"))?;
    let mut zip = zip::ZipWriter::new(file);
    let options = zip::write::SimpleFileOptions::default()
        .compression_method(zip::CompressionMethod::Deflated);
    zip.start_file("rho.npy", options)?;
    zip.write_all(&npy_f64(rho, &[ny, nx]))?;
    zip.start_file("ux.npy", options)?;
    zip.write_all(&npy_f64(ux, &[ny, nx]))?;
    zip.start_file("uy.npy", options)?;
    zip.write_all(&npy_f64(uy, &[ny, nx]))?;
    zip.start_file("step.npy", options)?;
    zip.write_all(&npy_i64(&[step as i64], &[]))?;
    zip.start_file("time.npy", options)?;
    zip.write_all(&npy_f64(&[time], &[]))?;
    if let Some(p) = part {
        zip.start_file("x_start.npy", options)?;
        zip.write_all(&npy_i64(&[p.x_start as i64], &[]))?;
        zip.start_file("y_start.npy", options)?;
        zip.write_all(&npy_i64(&[p.y_start as i64], &[]))?;
        zip.start_file("global_nx.npy", options)?;
        zip.write_all(&npy_i64(&[p.global_nx as i64], &[]))?;
        zip.start_file("global_ny.npy", options)?;
        zip.write_all(&npy_i64(&[p.global_ny as i64], &[]))?;
    }
    zip.finish()?;
    Ok(())
}

/// 从预先提取的物理场数组写出 ASCII Tecplot (.dat) 格式快照。
pub fn write_snapshot_tecplot_asc_raw(
    rho: &[f64], ux: &[f64], uy: &[f64],
    nx: usize, ny: usize,
    step: u64, time: f64, directory: &str,
    part: Option<PartitionInfo>,
) -> Result<()> {
    let (x0, y0) = part.map(|p| (p.x_start, p.y_start)).unwrap_or((0, 0));
    let path = format!("{}/fluid_{:06}.dat", directory, step);
    let mut file = std::fs::File::create(&path)
        .with_context(|| format!("failed to create Tecplot ASCII file: {path}"))?;
    writeln!(file, "TITLE = \"LBM Flow Field step={step:06} time={time:.3}\"")?;
    writeln!(file, "VARIABLES = \"X\" \"Y\" \"RHO\" \"UX\" \"UY\"")?;
    writeln!(file, "ZONE T=\"fluid\", I={nx}, J={ny}, K=1, DATAPACKING=POINT, SOLUTIONTIME={time}")?;
    for j in 0..ny {
        for i in 0..nx {
            let idx = j * nx + i;
            writeln!(file, "{:.4} {:.4} {:.8e} {:.8e} {:.8e}",
                (x0 + i) as f64 + 0.5, (y0 + j) as f64 + 0.5,
                rho[idx], ux[idx], uy[idx])?;
        }
    }
    Ok(())
}

/// 从预先提取的物理场数组写出二进制 Tecplot PLT（TDV112）格式快照。
pub fn write_snapshot_tecplot_bin_raw(
    rho: &[f64], ux: &[f64], uy: &[f64],
    nx: usize, ny: usize,
    step: u64, time: f64, directory: &str,
    part: Option<PartitionInfo>,
) -> Result<()> {
    use std::io::Write as _;
    let (x0, y0) = part.map(|p| (p.x_start, p.y_start)).unwrap_or((0, 0));
    let path = format!("{}/fluid_{:06}.plt", directory, step);
    let mut file = std::fs::File::create(&path)
        .with_context(|| format!("failed to create Tecplot binary file: {path}"))?;
    file.write_all(b"#!TDV112")?;
    file.write_all(&1_i32.to_le_bytes())?;
    file.write_all(&0_i32.to_le_bytes())?;
    let title = format!("LBM Flow Field step={step:06} time={time:.3}");
    write_tec_string(&mut file, &title)?;
    file.write_all(&5_i32.to_le_bytes())?;
    for name in &["X", "Y", "RHO", "UX", "UY"] { write_tec_string(&mut file, name)?; }
    file.write_all(&299.0_f32.to_le_bytes())?;
    write_tec_string(&mut file, "fluid")?;
    for v in &[-1_i32, -1_i32] { file.write_all(&v.to_le_bytes())?; }
    file.write_all(&time.to_le_bytes())?;
    for v in &[-1_i32, 0_i32, 0_i32, 0_i32, 0_i32] { file.write_all(&v.to_le_bytes())?; }
    file.write_all(&(nx as i32).to_le_bytes())?;
    file.write_all(&(ny as i32).to_le_bytes())?;
    file.write_all(&1_i32.to_le_bytes())?;
    file.write_all(&0_i32.to_le_bytes())?;
    file.write_all(&357.0_f32.to_le_bytes())?;
    file.write_all(&299.0_f32.to_le_bytes())?;
    for _ in 0..5 { file.write_all(&2_i32.to_le_bytes())?; }
    file.write_all(&0_i32.to_le_bytes())?;
    file.write_all(&0_i32.to_le_bytes())?;
    file.write_all(&(-1_i32).to_le_bytes())?;
    for _j in 0..ny {
        for i in 0..nx { file.write_all(&((x0 + i) as f64 + 0.5).to_le_bytes())?; }
    }
    for j in 0..ny {
        let y = (y0 + j) as f64 + 0.5;
        for _ in 0..nx { file.write_all(&y.to_le_bytes())?; }
    }
    for v in rho { file.write_all(&v.to_le_bytes())?; }
    for v in ux  { file.write_all(&v.to_le_bytes())?; }
    for v in uy  { file.write_all(&v.to_le_bytes())?; }
    Ok(())
}

/// 根据格式字符串从预提取数组写出快照（异步模式通用入口）。
pub fn write_snapshot_raw(
    format: &str,
    rho: &[f64], ux: &[f64], uy: &[f64],
    nx: usize, ny: usize,
    step: u64, time: f64, directory: &str,
    part: Option<PartitionInfo>,
) -> Result<()> {
    match format {
        "tecplot_asc" => write_snapshot_tecplot_asc_raw(rho, ux, uy, nx, ny, step, time, directory, part),
        "tecplot_bin" => write_snapshot_tecplot_bin_raw(rho, ux, uy, nx, ny, step, time, directory, part),
        _             => write_snapshot_npz_raw        (rho, ux, uy, nx, ny, step, time, directory, part),
    }
}

/// 根据格式字符串写出全局（combine_blocks gather 后）快照的通用入口。
pub fn write_global_snapshot_raw(
    format: &str,
    rho: &[f64], ux: &[f64], uy: &[f64],
    nx: usize, ny: usize,
    step: u64, time: f64, directory: &str,
) -> Result<()> {
    match format {
        "tecplot_asc" => write_global_snapshot_tecplot_asc(rho, ux, uy, nx, ny, step, time, directory),
        "tecplot_bin" => write_global_snapshot_tecplot_bin(rho, ux, uy, nx, ny, step, time, directory),
        _             => write_global_snapshot_npz        (rho, ux, uy, nx, ny, step, time, directory),
    }
}

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

    /// 验证 Tecplot ASCII 格式的关键结构：标题行、变量行、Zone 行。
    /// 本测试不需要真实 LbmGrid，仅验证辅助函数生成的字符串格式正确。
    #[test]
    fn test_tecplot_asc_header_structure() {
        // 模拟手动拼装一段 ASCII Tecplot 输出（格式正确性校验）
        let step: u64 = 100;
        let time: f64 = 100.0;
        let nx: usize = 4;
        let ny: usize = 4;

        let dir = std::env::temp_dir().join(format!(
            "lbm_tecplot_asc_test_{}", std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let dat_path = dir.join("test.dat");

        // 手动写出一个最小 Tecplot ASCII 文件，验证格式
        let mut f = std::fs::File::create(&dat_path).unwrap();
        writeln!(f, "TITLE = \"LBM Flow Field step={step:06} time={time:.3}\"").unwrap();
        writeln!(f, "VARIABLES = \"X\" \"Y\" \"RHO\" \"UX\" \"UY\"").unwrap();
        writeln!(f, "ZONE T=\"fluid\", I={nx}, J={ny}, K=1, DATAPACKING=POINT, SOLUTIONTIME={time}").unwrap();
        for j in 0..ny {
            for i in 0..nx {
                writeln!(f, "{:.4} {:.4} 1.00000000e0 0.00000000e0 0.00000000e0",
                    i as f64 + 0.5, j as f64 + 0.5).unwrap();
            }
        }
        drop(f);

        let text = std::fs::read_to_string(&dat_path).unwrap();
        // 验证标题行
        assert!(text.contains("TITLE = \"LBM Flow Field step=000100 time=100.000\""));
        // 验证变量行
        assert!(text.contains("VARIABLES = \"X\" \"Y\" \"RHO\" \"UX\" \"UY\""));
        // 验证 Zone 行包含 I=4, J=4
        assert!(text.contains("I=4"));
        assert!(text.contains("J=4"));
        // 验证数据行总数（ny * nx = 16）
        let data_lines = text.lines().skip(3).count();
        assert_eq!(data_lines, ny * nx);
    }

    /// 验证 Tecplot 二进制格式的魔数和字节序标志。
    #[test]
    fn test_tecplot_bin_magic_number() {
        let dir = std::env::temp_dir().join(format!(
            "lbm_tecplot_bin_test_{}", std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let plt_path = dir.join("test.plt");

        // 手动写出 TDV112 文件头并验证
        let mut f = std::fs::File::create(&plt_path).unwrap();
        // 魔数
        f.write_all(b"#!TDV112").unwrap();
        // 字节序标志 = 1
        f.write_all(&1_i32.to_le_bytes()).unwrap();
        drop(f);

        let bytes = std::fs::read(&plt_path).unwrap();
        // 验证魔数（前 8 字节）
        assert_eq!(&bytes[0..8], b"#!TDV112");
        // 验证字节序标志（字节 8-11，值 = 1）
        let byte_order = i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(byte_order, 1);
    }

    /// 验证 write_tec_string 将字符串按 i32 字符码序列写出，末尾有空字符。
    #[test]
    fn test_write_tec_string_null_terminated() {
        let dir = std::env::temp_dir().join(format!(
            "lbm_tecplot_str_test_{}", std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("tec_str.bin");
        let mut f = std::fs::File::create(&p).unwrap();
        write_tec_string(&mut f, "AB").unwrap();
        drop(f);

        let bytes = std::fs::read(&p).unwrap();
        // "AB" → [65, 0, 0, 0,  66, 0, 0, 0,  0, 0, 0, 0]
        assert_eq!(bytes.len(), 12); // 3 × i32
        let a = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        let b = i32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]);
        let z = i32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        assert_eq!(a, 'A' as i32);
        assert_eq!(b, 'B' as i32);
        assert_eq!(z, 0); // 空字符终止
    }

    /// Verify that CsvBuffer accumulates rows in memory and only writes on flush.
    #[test]
    fn test_csv_buffer_flush() {
        let dir = std::env::temp_dir().join(format!(
            "lbm_csvbuf_test_{}", std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let csv_path = dir.join("buf_monitor.csv");
        let p = csv_path.to_str().unwrap();

        let mut buf = CsvBuffer::new(p);

        // Before flush the file must not exist
        assert!(!csv_path.exists(), "file should not be created before flush");

        buf.push_row(1, 0.1, &[("ke", 1.0e-3)]);
        buf.push_row(2, 0.2, &[("ke", 2.0e-3)]);

        // Still no file
        assert!(!csv_path.exists(), "file should not be created before flush");

        buf.flush().unwrap();

        // Now the file should exist with header + 2 data rows
        let text = std::fs::read_to_string(&csv_path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3, "expected header + 2 rows");
        assert!(lines[0].starts_with("step,time,ke"));
        assert!(lines[1].starts_with("1,0.100000"));
        assert!(lines[2].starts_with("2,0.200000"));

        // Buffer should be empty after flush; a second flush is a no-op
        buf.flush().unwrap();
        let text2 = std::fs::read_to_string(&csv_path).unwrap();
        assert_eq!(text, text2, "second flush should not add rows");

        // Rows pushed after flush append correctly
        buf.push_row(3, 0.3, &[("ke", 3.0e-3)]);
        buf.flush().unwrap();
        let text3 = std::fs::read_to_string(&csv_path).unwrap();
        let lines3: Vec<&str> = text3.lines().collect();
        assert_eq!(lines3.len(), 4, "expected header + 3 rows after second flush");
    }

    /// Verify that write_snapshot_raw dispatches to the correct format.
    #[test]
    fn test_write_snapshot_raw_dispatch() {
        let dir = std::env::temp_dir().join(format!(
            "lbm_raw_snapshot_test_{}", std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let dir_str = dir.to_str().unwrap();

        let rho = vec![1.0f64; 4];
        let ux  = vec![0.1f64; 4];
        let uy  = vec![0.0f64; 4];

        // NPZ 格式
        write_snapshot_raw("npz", &rho, &ux, &uy, 2, 2, 1, 0.0, dir_str, None).unwrap();
        assert!(dir.join("fluid_000001.npz").exists(), "npz file should be created");

        // Tecplot ASCII 格式
        write_snapshot_raw("tecplot_asc", &rho, &ux, &uy, 2, 2, 2, 1.0, dir_str, None).unwrap();
        assert!(dir.join("fluid_000002.dat").exists(), "dat file should be created");

        // Tecplot binary 格式
        write_snapshot_raw("tecplot_bin", &rho, &ux, &uy, 2, 2, 3, 2.0, dir_str, None).unwrap();
        assert!(dir.join("fluid_000003.plt").exists(), "plt file should be created");
    }

    /// Verify that write_global_snapshot_raw dispatches correctly.
    #[test]
    fn test_write_global_snapshot_raw_dispatch() {
        let dir = std::env::temp_dir().join(format!(
            "lbm_global_raw_test_{}", std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let dir_str = dir.to_str().unwrap();

        let rho = vec![1.0f64; 4];
        let ux  = vec![0.1f64; 4];
        let uy  = vec![0.0f64; 4];

        write_global_snapshot_raw("npz", &rho, &ux, &uy, 2, 2, 1, 0.0, dir_str).unwrap();
        assert!(dir.join("fluid_000001.npz").exists());

        write_global_snapshot_raw("tecplot_asc", &rho, &ux, &uy, 2, 2, 2, 1.0, dir_str).unwrap();
        assert!(dir.join("fluid_000002.dat").exists());

        write_global_snapshot_raw("tecplot_bin", &rho, &ux, &uy, 2, 2, 3, 2.0, dir_str).unwrap();
        assert!(dir.join("fluid_000003.plt").exists());
    }
}
