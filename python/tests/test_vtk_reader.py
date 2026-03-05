"""
测试 lbm_post.vtk_reader —— 快照数据结构与 I/O（NPZ、ASCII Tecplot、二进制 Tecplot）。
"""

import numpy as np
import pytest

from lbm_post.vtk_reader import (
    FieldSnapshot,
    MarkerSnapshot,
    NpzReader,
    TecplotAscReader,
    TecplotBinReader,
    load_snapshot,
    make_synthetic_lid_cavity,
    save_snapshot_npz,
)


# ---------------------------------------------------------------------------
# FieldSnapshot derived quantities
# ---------------------------------------------------------------------------

class TestFieldSnapshotDerivatives:
    @pytest.fixture
    def snap(self):
        return make_synthetic_lid_cavity(nx=32, ny=32)

    def test_velocity_magnitude_shape(self, snap):
        mag = snap.velocity_magnitude()
        assert mag.shape == (snap.ny, snap.nx)

    def test_velocity_magnitude_non_negative(self, snap):
        assert (snap.velocity_magnitude() >= 0).all()

    def test_vorticity_shape(self, snap):
        omega = snap.vorticity()
        assert omega.shape == (snap.ny, snap.nx)

    def test_pressure_shape(self, snap):
        p = snap.pressure()
        assert p.shape == (snap.ny, snap.nx)

    def test_pressure_proportional_to_rho(self, snap):
        cs2 = 1.0 / 3.0
        p = snap.pressure(cs2=cs2)
        np.testing.assert_allclose(p, cs2 * snap.rho)

    def test_stream_function_shape(self, snap):
        psi = snap.stream_function()
        assert psi.shape == (snap.ny, snap.nx)

    def test_stream_function_zero_bottom(self, snap):
        psi = snap.stream_function()
        np.testing.assert_allclose(psi[0, :], 0.0)


# ---------------------------------------------------------------------------
# NpzReader round-trip
# ---------------------------------------------------------------------------

class TestNpzReader:
    def test_save_and_load(self, tmp_path):
        snap = make_synthetic_lid_cavity(nx=16, ny=16, step=500)
        path = save_snapshot_npz(snap, tmp_path / "fluid_000500.npz")
        assert path.exists()

        reader = NpzReader(tmp_path)
        assert len(reader) == 1

        loaded = reader.read(path)
        assert loaded.step == 500
        assert loaded.nx == 16
        assert loaded.ny == 16
        np.testing.assert_allclose(loaded.rho, snap.rho)
        np.testing.assert_allclose(loaded.ux,  snap.ux)
        np.testing.assert_allclose(loaded.uy,  snap.uy)

    def test_last(self, tmp_path):
        for step in [100, 500, 1000]:
            snap = make_synthetic_lid_cavity(step=step)
            save_snapshot_npz(snap, tmp_path / f"fluid_{step:06d}.npz")

        reader = NpzReader(tmp_path)
        last = reader.last()
        assert last.step == 1000

    def test_steps_list(self, tmp_path):
        for step in [200, 400]:
            snap = make_synthetic_lid_cavity(step=step)
            save_snapshot_npz(snap, tmp_path / f"fluid_{step:06d}.npz")

        reader = NpzReader(tmp_path)
        assert reader.steps() == [200, 400]

    def test_empty_directory_raises(self, tmp_path):
        reader = NpzReader(tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.last()

    def test_iteration(self, tmp_path):
        steps_written = [100, 200, 300]
        for s in steps_written:
            save_snapshot_npz(make_synthetic_lid_cavity(step=s),
                              tmp_path / f"fluid_{s:06d}.npz")
        steps_read = [snap.step for snap in NpzReader(tmp_path)]
        assert steps_read == steps_written

    def test_with_forces(self, tmp_path):
        snap = make_synthetic_lid_cavity(nx=8, ny=8, step=0)
        snap.force_x = np.ones((8, 8)) * 0.001
        snap.force_y = np.ones((8, 8)) * 0.002
        path = save_snapshot_npz(snap, tmp_path / "fluid_000000.npz")
        loaded = NpzReader(tmp_path).read(path)
        assert loaded.force_x is not None
        np.testing.assert_allclose(loaded.force_x, snap.force_x)


# ---------------------------------------------------------------------------
# load_snapshot auto-detect
# ---------------------------------------------------------------------------

class TestLoadSnapshot:
    def test_load_npz(self, tmp_path):
        snap = make_synthetic_lid_cavity(step=42)
        path = save_snapshot_npz(snap, tmp_path / "fluid_000042.npz")
        loaded = load_snapshot(path)
        assert loaded.step == 42

    def test_unsupported_format_raises(self, tmp_path):
        p = tmp_path / "data.xyz"
        p.write_text("dummy")
        with pytest.raises(ValueError, match="不支持的快照格式"):
            load_snapshot(p)


# ---------------------------------------------------------------------------
# make_synthetic_lid_cavity
# ---------------------------------------------------------------------------

class TestSyntheticData:
    def test_shape(self):
        snap = make_synthetic_lid_cavity(nx=48, ny=32)
        assert snap.rho.shape == (32, 48)
        assert snap.ux.shape == (32, 48)

    def test_density_uniform(self):
        snap = make_synthetic_lid_cavity()
        np.testing.assert_allclose(snap.rho, 1.0)

    def test_step_field(self):
        snap = make_synthetic_lid_cavity(step=9999)
        assert snap.step == 9999


# ---------------------------------------------------------------------------
# ASCII Tecplot 读取器（.dat）
# ---------------------------------------------------------------------------

def _write_tecplot_asc(path, snap: FieldSnapshot) -> None:
    """将 FieldSnapshot 写为 ASCII Tecplot .dat 文件（格式与 Rust 输出一致）。"""
    nx, ny = snap.nx, snap.ny
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'TITLE = "LBM Flow Field step={snap.step:06} time={snap.time:.3f}"\n')
        f.write('VARIABLES = "X" "Y" "RHO" "UX" "UY"\n')
        f.write(f"ZONE T=\"fluid\", I={nx}, J={ny}, K=1, DATAPACKING=POINT, SOLUTIONTIME={snap.time}\n")
        for j in range(ny):
            for i in range(nx):
                x = i + 0.5
                y = j + 0.5
                f.write(f"{x:.4f} {y:.4f} {snap.rho[j,i]:.8e} {snap.ux[j,i]:.8e} {snap.uy[j,i]:.8e}\n")


class TestTecplotAscReader:
    def test_round_trip(self, tmp_path):
        """写出 ASCII Tecplot 后读回，验证场数据一致。"""
        snap = make_synthetic_lid_cavity(nx=8, ny=8, step=200)
        dat_path = tmp_path / "fluid_000200.dat"
        _write_tecplot_asc(dat_path, snap)

        reader = TecplotAscReader(tmp_path)
        assert len(reader) == 1
        loaded = reader.read(200)

        assert loaded.step == 200
        assert loaded.nx == 8
        assert loaded.ny == 8
        np.testing.assert_allclose(loaded.rho, snap.rho, rtol=1e-6)
        np.testing.assert_allclose(loaded.ux,  snap.ux,  rtol=1e-6)
        np.testing.assert_allclose(loaded.uy,  snap.uy,  rtol=1e-6)

    def test_load_snapshot_dat(self, tmp_path):
        """load_snapshot 自动检测 .dat 格式。"""
        snap = make_synthetic_lid_cavity(nx=4, ny=4, step=10)
        dat_path = tmp_path / "fluid_000010.dat"
        _write_tecplot_asc(dat_path, snap)
        loaded = load_snapshot(dat_path)
        assert loaded.step == 10

    def test_empty_directory_raises(self, tmp_path):
        reader = TecplotAscReader(tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.last()

    def test_steps_list(self, tmp_path):
        for step in [100, 200]:
            snap = make_synthetic_lid_cavity(step=step)
            _write_tecplot_asc(tmp_path / f"fluid_{step:06d}.dat", snap)
        reader = TecplotAscReader(tmp_path)
        assert reader.steps() == [100, 200]


# ---------------------------------------------------------------------------
# 二进制 Tecplot 读取器（.plt，TDV112）
# ---------------------------------------------------------------------------

def _write_tecplot_bin(path, snap: FieldSnapshot) -> None:
    """
    将 FieldSnapshot 写为二进制 Tecplot TDV112 .plt 文件。
    格式与 Rust output::write_snapshot_tecplot_bin 完全一致。
    """
    import struct
    nx, ny = snap.nx, snap.ny
    n = nx * ny

    def w_i32(v):
        return struct.pack("<i", v)

    def w_f32(v):
        return struct.pack("<f", v)

    def w_f64(v):
        return struct.pack("<d", v)

    def w_str(s):
        b = b""
        for ch in s:
            b += w_i32(ord(ch))
        b += w_i32(0)  # 空字符终止
        return b

    buf = b""
    # 1. 魔数 + 字节序
    buf += b"#!TDV112"
    buf += w_i32(1)
    # 2. 文件类型
    buf += w_i32(0)
    # 3. 标题
    title = f"LBM Flow Field step={snap.step:06d} time={snap.time:.3f}"
    buf += w_str(title)
    # 4. 变量数量
    buf += w_i32(5)
    # 5. 变量名
    for name in ["X", "Y", "RHO", "UX", "UY"]:
        buf += w_str(name)
    # 6. Zone 头
    buf += w_f32(299.0)
    buf += w_str("fluid")
    buf += w_i32(-1)  # 父 Zone
    buf += w_i32(-1)  # Strand ID
    buf += w_f64(snap.time)
    buf += w_i32(-1)  # 颜色
    buf += w_i32(0)   # Zone 类型（Ordered）
    buf += w_i32(0)   # 变量位置（Nodal）
    buf += w_i32(0)   # 原始邻居
    buf += w_i32(0)   # 用户面邻居连接数
    buf += w_i32(nx)
    buf += w_i32(ny)
    buf += w_i32(1)
    buf += w_i32(0)   # 辅助数据对数
    # 7. EOH
    buf += w_f32(357.0)
    # 8. 数据区域
    buf += w_f32(299.0)
    # 变量格式（2 = float64）
    for _ in range(5):
        buf += w_i32(2)
    buf += w_i32(0)   # 无被动变量
    buf += w_i32(0)   # 无共享变量
    buf += w_i32(-1)  # 共享连接 Zone
    # X, Y, RHO, UX, UY 数据
    import numpy as np
    for j in range(ny):
        for i in range(nx):
            buf += w_f64(i + 0.5)
    for j in range(ny):
        for _ in range(nx):
            buf += w_f64(j + 0.5)
    for j in range(ny):
        for i in range(nx):
            buf += w_f64(float(snap.rho[j, i]))
    for j in range(ny):
        for i in range(nx):
            buf += w_f64(float(snap.ux[j, i]))
    for j in range(ny):
        for i in range(nx):
            buf += w_f64(float(snap.uy[j, i]))

    with open(path, "wb") as f:
        f.write(buf)


class TestTecplotBinReader:
    def test_round_trip(self, tmp_path):
        """写出二进制 Tecplot PLT 后读回，验证场数据一致。"""
        snap = make_synthetic_lid_cavity(nx=8, ny=8, step=500)
        plt_path = tmp_path / "fluid_000500.plt"
        _write_tecplot_bin(plt_path, snap)

        reader = TecplotBinReader(tmp_path)
        assert len(reader) == 1
        loaded = reader.read(500)

        assert loaded.step == 500
        assert loaded.nx == 8
        assert loaded.ny == 8
        np.testing.assert_allclose(loaded.rho, snap.rho, rtol=1e-10)
        np.testing.assert_allclose(loaded.ux,  snap.ux,  rtol=1e-10)
        np.testing.assert_allclose(loaded.uy,  snap.uy,  rtol=1e-10)

    def test_load_snapshot_plt(self, tmp_path):
        """load_snapshot 自动检测 .plt 格式。"""
        snap = make_synthetic_lid_cavity(nx=4, ny=4, step=20)
        plt_path = tmp_path / "fluid_000020.plt"
        _write_tecplot_bin(plt_path, snap)
        loaded = load_snapshot(plt_path)
        assert loaded.step == 20

    def test_empty_directory_raises(self, tmp_path):
        reader = TecplotBinReader(tmp_path)
        with pytest.raises(FileNotFoundError):
            reader.last()

    def test_bad_magic_raises(self, tmp_path):
        """非 TDV112 文件应抛出 ValueError。"""
        p = tmp_path / "fluid_000001.plt"
        p.write_bytes(b"NOTMAGIC" + b"\x00" * 64)
        with pytest.raises(ValueError, match="TDV112"):
            TecplotBinReader(tmp_path).read(p)
