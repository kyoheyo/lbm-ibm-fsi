# LBM-IBM-FSI MPI Parallel Implementation

This document describes the complete MPI (and MPI + OpenMP hybrid) parallel
workflow, covering: three parallel modes (XY block decomposition, independent
multi-process, nested multigrid), 1-D slices as a special case of 2-D block
decomposition, 3-D Z-direction decomposition, ghost-layer exchange
implementation, the `PhysicalBounds` mechanism, and Rust / C API usage
examples.

---

## Table of Contents

1. [Architecture and Mode Selection](#1-architecture-and-mode-selection)
2. [TOML Configuration Quick Reference](#2-toml-configuration-quick-reference)
3. [Mode 1: XY/XYZ Block Decomposition (`"block"`)](#3-mode-1-xyz-block-decomposition-block)
   - 3.1 Unified view: 1-D slices are a special case of 2-D blocks
   - 3.2 Process layout and coordinates
   - 3.3 Ghost-layer layout and halo exchange
   - 3.4 Correct halo-exchange timing (post-collision, pre-stream)
   - 3.5 `PhysicalBounds` (j_s / j_n / i_w / i_e) explained
   - 3.6 Boundary condition ownership
   - 3.7 Per-step execution order (time-step algorithm)
   - 3.8 C++ implementation: `MpiDecomp2D::create()`
   - 3.9 Ghost-layer exchange: `halo_exchange_d2q9_2d()`
   - 3.10 3-D extension: `MpiDecomp3D` and `halo_exchange_d3q19_3d()`
   - 3.11 Block-mode output strategy
4. [Mode 2: Independent (`"independent"`)](#4-mode-2-independent)
5. [Mode 3: Nested Multigrid (`"multigrid"`)](#5-mode-3-nested-multigrid)
6. [MPI + OpenMP Hybrid](#6-mpi--openmp-hybrid)
7. [Rust / C API Examples](#7-rust--c-api-examples)
8. [Build and Run](#8-build-and-run)
9. [Correctness Guarantees and Common Pitfalls](#9-correctness-guarantees-and-common-pitfalls)
10. [Performance Analysis and Tuning](#10-performance-analysis-and-tuning)

---

## 1. Architecture and Mode Selection

The project supports three MPI parallel modes, all controlled by compile-time
flags and TOML configuration:

| Mode | TOML `[mpi].mode` | Legacy names (still accepted) | Use case |
|------|-------------------|-------------------------------|----------|
| **XY/XYZ block decomp** | `"block"` (default) | `"1d_y"`, `"1d_x"`, `"2d_xy"` | Multi-node cluster parallel solve |
| **Independent** | `"independent"` | `"multi_grid"` | Parameter sweeps, batch processing |
| **Nested multigrid** | `"multigrid"` | — | AMR-style refinement (framework ready, time-sync pending) |

**Unified view of block decomposition:**

```
1D Y-slice  = block(nx_blocks=1, ny_blocks=nprocs)
1D X-slice  = block(nx_blocks=nprocs, ny_blocks=1)
2D XY block = block(nx_blocks=px, ny_blocks=py,  px*py=nprocs)
3D XYZ block= block(nx_blocks=px, ny_blocks=py, nz_blocks=pz, px*py*pz=nprocs)
```

---

## 2. TOML Configuration Quick Reference

### 2.1 OpenMP thread count

```toml
[parallel]
omp_num_threads = 8   # force 8 OpenMP threads per MPI rank (0 = system default)
```

### 2.2 1-D Y-slice (`nx_blocks=1`, `ny_blocks=nprocs`)

```toml
[mpi]
mode      = "block"
nx_blocks = 1   # do not split X
ny_blocks = 0   # 0 = auto: ny_blocks = nprocs
# mpirun -n 4 -> ny_blocks automatically set to 4
```

### 2.3 1-D X-slice (`nx_blocks=nprocs`, `ny_blocks=1`)

```toml
[mpi]
mode      = "block"
nx_blocks = 4   # requires mpirun -n 4
ny_blocks = 1
```

### 2.4 2-D XY block (4×2 = 8 processes)

```toml
[mpi]
mode      = "block"
nx_blocks = 4   # split X into 4 blocks
ny_blocks = 2   # split Y into 2 blocks (requires 4×2==8 == mpirun -n 8)
```

### 2.5 3-D XYZ block decomposition (2×2×2 = 8 processes)

```toml
[mpi]
mode      = "block"
nx_blocks = 2
ny_blocks = 2
nz_blocks = 2   # Z-direction splitting (D3Q19/D3Q27); requires nz > 1
# mpirun -n 8
```

### 2.6 Independent mode

```toml
[mpi]
mode = "independent"   # legacy "multi_grid" still works
```

### 2.7 Nested multigrid mode

```toml
[mpi]
mode = "multigrid"
# Nesting is configured in code via the LbmMgTree API.
# Current version: degrades to independent; MgTree framework usable standalone.
```

---

## 3. Mode 1: XY/XYZ Block Decomposition (`"block"`)

### 3.1 Unified view: 1-D slices are a special case of 2-D blocks

```
px=1, py=nprocs -> 1D Y-slice  (no X splitting, Y split uniformly)
px=nprocs, py=1 -> 1D X-slice  (no Y splitting, X split uniformly)
px*py=nprocs    -> 2D XY block (both axes split)
px*py*pz=nprocs -> 3D XYZ block (all three axes split)
```

All cases are implemented through `MpiDecomp2D` (2-D/1-D) or `MpiDecomp3D`
(3-D).  When `px=1`, there are no west/east ghost columns; the south/north
row exchange is identical to the original 1-D `MpiDecomp` behaviour.

### 3.2 Process layout and coordinates

```
Process layout (rank = row_rank * px + col_rank):

  row=py-1  +-----+-----+-----+  <- global north wall (row_rank==py-1 owns North BC)
            | r=6 | r=7 | r=8 |    example: px=3, py=3
  row=1     +-----+-----+-----+
            | r=3 | r=4 | r=5 |
  row=0     +-----+-----+-----+  <- global south wall (row_rank==0 owns South BC)
            col=0  col=1  col=2
            ^                 ^
          West BC           East BC
```

Uniform-partition formula (used by `uniform_partition()`):

```
local_n = total/nprocs + (r < total%nprocs ? 1 : 0)
start   = r * (total/nprocs) + min(r, total%nprocs)
```

### 3.3 Ghost-layer layout

Local grid with ghost layers (col_rank=1, row_rank=1 — all four neighbours
present):

```
  +------------------------------------------+
  | north ghost row  j=phys_y0+local_ny      | <- filled from rank_north's j=phys_y0
  +--+------------------------------------+--+
  |W | physical region                   |E |
  |g | i=phys_x0 .. phys_x0+local_nx-1  |g |
  |h | j=phys_y0 .. phys_y0+local_ny-1  |h |
  |o |                                   |o |
  |s |                                   |s |
  |t |                                   |t |
  +--+------------------------------------+--+
  | south ghost row  j=0                     | <- filled from rank_south's j=phys_y0+local_ny-1
  +------------------------------------------+
```

Key offsets (computed from `MpiDecomp2D` helper methods):

```cpp
phys_x0 = has_west_ghost()  ? 1 : 0;  // 0 for 1D Y-slice (no west ghost)
phys_y0 = has_south_ghost() ? 1 : 0;  // 1 for all interior ranks
```

### 3.4 Correct halo-exchange timing (post-collision, pre-stream)

**This is the most critical correctness point in the parallel implementation.**

In the push-scheme LBM, `stream()` writes each node's post-collision
distributions *outward* to its neighbours.  The ghost rows must therefore
contain the **post-collision** (not post-stream) distributions of the
neighbouring rank's physical boundary row *before* the stream loop begins.

```
Correct order (implemented in Solver::stream()):
  collide()              <- BGK/MRT relaxation, ghost nodes skipped
       |
       v
  halo_exchange()        <- MPI_Sendrecv: send post-collision boundary rows
       |                    to neighbours' ghost rows
       v
  stream loop            <- push neighbours' distributions using ghost rows
       |
       v
  std::swap(f, f_tmp)
       |
       v
  apply_boundary_conditions()
       |
       v
  compute_macroscopic()
```

If halo exchange were placed *after* `stream()`, the ghost rows would contain
post-stream data that has already been pushed to interior nodes — providing an
O(1) error at every partition interface and producing results that differ from
the single-rank reference.

**Implementation in `core/src/lbm/solver.cpp` (`Solver::stream()`):**

```cpp
void Solver::stream()
{
    // 1. Halo exchange BEFORE the stream loop (post-collision data)
#ifdef LBM_ENABLE_MPI
    if (mpi_decomp3d_)
        halo_exchange_d3q19_3d(grid_, *mpi_decomp3d_);
    else if (mpi_decomp2d_)
        halo_exchange_d2q9_2d(grid_, *mpi_decomp2d_);
    else if (mpi_decomp_)
        halo_exchange_d2q9(grid_, *mpi_decomp_);
#endif

    // 2. Push-scheme stream loop (reads from f, writes to f_tmp)
    const int n = grid_.size();
    for (int idx = 0; idx < n; ++idx) {
        // ... push distributions to neighbours
    }
}
```

### 3.5 `PhysicalBounds` (j_s / j_n / i_w / i_e) explained

`PhysicalBounds` is a lightweight struct that tells
`apply_boundary_conditions()` which rows/columns are *physical* (owned by
this rank) vs. ghost rows that must never receive a BC.

```cpp
struct PhysicalBounds {
    int j_s;  // first physical south row index  (local coords, inclusive)
    int j_n;  // last  physical north row index  (local coords, inclusive)
    int i_w;  // first physical west  column index
    int i_e;  // last  physical east  column index
    bool has_south_wall;  // true  => apply South-face BCs on this rank
    bool has_north_wall;  // true  => apply North-face BCs on this rank
    bool has_west_wall;   // true  => apply West-face  BCs on this rank
    bool has_east_wall;   // true  => apply East-face  BCs on this rank
};
```

**Field values by mode:**

| Mode | j_s | j_n | i_w | i_e |
|------|-----|-----|-----|-----|
| Non-MPI | 0 | ny-1 | 0 | nx-1 |
| 1D Y-slice (all ranks) | 1 | ny-2 | 0 | nx-1 |
| 2D block (interior rank) | phys_y0 | phys_y0+local_ny-1 | phys_x0 | phys_x0+local_nx-1 |

**Why this matters:**

Without `PhysicalBounds`, a bounce-back BC registered on the South face would
be applied to `j=0` — which in MPI mode is the *south ghost row* filled by
the southern neighbour's physical data.  Overwriting those ghost values would
corrupt the next timestep's stream step and produce a visible discontinuity at
the interface.

`j_s=1` (1-D MPI) ensures BC loops start at the first *physical* row, not the
ghost row.  Similarly `j_n=ny-2` ensures they stop before the north ghost row.

**`has_*_wall` flags:**

Interior ranks do not own a global physical wall.  If `has_south_wall=false`,
`apply_boundary_conditions()` skips all South-face BCs entirely, even though
`j_s` points to a valid physical row.  This prevents incorrect BC application
in the interior of the domain.

**Set in `Solver::step()` (from `core/src/lbm/solver.cpp`):**

```cpp
// 2D block decomp example:
PhysicalBounds pb;
pb.j_s = decomp2d_->phys_y0();
pb.j_n = decomp2d_->phys_y0() + decomp2d_->local_ny - 1;
pb.i_w = decomp2d_->phys_x0();
pb.i_e = decomp2d_->phys_x0() + decomp2d_->local_nx - 1;
pb.has_south_wall = decomp2d_->has_south_wall();
pb.has_north_wall = decomp2d_->has_north_wall();
pb.has_west_wall  = decomp2d_->has_west_wall();
pb.has_east_wall  = decomp2d_->has_east_wall();
apply_boundary_conditions(grid_, bcs_, pb);
```

### 3.6 Boundary condition ownership

| Face | Register BC when | Helper method |
|------|------------------|---------------|
| South | `row_rank == 0` | `decomp.has_south_wall()` |
| North | `row_rank == py-1` | `decomp.has_north_wall()` |
| West | `col_rank == 0` | `decomp.has_west_wall()` |
| East | `col_rank == px-1` | `decomp.has_east_wall()` |
| Bottom | `pz_rank == 0` | `decomp3d.has_bottom_wall()` (3D) |
| Top | `pz_rank == pz-1` | `decomp3d.has_top_wall()` (3D) |

1D Y-slice special case (`px=1`): `col_rank=0` means West *and* East are both
owned by every rank, so all ranks register West/East BCs.

### 3.7 Per-step execution order (time-step algorithm)

```
Solver::step():
  1. collide()                    BGK/MRT relaxation
                                  - ghost nodes skipped (CollideGuard)
                                  - writes post-collision f into f (in-place)
       |
       v
  2. stream()
     a. halo_exchange()           MPI_Sendrecv: push post-collision boundary
                                  rows/columns into neighbour ghost layers
     b. push-stream loop          for each node: f_tmp[neighbour] = f[node][dir]
       |
       v
  3. std::swap(f, f_tmp)          f now holds streamed distributions
       |
       v
  4. apply_boundary_conditions()  only on physically-owned walls (PhysicalBounds)
       |
       v
  5. compute_macroscopic()        update rho and u from f
```

### 3.8 C++ implementation: `MpiDecomp2D::create()`

**File:** `core/src/lbm/mpi_decomp.cpp`

```cpp
MpiDecomp2D MpiDecomp2D::create(int gnx, int gny, int in_px, int in_py)
{
    MpiDecomp2D d;
    d.global_nx = gnx;  d.global_ny = gny;
    d.px = in_px;       d.py = in_py;
    MPI_Comm_rank(MPI_COMM_WORLD, &d.rank);
    MPI_Comm_size(MPI_COMM_WORLD, &d.nprocs);

    if (d.px * d.py != d.nprocs)
        throw std::invalid_argument("MpiDecomp2D: px*py must equal MPI process count");

    d.col_rank = d.rank % d.px;
    d.row_rank = d.rank / d.px;

    // uniform_partition: ensures total >= local_n * nprocs with remainder handled
    uniform_partition(gnx, d.px, d.col_rank, d.local_nx, d.x_start);
    uniform_partition(gny, d.py, d.row_rank, d.local_ny, d.y_start);
    d.x_end = d.x_start + d.local_nx - 1;
    d.y_end = d.y_start + d.local_ny - 1;

    // Neighbour ranks (MPI_PROC_NULL is legal for boundary ranks)
    d.rank_south = (d.row_rank > 0)      ? (d.row_rank-1)*d.px + d.col_rank : MPI_PROC_NULL;
    d.rank_north = (d.row_rank < d.py-1) ? (d.row_rank+1)*d.px + d.col_rank : MPI_PROC_NULL;
    d.rank_west  = (d.col_rank > 0)      ? d.row_rank*d.px + d.col_rank-1   : MPI_PROC_NULL;
    d.rank_east  = (d.col_rank < d.px-1) ? d.row_rank*d.px + d.col_rank+1   : MPI_PROC_NULL;
    return d;
}
```

Ghost-layer grid sizes (including ghost rows/columns):

```cpp
int grid_nx() const {
    return local_nx + (has_west_ghost() ? 1 : 0) + (has_east_ghost() ? 1 : 0);
}
int grid_ny() const {
    return local_ny + (has_south_ghost() ? 1 : 0) + (has_north_ghost() ? 1 : 0);
}
int phys_x0() const { return has_west_ghost()  ? 1 : 0; }
int phys_y0() const { return has_south_ghost() ? 1 : 0; }
```

### 3.9 Ghost-layer exchange: `halo_exchange_d2q9_2d()`

**File:** `core/src/lbm/mpi_decomp.cpp`

Exchange is split into two phases to avoid deadlocks:

**Phase 1 (south/north, contiguous row data):**

```cpp
// Send top physical row to north neighbour, receive south ghost from south neighbour
MPI_Sendrecv(top_phys,   row_size, MPI_DOUBLE, decomp.rank_north, 10,
             south_ghost, row_size, MPI_DOUBLE, decomp.rank_south, 10,
             MPI_COMM_WORLD, &st);
// Send bottom physical row south, receive north ghost from north neighbour
MPI_Sendrecv(bot_phys,   row_size, MPI_DOUBLE, decomp.rank_south, 11,
             north_ghost, row_size, MPI_DOUBLE, decomp.rank_north, 11,
             MPI_COMM_WORLD, &st);
```

**Phase 2 (west/east, non-contiguous column data, must pack first):**

```cpp
// Pack westernmost/easternmost physical columns into contiguous send buffers
for (int j = 0; j < gny_all; ++j) {
    send_west[j*Q .. (j+1)*Q] = f[idx(phys_x0,         j)] * Q entries
    send_east[j*Q .. (j+1)*Q] = f[idx(phys_x0+lnx-1,   j)] * Q entries
}
MPI_Sendrecv(send_west, col_size, MPI_DOUBLE, decomp.rank_west, 20,
             recv_east,  col_size, MPI_DOUBLE, decomp.rank_east, 20, ...);
MPI_Sendrecv(send_east, col_size, MPI_DOUBLE, decomp.rank_east, 21,
             recv_west,  col_size, MPI_DOUBLE, decomp.rank_west, 21, ...);
// Unpack recv_east/recv_west into east/west ghost columns
```

**1D Y-slice optimisation:** When `px=1`, `has_west_ghost()=false` and
`has_east_ghost()=false`, so Phase 2 pack/unpack loops are zero-length and
`MPI_Sendrecv` targets `MPI_PROC_NULL` (no-op).  Performance equals the
original 1-D implementation.

### 3.10 3-D extension: `MpiDecomp3D` and `halo_exchange_d3q19_3d()`

**File:** `core/src/lbm/mpi_decomp.cpp`

`MpiDecomp3D` extends the 2-D decomposition to three dimensions:

```cpp
// Process layout: rank = pz_rank*(px*py) + row_rank*px + col_rank
// Six neighbour ranks: rank_west/east/south/north/bottom/top
// Ghost-layer helpers:
//   has_west/east/south/north/bottom/top_ghost()
//   grid_nx() = local_nx + west_ghost + east_ghost
//   grid_ny() = local_ny + south_ghost + north_ghost
//   grid_nz() = local_nz + bottom_ghost + top_ghost
//   phys_x0() = has_west_ghost()   ? 1 : 0
//   phys_y0() = has_south_ghost()  ? 1 : 0
//   phys_z0() = has_bottom_ghost() ? 1 : 0
static MpiDecomp3D create(int gnx, int gny, int gnz, int px, int py, int pz);
```

**`halo_exchange_d3q19_3d()` — six-face exchange in Z→Y→X order:**

```
Phase Z (bottom/top):  exchange whole XY slabs (contiguous in memory)
    MPI_Sendrecv(top_phys_slab,    slab_size, ..., rank_top,    30, ...)
    MPI_Sendrecv(bottom_phys_slab, slab_size, ..., rank_bottom, 31, ...)

Phase Y (south/north): pack entire XZ plane, then exchange
    pack:   for k in [0,gnz): send_north[k*row_size .. ] = row(j=phys_y0+lny-1, k)
    Sendrecv  north/south

Phase X (west/east):   pack entire YZ plane column-by-column, then exchange
    pack:   for k,j: send_east[(k*gny+j)*Q .. ] = node(i=phys_x0+lnx-1, j, k)
    Sendrecv  west/east
```

**Activation:** Create `LbmMpiDecomp3D` and call `solver.attach_mpi3d()`.
The orchestrator creates it automatically when `nz_blocks > 1` in TOML and
`cfg.fluid.nz > 1`.

```toml
[mpi]
mode      = "block"
nx_blocks = 2
ny_blocks = 2
nz_blocks = 2    # triggers 3D decomp; nz must be > 1
```

### 3.11 Block-mode output strategy

#### Per-rank snapshot output (default)

Each rank writes its local physical partition to its own subdirectory:

```
output/
+-- rank_0/
|   +-- fluid_000500.npz    <- includes x_start/y_start/global_nx/global_ny metadata
|   +-- monitor.csv
+-- rank_1/
|   +-- fluid_000500.npz
...
```

NPZ partition files include position metadata for reconstruction tools:

| Key | Type | Meaning |
|-----|------|---------|
| `x_start` | int64 | Partition X start in global coordinates |
| `y_start` | int64 | Partition Y start in global coordinates |
| `global_nx` | int64 | Total global X node count |
| `global_ny` | int64 | Total global Y node count |

#### In-situ global merge (`combine_blocks = true`)

```toml
[output]
write_interval = 500
directory      = "output/my_run"
format         = "npz"
combine_blocks = true   # default false; only effective in block mode
```

Rank-0 gathers all partitions via `MPI_Gatherv` and writes the global
snapshot to `<directory>/fluid_<NNNNNN>.<ext>`.

#### Offline merge after simulation

```bash
python3 python/examples/combine_blocks.py \
    --dir output/my_run --fmt dat --out output/my_run/global
```

---

## 4. Mode 2: Independent (`"independent"`)

Each MPI rank runs a **completely independent** simulation with no MPI
communication, suited for:

- **Parameter sweeps**: run Re=100/500/1000/5000 simultaneously
- **Integration tests**: validate different BC combinations in parallel
- **Restart generation**: start from different initial conditions

```toml
[mpi]
mode = "independent"
```

In this mode:
- Each rank allocates the **full** `nx × ny` global grid with no ghost layers.
- `step()` performs no MPI communication.
- All ranks execute the same config (differentiate via `mpi_rank()` if needed).
- Output is written to `output/rank_<N>/`.

---

## 5. Mode 3: Nested Multigrid (`"multigrid"`)

Nested multigrid embeds fine grids inside coarse grids:

```
Global coarse grid (level=0, 256x256)
  +-- Refined region A (level=1, centre 64x64, refine ratio 2)
  |     +-- Finer region A1 (level=2, centre 16x16, refine ratio 2)
  +-- Refined region B (level=1, lower-right 32x32, refine ratio 2)
```

### `MgTree` / `MgNode` data structure

**File:** `core/include/lbm/mg_tree.hpp`, `core/src/lbm/mg_tree.cpp`

```cpp
struct MgNode {
    MgExtent extent;          // spatial range in global coords
    int level;                // 0 = coarsest (root)
    int refine_ratio;         // local refinement factor
    LatticeGrid* grid;        // bound LatticeGrid (may be nullptr)
    std::vector<MgNode*> children;
    std::variant<std::monostate,
                 MpiDecomp2D*,
                 MpiDecomp3D*> decomp;  // optional per-node MPI decomp
};
```

### Tree construction

```cpp
MgTree tree({0,255, 0,255, 0,0});           // root: 256x256 coarse grid

auto* c1a = tree.add_level(root, {96,159, 96,159, 0,0}, 2);
auto* c1b = tree.add_level(root, {192,255, 0,63,  0,0}, 2);
auto* c2  = tree.add_level(c1a,  {112,143, 112,143, 0,0}, 2);
```

### Traversal

```cpp
// Coarse-to-fine (BFS): advance coarse first, then fine
tree.traverse_coarse_to_fine([](MgNode* node) {
    if (node->has_grid()) { /* run LBM step on this level */ }
});
// Fine-to-coarse (BFS reverse): for residual transfer (restriction)
tree.traverse_fine_to_coarse([](MgNode* node) { /* restriction */ });
```

### Implementation status

| Feature | Status | Note |
|---------|--------|------|
| `MgTree` / `MgNode` structures | Ready | N-ary tree, 2D/3D, arbitrary nesting |
| `add_level()` / `traverse_*()` | Ready | Bounds check + BFS traversal |
| `LbmMgTree` Rust wrapper | Ready | Full FFI + RAII |
| C ABI (`lbm_mg_tree_*`) | Ready | Create/free/query |
| `mg_prolong_rho_u` | Ready | Coarse→fine bilinear interpolation |
| `mg_restrict_rho_u` | Ready | Fine→coarse r×r volume average |
| `mg_prolong_f` | Ready | Prolongation via equilibrium reconstruction |
| `mg_apply_fringe_bc` | Ready | Overset fringe coupling |
| `mg_compute_refinement_indicator` | Ready | Density-gradient AMR criterion |
| Time-step synchronisation | Pending | Fine dt = 1/r * coarse dt |
| Automatic AMR refinement | Pending | Adaptive criterion + remeshing |

---

## 6. MPI + OpenMP Hybrid

### Thread count setting

```toml
[parallel]
omp_num_threads = 8   # each MPI rank uses 8 OpenMP threads
```

Equivalent to `omp_set_num_threads(8)`, taking priority over `OMP_NUM_THREADS`.

Rust API: `lbm_bindings::set_omp_num_threads(8)`
C API: `lbm_omp_set_num_threads(8)`

### Topology example (2 nodes, 4 ranks per node, 8 threads per rank)

```
Node A (ranks 0..3)                Node B (ranks 4..7)
+---------------------------+      +---------------------------+
| rank 0 (8 OMP threads)   |<-H-> | rank 4 (8 OMP threads)   |
| rank 1 (8 OMP threads)   |<-H-> | rank 5 (8 OMP threads)   |
| rank 2 (8 OMP threads)   |<-H-> | rank 6 (8 OMP threads)   |
| rank 3 (8 OMP threads)   |      | rank 7 (8 OMP threads)   |
+---------------------------+      +---------------------------+
Total parallelism = 8 ranks x 8 threads = 64 cores
```

**Recommendation:** `nprocs * omp_threads <= physical_cores`.  Assign 1 MPI
rank per NUMA domain with `omp_threads = cores_per_numa` for maximum NUMA
locality.

---

## 7. Rust / C API Examples

### 7.1 1-D Y-slice (nx_blocks=1, ny_blocks=nprocs)

```rust
use lbm_bindings::{
    mpi_init, mpi_finalize, mpi_rank, mpi_size,
    LbmGrid, LbmSolver, LbmMpiDecomp2D,
    LatticeModel, CollisionModel, BcType, Face,
};

fn main() {
    mpi_init();
    let rank   = mpi_rank();
    let nprocs = mpi_size();

    let (gnx, gny) = (256i32, 256i32);

    // 1D Y-slice = 2D block with px=1, py=nprocs
    let mut decomp = LbmMpiDecomp2D::new(gnx, gny, 1, nprocs)
        .expect("decomp creation failed");

    let local_nx = decomp.grid_nx();   // = gnx (no west/east ghosts when px=1)
    let local_ny = decomp.grid_ny();   // = local_ny + 2 ghost rows
    let y_start  = decomp.y_start();

    let mut grid   = LbmGrid::new(local_nx, local_ny, 1, LatticeModel::D2Q9);
    let mut solver = LbmSolver::new(&mut grid, 1.6, CollisionModel::Bgk);

    // Bind decomp (halo_exchange_d2q9_2d skips east/west when px=1)
    solver.attach_mpi2d(Some(&mut decomp));

    // Register BCs only on owned physical walls
    if decomp.y_start() == 0 {
        solver.add_boundary_condition(BcType::BounceBack, Face::South, 0.0, 0.0, 0.0, 1.0);
    }
    if (decomp.y_start() + decomp.local_ny()) as u64 == gny as u64 {
        solver.add_boundary_condition(BcType::ZouHeVelocity, Face::North, 0.1, 0.0, 0.0, 0.0);
    }
    // px=1: all ranks own West and East walls
    solver.add_boundary_condition(BcType::BounceBack, Face::West, 0.0, 0.0, 0.0, 1.0);
    solver.add_boundary_condition(BcType::BounceBack, Face::East, 0.0, 0.0, 0.0, 1.0);

    for _ in 0..5000 { solver.step(&mut grid); }
    mpi_finalize();
}
```

### 7.2 2-D block decomposition

```rust
// 2D block: px=4, py=2 -> requires mpirun -n 8
let (gnx, gny, px, py) = (512i32, 512i32, 4i32, 2i32);
assert_eq!(px * py, nprocs, "px*py must equal nprocs");

let mut decomp = LbmMpiDecomp2D::new(gnx, gny, px, py)
    .expect("decomp creation failed");

// Register BCs only on owned walls
if decomp.y_start() == 0 {
    solver.add_boundary_condition(BcType::BounceBack, Face::South, ...);
}
if decomp.x_start() == 0 {
    solver.add_boundary_condition(BcType::BounceBack, Face::West, ...);
}
// ... similarly for North and East
```

### 7.3 3-D block decomposition

```rust
// 3D block: px=2, py=2, pz=2 -> requires mpirun -n 8, and nz > 1 in TOML
let (gnx, gny, gnz, px, py, pz) = (64i32, 64i32, 64i32, 2, 2, 2);
let mut decomp3d = LbmMpiDecomp3D::new(gnx, gny, gnz, px, py, pz)
    .expect("3D decomp creation failed");

let mut grid = LbmGrid::new(decomp3d.grid_nx(), decomp3d.grid_ny(),
                             decomp3d.grid_nz(), LatticeModel::D3Q19);
let mut solver = LbmSolver::new(&mut grid, 1.6, CollisionModel::Bgk);
solver.attach_mpi3d(Some(&mut decomp3d));

// Register BCs based on z_start / y_start / x_start ownership
if decomp3d.z_start() == 0 {
    solver.add_boundary_condition(BcType::BounceBack, Face::Bottom, ...);
}
// ... similarly for Top, South, North, West, East
```

### 7.4 Independent mode

```rust
mpi_init();
let rank = mpi_rank();

// No decomp, no ghost layers, no communication
let mut grid   = LbmGrid::new(256, 256, 1, LatticeModel::D2Q9);
let mut solver = LbmSolver::new(&mut grid, 1.6, CollisionModel::Bgk);
// Can vary omega by rank: let omega = base_omega_for(rank);
for _ in 0..5000 { solver.step(&mut grid); }
// Output goes to output/rank_<rank>/ automatically
mpi_finalize();
```

### 7.5 C ABI quick reference

**MPI block decomposition**

| C function | Rust function | Description |
|------------|---------------|-------------|
| `lbm_mpi_init()` | `mpi_init()` | Initialise MPI |
| `lbm_mpi_finalize()` | `mpi_finalize()` | Finalise MPI |
| `lbm_mpi_rank()` | `mpi_rank()` | Current rank |
| `lbm_mpi_size()` | `mpi_size()` | Process count |
| `lbm_mpi_barrier()` | `mpi_barrier()` | Global barrier |
| `lbm_mpi_decomp2d_new(gnx,gny,px,py)` | `LbmMpiDecomp2D::new(...)` | Create 2-D block decomp |
| `lbm_mpi_decomp2d_free(h)` | `Drop for LbmMpiDecomp2D` | Free |
| `lbm_solver_attach_mpi2d(s,h)` | `solver.attach_mpi2d(d)` | Bind 2-D decomp |
| `lbm_mpi_decomp2d_grid_nx(h)` | `d.grid_nx()` | Local nx with ghosts |
| `lbm_mpi_decomp2d_grid_ny(h)` | `d.grid_ny()` | Local ny with ghosts |
| `lbm_mpi_decomp2d_local_nx(h)` | `d.local_nx()` | Physical columns |
| `lbm_mpi_decomp2d_local_ny(h)` | `d.local_ny()` | Physical rows |
| `lbm_mpi_decomp2d_phys_x0(h)` | `d.phys_x0()` | Physical X offset |
| `lbm_mpi_decomp2d_phys_y0(h)` | `d.phys_y0()` | Physical Y offset |
| `lbm_mpi_decomp2d_x_start(h)` | `d.x_start()` | Global X start |
| `lbm_mpi_decomp2d_y_start(h)` | `d.y_start()` | Global Y start |
| `lbm_mpi_decomp3d_new(gnx,gny,gnz,px,py,pz)` | `LbmMpiDecomp3D::new(...)` | Create 3-D block decomp |
| `lbm_mpi_decomp3d_free(h)` | `Drop for LbmMpiDecomp3D` | Free |
| `lbm_solver_attach_mpi3d(s,h)` | `solver.attach_mpi3d(d)` | Bind 3-D decomp |
| `lbm_mpi_decomp3d_grid_nz(h)` | `d.grid_nz()` | Local nz with ghosts |
| `lbm_mpi_decomp3d_local_nz(h)` | `d.local_nz()` | Physical layers |
| `lbm_mpi_decomp3d_phys_z0(h)` | `d.phys_z0()` | Physical Z offset |
| `lbm_mpi_decomp3d_z_start(h)` | `d.z_start()` | Global Z start |

**Multigrid tree**

| C function | Rust function | Description |
|------------|---------------|-------------|
| `lbm_mg_tree_new(x0,x1,y0,y1,z0,z1,is_3d)` | `LbmMgTree::new(...)` | Create tree |
| `lbm_mg_tree_free(h)` | `Drop for LbmMgTree` | Free all nodes |
| `lbm_mg_tree_add_level(...)` | `tree.add_level(...)` | Add refined child region |
| `lbm_mg_tree_root(h)` | `tree.root()` | Root node |
| `lbm_mg_tree_max_level(h)` | `tree.max_level()` | Deepest level |
| `lbm_mg_tree_node_count(h)` | `tree.node_count()` | Total node count |

**OpenMP**

| C function | Rust function | Description |
|------------|---------------|-------------|
| `lbm_omp_set_num_threads(n)` | `set_omp_num_threads(n)` | Set thread count |

---

## 8. Build and Run

### 8.1 Compile flags

| Cargo env var | CMake flag | Description |
|---------------|-----------|-------------|
| `LBM_ENABLE_MPI=ON` | `-DENABLE_MPI=ON` | Enable MPI (default OFF) |
| `LBM_ENABLE_OPENMP=ON` | `-DENABLE_OPENMP=ON` | Enable OpenMP (default OFF) |
| `LBM_ENABLE_CUDA=ON` | `-DENABLE_CUDA=ON` | Enable CUDA GPU backend |

### 8.2 Build commands

**Linux / macOS:**
```bash
LBM_ENABLE_MPI=ON LBM_ENABLE_OPENMP=ON cargo build --release
```

**Windows PowerShell:**
```powershell
$env:LBM_ENABLE_MPI = "ON"; $env:LBM_ENABLE_OPENMP = "ON"
cargo build --release
```

### 8.3 Run commands

**1-D Y-slice (4 processes):**
```bash
# [mpi] mode="block" nx_blocks=1 ny_blocks=0
mpirun -n 4 ./target/release/lbm-ibm-fsi configs/my.toml
```

**2-D XY block (4x2 = 8 processes):**
```bash
# [mpi] mode="block" nx_blocks=4 ny_blocks=2
mpirun -n 8 ./target/release/lbm-ibm-fsi configs/my.toml
```

**3-D XYZ block (2x2x2 = 8 processes):**
```bash
# [mpi] mode="block" nx_blocks=2 ny_blocks=2 nz_blocks=2
mpirun -n 8 ./target/release/lbm-ibm-fsi configs/my.toml
```

**Independent (4 independent simulations):**
```bash
# [mpi] mode="independent"
mpirun -n 4 ./target/release/lbm-ibm-fsi configs/my.toml
# Output: output/rank_0/ output/rank_1/ output/rank_2/ output/rank_3/
```

**Windows (MS-MPI):**
```powershell
mpiexec -n 4 .\target\release\lbm-ibm-fsi.exe configs\my.toml
```

### 8.4 SLURM cluster script

**2-D block + OpenMP hybrid (2 nodes, 4 ranks/node, 4 threads/rank):**
```bash
#!/bin/bash
#SBATCH --nodes=2 --ntasks=8 --ntasks-per-node=4 --cpus-per-task=4
export OMP_PROC_BIND=close; export OMP_PLACES=cores
# TOML: [mpi] mode="block" nx_blocks=4 ny_blocks=2
#        [parallel] omp_num_threads=4
mpirun -n 8 ./target/release/lbm-ibm-fsi configs/hybrid.toml
```

---

## 9. Correctness Guarantees and Common Pitfalls

### 9.1 Ghost nodes are never collided

Ghost nodes hold post-collision data from the neighbouring rank.  Colliding
them again would apply relaxation twice ("double-relaxation" error).
`collide_bgk()` / `collide_mrt()` skip all ghost nodes using `CollideGuard`
bounds (which checks `is_ghost(i)` for each flat index).

### 9.2 Halo exchange must occur before the stream loop

See §3.4 for the full explanation.  Moving the exchange to after `stream()`
causes O(1) errors at every partition interface and makes multi-rank results
differ from the single-rank reference.  This was a known bug in earlier
versions and is now fixed: the exchange is placed at the start of `stream()`.

### 9.3 BCs are registered only on the rank that owns the wall

```rust
if decomp2d.y_start() == 0 {                          // row_rank==0 -> South
    solver.add_boundary_condition(BcType::BounceBack, Face::South, ...);
}
if decomp2d.x_start() == 0 {                          // col_rank==0 -> West
    solver.add_boundary_condition(BcType::BounceBack, Face::West, ...);
}
// Similarly for North and East
```

### 9.4 `nx_blocks * ny_blocks` (or `* nz_blocks`) must equal `nprocs`

If the product does not match, the orchestrator prints a warning and falls back
to 1-D Y-slice automatically:
```
[warn] mpi.nx_blocks(4) * mpi.ny_blocks(2) = 8 != nprocs(4).
       Falling back to 1D Y slice (nx_blocks=1, ny_blocks=nprocs).
```

### 9.5 `MgTree` child extent must lie within parent extent

`add_level()` throws `std::invalid_argument` if `child_extent` is not a
subset of `parent->extent`.  For 2-D trees set `z_start=z_end=0`.

---

## 10. Performance Analysis and Tuning

### Communication volume comparison

| Mode | Communication per step (D2Q9, N² nodes, P ranks) | Notes |
|------|---------------------------------------------------|-------|
| 1D Y (px=1, py=P) | 2 × nx × 9 × 8 B × P | 2 Sendrecv, no packing |
| 1D X (px=P, py=1) | 2 × ny × 9 × 8 B × P | 2 Sendrecv, packing overhead |
| 2D XY (px×py=P) | 2 × (lnx+lny) × 9 × 8 B × P | 4 Sendrecv, east/west packing |
| 3D XYZ (px×py×pz=P) | slabs + rows + cols | 6 Sendrecv phases |

**2-D advantage:** For a square grid (nx≈ny), per-rank area O(N²/P), boundary
perimeter O(N/√P), communication-to-computation ratio O(1/√P) — much better
than 1-D's O(1/P^{1/2}).

### Decomposition selection guide

- **nx << ny**: use 1D Y (py=P, px=1)
- **ny << nx**: use 1D X (px=P, py=1)
- **nx ≈ ny**: use 2D XY (px≈py≈√P)
- **3D domain**: use XYZ block (px≈py≈pz≈P^{1/3})
- **No communication needed, parameter sweeps**: use `independent`
- **Local high-resolution needs**: use `MgTree` for nested refinement

---

*Code reference: `core/include/lbm/mpi_decomp.hpp`, `core/include/lbm/solver.hpp`,
`core/src/lbm/mpi_decomp.cpp`, `core/src/lbm/solver.cpp`, `core/include/lbm/boundary.hpp`,
`core/src/lbm/boundary.cpp`, `core/include/lbm/mg_tree.hpp`, `core/src/lbm/mg_tree.cpp`,
`core/src/capi/lbm_capi.cpp`, `bindings/src/lib.rs`, `orchestrator/src/config.rs`,
`orchestrator/src/main.rs`.*
