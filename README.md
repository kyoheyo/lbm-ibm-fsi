# lbm-ibm-fsi

A high-performance Lattice Boltzmann Method (LBM) + Immersed Boundary Method (IBM) + Fluid-Structure Interaction (FSI) solver framework.

## Architecture Overview

This project uses a **multi-language hybrid architecture** to balance performance, safety, and extensibility:

| Layer | Language | Role |
|-------|----------|------|
| Computational Core | **C++17** | LBM solver, IBM coupling, FSI structural solver; MPI+OpenMP parallelism |
| Orchestration & CLI | **Rust** | Safe driver, configuration loading, FFI bindings to C++ core, I/O |
| Pre/Post-processing | **Python 3.11+** | gmsh mesh generation, config generation, visualisation, result analysis |
| Build System | **CMake + Cargo** | C++ build (with MPI/OpenMP/CUDA detection); Rust workspace |

### Why this combination?

- **C++** has the most mature ecosystem for HPC numerics (MPI, OpenMP, CUDA, PETSc, Eigen, FFTW). Its zero-overhead abstractions and direct SIMD/intrinsic access make it ideal for the inner computational loops.
- **Rust** provides memory-safety guarantees for the driver/orchestration layer, preventing data-race bugs when managing simulation state across threads. Its `cargo` tooling makes dependency management and cross-platform builds straightforward. The `mpi` crate and `rayon` crate offer additional parallelism options at the orchestration level.
- The **Rust ↔ C++ FFI** boundary is thin and well-defined via a C ABI layer, keeping the two languages loosely coupled.
- **Python** is the natural choice for the pre/post-processing layer: `gmsh` provides a mature Python API for mesh generation; `numpy`/`scipy` and `matplotlib` are the de-facto standard for scientific computing and visualisation in Python. No performance-critical code runs in Python.

## Project Structure

```
lbm-ibm-fsi/
├── CMakeLists.txt           # Top-level CMake (C++ core)
├── Cargo.toml               # Rust workspace
├── README.md
├── .gitignore
│
├── core/                    # C++ computational core
│   ├── CMakeLists.txt
│   ├── include/
│   │   ├── lbm/
│   │   │   ├── lattice.hpp          # Lattice model (D2Q9 / D3Q19 / D3Q27)
│   │   │   ├── solver.hpp           # BGK/MRT collision + streaming
│   │   │   └── boundary.hpp        # Boundary conditions (bounce-back, ZOU-HE, etc.)
│   │   ├── ibm/
│   │   │   ├── marker.hpp           # Lagrangian marker points
│   │   │   └── interpolation.hpp   # Delta-function spreading/interpolation
│   │   └── fsi/
│   │       ├── structure.hpp        # Flexible beam/shell FEM
│   │       └── coupling.hpp        # Partitioned FSI coupling
│   └── src/
│       ├── lbm/
│       ├── ibm/
│       ├── fsi/
│       └── capi/                   # C ABI bridge for Rust FFI
│
├── bindings/                # Rust crate — thin C ABI + safe Rust wrappers
│   ├── Cargo.toml
│   ├── build.rs             # Invokes cmake to build the C++ core
│   └── src/lib.rs
│
├── orchestrator/            # Rust crate — CLI driver + simulation loop
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       └── config.rs
│
├── python/                  # Python pre/post-processing
│   ├── pyproject.toml       # Package metadata & dependencies
│   ├── lbm_pre/             # Pre-processing package
│   │   ├── mesh.py          # gmsh domain & obstacle mesh generation
│   │   ├── geometry.py      # IBM Lagrangian marker constructors
│   │   └── config_gen.py    # Generate solver TOML configs from Python
│   ├── lbm_post/            # Post-processing package
│   │   ├── vtk_reader.py    # Read .npz / .vtu solver output
│   │   ├── plot.py          # Contour, streamline, vorticity, beam plots
│   │   └── analysis.py      # Drag/lift, Q-criterion, error norms, convergence
│   ├── tests/               # pytest suite (105 tests)
│   └── examples/
│       ├── lid_driven_cavity.py   # End-to-end pre/post example (synthetic data)
│       └── plot_solver_output.py  # Visualise real solver NPZ output
│
└── configs/                 # Example TOML configuration files
    └── lid_driven_cavity.toml
```

## Parallelism Strategy

### Multi-CPU (current target)
- **MPI** (via `OpenMPI` / `MPICH`): domain decomposition across nodes; each MPI rank owns a sub-domain of the Eulerian LBM grid and a subset of Lagrangian IBM markers.
- **OpenMP**: thread-level parallelism within each MPI rank for the collision and streaming loops.
- Communication pattern: halo exchange at MPI boundaries after each streaming step; IBM force reduction is an `MPI_Allreduce`.

### GPU (future extension)
- The C++ core is structured so the hot loops (`collision`, `streaming`, `ibm_spread`, `ibm_interpolate`) are isolated in translation units that can be replaced with CUDA/HIP/SYCL kernels.
- CMake already detects `CUDA` as an optional component; enable with `-DENABLE_CUDA=ON`.

## Building

### Prerequisites
- CMake ≥ 3.20
- A C++17 compiler (GCC ≥ 11 or Clang ≥ 14)
- MPI implementation (OpenMPI ≥ 4.0 or MPICH ≥ 3.4)
- OpenMP (bundled with most compilers)
- Rust toolchain ≥ 1.70 (`rustup`)
- Python ≥ 3.11 with `pip`
- (Optional) CUDA ≥ 11.8 for GPU support

### Quick start

```bash
# Build C++ core and Rust orchestrator
cargo build --release

# Run the lid-driven cavity example (single process)
./target/release/lbm-ibm-fsi --config configs/lid_driven_cavity.toml

# Run with MPI (4 processes)
mpirun -n 4 ./target/release/lbm-ibm-fsi --config configs/lid_driven_cavity.toml
```

### Build options

```bash
# Enable CUDA GPU backend
cmake -B build -DENABLE_CUDA=ON
cargo build --release --features cuda
```

## Testing

```bash
# Run C++ unit tests
cmake --build build --target tests && ctest --test-dir build

# Run Rust unit + integration tests
cargo test

# Run Python pre/post-processing tests
cd python && python3 -m pytest tests/ -v
```

## Python Pre/Post-Processing

### Installation

```bash
cd python
pip install -e ".[test]"          # installs gmsh, numpy, matplotlib, scipy, pytest
```

### Pre-processing

```python
from lbm_pre.mesh import MeshBuilder, export_surface_nodes
from lbm_pre.geometry import circle_markers, filament_markers, write_markers_csv
from lbm_pre.config_gen import SolverConfig, reynolds_to_nu

# Generate a solver config for lid-driven cavity Re=100
cfg = SolverConfig.lid_driven_cavity(nx=100, ny=100, Re=100.0)
cfg.write("configs/my_run.toml")

# Build a gmsh mesh of the domain with a circular obstacle
with MeshBuilder(200, 80).set_mesh_size(3.0) as mb:
    mb.add_circle_obstacle(cx=50, cy=40, radius=10, mesh_size=0.8)
    mb.build().write("domain.msh")

# Generate IBM Lagrangian markers from geometry helpers
markers = circle_markers(cx=50, cy=40, radius=10, n=128)
write_markers_csv(markers, "markers.csv")
```

### Post-processing

```python
from lbm_post.vtk_reader import NpzReader
from lbm_post.plot import plot_velocity_magnitude, plot_vorticity, plot_streamlines, save_figure
from lbm_post.analysis import compute_bulk_quantities, drag_lift_coefficients

# Load solver snapshots
reader = NpzReader("output/my_run/")
snap = reader.last()

# Plots
fig, _ = plot_velocity_magnitude(snap)
save_figure(fig, "vel_mag.png")

fig, _ = plot_vorticity(snap)
save_figure(fig, "vorticity.png")

fig, _ = plot_streamlines(snap)
save_figure(fig, "streamlines.png")

# Bulk statistics
bq = compute_bulk_quantities(snap)
print(f"KE={bq.ke:.4f}  enstrophy={bq.enstrophy:.4f}  max|∇·u|={bq.div_max:.2e}")
```

### Full worked example

```bash
cd python
PYTHONPATH=. python3 examples/lid_driven_cavity.py
```

This generates config, mesh, synthetic snapshots, and five publication-ready plots in
`python/examples/output/lid_cavity/`.

### Visualising real solver output

After building and running the solver:

```bash
# Build
cargo build --release

# Run the lid-driven cavity simulation
./target/release/lbm-ibm-fsi --config configs/lid_driven_cavity.toml
```

The solver writes `output/lid_cavity/fluid_NNNNNN.npz` snapshots at every
`write_interval` step.  To visualise the results:

```bash
# Plot from anywhere — the script finds the output directory automatically
python3 python/examples/plot_solver_output.py

# Or point at a specific directory
python3 python/examples/plot_solver_output.py --input output/lid_cavity

# Save plots to a separate directory
python3 python/examples/plot_solver_output.py \
    --input output/lid_cavity --output my_plots/
```

The `[python]` section already added to `configs/lid_driven_cavity.toml` wires
`post_script` so that the solver runs this visualisation step automatically
once the simulation loop finishes.

Plots produced:

| File | Contents |
|------|----------|
| `velocity_magnitude.png` | Filled contour of \|u\| = √(ux²+uy²) |
| `streamlines.png` | Integrated streamlines coloured by \|u\| |
| `vorticity.png` | Vorticity ωz = ∂uy/∂x − ∂ux/∂y |
| `ux_profile.png` | ux vertical profile at cavity centre (x = nx/2) |
| `uy_profile.png` | uy horizontal profile at cavity centre (y = ny/2) |
| `monitor_velocity.png` | \|u\| time series at the centre monitor point |

## References

- Aidun, C.K. & Clausen, J.R. (2010). *Lattice-Boltzmann method for complex flows*. Annual Review of Fluid Mechanics.
- Peskin, C.S. (2002). *The immersed boundary method*. Acta Numerica.
- Hu, H.H., Patankar, N.A. & Zhu, M.Y. (2001). *Direct numerical simulations of fluid-solid systems*. JCP.
