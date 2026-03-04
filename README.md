# lbm-ibm-fsi

A high-performance Lattice Boltzmann Method (LBM) + Immersed Boundary Method (IBM) + Fluid-Structure Interaction (FSI) solver framework.

## Architecture Overview

This project uses a **multi-language hybrid architecture** to balance performance, safety, and extensibility:

| Layer | Language | Role |
|-------|----------|------|
| Computational Core | **C++17** | LBM solver, IBM coupling, FSI structural solver; MPI+OpenMP parallelism |
| Orchestration & CLI | **Rust** | Safe driver, configuration loading, FFI bindings to C++ core, I/O |
| Build System | **CMake + Cargo** | C++ build (with MPI/OpenMP/CUDA detection); Rust workspace |

### Why this combination?

- **C++** has the most mature ecosystem for HPC numerics (MPI, OpenMP, CUDA, PETSc, Eigen, FFTW). Its zero-overhead abstractions and direct SIMD/intrinsic access make it ideal for the inner computational loops.
- **Rust** provides memory-safety guarantees for the driver/orchestration layer, preventing data-race bugs when managing simulation state across threads. Its `cargo` tooling makes dependency management and cross-platform builds straightforward. The `mpi` crate and `rayon` crate offer additional parallelism options at the orchestration level.
- The **Rust ↔ C++ FFI** boundary is thin and well-defined via a C ABI layer, keeping the two languages loosely coupled.

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
│       └── fsi/
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
```

## References

- Aidun, C.K. & Clausen, J.R. (2010). *Lattice-Boltzmann method for complex flows*. Annual Review of Fluid Mechanics.
- Peskin, C.S. (2002). *The immersed boundary method*. Acta Numerica.
- Hu, H.H., Patankar, N.A. & Zhu, M.Y. (2001). *Direct numerical simulations of fluid-solid systems*. JCP.
