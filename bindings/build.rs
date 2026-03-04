// build.rs — build the C++ core via CMake and emit linker directives
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.parent().unwrap();

    // Build the C++ core library with CMake
    let dst = cmake::Config::new(repo_root)
        .define("BUILD_TESTS", "OFF")
        // Disable MPI/OpenMP by default in Rust builds to avoid requiring them as
        // host prerequisites; enable explicitly via env vars or feature flags.
        // Standalone CMake builds default to ON as documented in CMakeLists.txt.
        .define("ENABLE_MPI",    std::env::var("LBM_ENABLE_MPI").unwrap_or_else(|_| "OFF".into()))
        .define("ENABLE_OPENMP", std::env::var("LBM_ENABLE_OPENMP").unwrap_or_else(|_| "OFF".into()))
        .build();

    // Tell rustc where to find the static library
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=lbm_core");

    // Link against C++ standard library
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");
    #[cfg(not(target_os = "macos"))]
    println!("cargo:rustc-link-lib=stdc++");

    // Re-run this build script if any C++ source files change
    println!("cargo:rerun-if-changed=../core/src");
    println!("cargo:rerun-if-changed=../core/include");
    println!("cargo:rerun-if-env-changed=LBM_ENABLE_MPI");
    println!("cargo:rerun-if-env-changed=LBM_ENABLE_OPENMP");
}
