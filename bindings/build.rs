// build.rs — 驱动 C++ 核心库编译并向 rustc 发出链接指令
// 详见项目根目录 docs/构建系统详解.md
use std::path::PathBuf;

fn main() {
    // 步骤 1：确定项目根目录（bindings/ 的上一级）
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.parent().unwrap();

    // 步骤 2：推断 cmake 构建类型（与 Cargo OPT_LEVEL 对齐，防止 MSVC 运行时库冲突）
    let cmake_build_type = if std::env::var("OPT_LEVEL").unwrap_or_default() == "0" {
        "Debug"
    } else {
        "Release"
    };

    // 步骤 3：调用 cmake 编译 liblbm_core.a
    let dst = cmake::Config::new(repo_root)
        .define("BUILD_TESTS", "OFF")
        .define("ENABLE_MPI",    std::env::var("LBM_ENABLE_MPI").unwrap_or_else(|_| "OFF".into()))
        .define("ENABLE_OPENMP", std::env::var("LBM_ENABLE_OPENMP").unwrap_or_else(|_| "OFF".into()))
        .define("CMAKE_BUILD_TYPE", cmake_build_type)
        .build();

    // 步骤 4：链接器搜索路径（<dst>/lib/liblbm_core.a）
    println!("cargo:rustc-link-search=native={}/lib", dst.display());

    // 步骤 5：静态链接 C++ 核心库
    println!("cargo:rustc-link-lib=static=lbm_core");

    // 步骤 6：链接 C++ 标准库（平台相关；Windows/MSVC 自动处理）
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" | "ios" => println!("cargo:rustc-link-lib=c++"),
        "windows"       => {}
        _               => println!("cargo:rustc-link-lib=stdc++"),
    }

    // 步骤 7：增量构建守卫
    println!("cargo:rerun-if-changed=../core/src");
    println!("cargo:rerun-if-changed=../core/include");
    println!("cargo:rerun-if-env-changed=LBM_ENABLE_MPI");
    println!("cargo:rerun-if-env-changed=LBM_ENABLE_OPENMP");
}
