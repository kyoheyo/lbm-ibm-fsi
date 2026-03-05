// build.rs — 驱动 C++ 核心库编译并向 rustc 发出链接指令
// 详见项目根目录 docs/构建系统详解.md
use std::path::PathBuf;

fn main() {
    // 步骤 1：确定项目根目录（bindings/ 的上一级）
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.parent().unwrap();

    // 步骤 2：推断 cmake 构建类型（与 Cargo OPT_LEVEL 对齐，防止 MSVC 运行时库冲突）
    // OPT_LEVEL=0 → cargo build（Debug）；OPT_LEVEL≥1 → cargo build --release（Release）
    let cmake_build_type = if std::env::var("OPT_LEVEL").unwrap_or_default() == "0" {
        "Debug"
    } else {
        "Release"
    };

    // 步骤 3：调用 cmake 编译 liblbm_core.a
    // .define() 等价于 cmake 命令行的 -D<KEY>=<VALUE>，覆盖 CMakeLists.txt 中的 option() 默认值
    let dst = cmake::Config::new(repo_root)
        .define("BUILD_TESTS", "OFF")                  // 不编译 test_lbm，加速构建
        .define("ENABLE_MPI",    std::env::var("LBM_ENABLE_MPI").unwrap_or_else(|_| "OFF".into()))    // 可用 LBM_ENABLE_MPI=ON 覆盖
        .define("ENABLE_OPENMP", std::env::var("LBM_ENABLE_OPENMP").unwrap_or_else(|_| "OFF".into())) // 可用 LBM_ENABLE_OPENMP=ON 覆盖
        .define("ENABLE_CUDA",   std::env::var("LBM_ENABLE_CUDA").unwrap_or_else(|_| "OFF".into()))   // 可用 LBM_ENABLE_CUDA=ON 覆盖（需 nvcc）
        .define("CMAKE_BUILD_TYPE", cmake_build_type)  // 单配置生成器（Ninja）使用此值
        .build(); // 依次执行：cmake configure → cmake --build → cmake --install，返回安装前缀

    // 步骤 4：链接器搜索路径（<dst>/lib/liblbm_core.a）
    println!("cargo:rustc-link-search=native={}/lib", dst.display());

    // 步骤 5：静态链接 C++ 核心库
    println!("cargo:rustc-link-lib=static=lbm_core");

    // 步骤 6：链接 C++ 标准库（平台相关；Windows/MSVC 自动处理）
    // 使用 CARGO_CFG_TARGET_OS（目标平台）而非 cfg!(target_os)（宿主平台），支持交叉编译
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "macos" | "ios" => println!("cargo:rustc-link-lib=c++"),   // Apple Clang 默认使用 libc++
        "windows"       => {}                                        // MSVC 工具链自动链接 CRT，无需声明
        _               => println!("cargo:rustc-link-lib=stdc++"), // Linux/FreeBSD 等使用 libstdc++
    }

    // 步骤 7：增量构建守卫
    println!("cargo:rerun-if-changed=../core/src");
    println!("cargo:rerun-if-changed=../core/include");
    println!("cargo:rerun-if-env-changed=LBM_ENABLE_MPI");
    println!("cargo:rerun-if-env-changed=LBM_ENABLE_OPENMP");
    println!("cargo:rerun-if-env-changed=LBM_ENABLE_CUDA");
}
