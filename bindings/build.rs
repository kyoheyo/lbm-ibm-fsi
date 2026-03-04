// build.rs — 通过 cmake crate 驱动 C++ 核心库的编译，并向 rustc 发出链接指令
//
// ============================================================
// 执行时机
// ============================================================
// Cargo 在编译 `lbm-bindings` crate 之前会自动运行此脚本。
// 触发条件（见脚本末尾的 `cargo:rerun-if-*` 指令）：
//   - `core/src/` 或 `core/include/` 下有文件被修改；
//   - 环境变量 `LBM_ENABLE_MPI` 或 `LBM_ENABLE_OPENMP` 发生变化。
// 若以上条件均未触发，Cargo 直接使用上次的缓存结果，不会重新编译 C++。
//
// ============================================================
// 与 Cargo 的交互协议
// ============================================================
// 本脚本通过向标准输出打印以 `cargo:` 开头的特殊指令与 Cargo 通信：
//   cargo:rustc-link-search=native=<路径>  → 告诉 rustc 在该目录中搜索 .a 文件
//   cargo:rustc-link-lib=static=<名称>     → 告诉 rustc 链接 lib<名称>.a 静态库
//   cargo:rustc-link-lib=<名称>            → 告诉 rustc 链接动态系统库
//   cargo:rerun-if-changed=<路径>          → 若指定路径变化则重新运行本脚本
//   cargo:rerun-if-env-changed=<变量名>    → 若环境变量变化则重新运行本脚本
use std::path::PathBuf;

fn main() {
    // ------------------------------------------------------------------
    // 第一步：确定项目根目录
    // ------------------------------------------------------------------
    // `CARGO_MANIFEST_DIR` 是 Cargo 为每个 crate 自动设置的环境变量，
    // 其值为该 crate 的 Cargo.toml 所在目录（即 bindings/）的绝对路径。
    // 向上一级即为包含顶层 CMakeLists.txt 的项目根目录。
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let repo_root = manifest_dir.parent().unwrap();

    // ------------------------------------------------------------------
    // 第二步：使用 cmake crate 调用 CMake 构建 C++ 核心库
    // ------------------------------------------------------------------
    // `cmake::Config::new(src_dir)` 指定 CMake 源码树的根目录，
    // 该目录下必须有顶层 CMakeLists.txt。
    //
    // `.define(KEY, VALUE)` 等价于 cmake 命令行的 `-DKEY=VALUE`，
    // 用于覆盖 CMakeLists.txt 中通过 `option()` 定义的默认值。
    //
    // `.build()` 内部依次执行：
    //   1. cmake <src_dir> -DKEY=VALUE ... -B <out_dir>/build
    //      （out_dir 由 Cargo 自动管理，位于 target/*/build/lbm-bindings-*/out/）
    //   2. cmake --build <out_dir>/build
    //      （使用 make / ninja 等本地构建工具编译所有目标）
    //   3. cmake --install <out_dir>/build --prefix <out_dir>
    //      （将 liblbm_core.a 安装到 <out_dir>/lib/）
    // 返回值 `dst` 是 CMake 的安装前缀，即 <out_dir>。
    let dst = cmake::Config::new(repo_root)
        // 关闭 C++ 测试构建：Rust 构建链路中只需要 liblbm_core.a，
        // 不需要编译 test_lbm 可执行文件，禁用可显著加快构建速度。
        .define("BUILD_TESTS", "OFF")
        // MPI / OpenMP 默认关闭，避免在没有相关依赖的 CI 或开发机上构建失败。
        // 若需要启用并行计算，可在 shell 中设置环境变量后重新运行 cargo build：
        //   LBM_ENABLE_MPI=ON LBM_ENABLE_OPENMP=ON cargo build
        // 注意：独立执行 cmake（不经过 cargo build）时，CMakeLists.txt 的
        // 默认值是 ON，与此处行为不同——参见项目文档中的对比表。
        .define("ENABLE_MPI",    std::env::var("LBM_ENABLE_MPI").unwrap_or_else(|_| "OFF".into()))
        .define("ENABLE_OPENMP", std::env::var("LBM_ENABLE_OPENMP").unwrap_or_else(|_| "OFF".into()))
        .build();

    // ------------------------------------------------------------------
    // 第三步：向 rustc 发出链接器搜索路径指令
    // ------------------------------------------------------------------
    // cmake::Config::build() 安装完成后，liblbm_core.a 位于：
    //   <dst>/lib/liblbm_core.a
    // 以下指令告诉 rustc 在该目录中搜索 .a / .so 文件。
    println!("cargo:rustc-link-search=native={}/lib", dst.display());

    // ------------------------------------------------------------------
    // 第四步：声明静态链接 C++ 核心库
    // ------------------------------------------------------------------
    // 告诉 rustc 链接 liblbm_core.a（`static=` 前缀表示静态链接）。
    // 该库由上方 CMake 步骤编译生成，包含所有 C++ 计算内核：
    //   LBM 格子 Boltzmann 求解器 + IBM 浸入边界法 + FSI 流固耦合 +
    //   C ABI 桥接层 + 插件注册中心
    println!("cargo:rustc-link-lib=static=lbm_core");

    // ------------------------------------------------------------------
    // 第五步：链接 C++ 标准库（平台相关）
    // ------------------------------------------------------------------
    // C++ 代码使用了标准库容器（std::vector、std::array 等），
    // 必须链接对应的 C++ 运行时库，否则会出现符号未定义错误。
    //   - macOS（Apple Clang）：使用 libc++
    //   - Linux / Windows（GCC / MSVC）：使用 libstdc++
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-lib=c++");
    #[cfg(not(target_os = "macos"))]
    println!("cargo:rustc-link-lib=stdc++");

    // ------------------------------------------------------------------
    // 第六步：增量构建守卫
    // ------------------------------------------------------------------
    // 以下指令告诉 Cargo 只有当指定路径或环境变量发生变化时，
    // 才需要重新运行本脚本（进而重新触发 cmake 增量编译）。
    // 没有变化时，Cargo 直接复用上次的 liblbm_core.a 缓存，
    // 大幅缩短二次构建的等待时间。
    println!("cargo:rerun-if-changed=../core/src");     // C++ 源文件目录
    println!("cargo:rerun-if-changed=../core/include"); // C++ 头文件目录
    println!("cargo:rerun-if-env-changed=LBM_ENABLE_MPI");    // MPI 开关
    println!("cargo:rerun-if-env-changed=LBM_ENABLE_OPENMP"); // OpenMP 开关
}
