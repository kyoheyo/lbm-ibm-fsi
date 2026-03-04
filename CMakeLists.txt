# ===========================================================================
# 顶层 CMakeLists.txt — 项目全局配置
# ===========================================================================
# 作用：
#   1. 声明 CMake 最低版本要求和项目元信息；
#   2. 配置 C++ 语言标准（C++17）；
#   3. 定义可通过 -D 选项覆盖的编译开关（MPI、OpenMP、CUDA、测试）；
#   4. 配置编译器优化与警告选项；
#   5. 查找系统中的 MPI / OpenMP / CUDA 依赖；
#   6. 递归进入 core/ 子目录，驱动 C++ 核心库的编译；
#   7. （可选）递归进入 core/tests/ 子目录，编译 C++ 单元测试。
#
# 执行时机：
#   - 直接运行 cmake 时（独立构建）：最先被处理，MPI/OpenMP 默认开启；
#   - 通过 cargo build 驱动时：由 bindings/build.rs 中的 cmake::Config 调用，
#     BUILD_TESTS 被强制关闭，MPI/OpenMP 默认也被关闭（可通过环境变量覆盖）。
# ===========================================================================

# ---------------------------------------------------------------------------
# CMake 版本要求
# ---------------------------------------------------------------------------
# 3.20 起支持更完善的 target 属性传播和 FetchContent，也是现代 CMake 的基准版本。
cmake_minimum_required(VERSION 3.20)

# ---------------------------------------------------------------------------
# 项目声明
# ---------------------------------------------------------------------------
# 指定项目名称、版本号及使用的编程语言（此处仅 CXX，CUDA 按需启用）。
project(lbm_ibm_fsi VERSION 0.1.0 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# C++ 语言标准配置
# ---------------------------------------------------------------------------
# 要求 C++17（用到了结构化绑定、if constexpr 等特性）。
# REQUIRED = ON：编译器必须支持该标准，否则报错。
# EXTENSIONS = OFF：禁用 GNU/MSVC 特有扩展，保持代码可移植性。
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---------------------------------------------------------------------------
# 编译开关（Options）
# ---------------------------------------------------------------------------
# 这些开关可在 cmake 命令行通过 -D<选项名>=ON/OFF 覆盖。
# cargo build 通过 build.rs 传入不同的默认值（见注释）。

# ENABLE_MPI：启用 MPI 并行（多进程分布式计算）
#   直接 cmake 时默认 ON；cargo build 时默认 OFF（可通过 LBM_ENABLE_MPI=ON 覆盖）
option(ENABLE_MPI    "Enable MPI parallelism"   ON)

# ENABLE_OPENMP：启用 OpenMP 多线程并行（单节点共享内存）
option(ENABLE_OPENMP "Enable OpenMP parallelism" ON)

# ENABLE_CUDA：启用 CUDA GPU 加速后端（当前默认关闭，需要 NVCC）
option(ENABLE_CUDA   "Enable CUDA GPU backend"   OFF)

# BUILD_TESTS：是否编译 C++ 单元测试（test_lbm 可执行文件）
#   cargo build 时传入 OFF 以减少不必要的编译；独立构建时保持 ON
option(BUILD_TESTS   "Build unit tests"          ON)

# ---------------------------------------------------------------------------
# 编译器优化与警告选项
# ---------------------------------------------------------------------------
# -O3：最高优化级别（内联、向量化、循环展开等）
# -march=native：针对当前主机 CPU 指令集生成最优代码（不可跨机器移植）
# -Wall -Wextra：开启几乎所有有用的编译警告，有助于尽早发现潜在问题
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# ---------------------------------------------------------------------------
# MPI 依赖查找
# ---------------------------------------------------------------------------
# 若 ENABLE_MPI=ON，查找系统中安装的 MPI 实现（OpenMPI、MPICH 等）。
# REQUIRED：若找不到则立即终止 cmake 配置并打印错误信息。
# 查找成功后，core/CMakeLists.txt 中通过 MPI::MPI_CXX 目标隐式传递
# 编译/链接参数，无需手动填写 -I / -L / -l 选项。
if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
endif()

# ---------------------------------------------------------------------------
# OpenMP 依赖查找
# ---------------------------------------------------------------------------
# 若 ENABLE_OPENMP=ON，查找编译器内置的 OpenMP 支持。
# 查找成功后，通过 OpenMP::OpenMP_CXX 目标传递 -fopenmp 等选项。
if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP found: version ${OpenMP_CXX_VERSION}")
endif()

# ---------------------------------------------------------------------------
# CUDA 依赖查找（可选）
# ---------------------------------------------------------------------------
# 仅当 ENABLE_CUDA=ON 时才启用 CUDA 语言支持并查找 CUDA Toolkit。
# 当前框架中 CUDA 后端尚未实现，该开关仅作为未来扩展预留。
if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
endif()

# ---------------------------------------------------------------------------
# C++ 核心库子目录
# ---------------------------------------------------------------------------
# 进入 core/ 目录并处理其 CMakeLists.txt（第二步）。
# 该步骤编译所有 C++ 源文件并生成静态库 liblbm_core.a。
add_subdirectory(core)

# ---------------------------------------------------------------------------
# C++ 单元测试子目录（条件编译）
# ---------------------------------------------------------------------------
# 仅当 BUILD_TESTS=ON 时才进入 core/tests/（第三步）。
# cargo build 通过 build.rs 传入 BUILD_TESTS=OFF，故该子目录在 Rust 构建时被跳过。
if(BUILD_TESTS)
    enable_testing()          # 初始化 CTest 框架，使 add_test() 生效
    add_subdirectory(core/tests)
endif()
