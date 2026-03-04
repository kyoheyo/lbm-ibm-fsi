# 顶层 CMakeLists.txt — 项目全局配置
# 详见 docs/构建系统详解.md

cmake_minimum_required(VERSION 3.20)
project(lbm_ibm_fsi VERSION 0.1.0 LANGUAGES CXX)

# C++17，禁用编译器扩展
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译开关（cargo build 时由 build.rs 覆盖默认值，详见文档）
# 可在命令行通过 -D<选项>=ON/OFF 覆盖，或在 cargo build 时通过 LBM_ENABLE_* 环境变量覆盖
option(ENABLE_MPI    "Enable MPI parallelism"   ON)   # cargo build 默认 OFF
option(ENABLE_OPENMP "Enable OpenMP parallelism" ON)  # cargo build 默认 OFF
option(ENABLE_CUDA   "Enable CUDA GPU backend"   OFF) # 当前后端尚未实现，预留扩展
option(BUILD_TESTS   "Build unit tests"          ON)  # cargo build 固定传入 OFF

# GCC/Clang 优化与警告
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-O3 -march=native -Wall -Wextra)
endif()

# MSVC：强制所有配置使用 /MD，防止与 Rust 链接时出现 LNK1120
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
    add_compile_options(/EHsc)
endif()

# 依赖查找
if(ENABLE_MPI)
    find_package(MPI REQUIRED)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
endif()

if(ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
    message(STATUS "OpenMP found: version ${OpenMP_CXX_VERSION}")
endif()

if(ENABLE_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA found: ${CUDAToolkit_VERSION}")
endif()

# 子目录
add_subdirectory(core)

if(BUILD_TESTS)
    enable_testing()               # 初始化 CTest 框架，使 add_test() 生效
    add_subdirectory(core/tests)   # 编译 test_lbm 可执行文件并注册测试用例
endif()
