# ===========================================================================
# core/tests/CMakeLists.txt — C++ 单元测试
# ===========================================================================
# 作用：
#   将 4 个测试源文件编译为单一可执行程序 test_lbm，
#   并将其注册为 CTest 测试用例，可通过 ctest 命令运行。
#
# 执行时机：
#   由顶层 CMakeLists.txt 中的 add_subdirectory(core/tests) 触发。
#   仅当 BUILD_TESTS=ON（顶层 cmake 默认值）时才被处理。
#   cargo build 通过 build.rs 传入 BUILD_TESTS=OFF，因此此文件
#   在 Rust 构建流程中完全被跳过，不会影响 liblbm_core.a 的生成速度。
#
# 手动运行 C++ 测试的完整步骤：
#   cmake -S . -B build -DBUILD_TESTS=ON
#   cmake --build build --parallel 4
#   ctest --test-dir build --output-on-failure
# ===========================================================================

# ---------------------------------------------------------------------------
# 测试可执行文件
# ---------------------------------------------------------------------------
# 将四个测试模块编译为一个可执行文件 test_lbm：
#   test_runner.cpp  — 主函数，调用各测试模块，统计 PASS / FAIL 数量
#   test_lbm.cpp     — LBM 求解器测试（质量守恒、f_eq 归一化）
#   test_ibm.cpp     — IBM 标记点测试（圆形/丝状体创建、δ 函数归一化）
#   test_fsi.cpp     — FSI 结构求解器测试（节点数、静止平衡、尖端挠度方向）
add_executable(test_lbm
    test_runner.cpp
    test_lbm.cpp
    test_ibm.cpp
    test_fsi.cpp
)

# ---------------------------------------------------------------------------
# 链接核心库
# ---------------------------------------------------------------------------
# PRIVATE：lbm_core 的链接依赖仅对 test_lbm 可执行文件本身有效，
# 不会传播给依赖 test_lbm 的其他目标（此处无其他目标，仅作规范声明）。
# 链接后，lbm_core 的 PUBLIC 头文件路径（core/include/）自动对 test_lbm 可见，
# 无需再次调用 target_include_directories。
target_link_libraries(test_lbm PRIVATE lbm_core)

# ---------------------------------------------------------------------------
# CTest 测试注册
# ---------------------------------------------------------------------------
# 向 CTest 框架注册名为 lbm_tests 的测试项，运行命令为 test_lbm。
# 之后可通过以下命令执行：
#   ctest --test-dir build                  （静默模式）
#   ctest --test-dir build --output-on-failure  （失败时打印输出）
#   ctest --test-dir build -V               （详细模式，打印所有输出）
add_test(NAME lbm_tests COMMAND test_lbm)
