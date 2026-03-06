# core/tests/CMakeLists.txt — C++ 单元测试
# 详见 docs/构建系统详解.md
# 手动运行：cmake -S . -B build -DBUILD_TESTS=ON && cmake --build build && ctest --test-dir build -V

add_executable(test_lbm
    test_runner.cpp   # 主函数，统计 PASS/FAIL
    test_lbm.cpp      # LBM：质量守恒、f_eq 归一化
    test_ibm.cpp      # IBM：标记点创建、δ 函数归一化
    test_fsi.cpp      # FSI：节点数、静止平衡、尖端挠度方向
)

target_link_libraries(test_lbm PRIVATE lbm_core) # PRIVATE：lbm_core 依赖不向外传播

add_test(NAME lbm_tests COMMAND test_lbm) # 注册 CTest 测试用例，可用 ctest --test-dir build 运行
