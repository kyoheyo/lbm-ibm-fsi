# core/CMakeLists.txt — 编译 C++ 计算核心静态库 liblbm_core.a
# 详见 docs/构建系统详解.md

# 源文件列表（按物理模块分组）
set(CORE_SOURCES
    # LBM
    src/lbm/lattice.cpp
    src/lbm/solver.cpp
    src/lbm/boundary.cpp
    # IBM
    src/ibm/marker.cpp
    src/ibm/interpolation.cpp
    # FSI
    src/fsi/structure.cpp
    src/fsi/coupling.cpp
    # C ABI 桥接层
    src/capi/lbm_capi.cpp
    # 插件系统
    src/plugins/plugin_registry.cpp
)

# 静态库目标
add_library(lbm_core STATIC ${CORE_SOURCES})

# 头文件搜索路径（PUBLIC：对 lbm_core 及所有链接它的目标均可见）
target_include_directories(lbm_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# MPI
if(ENABLE_MPI)
    target_link_libraries(lbm_core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(lbm_core PUBLIC LBM_ENABLE_MPI)
endif()

# OpenMP
if(ENABLE_OPENMP)
    target_link_libraries(lbm_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(lbm_core PUBLIC LBM_ENABLE_OPENMP)
endif()

# CUDA（宏定义，后端尚未完整实现）
if(ENABLE_CUDA)
    target_compile_definitions(lbm_core PUBLIC LBM_ENABLE_CUDA)
endif()

# 输出路径（与 build.rs 中的 cargo:rustc-link-search 对应）
set_target_properties(lbm_core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# 安装规则（供非 Cargo 环境复用）
install(TARGETS lbm_core ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
