# core/CMakeLists.txt — 编译 C++ 计算核心静态库 liblbm_core.a
# 详见 docs/构建系统详解.md

# 源文件列表（按物理模块分组）
set(CORE_SOURCES
    # LBM — 格子 Boltzmann 方法
    src/lbm/lattice.cpp       # 格子模型初始化（D2Q9/D3Q19）与宏观量（ρ、u）计算
    src/lbm/solver.cpp        # BGK/MRT 碰撞 + 流式迁移 + Guo 体力修正
    src/lbm/boundary.cpp      # 反弹边界、Zou-He 速度/压力边界条件
    src/lbm/mpi_decomp.cpp    # MPI 一/二维域分解与幽灵层交换（始终编译；MPI 未启用时为空实现）
    src/lbm/mg_tree.cpp       # ★ 多重网格嵌套关系树（MgTree/MgNode，支持 2D/3D AMR）
    # IBM — 浸入边界法
    src/ibm/marker.cpp        # 拉格朗日标记点集创建（圆形截面、柔性丝状体）
    src/ibm/interpolation.cpp # 速度插值（欧拉→拉格朗日）与力展布（拉格朗日→欧拉）
    # FSI — 流固耦合
    src/fsi/structure.cpp     # Euler-Bernoulli 梁有限元（Newmark-β 时间积分）
    src/fsi/coupling.cpp      # 分区显式 FSI 耦合步：插值→直接力→展布→结构推进→速度同步
    # C ABI 桥接层（供 Rust FFI 调用）
    src/capi/lbm_capi.cpp     # lbm_grid_* / lbm_solver_* / lbm_mpi_* / lbm_gpu_* C 函数实现
    # 插件系统
    src/plugins/plugin_registry.cpp  # C 函数指针→C++ 虚函数适配器 + 全局单例注册中心
)

# CUDA 源文件（仅在 ENABLE_CUDA=ON 时加入）
if(ENABLE_CUDA)
    list(APPEND CORE_SOURCES src/lbm/gpu_solver.cu)
endif()

# 静态库目标（产物：liblbm_core.a / lbm_core.lib）
add_library(lbm_core STATIC ${CORE_SOURCES})

# 头文件搜索路径（PUBLIC：对 lbm_core 及所有链接它的目标均可见）
target_include_directories(lbm_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# MPI：链接导入目标（自动传递 -I/-L/-l）并定义预处理宏供 #ifdef 使用
if(ENABLE_MPI)
    target_link_libraries(lbm_core PUBLIC MPI::MPI_CXX)       # PUBLIC：传播给依赖 lbm_core 的目标
    target_compile_definitions(lbm_core PUBLIC LBM_ENABLE_MPI)
endif()

# OpenMP：传递 -fopenmp 等编译/链接标志，使 #pragma omp 指令生效
if(ENABLE_OPENMP)
    target_link_libraries(lbm_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(lbm_core PUBLIC LBM_ENABLE_OPENMP)
endif()

# CUDA（宏定义 + 链接 CUDA Runtime）
if(ENABLE_CUDA)
    target_link_libraries(lbm_core PUBLIC CUDA::cudart)
    target_compile_definitions(lbm_core PUBLIC LBM_ENABLE_CUDA)
    # CUDA 源文件使用 C++17 标准
    set_property(SOURCE src/lbm/gpu_solver.cu PROPERTY CUDA_STANDARD 17)
endif()

# 输出路径（与 build.rs 中的 cargo:rustc-link-search 对应）
set_target_properties(lbm_core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# 安装规则（供非 Cargo 环境复用）
install(TARGETS lbm_core ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
