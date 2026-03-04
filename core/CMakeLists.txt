# ===========================================================================
# core/CMakeLists.txt — C++ 计算核心库
# ===========================================================================
# 作用：
#   将 core/src/ 下的全部 C++ 源文件编译并打包为静态库，
#   并向依赖该库的目标（包括 Rust bindings crate）暴露头文件路径。
#
# 执行时机：
#   由顶层 CMakeLists.txt 中的 add_subdirectory(core) 触发，
#   在顶层文件完成全局配置（编译器标准、MPI/OpenMP 查找）之后执行。
#
# 产物（名称随平台变化）：
#   <CMAKE_BINARY_DIR>/lib/liblbm_core.a   （Unix：Linux / macOS，GCC / Clang）
#   <CMAKE_BINARY_DIR>/lib/lbm_core.lib    （Windows：MSVC）
#   （CMAKE_BINARY_DIR 由顶层 cmake -B <build_dir> 参数指定，
#    通过 cargo build 时为 Cargo 管理的临时目录下的 out/build/）
# ===========================================================================

# ---------------------------------------------------------------------------
# 源文件列表
# ---------------------------------------------------------------------------
# 列出需要编译的所有 .cpp 文件，按物理模块分组以便阅读。
set(CORE_SOURCES
    # LBM 模块：格子 Boltzmann 方法核心
    src/lbm/lattice.cpp          # 格子模型（D2Q9/D3Q19）初始化与宏观量计算
    src/lbm/solver.cpp           # BGK / MRT 碰撞 + 流式迁移 + Guo 体力修正
    src/lbm/boundary.cpp         # 边界条件（反弹、Zou-He 速度/压力）

    # IBM 模块：浸入边界法
    src/ibm/marker.cpp           # 拉格朗日标记点集的创建（圆形、丝状体）
    src/ibm/interpolation.cpp    # 速度插值（欧拉→拉格朗日）与力展布（拉格朗日→欧拉）

    # FSI 模块：流固耦合
    src/fsi/structure.cpp        # Euler-Bernoulli 梁有限元求解器（Newmark-β 积分）
    src/fsi/coupling.cpp         # 分区显式 FSI 耦合步（插值→直接力→展布→结构推进→同步）

    # C ABI 桥接层（供 Rust FFI 调用）
    src/capi/lbm_capi.cpp        # lbm_grid_*/lbm_solver_* C 函数实现

    # 插件系统
    src/plugins/plugin_registry.cpp  # C 函数指针 → C++ 虚函数适配器 + 全局注册
)

# ---------------------------------------------------------------------------
# 声明静态库目标
# ---------------------------------------------------------------------------
# 将上述所有源文件编译并归档为静态库 liblbm_core.a。
# STATIC：生成 .a 文件；链接到 Rust 时无需额外的动态库依赖。
add_library(lbm_core STATIC ${CORE_SOURCES})

# ---------------------------------------------------------------------------
# 头文件搜索路径
# ---------------------------------------------------------------------------
# PUBLIC：该路径对 lbm_core 自身及所有链接它的目标均可见。
# BUILD_INTERFACE：仅在当前构建树中有效（用于源码内构建）。
# INSTALL_INTERFACE：仅在 cmake --install 后的安装树中有效。
# 这样 bindings/src/lib.rs 对应的 C FFI 声明所需的头文件可以被编译器找到。
target_include_directories(lbm_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# ---------------------------------------------------------------------------
# MPI 链接与宏定义
# ---------------------------------------------------------------------------
# 若顶层启用了 MPI（ENABLE_MPI=ON），则：
#   - 链接 MPI::MPI_CXX（自动传递 -I<mpi_include> -L<mpi_lib> -lmpi_cxx 等）；
#   - 定义预处理宏 LBM_ENABLE_MPI，使 core/src/ 中的 #ifdef 分支生效。
if(ENABLE_MPI)
    target_link_libraries(lbm_core PUBLIC MPI::MPI_CXX)
    target_compile_definitions(lbm_core PUBLIC LBM_ENABLE_MPI)
endif()

# ---------------------------------------------------------------------------
# OpenMP 链接与宏定义
# ---------------------------------------------------------------------------
# 若启用了 OpenMP（ENABLE_OPENMP=ON），则：
#   - 链接 OpenMP::OpenMP_CXX（传递 -fopenmp 等编译/链接标志）；
#   - 定义预处理宏 LBM_ENABLE_OPENMP，使 #pragma omp 指令生效。
if(ENABLE_OPENMP)
    target_link_libraries(lbm_core PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(lbm_core PUBLIC LBM_ENABLE_OPENMP)
endif()

# ---------------------------------------------------------------------------
# CUDA 宏定义（仅影响编译器预处理，不链接 CUDA 库）
# ---------------------------------------------------------------------------
# 当前 CUDA 后端尚未完整实现，此处仅定义宏供将来的 #ifdef LBM_ENABLE_CUDA 使用。
if(ENABLE_CUDA)
    target_compile_definitions(lbm_core PUBLIC LBM_ENABLE_CUDA)
endif()

# ---------------------------------------------------------------------------
# 库文件输出路径
# ---------------------------------------------------------------------------
# 将 liblbm_core.a 放置在 <CMAKE_BINARY_DIR>/lib/ 目录下。
# build.rs 中的 `cargo:rustc-link-search=native=<dst>/lib` 指向该目录，
# 确保 rustc 能够找到并链接此静态库。
set_target_properties(lbm_core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# ---------------------------------------------------------------------------
# 安装规则（可选，供独立安装使用）
# ---------------------------------------------------------------------------
# cmake --install 时将 liblbm_core.a 和头文件分别安装到系统目录，
# 供非 Cargo 构建环境（如纯 C++ 项目）复用本库。
install(TARGETS lbm_core
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)
