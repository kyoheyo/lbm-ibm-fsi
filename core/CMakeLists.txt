# -----------------------------------------------------------------------
# core/CMakeLists.txt  — C++ computational core library
# -----------------------------------------------------------------------

set(CORE_SOURCES
    src/lbm/lattice.cpp
    src/lbm/solver.cpp
    src/lbm/boundary.cpp
    src/ibm/marker.cpp
    src/ibm/interpolation.cpp
    src/fsi/structure.cpp
    src/fsi/coupling.cpp
    src/capi/lbm_capi.cpp
)

add_library(lbm_core STATIC ${CORE_SOURCES})

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

# CUDA
if(ENABLE_CUDA)
    target_compile_definitions(lbm_core PUBLIC LBM_ENABLE_CUDA)
endif()

# Export a pkg-config-like target for the Rust build.rs to find
set_target_properties(lbm_core PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

install(TARGETS lbm_core
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)
