add_executable(test_lbm
    test_runner.cpp
    test_lbm.cpp
    test_ibm.cpp
    test_fsi.cpp
)

target_link_libraries(test_lbm PRIVATE lbm_core)

add_test(NAME lbm_tests COMMAND test_lbm)
