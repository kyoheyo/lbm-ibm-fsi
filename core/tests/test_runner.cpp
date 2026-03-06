// 最小测试运行器 — 逐一打印每个测试的 PASS/FAIL 结果
#include <cstdio>
#include <cstdlib>

// 前向声明
int test_lbm_main();
int test_ibm_main();
int test_fsi_main();

int main()
{
    int failures = 0;
    failures += test_lbm_main();
    failures += test_ibm_main();
    failures += test_fsi_main();

    if (failures == 0) {
        std::puts("All tests PASSED");
        return EXIT_SUCCESS;
    } else {
        std::printf("%d test(s) FAILED\n", failures);
        return EXIT_FAILURE;
    }
}
