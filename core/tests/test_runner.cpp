// Minimal test runner — prints PASS/FAIL for each test
#include <cstdio>
#include <cstdlib>

// Forward declarations
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
