// Test: IBM marker creation and delta-function properties
#include "ibm/marker.hpp"
#include "ibm/interpolation.hpp"
#include <cmath>
#include <cstdio>
#include <numeric>

static bool approx(double a, double b, double tol = 1e-10) {
    return std::abs(a - b) < tol;
}

// Test 1: Circle marker count matches request
static int test_circle_marker_count()
{
    const int n = 64;
    auto ms = ibm::MarkerSet::make_circle(0.5, 0.5, 0.2, n);
    const bool ok = (ms.size() == n);
    std::printf("[IBM] circle marker count: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Test 2: Circle markers lie on the circle
static int test_circle_marker_positions()
{
    const double cx = 0.5, cy = 0.5, r = 0.25;
    auto ms = ibm::MarkerSet::make_circle(cx, cy, r, 32);
    bool ok = true;
    for (const auto& m : ms.markers) {
        const double dx = m.x - cx;
        const double dy = m.y - cy;
        const double dist = std::sqrt(dx * dx + dy * dy);
        if (std::abs(dist - r) > 1e-12) { ok = false; break; }
    }
    std::printf("[IBM] circle marker positions on circle: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Test 3: Delta function integrates to ~1 over its support
static int test_delta_normalisation()
{
    const double h = 1.0;
    double sum = 0.0;
    const int N = 1000;
    const double dr = 4.0 * h / N;
    for (int i = 0; i < N; ++i) {
        const double r = -2.0 * h + (i + 0.5) * dr;
        sum += ibm::delta_phi(r, h, ibm::DeltaKernel::FourPoint) * dr;
    }
    const bool ok = approx(sum, 1.0, 1e-6);
    std::printf("[IBM] delta_phi 4-point integral ≈ 1: %s  (sum=%.8f)\n",
                ok ? "PASS" : "FAIL", sum);
    return ok ? 0 : 1;
}

// Test 4: Filament marker spacing is uniform
static int test_filament_spacing()
{
    const int n = 11;
    const double len = 1.0;
    auto ms = ibm::MarkerSet::make_filament(0.0, 0.5, len, n);
    const double expected_dl = len / (n - 1);
    bool ok = true;
    for (const auto& m : ms.markers) {
        if (std::abs(m.ds - expected_dl) > 1e-12) { ok = false; break; }
    }
    std::printf("[IBM] filament uniform spacing: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

int test_ibm_main()
{
    int failures = 0;
    failures += test_circle_marker_count();
    failures += test_circle_marker_positions();
    failures += test_delta_normalisation();
    failures += test_filament_spacing();
    return failures;
}
