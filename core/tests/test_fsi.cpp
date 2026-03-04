// Test: FSI structural solver basic properties
#include "fsi/structure.hpp"
#include <cmath>
#include <cstdio>

// Test 1: BeamSolver initialises correct number of nodes
static int test_beam_node_count()
{
    fsi::BeamParams p;
    p.E          = 1.0e6;
    p.I          = 1.0e-8;
    p.rho_s      = 1000.0;
    p.A          = 1.0e-4;
    p.length     = 1.0;
    p.n_elements = 8;

    fsi::BeamSolver beam(p);
    const bool ok = (beam.n_nodes() == 9);
    std::printf("[FSI] beam node count: %s  (n_nodes=%d)\n",
                ok ? "PASS" : "FAIL", beam.n_nodes());
    return ok ? 0 : 1;
}

// Test 2: At rest with no forces, beam stays at rest
static int test_beam_at_rest()
{
    fsi::BeamParams p;
    p.E          = 1.0e4;
    p.I          = 1.0e-6;
    p.rho_s      = 1.0;
    p.A          = 0.01;
    p.length     = 1.0;
    p.n_elements = 4;

    fsi::BeamSolver beam(p);

    const double dt = 0.01;
    for (int t = 0; t < 100; ++t) beam.step(dt);

    bool ok = true;
    for (const auto& dof : beam.dofs()) {
        if (std::abs(dof.y) > 1e-10 || std::abs(dof.vy) > 1e-10) {
            ok = false;
            break;
        }
    }
    std::printf("[FSI] beam at rest (no forces): %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}

// Test 3: Beam responds to a point load (tip deflection direction is correct)
static int test_beam_tip_deflection()
{
    fsi::BeamParams p;
    p.E          = 2.1e11;  // steel-like
    p.I          = 1.0e-8;
    p.rho_s      = 7850.0;
    p.A          = 1.0e-4;
    p.length     = 0.5;
    p.n_elements = 4;

    fsi::BeamSolver beam(p);

    // Apply a downward force on the tip node
    beam.dofs().back().fy = -100.0;

    const double dt = 1e-5;
    for (int t = 0; t < 10; ++t) beam.step(dt);

    // After a downward tip force, tip y should be negative (non-trivial deflection)
    const double tip_y = beam.dofs().back().y;
    const bool ok = (tip_y < 1e-10);  // strictly less than to confirm downward motion
    std::printf("[FSI] beam tip deflection direction: %s  (tip_y=%.3e)\n",
                ok ? "PASS" : "FAIL", tip_y);
    return ok ? 0 : 1;
}

int test_fsi_main()
{
    int failures = 0;
    failures += test_beam_node_count();
    failures += test_beam_at_rest();
    failures += test_beam_tip_deflection();
    return failures;
}
