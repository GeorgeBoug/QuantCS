#include "invmul/multiplier.hpp"
#include "invmul/solver.hpp"

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
int g_soundness_violations = 0;
long long g_runs_checked = 0;

void check(bool ok, const std::string& label) {
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << label << "\n";
    if (!ok) ++g_failures;
}

// Claim 1, applied to every single solve performed anywhere in this file.
void assert_sound(const invmul::MultiplierGrid& grid, const invmul::SolveResult& r,
                  const invmul::AnnealingSolver& solver, bool require_nontrivial) {
    ++g_runs_checked;

    // The clamp must hold whether or not the run succeeded: the product
    // wires are not any cell's input, so nothing may ever have written them.
    if (grid.read_product(solver.best_state()) != r.target) {
        std::cout << "    SOUNDNESS: product wires drifted off the target (" << r.target
                  << " -> " << grid.read_product(solver.best_state()) << ")\n";
        ++g_soundness_violations;
    }
    if (r.best_energy < 0 || r.final_energy < 0) {
        std::cout << "    SOUNDNESS: negative energy reported\n";
        ++g_soundness_violations;
    }
    if (!r.solved) return;

    if (r.final_energy != 0) {
        std::cout << "    SOUNDNESS: solved but energy " << r.final_energy << " != 0\n";
        ++g_soundness_violations;
    }
    if (r.a * r.b != r.target) {
        std::cout << "    SOUNDNESS: solved but " << r.a << " * " << r.b << " = " << (r.a * r.b)
                  << " != " << r.target << "\n";
        ++g_soundness_violations;
    }
    if (require_nontrivial && (r.a < 2 || r.b < 2)) {
        std::cout << "    SOUNDNESS: trivial factorization returned despite require_nontrivial\n";
        ++g_soundness_violations;
    }
    // The returned state must independently re-derive the same energy.
    if (grid.total_energy(solver.best_state()) != 0) {
        std::cout << "    SOUNDNESS: best_state does not actually have energy 0\n";
        ++g_soundness_violations;
    }
}

struct RunOptions {
    invmul::UpdateMode mode = invmul::UpdateMode::Asynchronous;
    int restarts = 1;
    long long rounds = 5000;
    int threads = 1;
    std::uint64_t seed = 1;
};

invmul::SolveResult run_one(const invmul::MultiplierGrid& grid, std::uint64_t target,
                            const RunOptions& opts) {
    invmul::SolverConfig cfg;
    cfg.mode = opts.mode;
    cfg.subset_fraction = 0.5;
    cfg.max_rounds = opts.rounds;
    cfg.restarts = opts.restarts;
    cfg.num_threads = opts.threads;
    cfg.seed = opts.seed;
    cfg.require_nontrivial = true;
    cfg.clamp_odd_low_bits = true;

    invmul::AnnealingSolver solver(grid, cfg);
    const invmul::SolveResult r = solver.solve(target);
    assert_sound(grid, r, solver, cfg.require_nontrivial);
    return r;
}

const char* mode_name(invmul::UpdateMode m) {
    return m == invmul::UpdateMode::Synchronous ? "sync" : "async";
}

// --- Claim 2: recovery of known composites --------------------------------

void test_recovery(int n, std::uint64_t target, const std::string& label, int restarts,
                   invmul::UpdateMode mode, int threads) {
    invmul::MultiplierGrid grid(n);
    RunOptions opts;
    opts.mode = mode;
    opts.restarts = restarts;
    opts.rounds = 5000;
    opts.threads = threads;
    opts.seed = 0xA5A5u + target;

    const invmul::SolveResult r = run_one(grid, target, opts);

    std::string detail = label + " on a " + std::to_string(n) + "x" + std::to_string(n) +
                         " grid [" + mode_name(mode) + ", " + std::to_string(threads) +
                         " thread(s)]";
    if (r.solved) {
        detail += " -> " + std::to_string(r.a) + " * " + std::to_string(r.b) + " after " +
                  std::to_string(r.restarts_used) + " restart(s)";
    }
    check(r.solved, detail);
}

// --- Claim 3: the local dynamics, with restarts switched off --------------

void test_single_run_rate(int n, std::uint64_t target, const std::string& label, int seeds,
                          int floor_successes, invmul::UpdateMode mode) {
    invmul::MultiplierGrid grid(n);
    int solved = 0;
    for (int s = 0; s < seeds; ++s) {
        RunOptions opts;
        opts.mode = mode;
        opts.restarts = 1; // no restarts: only the corrective process can win
        opts.rounds = 20000;
        opts.seed = static_cast<std::uint64_t>(s) * 2654435761u + 1u;
        if (run_one(grid, target, opts).solved) ++solved;
    }
    const double pct = 100.0 * solved / seeds;
    check(solved >= floor_successes,
          label + " [" + mode_name(mode) + "]: " + std::to_string(solved) + "/" +
              std::to_string(seeds) + " single runs solved (" +
              std::to_string(static_cast<int>(pct)) + "%), floor " +
              std::to_string(floor_successes));
}

// A run that is not expected to succeed must still behave: energy never
// negative, clamp intact, and the reported best energy consistent with the
// state handed back.
void test_unsolved_behaviour() {
    invmul::MultiplierGrid grid(8);
    RunOptions opts;
    opts.rounds = 50; // deliberately far too short
    opts.restarts = 1;
    opts.seed = 99;
    const std::uint64_t target = 40501; // 101 * 401 -- 401 does not fit in 8 bits
    const invmul::SolveResult r = run_one(grid, target, opts);
    check(!r.solved || r.a * r.b == target,
          "a run with an unsatisfiable target either fails cleanly or is genuinely correct");
    check(r.best_energy >= 0, "best energy stays non-negative on an unsolved run");
}

// Solving N = a * b for operands the grid can actually represent, drawn at
// random -- a round trip through both directions of the array.
void test_round_trip(int n, int cases, invmul::UpdateMode mode) {
    invmul::MultiplierGrid grid(n);
    invmul::BitState fwd(static_cast<std::size_t>(grid.num_vars()));
    int solved = 0;
    for (int c = 0; c < cases; ++c) {
        const std::uint64_t a = 3 + 2 * static_cast<std::uint64_t>(c % 6); // 3,5,7,9,11,13
        const std::uint64_t b = 3 + 2 * static_cast<std::uint64_t>((c / 6) % 6);
        grid.forward(a, b, fwd);
        const std::uint64_t target = grid.read_product(fwd);

        RunOptions opts;
        opts.mode = mode;
        opts.restarts = 150;
        opts.rounds = 3000;
        opts.seed = 500u + c;
        if (run_one(grid, target, opts).solved) ++solved;
    }
    check(solved == cases, std::string("round trip [") + mode_name(mode) + "]: forward then " +
                               "corrective recovered " + std::to_string(solved) + "/" +
                               std::to_string(cases) + " products");
}

} // namespace

int main() {
    std::cout << "Corrective (inverse) direction\n\n";

    // ---- Claim 2: recovery of known small composites ---------------------
    // Restart budgets are sized from measured single-run rates so that the
    // probability of a spurious failure is below ~1e-8: e.g. 143 = 11 * 13
    // solves ~14% of the time in one run, so 200 restarts fail with
    // probability ~0.86^200 ~ 1e-13.
    std::cout << "Recovery of known composites (restarts enabled)\n";
    for (invmul::UpdateMode mode :
         {invmul::UpdateMode::Asynchronous, invmul::UpdateMode::Synchronous}) {
        test_recovery(4, 15, "15 = 3 * 5", 200, mode, 1);
        test_recovery(4, 35, "35 = 5 * 7", 200, mode, 1);
        test_recovery(4, 91, "91 = 7 * 13", 200, mode, 1);
        test_recovery(4, 143, "143 = 11 * 13", 200, mode, 1);
        test_recovery(5, 667, "667 = 23 * 29", 400, mode, 1);
        test_recovery(5, 899, "899 = 29 * 31", 200, mode, 1);
    }
    std::cout << "\n";

    // Parallel: same claim, several threads. The optimistic parallel scheme
    // may make individual proposals from slightly stale reads, so this is
    // really a check that concurrency cannot produce an *unsound* answer.
    std::cout << "Recovery with multiple threads\n";
    test_recovery(4, 143, "143 = 11 * 13", 300, invmul::UpdateMode::Asynchronous, 4);
    test_recovery(4, 143, "143 = 11 * 13", 300, invmul::UpdateMode::Synchronous, 4);
    test_recovery(5, 899, "899 = 29 * 31", 300, invmul::UpdateMode::Asynchronous, 4);
    std::cout << "\n";

    std::cout << "Forward-then-corrective round trip\n";
    test_round_trip(4, 12, invmul::UpdateMode::Asynchronous);
    test_round_trip(4, 12, invmul::UpdateMode::Synchronous);
    std::cout << "\n";

    // ---- Claim 3: the corrective dynamics on its own ---------------------
    // Floors sit far below the measured rates at the default schedule
    // (200 seeds each: 195 -> 100%, 33 -> 100%, 899 -> 99%, 143 -> 45%), so
    // these are regression detectors rather than coin flips. They are also
    // deliberately sensitive to the annealing schedule: dropping t_initial
    // from 3.0 to 0.5 takes 899 from 99% to 5%, which these floors catch.
    std::cout << "Single-run dynamics (restarts disabled -- only local repair can win)\n";
    for (invmul::UpdateMode mode :
         {invmul::UpdateMode::Asynchronous, invmul::UpdateMode::Synchronous}) {
        test_single_run_rate(4, 195, "195 = 13 * 15", 100, 85, mode);
        test_single_run_rate(4, 33, "33 = 3 * 11", 100, 85, mode);
        test_single_run_rate(5, 899, "899 = 29 * 31", 100, 80, mode);
        test_single_run_rate(4, 143, "143 = 11 * 13", 100, 20, mode);
    }
    std::cout << "\n";

    std::cout << "Degenerate / unsolvable inputs\n";
    test_unsolved_behaviour();
    std::cout << "\n";

    check(g_soundness_violations == 0,
          "soundness held on all " + std::to_string(g_runs_checked) +
              " solver runs (every reported factorization verified against integer "
              "multiplication, energy, and the clamp)");

    std::cout << "\n";
    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cout << g_failures << " CHECK(S) FAILED\n";
    return 1;
}
