#include "invmul/multiplier.hpp"
#include "invmul/solver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

const char* mode_name(invmul::UpdateMode m) {
    return m == invmul::UpdateMode::Synchronous ? "sync" : "async";
}

const char* init_name(invmul::InitMode m) {
    return m == invmul::InitMode::RandomWires ? "random_wires" : "forward";
}

bool is_prime(std::uint64_t x) {
    if (x < 2) return false;
    for (std::uint64_t d = 2; d * d <= x; ++d) {
        if (x % d == 0) return false;
    }
    return true;
}

// A prime with exactly n significant bits, so that p * q genuinely needs
// the full 2n-bit output -- the hardest kind of target for a given grid.
std::uint64_t random_nbit_prime(int n, std::mt19937_64& rng) {
    const std::uint64_t lo = std::uint64_t{1} << (n - 1);
    const std::uint64_t hi = (std::uint64_t{1} << n) - 1;
    std::uniform_int_distribution<std::uint64_t> dist(lo, hi);
    for (;;) {
        const std::uint64_t v = dist(rng) | 1u;
        if (is_prime(v)) return v;
    }
}

std::string frac(int num, int den) { return std::to_string(num) + "/" + std::to_string(den); }

// ---------------------------------------------------------------- section 1
void bench_forward(const std::string& csv_path) {
    std::ofstream csv(csv_path);
    csv << "n,cells,wires,products,ms,products_per_sec\n";

    std::cout << "\n=== 1. Forward multiplication throughput ===\n";
    std::cout << std::left << std::setw(6) << "n" << std::setw(9) << "cells" << std::setw(9)
              << "wires" << std::setw(14) << "ms" << std::setw(16) << "products/s" << "\n";

    for (int n : {4, 8, 12, 16, 20, 24, 28}) {
        invmul::MultiplierGrid grid(n);
        invmul::BitState state(static_cast<std::size_t>(grid.num_vars()));

        std::mt19937_64 rng(1234 + n);
        const std::uint64_t limit = (std::uint64_t{1} << n) - 1;
        std::uniform_int_distribution<std::uint64_t> dist(0, limit);

        // Scale the repeat count so every size takes a comparable time.
        const int reps = std::max(200, 400000 / (n * n));
        std::vector<std::uint64_t> as(reps), bs(reps);
        for (int i = 0; i < reps; ++i) {
            as[i] = dist(rng);
            bs[i] = dist(rng);
        }

        std::uint64_t checksum = 0;
        const auto start = Clock::now();
        for (int i = 0; i < reps; ++i) {
            grid.forward(as[i], bs[i], state);
            checksum += grid.read_product(state);
        }
        const double ms = ms_since(start);
        if (checksum == 0xDEADBEEFu) std::cout << ""; // keep the loop from being elided

        const double per_sec = reps / (ms / 1000.0);
        std::cout << std::left << std::setw(6) << n << std::setw(9) << grid.num_cells()
                  << std::setw(9) << grid.num_vars() << std::fixed << std::setprecision(2)
                  << std::setw(14) << ms << std::setprecision(0) << std::setw(16) << per_sec
                  << std::defaultfloat << "\n";
        csv << n << ',' << grid.num_cells() << ',' << grid.num_vars() << ',' << reps << ','
            << std::fixed << std::setprecision(4) << ms << ',' << std::setprecision(1) << per_sec
            << std::defaultfloat << '\n';
        csv.flush();
    }
    std::cout << "  -> " << csv_path << "\n";
}

// ---------------------------------------------------------------- section 2
// Convergence is a stochastic trajectory, so a single run is an anecdote.
// The summary below averages over several instances, and every individual
// trajectory goes to the CSV tagged by instance, rather than presenting one
// curve as if it were representative.
void bench_convergence(const std::string& csv_path, int instances) {
    std::ofstream csv(csv_path);
    csv << "n,mode,init,instance,round,energy,best_energy,frontier,temperature\n";

    std::cout << "\n=== 2. Convergence of the corrective process ===\n";
    std::cout << "  Mean over " << instances
              << " instances per row; full per-instance curves in the CSV.\n";
    std::cout << "  'improved' counts runs that got below their own starting energy.\n\n";
    std::cout << std::left << std::setw(6) << "n" << std::setw(8) << "mode" << std::setw(14)
              << "init" << std::setw(12) << "E(start)" << std::setw(12) << "E(best)"
              << std::setw(12) << "improved" << std::setw(10) << "solved" << "\n";

    for (int n : {8, 12, 20}) {
        std::mt19937_64 rng(9000 + n);
        std::vector<std::uint64_t> targets;
        for (int i = 0; i < instances; ++i) {
            targets.push_back(random_nbit_prime(n, rng) * random_nbit_prime(n, rng));
        }

        for (invmul::InitMode init :
             {invmul::InitMode::ForwardFromRandomOperands, invmul::InitMode::RandomWires}) {
            for (invmul::UpdateMode mode :
                 {invmul::UpdateMode::Asynchronous, invmul::UpdateMode::Synchronous}) {
                double sum_start = 0.0;
                double sum_best = 0.0;
                int improved = 0;
                int solved = 0;

                for (int i = 0; i < instances; ++i) {
                    invmul::MultiplierGrid grid(n);
                    invmul::SolverConfig cfg;
                    cfg.mode = mode;
                    cfg.init = init;
                    cfg.subset_fraction = 0.5;
                    cfg.max_rounds = 20000;
                    cfg.restarts = 1; // one run: we want the trajectory, not the answer
                    cfg.num_threads = 1;
                    cfg.seed = 4242u + static_cast<std::uint64_t>(i);
                    cfg.require_nontrivial = true;
                    cfg.clamp_odd_low_bits = true;
                    cfg.record_trace = true;
                    cfg.trace_interval = 50;

                    invmul::AnnealingSolver solver(grid, cfg);
                    const invmul::SolveResult r =
                        solver.solve(targets[static_cast<std::size_t>(i)]);

                    for (const invmul::TracePoint& tp : r.trace) {
                        csv << n << ',' << mode_name(mode) << ',' << init_name(init) << ',' << i
                            << ',' << tp.round << ',' << tp.energy << ',' << tp.best_energy << ','
                            << tp.frontier_size << ',' << std::fixed << std::setprecision(6)
                            << tp.temperature << std::defaultfloat << '\n';
                    }
                    csv.flush();

                    const long long e_start = r.trace.empty() ? 0 : r.trace.front().energy;
                    sum_start += static_cast<double>(e_start);
                    sum_best += static_cast<double>(r.best_energy);
                    if (r.best_energy < e_start) ++improved;
                    if (r.solved) ++solved;
                }

                std::cout << std::left << std::setw(6) << n << std::setw(8) << mode_name(mode)
                          << std::setw(14) << init_name(init) << std::fixed << std::setprecision(2)
                          << std::setw(12) << (sum_start / instances) << std::setw(12)
                          << (sum_best / instances) << std::defaultfloat << std::setw(12)
                          << frac(improved, instances) << std::setw(10) << frac(solved, instances)
                          << "\n";
            }
        }
    }
    std::cout << "  -> " << csv_path << "\n";
}

// ---------------------------------------------------------------- section 3
struct QualityConfig {
    const char* label;
    int restarts;
    long long rounds;
};

void bench_quality(const std::string& csv_path, int small_instances, int large_instances) {
    std::ofstream csv(csv_path);
    csv << "n,budget,mode,restarts,rounds,instances,solved,success_rate,avg_ms,avg_best_energy\n";

    // Three ways of spending a comparable number of rounds, so the
    // comparison is about *how* the budget is allocated.
    const QualityConfig budgets[] = {
        {"draw_only", 2000, 1},      // no local search at all: the baseline
        {"single_anneal", 1, 20000}, // the corrective dynamics on its own
        {"restart_heavy", 2000, 20}, // the same budget split across restarts
    };

    std::cout << "\n=== 3. Solve quality vs grid size ===\n";
    std::cout << "  'draw_only' does no local search (1 round per restart): it is just the\n"
                 "  chance that random operands happen to be right. Whatever the corrective\n"
                 "  process contributes has to show up as a gap above that baseline.\n"
                 "  Targets are balanced semiprimes p*q with p, q both exactly n bits --\n"
                 "  the hardest case for a given grid.\n\n";
    std::cout << std::left << std::setw(5) << "n" << std::setw(16) << "budget" << std::setw(8)
              << "mode" << std::setw(12) << "solved" << std::setw(10) << "rate" << std::setw(12)
              << "avg_ms" << std::setw(12) << "avg_bestE" << "\n";

    for (int n : {4, 5, 6, 7, 8, 10, 12, 16, 20}) {
        // Large grids are slower per run and (as the table shows) solve far
        // less often, so they get fewer instances.
        const int instances = (n <= 8) ? small_instances : large_instances;

        // The same instances are used for every configuration at this size.
        std::mt19937_64 rng(31337 + n);
        std::vector<std::uint64_t> targets;
        for (int i = 0; i < instances; ++i) {
            targets.push_back(random_nbit_prime(n, rng) * random_nbit_prime(n, rng));
        }

        for (const QualityConfig& budget : budgets) {
            for (invmul::UpdateMode mode :
                 {invmul::UpdateMode::Asynchronous, invmul::UpdateMode::Synchronous}) {
                // The draw-only baseline never makes a move, so it does not
                // depend on the update mode: measure it once.
                if (std::string(budget.label) == "draw_only" &&
                    mode == invmul::UpdateMode::Synchronous) {
                    continue;
                }

                int solved = 0;
                double total_ms = 0.0;
                double total_best = 0.0;
                for (int i = 0; i < instances; ++i) {
                    invmul::MultiplierGrid grid(n);
                    invmul::SolverConfig cfg;
                    cfg.mode = mode;
                    cfg.subset_fraction = 0.5;
                    cfg.max_rounds = budget.rounds;
                    cfg.restarts = budget.restarts;
                    cfg.num_threads = 1;
                    cfg.seed = 77000u + static_cast<std::uint64_t>(i);
                    cfg.require_nontrivial = true;
                    cfg.clamp_odd_low_bits = true;

                    invmul::AnnealingSolver solver(grid, cfg);
                    const invmul::SolveResult r = solver.solve(targets[static_cast<std::size_t>(i)]);
                    if (r.solved) ++solved;
                    total_ms += r.elapsed_ms;
                    total_best += static_cast<double>(r.best_energy);
                }

                const double rate = 100.0 * solved / instances;
                std::cout << std::left << std::setw(5) << n << std::setw(16) << budget.label
                          << std::setw(8) << mode_name(mode) << std::setw(12)
                          << frac(solved, instances) << std::setw(10)
                          << (std::to_string(static_cast<int>(rate + 0.5)) + "%") << std::fixed
                          << std::setprecision(2) << std::setw(12) << (total_ms / instances)
                          << std::setw(12) << (total_best / instances) << std::defaultfloat << "\n";

                csv << n << ',' << budget.label << ',' << mode_name(mode) << ',' << budget.restarts
                    << ',' << budget.rounds << ',' << instances << ',' << solved << ','
                    << std::fixed << std::setprecision(4) << rate << ',' << (total_ms / instances)
                    << ',' << (total_best / instances) << std::defaultfloat << '\n';
                csv.flush();
            }
        }
        std::cout << "\n";
    }
    std::cout << "  -> " << csv_path << "\n";
}

// ---------------------------------------------------------------- section 4
void bench_scaling(const std::string& csv_path, int max_threads) {
    std::ofstream csv(csv_path);
    csv << "n,cells,init,mode,threads,rounds,moves_proposed,ms,moves_per_sec,rounds_per_sec,"
           "throughput_speedup,avg_frontier\n";

    std::cout << "\n=== 4. Parallel scaling ===\n";
    std::cout << "  'speedup' is moves/s relative to 1 thread, NOT a wall-clock ratio:\n"
                 "  different thread counts follow different trajectories and do different\n"
                 "  amounts of work, so elapsed time alone is not comparable.\n"
                 "  'avg_front' is the mean frontier size -- it is the ceiling on how much\n"
                 "  parallel work exists at all.\n\n";
    std::cout << std::left << std::setw(5) << "n" << std::setw(14) << "init" << std::setw(8)
              << "mode" << std::setw(9) << "threads" << std::setw(12) << "ms" << std::setw(15)
              << "moves/s" << std::setw(11) << "speedup" << std::setw(12) << "avg_front" << "\n";

    std::vector<int> thread_counts;
    for (int t = 1; t <= max_threads; t *= 2) thread_counts.push_back(t);

    for (int n : {12, 20, 28}) {
        std::mt19937_64 rng(555 + n);
        const std::uint64_t target = random_nbit_prime(n, rng) * random_nbit_prime(n, rng);

        for (invmul::InitMode init :
             {invmul::InitMode::RandomWires, invmul::InitMode::ForwardFromRandomOperands}) {
            for (invmul::UpdateMode mode :
                 {invmul::UpdateMode::Asynchronous, invmul::UpdateMode::Synchronous}) {
                double baseline_moves_per_sec = -1.0;
                for (int threads : thread_counts) {
                    invmul::MultiplierGrid grid(n);
                    invmul::SolverConfig cfg;
                    cfg.mode = mode;
                    cfg.init = init;
                    // Take the whole frontier each round, so the available
                    // parallel work per round is the frontier itself.
                    cfg.subset_fraction = 1.0;
                    cfg.max_rounds = 3000;
                    cfg.restarts = 1;
                    cfg.num_threads = threads;
                    cfg.seed = 24680;
                    cfg.require_nontrivial = true;
                    cfg.clamp_odd_low_bits = true;
                    cfg.record_trace = true;
                    cfg.trace_interval = 50;

                    invmul::AnnealingSolver solver(grid, cfg);
                    const invmul::SolveResult r = solver.solve(target);

                    double avg_frontier = 0.0;
                    for (const invmul::TracePoint& tp : r.trace) {
                        avg_frontier += tp.frontier_size;
                    }
                    if (!r.trace.empty()) avg_frontier /= static_cast<double>(r.trace.size());

                    const double moves_per_sec =
                        r.elapsed_ms > 0.0
                            ? static_cast<double>(r.moves_proposed) / (r.elapsed_ms / 1000.0)
                            : 0.0;
                    const double rounds_per_sec =
                        r.elapsed_ms > 0.0
                            ? static_cast<double>(r.rounds) / (r.elapsed_ms / 1000.0)
                            : 0.0;
                    if (threads == thread_counts.front()) baseline_moves_per_sec = moves_per_sec;
                    const double speedup =
                        baseline_moves_per_sec > 0.0 ? moves_per_sec / baseline_moves_per_sec : 0.0;

                    std::cout << std::left << std::setw(5) << n << std::setw(14) << init_name(init)
                              << std::setw(8) << mode_name(mode) << std::setw(9) << threads
                              << std::fixed << std::setprecision(2) << std::setw(12) << r.elapsed_ms
                              << std::setprecision(0) << std::setw(15) << moves_per_sec
                              << std::setprecision(2) << std::setw(11) << speedup << std::setw(12)
                              << avg_frontier << std::defaultfloat << "\n";

                    csv << n << ',' << grid.num_cells() << ',' << init_name(init) << ','
                        << mode_name(mode) << ',' << threads << ',' << r.rounds << ','
                        << r.moves_proposed << ',' << std::fixed << std::setprecision(4)
                        << r.elapsed_ms << ',' << std::setprecision(1) << moves_per_sec << ','
                        << rounds_per_sec << ',' << std::setprecision(4) << speedup << ','
                        << avg_frontier << std::defaultfloat << '\n';
                    csv.flush();
                }
                std::cout << "\n";
            }
        }
    }
    std::cout << "  -> " << csv_path << "\n";
}

}

int main(int argc, char** argv) {
    const std::string prefix = argc > 1 ? argv[1] : "invmul_bench";
#ifdef _OPENMP
    const int hw_threads = omp_get_max_threads();
#else
    const int hw_threads = 1;
#endif
    const int max_threads = argc > 2 ? std::atoi(argv[2]) : hw_threads;

    std::cout << "invmul benchmarks\n";
#ifdef _OPENMP
    std::cout << "  OpenMP enabled, " << hw_threads << " hardware thread(s); sweeping up to "
              << max_threads << "\n";
#else
    std::cout << "  built without OpenMP -- everything runs serially\n";
#endif
    std::cout << "  CSV prefix: " << prefix << "\n";

    bench_forward(prefix + "_forward.csv");
    bench_convergence(prefix + "_convergence.csv", 5);
    bench_quality(prefix + "_quality.csv", 60, 25);
    bench_scaling(prefix + "_scaling.csv", max_threads);

    std::cout << "\nDone.\n";
    return 0;
}
