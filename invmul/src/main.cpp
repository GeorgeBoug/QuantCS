#include "invmul/multiplier.hpp"
#include "invmul/solver.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

void print_forward_demo(int n, std::uint64_t a, std::uint64_t b) {
    invmul::MultiplierGrid grid(n);
    invmul::BitState state(static_cast<std::size_t>(grid.num_vars()));
    grid.forward(a, b, state);

    std::cout << "--- Forward operation (" << n << "x" << n << " grid, " << grid.num_cells()
              << " cells) ---\n";
    std::cout << "  operands in from north (a) and east (b): a = " << a << ", b = " << b << "\n";
    std::cout << "  product out on west and south edges:     " << grid.read_product(state) << "\n";
    std::cout << "  plain integer multiplication:            " << (a * b) << "\n";
    std::cout << "  total energy (0 == every cell consistent): " << grid.total_energy(state)
              << "\n\n";
}

void print_inverse_demo(int n, std::uint64_t target, invmul::UpdateMode mode, int threads) {
    invmul::MultiplierGrid grid(n);

    invmul::SolverConfig cfg;
    cfg.mode = mode;
    cfg.subset_fraction = 0.5;
    cfg.max_rounds = 60000;
    cfg.t_initial = 3.0;
    cfg.t_final = 0.02;
    cfg.restarts = 12;
    cfg.num_threads = threads;
    cfg.seed = 20260904u;
    cfg.require_nontrivial = true;
    cfg.clamp_odd_low_bits = true;

    invmul::AnnealingSolver solver(grid, cfg);
    const invmul::SolveResult result = solver.solve(target);

    std::cout << "--- Corrective / inverse operation ---\n";
    std::cout << "  target clamped onto the output edges: " << target << "\n";
    std::cout << "  update mode: "
              << (mode == invmul::UpdateMode::Synchronous ? "synchronous (cellular automaton)"
                                                          : "asynchronous (random subset)")
              << ", threads: " << cfg.num_threads << "\n";
    std::cout << "  annealing: T " << cfg.t_initial << " -> " << cfg.t_final << " geometric over "
              << cfg.max_rounds << " rounds (alpha = " << std::setprecision(8)
              << invmul::annealing_alpha(cfg.t_initial, cfg.t_final, cfg.max_rounds) << ")\n"
              << std::setprecision(6);

    if (result.solved) {
        std::cout << "  SOLVED: " << result.a << " * " << result.b << " = "
                  << (result.a * result.b) << "  (energy " << result.final_energy << ")\n";
    } else {
        std::cout << "  no zero-energy state found; best energy = " << result.best_energy
                  << " with a = " << result.a << ", b = " << result.b << "\n";
    }
    std::cout << "  rounds: " << result.rounds << ", restarts used: " << result.restarts_used
              << ", reheats: " << result.reheats << "\n";
    std::cout << "  moves proposed/accepted: " << result.moves_proposed << " / "
              << result.moves_accepted << " (" << result.uphill_accepted
              << " uphill, accepted by the Metropolis rule)\n";
    std::cout << "  trivial 1*N solutions rejected: " << result.trivial_rejected << "\n";
    std::cout << "  wall time: " << std::fixed << std::setprecision(2) << result.elapsed_ms
              << " ms\n\n"
              << std::defaultfloat;
}

}

int main(int argc, char** argv) {
    const int n = argc > 1 ? std::atoi(argv[1]) : 4;
    const std::uint64_t target =
        argc > 2 ? std::strtoull(argv[2], nullptr, 10) : std::uint64_t{143};
    const invmul::UpdateMode mode =
        (argc > 3 && std::string(argv[3]) == "sync") ? invmul::UpdateMode::Synchronous
                                                     : invmul::UpdateMode::Asynchronous;
    const int threads = argc > 4 ? std::atoi(argv[4]) : 1;

    if (n < 1 || n > 31) {
        std::cerr << "n must be in [1, 31]\n";
        return 1;
    }
    const int width = 2 * n;
    const std::uint64_t limit =
        (width >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << width) - 1);
    if (target > limit) {
        std::cerr << "target " << target << " does not fit in " << width << " bits\n";
        return 1;
    }

    const std::uint64_t demo_limit = (std::uint64_t{1} << n) - 1;
    print_forward_demo(n, std::min<std::uint64_t>(11, demo_limit),
                       std::min<std::uint64_t>(13, demo_limit));
    print_inverse_demo(n, target, mode, threads);
    return 0;
}
