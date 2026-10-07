// Forward-direction correctness for the n x n array multiplier.
//
// This is the test that has to pass before the corrective/inverse mode
// means anything: if the grid does not compute a * b in the normal
// direction, then "every cell is locally consistent" is not the same
// statement as "these are the factors", and the whole energy formulation
// is built on sand.
//
// Three things are checked:
//   1. read_product(forward(a, b)) == a * b, exhaustively for small n and
//      on random operands up to a 20x20 grid.
//   2. A forward-evaluated grid has total energy 0 -- forward evaluation
//      and the energy function agree on what "consistent" means. (They are
//      written independently: forward() assigns outputs from inputs,
//      cell_energy() scores the full-adder residual.)
//   3. Corrupting any single wire makes the energy strictly positive, and
//      only in cells adjacent to that wire in the interaction graph.

#include "invmul/multiplier.hpp"

#include <cstdint>
#include <iostream>
#include <random>
#include <string>

namespace {

int g_failures = 0;

void check(bool ok, const std::string& label) {
    if (!ok) {
        std::cout << "  [FAIL] " << label << "\n";
        ++g_failures;
    }
}

void pass(const std::string& label) { std::cout << "  [PASS] " << label << "\n"; }

// Exhaustive over every (a, b) pair an n-bit grid can accept.
void test_exhaustive(int n) {
    invmul::MultiplierGrid grid(n);
    invmul::BitState state(static_cast<std::size_t>(grid.num_vars()));

    const std::uint64_t limit = std::uint64_t{1} << n;
    bool product_ok = true;
    bool energy_ok = true;
    std::uint64_t bad_a = 0, bad_b = 0, bad_got = 0, bad_want = 0;

    for (std::uint64_t a = 0; a < limit && product_ok && energy_ok; ++a) {
        for (std::uint64_t b = 0; b < limit; ++b) {
            grid.forward(a, b, state);
            const std::uint64_t got = grid.read_product(state);
            const std::uint64_t want = a * b;
            if (got != want) {
                product_ok = false;
                bad_a = a; bad_b = b; bad_got = got; bad_want = want;
                break;
            }
            if (grid.total_energy(state) != 0) {
                energy_ok = false;
                bad_a = a; bad_b = b;
                break;
            }
        }
    }

    check(product_ok, "n=" + std::to_string(n) + " exhaustive product: " +
                          std::to_string(bad_a) + " * " + std::to_string(bad_b) + " gave " +
                          std::to_string(bad_got) + ", expected " + std::to_string(bad_want));
    check(energy_ok, "n=" + std::to_string(n) + " exhaustive energy: forward state for " +
                         std::to_string(bad_a) + " * " + std::to_string(bad_b) +
                         " has nonzero energy");
    if (product_ok && energy_ok) {
        pass("n=" + std::to_string(n) + ": all " + std::to_string(limit * limit) +
             " operand pairs multiply correctly and settle at energy 0");
    }
}

// Random operands at sizes too large to enumerate -- including the 20x20
// grid the brief calls out.
void test_random(int n, int trials, std::uint64_t seed) {
    invmul::MultiplierGrid grid(n);
    invmul::BitState state(static_cast<std::size_t>(grid.num_vars()));
    std::mt19937_64 rng(seed);
    const std::uint64_t limit = std::uint64_t{1} << n;
    std::uniform_int_distribution<std::uint64_t> dist(0, limit - 1);

    bool ok = true;
    for (int t = 0; t < trials && ok; ++t) {
        // Bias the first few trials towards the all-ones worst case, which
        // exercises the longest carry chains in the array.
        const std::uint64_t a = (t == 0) ? limit - 1 : dist(rng);
        const std::uint64_t b = (t == 0) ? limit - 1 : dist(rng);
        grid.forward(a, b, state);
        if (grid.read_product(state) != a * b || grid.total_energy(state) != 0) {
            ok = false;
            std::cout << "    mismatch at a=" << a << " b=" << b
                      << " got=" << grid.read_product(state)
                      << " want=" << (a * b)
                      << " energy=" << grid.total_energy(state) << "\n";
        }
    }
    check(ok, "n=" + std::to_string(n) + " random operands");
    if (ok) {
        pass("n=" + std::to_string(n) + " (" + std::to_string(n * n) + " cells): " +
             std::to_string(trials) + " random products correct, energy 0");
    }
}

// Flipping one wire must cost energy, and only locally. This is what
// justifies the incremental delta-E in the solver: the energy change from
// touching variable v is fully captured by grid.cells_touching(v).
void test_local_energy_structure(int n, std::uint64_t seed) {
    invmul::MultiplierGrid grid(n);
    invmul::BitState state(static_cast<std::size_t>(grid.num_vars()));
    std::mt19937_64 rng(seed);
    const std::uint64_t limit = std::uint64_t{1} << n;
    std::uniform_int_distribution<std::uint64_t> dist(0, limit - 1);

    grid.forward(dist(rng), dist(rng), state);

    bool all_positive = true;
    bool all_local = true;
    for (invmul::VarIndex v = 0; v < grid.num_vars(); ++v) {
        const std::vector<int> before_cells = grid.cells_touching(v);

        std::vector<int> before(static_cast<std::size_t>(grid.num_cells()));
        for (int c = 0; c < grid.num_cells(); ++c) before[static_cast<std::size_t>(c)] = grid.cell_energy(state, c);

        state.set(v, static_cast<std::uint8_t>(state.get(v) ^ 1u));

        long long after_total = 0;
        for (int c = 0; c < grid.num_cells(); ++c) {
            const int e = grid.cell_energy(state, c);
            after_total += e;
            if (e != before[static_cast<std::size_t>(c)]) {
                // This cell's energy moved, so it must be listed as
                // touching v.
                bool listed = false;
                for (int lc : before_cells) {
                    if (lc == c) { listed = true; break; }
                }
                if (!listed) all_local = false;
            }
        }
        if (after_total <= 0) all_positive = false;

        state.set(v, static_cast<std::uint8_t>(state.get(v) ^ 1u)); // restore
    }

    check(all_positive, "n=" + std::to_string(n) + ": every single-wire flip raises energy above 0");
    check(all_local, "n=" + std::to_string(n) +
                         ": energy changes from flipping v are confined to cells_touching(v)");
    if (all_positive && all_local) {
        pass("n=" + std::to_string(n) + ": energy is local and every wire matters (" +
             std::to_string(grid.num_vars()) + " wires checked)");
    }
}

// The product-bit map must name 2n distinct wires on the west/south edges.
void test_product_bit_map(int n) {
    invmul::MultiplierGrid grid(n);
    bool distinct = true;
    for (int k = 0; k < 2 * n; ++k) {
        if (grid.product_var(k) == invmul::kConstZero) distinct = false;
        for (int m = k + 1; m < 2 * n; ++m) {
            if (grid.product_var(k) == grid.product_var(m)) distinct = false;
        }
    }
    check(distinct, "n=" + std::to_string(n) + ": the 2n product bits are 2n distinct wires");
    if (distinct) pass("n=" + std::to_string(n) + ": product-bit map is well formed");
}

}

int main() {
    std::cout << "Forward multiplication (normal gate direction)\n";
    for (int n = 1; n <= 5; ++n) test_exhaustive(n);

    std::cout << "\nLarger grids, random operands\n";
    test_random(8, 2000, 0xC0FFEEu);
    test_random(12, 2000, 0xBEEFu);
    test_random(16, 1000, 0xFEEDu);
    test_random(20, 1000, 0xD00Du); // the 20x20 grid from the brief
    test_random(24, 500, 0xFACEu);

    std::cout << "\nEnergy structure\n";
    test_local_energy_structure(4, 0x1234u);
    test_local_energy_structure(7, 0x5678u);

    std::cout << "\nProduct-bit map\n";
    for (int n : {1, 2, 3, 8, 20}) test_product_bit_map(n);

    std::cout << "\n";
    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    }
    std::cout << g_failures << " CHECK(S) FAILED\n";
    return 1;
}
