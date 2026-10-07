#pragma once

#include "invmul/multiplier.hpp"

#include <cstdint>
#include <random>
#include <vector>

// =========================================================================
// Running the multiplier backwards: local search for a zero-energy state
// =========================================================================
//
// invmul/multiplier.hpp defines a total energy
//
//     E = sum over cells of (pp + sum_in + carry_in - sum_out - 2*carry_out)^2
//
// which is 0 exactly when every cell is a correct AND-plus-full-adder. If
// we pin the 2n product bits to a target N and find any configuration with
// E = 0, the operand wires spell out a factorization of N. This header is
// the search that tries to find one.
//
// -------------------------------------------------------------------------
// The local rule ("a gate that runs backwards")
// -------------------------------------------------------------------------
// A cell in the forward direction reads its inputs and writes its outputs.
// In corrective mode it does the opposite: it reads *both* its inputs and
// its outputs, and
//
//   * if they satisfy pp + sum_in + carry_in == sum_out + 2*carry_out,
//     it does nothing (its energy term is already 0);
//   * otherwise it changes one or more of its own INPUT wires to repair the
//     mismatch, leaving its outputs alone.
//
// Concretely, propose_repair() enumerates every assignment to the cell's
// writable input wires (at most 2^4 = 16 of them), scores each by the
// cell's local energy, and picks uniformly at random among the assignments
// that minimize it. The current assignment is excluded, so an incorrect
// cell always moves.
//
// Two structural facts make this well defined:
//
//   * A cell never writes its own outputs. Its outputs are some *other*
//     cell's inputs, and that downstream cell is the one allowed to change
//     them. So every internal wire has exactly one cell that may write it
//     (its consumer), and information flows backwards through the array --
//     which is what "bidirectional" means here.
//   * The 2n product wires are never any cell's input (they are exactly the
//     wires whose consumer would fall outside the grid: the west-edge sums,
//     the south-edge sums, and the south-east carry). So clamping them is
//     automatically respected by the local rule, and the clamp in
//     SolverConfig is belt-and-braces rather than load-bearing.
//
// The operand wires a_j and b_i are the exception to "one writer": a_j is
// an input of all n cells in column j, b_i of all n cells in row i. That is
// precisely why a purely local repair is not enough on its own -- fixing
// one cell by flipping a_j can break the other n-1 cells in that column.
//
// -------------------------------------------------------------------------
// Why Metropolis on top of the local rule
// -------------------------------------------------------------------------
// Each cell's proposal is locally greedy, but its effect on the *total*
// energy can be positive, because the wire it flips is shared (a_j, b_i) or
// feeds a neighbour. So we score every proposal by the exact global change
// dE -- cheap, because grid.cells_touching(v) bounds the affected cells to
// O(n) -- and accept it with the Metropolis probability
//
//     P(accept) = 1                if dE <= 0
//               = exp(-dE / T)     if dE > 0
//
// T falls geometrically from t_initial to t_final over the run, i.e.
// simulated annealing: early on the search accepts uphill moves freely and
// wanders; late on it is nearly greedy and settles into a minimum.
//
// Choosing the endpoints. The energy quantum here is 1 (residuals are
// integers, so the smallest possible uphill move is dE = +1, and a single
// cell can contribute at most 9). t_initial = 3.0 accepts a dE = +1 move
// with probability exp(-1/3) ~ 0.72 and a dE = +9 move with ~0.05, so the
// search starts genuinely mobile. t_final = 0.02 gives exp(-1/0.02) ~ 2e-22
// -- effectively greedy, so the run ends in a local minimum rather than
// still jittering out of one. Geometric (rather than linear) cooling spends
// proportionally equal time at each *scale* of energy barrier, which is the
// standard choice when you do not know in advance which barrier heights
// matter.
//
// t_initial = 3.0 was measured, not guessed, and the measurement is worth
// repeating because the sensitivity is severe. Single-run solve rate over
// 200 seeds, 20000 rounds, restarts disabled:
//
//     t_initial      0.25   0.5    1.0    2.0    3.0    5.0    8.0
//     N = 195         13%    34%    96%   100%   100%   100%   100%
//     N = 899          1%     5%    81%   100%    99%   100%   100%
//     N = 143          7%    11%    14%    33%    45%    42%    39%
//
// Everything from 2.0 upward is flat, and everything below 1.0 falls off a
// cliff. 3.0 sits in the middle of the flat region.
//
// -------------------------------------------------------------------------
// What the measurements actually say (see invmul/README.md for the data)
// -------------------------------------------------------------------------
// Being honest about this matters more than the machinery looking good:
//
//   * The initialization dominates everything. Starting from a forward
//     evaluation of random operands and then clamping the product edge
//     (InitMode::ForwardFromRandomOperands) beats melting every wire
//     (InitMode::RandomWires) by an enormous margin -- on balanced 7- and
//     8-bit-factor semiprimes the molten start solved 0 of 40 instances
//     under every schedule tried, while the forward start solved most of
//     them. Starting on the constraint surface and perturbing only the
//     output is what makes the local rule tractable.
//
//   * Annealing genuinely works, in the regime it is meant for. With
//     restarts disabled, a proper anneal beats a greedy descent by a wide
//     margin -- the table above is exactly that comparison, since t_initial
//     = 0.25 is very nearly greedy. Temperature is doing real work: it is
//     what lets the search cross the barrier between two operand basins,
//     which costs a coordinated flip of an operand bit *plus* the repair of
//     its whole row or column, a dE of order n.
//
//   * But at a fixed CPU budget, restarting still beats annealing longer.
//     A run relaxes within a few tens of rounds and then makes progress
//     only slowly, so splitting the budget into many short independent runs
//     scores better than one long slow cooling schedule. In that
//     restart-heavy regime each run is far too short for the schedule to
//     matter, and greedy, constant-T and annealed runs all score the same.
//     That is a statement about the budget allocation, not about annealing
//     being useless -- the two regimes are easy to confuse, and measuring
//     only the restart-heavy one is what makes annealing look pointless.
//
//   * The corrective process is worth more than random guessing, though by
//     one to two orders of magnitude rather than three. A single run recovers
//     the factors of a 4x4-grid semiprime roughly 15-45% of the time
//     depending on the target; one random operand draw is correct 3.1% of the
//     time at that size (2 correct ordered pairs out of 4^(n-1), given the
//     odd-factor reduction), so the margin is ~14x at N = 143 and ~127x at
//     N = 899 on a 5x5 grid. Beware comparing a small-grid solve rate against
//     a large-grid draw probability -- the draw probability falls by 4x per
//     unit of n, so mismatching the sizes inflates the margin enormously.
//
//   * It still scales badly. Balanced semiprimes stop being solvable in a
//     single run somewhere around a 6x6 grid. This is a property of the
//     landscape, not of the implementation.
//
// So: keep restarts = 1 when you want to measure the corrective dynamics,
// and raise it when you just want the factors. bench_invmul reports the
// random-draw baseline alongside the solver so the difference stays
// visible.
//
// -------------------------------------------------------------------------
// Frontier, subsets, and synchronization
// -------------------------------------------------------------------------
// Only incorrect cells need attention, so the solver keeps an explicit
// frontier: the list of cells with nonzero energy. Each round it
//
//   1. rebuilds the frontier and the exact total energy (this is the
//      synchronization barrier),
//   2. selects a random subset of the frontier -- subset_fraction of it,
//      without replacement,
//   3. splits that subset contiguously across threads, each of which
//      repairs its share (one or more moves) using a thread-local RNG,
//   4. cools T by one geometric step.
//
// The two update modes differ in step 3:
//
//   * Asynchronous -- selected cells are repaired one at a time, and each
//     sees the writes of the ones before it. dE is exact (single-threaded)
//     and Metropolis is a correct Markov chain.
//
//   * Synchronous -- the cellular-automaton rule. Every selected cell
//     proposes against the *same* read-only snapshot taken at the top of
//     the round, decides accept/reject independently, and all accepted
//     writes are applied together at the end. Where two cells write the
//     same wire (only possible for a shared a_j / b_i), one of the writers
//     is chosen uniformly at random. With subset_fraction = 1.0 this is
//     literally "every incorrect gate is corrected at the same time".
//     The trade-off is that each cell's dE is measured against the
//     snapshot, so it does not account for what the other cells did in the
//     same round; the exact energy is restored at the next barrier.
//
// Parallelism follows the same split. In asynchronous mode threads write
// concurrently between barriers without locking -- an optimistic /
// "Hogwild" scheme. Threads own contiguous slices of the selected subset,
// so conflicts are confined to the shared operand wires; a stale read just
// makes one proposal's dE slightly wrong, which an annealer absorbs as
// extra noise, and the total energy is recomputed exactly at every barrier
// so nothing the caller sees is ever approximate. Wire storage is relaxed
// byte atomics (see BitState) so this is defined behaviour, not a race.

namespace invmul {

// How a run's starting configuration is built.
enum class InitMode {
    // Evaluate the grid forward on random operands, then overwrite the
    // product edge with the target. Only the cells adjacent to the
    // disturbed output wires start inconsistent.
    ForwardFromRandomOperands,
    // Randomize every unclamped wire. The grid starts globally frustrated.
    RandomWires
};

enum class UpdateMode {
    // Selected cells repair one at a time; each sees the previous writes.
    Asynchronous,
    // Cellular automaton: all selected cells repair against one snapshot
    // and their writes land together.
    Synchronous
};

// One sample of the run, recorded at a synchronization barrier.
struct TracePoint {
    long long round = 0;
    long long energy = 0;      // exact total energy at this barrier
    long long best_energy = 0; // best seen so far in this run
    int frontier_size = 0;
    double temperature = 0.0;
};

struct SolverConfig {
    UpdateMode mode = UpdateMode::Asynchronous;

    // Fraction of the current frontier repaired per round, and a floor so
    // that a tiny frontier still gets worked on.
    double subset_fraction = 0.25;
    int min_selected = 1;

    long long max_rounds = 200000;

    // Geometric annealing schedule; see the header comment for why these
    // defaults.
    double t_initial = 3.0;
    double t_final = 0.02;

    InitMode init = InitMode::ForwardFromRandomOperands;

    // Tie-break among the assignments that minimize a cell's local energy.
    //
    // false (default): pick uniformly among all of them.
    // true: prefer the ones that change the fewest input wires.
    //
    // There is a plausible argument that "minimal repair" should be
    // harmful: driving pp from 0 to 1 requires setting *both* a_j and b_i
    // (two flips), whereas flipping sum_in or carry_in achieves the same
    // local residual with one, so preferring minimal repairs should make
    // cells route around the operand wires -- precisely the unknowns being
    // solved for. Measured, the effect is within noise at every size tried,
    // so the argument does not survive contact with the data. Uniform
    // selection is the default because it is the less biased rule, not
    // because it measured better.
    bool prefer_minimal_repair = false;

    // Reheat if the best energy has not improved for this many rounds since
    // the last reheat (0 disables). The idea is to restore mobility without
    // discarding the current configuration. Measured on this landscape it
    // does not pay for itself -- an equivalent budget spent on extra
    // restarts scores better at every size tried -- so it is off by
    // default and kept for comparison.
    long long reheat_after_rounds = 0;
    double reheat_factor = 25.0;

    // Independent annealing runs (fresh random operands, fresh schedule).
    // solve() returns on the first success, or the best state seen.
    int restarts = 1;

    int num_threads = 1;

    std::uint64_t seed = 1;

    // Treat a < 2 or b < 2 as a non-solution and keep searching. Without
    // this, N = 1 * N is a perfectly good zero-energy state.
    bool require_nontrivial = false;

    // If the target is odd then both factors must be odd, so a_0 = b_0 = 1
    // is a sound (not heuristic) reduction that removes two variables and a
    // lot of the search space. Ignored when the target is even.
    bool clamp_odd_low_bits = false;

    bool record_trace = false;
    long long trace_interval = 100;
};

struct SolveResult {
    bool solved = false;
    std::uint64_t target = 0;
    std::uint64_t a = 0;
    std::uint64_t b = 0;

    long long final_energy = 0; // energy of the returned state
    long long best_energy = 0;  // best energy seen across all restarts
    long long rounds = 0;       // synchronization barriers executed, total
    long long moves_proposed = 0;
    long long moves_accepted = 0;
    long long uphill_accepted = 0; // accepted with dE > 0 (the Metropolis part)
    long long reheats = 0;
    long long trivial_rejected = 0;
    int restarts_used = 0;
    double elapsed_ms = 0.0;

    std::vector<TracePoint> trace;
};

class AnnealingSolver {
public:
    // `grid` must outlive the solver.
    AnnealingSolver(const MultiplierGrid& grid, const SolverConfig& config);

    // Clamps the product wires to `target` and searches for a zero-energy
    // configuration. Throws std::invalid_argument if target does not fit in
    // 2n bits.
    SolveResult solve(std::uint64_t target);

    // Lowest-energy configuration found by the most recent solve().
    const BitState& best_state() const { return m_best_state; }

    const SolverConfig& config() const { return m_config; }

private:
    // A candidate repair: up to four wires to rewrite. (A cell has at most
    // four input ports, and it only ever writes its own input ports.)
    struct Move {
        int count = 0;
        VarIndex vars[4] = {kConstZero, kConstZero, kConstZero, kConstZero};
        std::uint8_t vals[4] = {0, 0, 0, 0};
    };

    // Per-thread scratch. Padded to a cache line so the counters in
    // adjacent workspaces do not false-share.
    struct alignas(64) Workspace {
        std::mt19937_64 rng;
        std::vector<long long> stamp; // per-cell "seen in this dE" marker
        long long generation = 0;
        std::vector<int> affected;
        std::vector<std::pair<VarIndex, std::uint8_t>> writes; // synchronous mode
        long long proposed = 0;
        long long accepted = 0;
        long long uphill = 0;
    };

    void reset_for_run(std::uint64_t target, std::uint64_t run_seed);
    void rebuild_frontier_and_energy();

    // Picks a repair for `cell` by reading wires through `read`. Returns
    // false if the cell has no writable input wire.
    template <typename Reader>
    bool propose_repair(int cell, Reader read, std::mt19937_64& rng, Move& out) const;

    // Applies `move` to the live state, returning the exact energy change
    // over the affected cells. Caller reverts on rejection.
    long long apply_and_measure(const Move& move, Workspace& ws, std::uint8_t (&saved)[4]);
    void revert(const Move& move, const std::uint8_t (&saved)[4]);

    // Energy change `move` would cause, measured against a read-only
    // snapshot (synchronous mode; nothing is written).
    long long measure_against_snapshot(const Move& move, const std::vector<std::uint8_t>& snap,
                                       Workspace& ws) const;

    void run_round_async(int num_selected, double temperature);
    void run_round_sync(int num_selected, double temperature);

    const MultiplierGrid& m_grid;
    SolverConfig m_config;

    BitState m_state;
    BitState m_best_state;
    std::vector<std::uint8_t> m_clamped;
    std::vector<int> m_frontier;
    long long m_energy = 0;

    std::vector<Workspace> m_workspaces;

    // Synchronous-mode merge buffers (serial, reused across rounds).
    std::vector<long long> m_var_stamp;
    std::vector<int> m_var_writers;
    std::vector<std::uint8_t> m_var_value;
    std::vector<VarIndex> m_touched_vars;
    long long m_merge_generation = 0;
};

// Convenience: geometric cooling factor taking t_initial to t_final in
// `rounds` steps. Exposed so benchmarks can report the schedule they used.
double annealing_alpha(double t_initial, double t_final, long long rounds);

}
