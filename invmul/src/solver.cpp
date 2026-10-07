#include "invmul/solver.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace invmul {
namespace {

// Number of set bits in a 4-bit flip mask. Small and portable -- avoids
// depending on __builtin_popcount / <bit>.
inline int popcount4(int mask) {
    return ((mask >> 0) & 1) + ((mask >> 1) & 1) + ((mask >> 2) & 1) + ((mask >> 3) & 1);
}

// Contiguous slice [lo, hi) of `total` items belonging to thread `tid` of
// `threads`.
inline void slice_for(int tid, int threads, int total, int& lo, int& hi) {
    lo = static_cast<int>(static_cast<long long>(total) * tid / threads);
    hi = static_cast<int>(static_cast<long long>(total) * (tid + 1) / threads);
}

} // namespace

double annealing_alpha(double t_initial, double t_final, long long rounds) {
    if (rounds <= 1 || t_initial <= 0.0 || t_final <= 0.0 || t_final >= t_initial) {
        return 1.0;
    }
    return std::pow(t_final / t_initial, 1.0 / static_cast<double>(rounds));
}

AnnealingSolver::AnnealingSolver(const MultiplierGrid& grid, const SolverConfig& config)
    : m_grid(grid), m_config(config) {
    if (m_config.num_threads < 1) m_config.num_threads = 1;
#ifdef _OPENMP
    m_config.num_threads = std::min(m_config.num_threads, omp_get_max_threads());
#else
    m_config.num_threads = 1; // no OpenMP in this build: everything is serial
#endif
    if (m_config.subset_fraction <= 0.0) m_config.subset_fraction = 1e-9;
    if (m_config.subset_fraction > 1.0) m_config.subset_fraction = 1.0;
    if (m_config.min_selected < 1) m_config.min_selected = 1;
    if (m_config.restarts < 1) m_config.restarts = 1;

    m_state = BitState(static_cast<std::size_t>(m_grid.num_vars()));
    m_best_state = BitState(static_cast<std::size_t>(m_grid.num_vars()));
    m_clamped.assign(static_cast<std::size_t>(m_grid.num_vars()), 0);

    m_workspaces.resize(static_cast<std::size_t>(m_config.num_threads));
    for (auto& ws : m_workspaces) {
        ws.stamp.assign(static_cast<std::size_t>(m_grid.num_cells()), 0);
        ws.affected.reserve(static_cast<std::size_t>(2 * m_grid.n() + 8));
    }

    m_var_stamp.assign(static_cast<std::size_t>(m_grid.num_vars()), 0);
    m_var_writers.assign(static_cast<std::size_t>(m_grid.num_vars()), 0);
    m_var_value.assign(static_cast<std::size_t>(m_grid.num_vars()), 0);

    m_frontier.reserve(static_cast<std::size_t>(m_grid.num_cells()));
}

template <typename Reader>
bool AnnealingSolver::propose_repair(int cell, Reader read, std::mt19937_64& rng, Move& out) const {
    const CellWiring& w = m_grid.wiring(cell);

    // The cell's four input ports, in a fixed order matching the argument
    // order of cell_energy_from_wires.
    const VarIndex ports[4] = {w.a_var, w.b_var, w.sum_in_var, w.carry_in_var};
    std::uint8_t cur[4];
    for (int i = 0; i < 4; ++i) cur[i] = read(ports[i]);

    // A port is writable unless it is a boundary constant or clamped. The
    // four ports are always four distinct variables (a and b live in
    // different index ranges; sum_in and carry_in are in different rows or
    // columns), so there is no aliasing to worry about here.
    int free_idx[4];
    int num_free = 0;
    for (int i = 0; i < 4; ++i) {
        const VarIndex v = ports[i];
        if (v != kConstZero && !m_clamped[static_cast<std::size_t>(v)]) {
            free_idx[num_free++] = i;
        }
    }
    if (num_free == 0) return false; // nothing this cell is allowed to change

    // The cell reads its own outputs but never writes them: they belong to
    // the downstream cells that consume them.
    const std::uint8_t s_out = read(w.sum_out_var);
    const std::uint8_t c_out = read(w.carry_out_var);

    // Enumerate every flip mask over the writable ports. mask == 0 is the
    // current assignment and is deliberately skipped, so an incorrect cell
    // always changes at least one input -- "if they are not consistent, it
    // randomly changes one or more of its own inputs".
    const int num_masks = 1 << num_free;
    int best_energy = std::numeric_limits<int>::max();
    int best_hamming = std::numeric_limits<int>::max();
    int ties = 0;
    int chosen_mask = 0;

    for (int mask = 1; mask < num_masks; ++mask) {
        std::uint8_t vals[4] = {cur[0], cur[1], cur[2], cur[3]};
        for (int k = 0; k < num_free; ++k) {
            if ((mask >> k) & 1) vals[free_idx[k]] ^= std::uint8_t{1};
        }
        const int e = cell_energy_from_wires(vals[0], vals[1], vals[2], vals[3], s_out, c_out);
        const int hamming = popcount4(mask);

        // Rank by local energy, optionally then by how few inputs the
        // repair disturbs (see SolverConfig::prefer_minimal_repair for why
        // that secondary key is off by default). Ties are broken uniformly
        // at random by reservoir sampling -- the modulo bias is ~2^-60 for
        // ties <= 15, which is the most a 4-port cell can produce.
        const int rank_hamming = m_config.prefer_minimal_repair ? hamming : 0;
        if (e < best_energy || (e == best_energy && rank_hamming < best_hamming)) {
            best_energy = e;
            best_hamming = rank_hamming;
            ties = 1;
            chosen_mask = mask;
        } else if (e == best_energy && rank_hamming == best_hamming) {
            ++ties;
            if (rng() % static_cast<std::uint64_t>(ties) == 0) chosen_mask = mask;
        }
    }

    out.count = 0;
    for (int k = 0; k < num_free; ++k) {
        if ((chosen_mask >> k) & 1) {
            const int idx = free_idx[k];
            out.vars[out.count] = ports[idx];
            out.vals[out.count] = static_cast<std::uint8_t>(cur[idx] ^ 1u);
            ++out.count;
        }
    }
    return out.count > 0;
}

long long AnnealingSolver::apply_and_measure(const Move& move, Workspace& ws,
                                             std::uint8_t (&saved)[4]) {
    // Collect the cells whose energy can move. By construction of the
    // interaction graph this is the complete set, so the delta below is
    // exact (up to concurrent writes by other threads in parallel
    // asynchronous mode -- see the header).
    ++ws.generation;
    ws.affected.clear();
    for (int m = 0; m < move.count; ++m) {
        for (int c : m_grid.cells_touching(move.vars[m])) {
            if (ws.stamp[static_cast<std::size_t>(c)] != ws.generation) {
                ws.stamp[static_cast<std::size_t>(c)] = ws.generation;
                ws.affected.push_back(c);
            }
        }
    }

    long long before = 0;
    for (int c : ws.affected) before += m_grid.cell_energy(m_state, c);

    for (int m = 0; m < move.count; ++m) {
        saved[m] = m_state.get(move.vars[m]);
        m_state.set(move.vars[m], move.vals[m]);
    }

    long long after = 0;
    for (int c : ws.affected) after += m_grid.cell_energy(m_state, c);

    return after - before;
}

void AnnealingSolver::revert(const Move& move, const std::uint8_t (&saved)[4]) {
    for (int m = 0; m < move.count; ++m) m_state.set(move.vars[m], saved[m]);
}

long long AnnealingSolver::measure_against_snapshot(const Move& move,
                                                    const std::vector<std::uint8_t>& snap,
                                                    Workspace& ws) const {
    ++ws.generation;
    ws.affected.clear();
    for (int m = 0; m < move.count; ++m) {
        for (int c : m_grid.cells_touching(move.vars[m])) {
            if (ws.stamp[static_cast<std::size_t>(c)] != ws.generation) {
                ws.stamp[static_cast<std::size_t>(c)] = ws.generation;
                ws.affected.push_back(c);
            }
        }
    }

    // Read the post-move value of a wire without writing anything: the move
    // touches at most four wires, so a linear scan beats copying the whole
    // snapshot.
    auto after_read = [&](VarIndex v) -> std::uint8_t {
        for (int m = 0; m < move.count; ++m) {
            if (move.vars[m] == v) return move.vals[m];
        }
        return snapshot_get(snap, v);
    };

    long long before = 0;
    long long after = 0;
    for (int c : ws.affected) {
        before += m_grid.cell_energy(snap, c);
        const CellWiring& w = m_grid.wiring(c);
        after += cell_energy_from_wires(after_read(w.a_var), after_read(w.b_var),
                                        after_read(w.sum_in_var), after_read(w.carry_in_var),
                                        after_read(w.sum_out_var), after_read(w.carry_out_var));
    }
    return after - before;
}

void AnnealingSolver::rebuild_frontier_and_energy() {
    m_frontier.clear();
    const int cells = m_grid.num_cells();
    long long total = 0;

#ifdef _OPENMP
    if (m_config.num_threads > 1) {
#pragma omp parallel num_threads(m_config.num_threads) reduction(+ : total)
        {
            std::vector<int> local;
            local.reserve(64);
#pragma omp for schedule(static) nowait
            for (int c = 0; c < cells; ++c) {
                const int e = m_grid.cell_energy(m_state, c);
                total += e;
                if (e > 0) local.push_back(c);
            }
#pragma omp critical
            m_frontier.insert(m_frontier.end(), local.begin(), local.end());
        }
        m_energy = total;
        return;
    }
#endif

    for (int c = 0; c < cells; ++c) {
        const int e = m_grid.cell_energy(m_state, c);
        total += e;
        if (e > 0) m_frontier.push_back(c);
    }
    m_energy = total;
}

void AnnealingSolver::run_round_async(int num_selected, double temperature) {
    const int threads = m_config.num_threads;

#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        Workspace& ws = m_workspaces[static_cast<std::size_t>(tid)];
        int lo = 0, hi = 0;
        slice_for(tid, threads, num_selected, lo, hi);

        std::uniform_real_distribution<double> unit(0.0, 1.0);
        Move move;
        std::uint8_t saved[4];

        auto read = [this](VarIndex v) { return m_state.get(v); };

        for (int k = lo; k < hi; ++k) {
            const int cell = m_frontier[static_cast<std::size_t>(k)];
            // An earlier move in this round may already have repaired it.
            if (m_grid.cell_energy(m_state, cell) == 0) continue;
            if (!propose_repair(cell, read, ws.rng, move)) continue;

            ++ws.proposed;
            const long long delta = apply_and_measure(move, ws, saved);
            bool accept = delta <= 0;
            if (!accept) {
                accept = unit(ws.rng) < std::exp(-static_cast<double>(delta) / temperature);
            }
            if (accept) {
                ++ws.accepted;
                if (delta > 0) ++ws.uphill;
            } else {
                revert(move, saved);
            }
        }
    }
}

void AnnealingSolver::run_round_sync(int num_selected, double temperature) {
    // One read-only snapshot for the whole round: this is what makes the
    // update a cellular automaton rather than a sequential sweep. Every
    // selected cell sees the same world.
    const std::vector<std::uint8_t> snap = m_state.snapshot();
    const int threads = m_config.num_threads;

#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
#ifdef _OPENMP
        const int tid = omp_get_thread_num();
#else
        const int tid = 0;
#endif
        Workspace& ws = m_workspaces[static_cast<std::size_t>(tid)];
        ws.writes.clear();
        int lo = 0, hi = 0;
        slice_for(tid, threads, num_selected, lo, hi);

        std::uniform_real_distribution<double> unit(0.0, 1.0);
        Move move;

        auto read = [&snap](VarIndex v) { return snapshot_get(snap, v); };

        for (int k = lo; k < hi; ++k) {
            const int cell = m_frontier[static_cast<std::size_t>(k)];
            if (m_grid.cell_energy(snap, cell) == 0) continue;
            if (!propose_repair(cell, read, ws.rng, move)) continue;

            ++ws.proposed;
            const long long delta = measure_against_snapshot(move, snap, ws);
            bool accept = delta <= 0;
            if (!accept) {
                accept = unit(ws.rng) < std::exp(-static_cast<double>(delta) / temperature);
            }
            if (accept) {
                ++ws.accepted;
                if (delta > 0) ++ws.uphill;
                for (int m = 0; m < move.count; ++m) {
                    ws.writes.emplace_back(move.vars[m], move.vals[m]);
                }
            }
        }
    }

    // Serial merge. Several cells may have accepted a write to the same
    // wire -- only possible for the shared operand bits a_j / b_i, since
    // every internal wire has a single consumer. Pick one writer uniformly
    // at random (reservoir sampling over the writers of each wire).
    ++m_merge_generation;
    m_touched_vars.clear();
    std::mt19937_64& rng = m_workspaces[0].rng;

    for (int t = 0; t < threads; ++t) {
        for (const auto& write : m_workspaces[static_cast<std::size_t>(t)].writes) {
            const std::size_t v = static_cast<std::size_t>(write.first);
            if (m_var_stamp[v] != m_merge_generation) {
                m_var_stamp[v] = m_merge_generation;
                m_var_writers[v] = 1;
                m_var_value[v] = write.second;
                m_touched_vars.push_back(write.first);
            } else {
                ++m_var_writers[v];
                if (rng() % static_cast<std::uint64_t>(m_var_writers[v]) == 0) {
                    m_var_value[v] = write.second;
                }
            }
        }
    }

    for (VarIndex v : m_touched_vars) {
        m_state.set(v, m_var_value[static_cast<std::size_t>(v)]);
    }
}

void AnnealingSolver::reset_for_run(std::uint64_t target, std::uint64_t run_seed) {
    for (std::size_t t = 0; t < m_workspaces.size(); ++t) {
        // Distinct, well-separated streams per thread. The odd multiplier
        // is the 64-bit golden-ratio constant used by splitmix64.
        m_workspaces[t].rng.seed(run_seed + 0x9E3779B97F4A7C15ull * (t + 1));
        m_workspaces[t].generation = 0;
        std::fill(m_workspaces[t].stamp.begin(), m_workspaces[t].stamp.end(), 0);
    }

    // Start from a *valid* forward evaluation of random operands, then
    // overwrite the product edge with the target. Only the cells near the
    // disturbed output wires are inconsistent to begin with, so the initial
    // frontier is small and the search starts close to the constraint
    // surface instead of in a uniformly random configuration.
    const int n = m_grid.n();
    const std::uint64_t limit = std::uint64_t{1} << n;
    std::uniform_int_distribution<std::uint64_t> dist(0, limit - 1);
    std::mt19937_64& rng = m_workspaces[0].rng;

    std::uint64_t a = dist(rng);
    std::uint64_t b = dist(rng);
    if (m_config.clamp_odd_low_bits && (target & 1u)) {
        a |= 1u;
        b |= 1u;
    }
    m_grid.forward(a, b, m_state);

    if (m_config.init == InitMode::RandomWires) {
        // Melt the interior: randomize every internal wire, keeping the
        // operands just drawn. forward() above has already sized the state
        // and set the operand bits.
        for (VarIndex v = 2 * n; v < m_grid.num_vars(); ++v) {
            m_state.set(v, static_cast<std::uint8_t>(rng() & 1u));
        }
    }

    m_grid.write_product(target, m_state);
}

SolveResult AnnealingSolver::solve(std::uint64_t target) {
    const int width = m_grid.num_product_bits();
    const std::uint64_t target_limit =
        (width >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << width) - 1);
    if (target > target_limit) {
        throw std::invalid_argument("AnnealingSolver::solve: target does not fit in 2n bits");
    }

    SolveResult result;
    result.target = target;

    // Clamp the product wires to the target. (As noted in the header these
    // wires are never any cell's input, so nothing would write them anyway;
    // the clamp makes the intent explicit and also carries the optional
    // odd-factor reduction on a_0 / b_0.)
    std::fill(m_clamped.begin(), m_clamped.end(), std::uint8_t{0});
    for (int k = 0; k < width; ++k) {
        m_clamped[static_cast<std::size_t>(m_grid.product_var(k))] = 1;
    }
    if (m_config.clamp_odd_low_bits && (target & 1u)) {
        m_clamped[static_cast<std::size_t>(m_grid.a_var(0))] = 1;
        m_clamped[static_cast<std::size_t>(m_grid.b_var(0))] = 1;
    }

    for (auto& ws : m_workspaces) {
        ws.proposed = 0;
        ws.accepted = 0;
        ws.uphill = 0;
    }

    const auto wall_start = std::chrono::steady_clock::now();

    long long best_overall = std::numeric_limits<long long>::max();
    std::vector<std::uint8_t> best_overall_bits;
    const double alpha = annealing_alpha(m_config.t_initial, m_config.t_final, m_config.max_rounds);
    bool solved = false;

    for (int run = 0; run < m_config.restarts && !solved; ++run) {
        ++result.restarts_used;
        reset_for_run(target, m_config.seed + 0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(run + 1));

        double temperature = m_config.t_initial;
        long long best_this_run = std::numeric_limits<long long>::max();
        long long rounds_since_improvement = 0;

        for (long long round = 0; round < m_config.max_rounds; ++round) {
            // --- synchronization barrier: exact energy, fresh frontier ---
            rebuild_frontier_and_energy();
            ++result.rounds;

            if (m_energy < best_this_run) {
                best_this_run = m_energy;
                rounds_since_improvement = 0;
            } else {
                ++rounds_since_improvement;
            }
            if (m_energy < best_overall) {
                best_overall = m_energy;
                best_overall_bits = m_state.snapshot();
            }

            if (m_config.record_trace &&
                (round % std::max<long long>(1, m_config.trace_interval) == 0)) {
                TracePoint tp;
                tp.round = result.rounds;
                tp.energy = m_energy;
                tp.best_energy = best_overall;
                tp.frontier_size = static_cast<int>(m_frontier.size());
                tp.temperature = temperature;
                result.trace.push_back(tp);
            }

            if (m_energy == 0) {
                const std::uint64_t a = m_grid.read_a(m_state);
                const std::uint64_t b = m_grid.read_b(m_state);
                if (m_config.require_nontrivial && (a < 2 || b < 2)) {
                    // A perfectly consistent state, but it says N = 1 * N.
                    // Reject it and restart this run's configuration rather
                    // than reporting a factorization nobody wanted.
                    ++result.trivial_rejected;
                    reset_for_run(target, m_config.seed + 0xD1B54A32D192ED03ull *
                                              static_cast<std::uint64_t>(result.trivial_rejected + 1));
                    temperature = m_config.t_initial;
                    best_this_run = std::numeric_limits<long long>::max();
                    rounds_since_improvement = 0;
                    continue;
                }
                solved = true;
                break;
            }

            const int frontier_size = static_cast<int>(m_frontier.size());
            int num_selected = static_cast<int>(
                std::ceil(m_config.subset_fraction * static_cast<double>(frontier_size)));
            num_selected = std::max(num_selected, m_config.min_selected);
            num_selected = std::min(num_selected, frontier_size);

            // Random subset of the frontier, without replacement: a partial
            // Fisher-Yates shuffle moves the chosen cells to the front. The
            // frontier is rebuilt next round, so permuting it is free.
            std::mt19937_64& rng = m_workspaces[0].rng;
            for (int i = 0; i < num_selected; ++i) {
                const int span = frontier_size - i;
                const int j = i + static_cast<int>(rng() % static_cast<std::uint64_t>(span));
                std::swap(m_frontier[static_cast<std::size_t>(i)],
                          m_frontier[static_cast<std::size_t>(j)]);
            }

            if (m_config.mode == UpdateMode::Asynchronous) {
                run_round_async(num_selected, temperature);
            } else {
                run_round_sync(num_selected, temperature);
            }

            temperature *= alpha;
            if (temperature < 1e-12) temperature = 1e-12; // keep exp() well behaved

            if (m_config.reheat_after_rounds > 0 &&
                rounds_since_improvement >= m_config.reheat_after_rounds) {
                temperature = std::min(m_config.t_initial, temperature * m_config.reheat_factor);
                rounds_since_improvement = 0;
                // Stagnation is measured since the *last reheat*, not
                // against the run's all-time best. Comparing against the
                // all-time best would retrigger every reheat_after_rounds
                // rounds forever once the best stops improving, pinning the
                // temperature near t_initial and destroying the anneal.
                best_this_run = m_energy;
                ++result.reheats;
            }
        }
    }

    const auto wall_end = std::chrono::steady_clock::now();
    result.elapsed_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();

    if (solved) {
        best_overall = 0;
        best_overall_bits = m_state.snapshot();
    }
    if (!best_overall_bits.empty()) {
        m_best_state.restore(best_overall_bits);
    } else {
        m_best_state.restore(m_state.snapshot());
    }

    result.solved = solved;
    result.best_energy = (best_overall == std::numeric_limits<long long>::max()) ? m_energy : best_overall;
    result.final_energy = m_grid.total_energy(m_best_state);
    result.a = m_grid.read_a(m_best_state);
    result.b = m_grid.read_b(m_best_state);

    for (const auto& ws : m_workspaces) {
        result.moves_proposed += ws.proposed;
        result.moves_accepted += ws.accepted;
        result.uphill_accepted += ws.uphill;
    }
    return result;
}

} // namespace invmul
