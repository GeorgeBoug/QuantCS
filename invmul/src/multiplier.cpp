#include "invmul/multiplier.hpp"

#include <sstream>
#include <stdexcept>

namespace invmul {

std::vector<std::uint8_t> BitState::snapshot() const {
    std::vector<std::uint8_t> out(m_bits.size());
    for (std::size_t i = 0; i < m_bits.size(); ++i) {
        out[i] = m_bits[i].load(std::memory_order_relaxed);
    }
    return out;
}

void BitState::restore(const std::vector<std::uint8_t>& values) {
    if (values.size() != m_bits.size()) {
        throw std::invalid_argument("BitState::restore: size mismatch");
    }
    for (std::size_t i = 0; i < m_bits.size(); ++i) {
        m_bits[i].store(values[i], std::memory_order_relaxed);
    }
}

MultiplierGrid::MultiplierGrid(int n) : m_n(n) {
    /* n = 31 already means a 62-bit product; beyond that the uint64_t
    convenience accessors (read_product etc.) would overflow, and the
    grid is well past any size this local search can solve anyway. */
    if (n < 1 || n > 31) {
        throw std::invalid_argument("MultiplierGrid: n must be in [1, 31]");
    }

    m_num_vars = 2 * m_n + 2 * m_n * m_n;
    m_wiring.resize(static_cast<std::size_t>(num_cells()));
    m_cells_touching.resize(static_cast<std::size_t>(m_num_vars));

    for (int i = 0; i < m_n; ++i) {
        for (int j = 0; j < m_n; ++j) {
            CellWiring w;
            w.a_var = a_var(j);
            w.b_var = b_var(i);
            w.sum_out_var = sum_var(i, j);
            w.carry_out_var = carry_var(i, j);

            // Carry ripples east along a row; column 0 has no west
            // neighbour, so its carry_in is hard 0.
            w.carry_in_var = (j == 0) ? kConstZero : carry_var(i, j - 1);
            /* Sums propagate diagonally downward and to the left, from each row into the
            next. Row 0 has no row above it, so it receives no incoming sum. The
            easternmost column has no cell diagonally above and to its right, so its
            otherwise unused sum_in port is used to capture the carry that propagates
            off the eastern end of the row above. */
            if (i == 0) {
                w.sum_in_var = kConstZero;
            } else if (j == m_n - 1) {
                w.sum_in_var = carry_var(i - 1, m_n - 1);
            } else {
                w.sum_in_var = sum_var(i - 1, j + 1);
            }

            const int cell = cell_id(i, j);
            m_wiring[static_cast<std::size_t>(cell)] = w;

            /* Construct the interaction graph. The six wires incident to a cell are
            always six distinct variables whenever they are variables at all: a and b
            are stored in separate arrays, sum_in and sum_out belong to different
            rows, and carry_in and carry_out belong to different columns or different
            rows. No deduplication is required at this point. */
            const VarIndex incident[6] = {w.a_var,       w.b_var,       w.sum_in_var,
                                          w.carry_in_var, w.sum_out_var, w.carry_out_var};
            for (VarIndex v : incident) {
                if (v != kConstZero) {
                    m_cells_touching[static_cast<std::size_t>(v)].push_back(cell);
                }
            }
        }
    }

    /* Product bits: the west edge carries p_0 through p_{n-1}, the south edge
    carries p_n through p_{2n-2}, and the carry of the south-east corner cell
    is the most significant bit, p_{2n-1}. The derivation is given in the
    header. */
    m_product_vars.assign(static_cast<std::size_t>(2 * m_n), kConstZero);
    for (int k = 0; k < m_n; ++k) {
        m_product_vars[static_cast<std::size_t>(k)] = sum_var(k, 0);
    }
    for (int j = 1; j < m_n; ++j) {
        m_product_vars[static_cast<std::size_t>(m_n - 1 + j)] = sum_var(m_n - 1, j);
    }
    m_product_vars[static_cast<std::size_t>(2 * m_n - 1)] = carry_var(m_n - 1, m_n - 1);
}

void MultiplierGrid::forward(std::uint64_t a, std::uint64_t b, BitState& state) const {
    const std::uint64_t limit = (m_n >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << m_n) - 1);
    if (a > limit || b > limit) {
        throw std::invalid_argument("MultiplierGrid::forward: operand does not fit in n bits");
    }
    if (state.size() != static_cast<std::size_t>(m_num_vars)) {
        state = BitState(static_cast<std::size_t>(m_num_vars));
    }

    for (int j = 0; j < m_n; ++j) {
        state.set(a_var(j), static_cast<std::uint8_t>((a >> j) & 1u));
    }
    for (int i = 0; i < m_n; ++i) {
        state.set(b_var(i), static_cast<std::uint8_t>((b >> i) & 1u));
    }

    /* Row-major evaluation constitutes a valid topological order: cell (i, j)
    reads carry_out(i, j-1), which was produced earlier in the same row, and
    either sum_out(i-1, j+1) or carry_out(i-1, n-1), both of which were
    produced in the row above. */
    for (int i = 0; i < m_n; ++i) {
        for (int j = 0; j < m_n; ++j) {
            const CellWiring& w = wiring(cell_id(i, j));
            const int pp = (state.get(w.a_var) & state.get(w.b_var)) ? 1 : 0;
            const int total = pp + state.get(w.sum_in_var) + state.get(w.carry_in_var);
            state.set(w.sum_out_var, static_cast<std::uint8_t>(total & 1));
            state.set(w.carry_out_var, static_cast<std::uint8_t>(total >> 1));
        }
    }
}

std::uint64_t MultiplierGrid::read_a(const BitState& state) const {
    std::uint64_t a = 0;
    for (int j = 0; j < m_n; ++j) {
        a |= static_cast<std::uint64_t>(state.get(a_var(j))) << j;
    }
    return a;
}

std::uint64_t MultiplierGrid::read_b(const BitState& state) const {
    std::uint64_t b = 0;
    for (int i = 0; i < m_n; ++i) {
        b |= static_cast<std::uint64_t>(state.get(b_var(i))) << i;
    }
    return b;
}

std::uint64_t MultiplierGrid::read_product(const BitState& state) const {
    std::uint64_t p = 0;
    for (int k = 0; k < 2 * m_n; ++k) {
        p |= static_cast<std::uint64_t>(state.get(product_var(k))) << k;
    }
    return p;
}

void MultiplierGrid::write_product(std::uint64_t product, BitState& state) const {
    const int width = 2 * m_n;
    const std::uint64_t limit =
        (width >= 64) ? ~std::uint64_t{0} : ((std::uint64_t{1} << width) - 1);
    if (product > limit) {
        throw std::invalid_argument("MultiplierGrid::write_product: target does not fit in 2n bits");
    }
    for (int k = 0; k < width; ++k) {
        state.set(product_var(k), static_cast<std::uint8_t>((product >> k) & 1u));
    }
}

int MultiplierGrid::cell_energy(const BitState& state, int cell) const {
    const CellWiring& w = wiring(cell);
    return cell_energy_from_wires(state.get(w.a_var), state.get(w.b_var), state.get(w.sum_in_var),
                                  state.get(w.carry_in_var), state.get(w.sum_out_var),
                                  state.get(w.carry_out_var));
}

int MultiplierGrid::cell_energy(const std::vector<std::uint8_t>& snap, int cell) const {
    const CellWiring& w = wiring(cell);
    return cell_energy_from_wires(snapshot_get(snap, w.a_var), snapshot_get(snap, w.b_var),
                                  snapshot_get(snap, w.sum_in_var),
                                  snapshot_get(snap, w.carry_in_var),
                                  snapshot_get(snap, w.sum_out_var),
                                  snapshot_get(snap, w.carry_out_var));
}

long long MultiplierGrid::total_energy(const BitState& state) const {
    long long total = 0;
    const int cells = num_cells();
    for (int cell = 0; cell < cells; ++cell) {
        total += cell_energy(state, cell);
    }
    return total;
}

std::string MultiplierGrid::to_string(const BitState& state) const {
    std::ostringstream oss;
    oss << "a = " << read_a(state) << ", b = " << read_b(state)
        << ", product bits = " << read_product(state) << ", energy = " << total_energy(state)
        << "\n";
    oss << "per-cell energy (row 0 = north edge, col 0 = west edge):\n";
    for (int i = 0; i < m_n; ++i) {
        oss << "  ";
        for (int j = 0; j < m_n; ++j) {
            oss << cell_energy(state, cell_id(i, j)) << ' ';
        }
        oss << '\n';
    }
    return oss.str();
}

}
