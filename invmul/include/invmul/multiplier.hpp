#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// =========================================================================
// invmul -- a bidirectional (invertible) array multiplier
// =========================================================================
//
// This module builds an n x n array multiplier out of identical logic
// cells, then runs it *backwards*: instead of feeding in two factors and
// reading out a product, we clamp the product to a target N and let the
// cells locally repair each other until every cell is simultaneously
// consistent. A fully consistent state is, by construction, a
// factorization of N.
//
// -------------------------------------------------------------------------
// 1. Grid topology (which cell talks to which)
// -------------------------------------------------------------------------
// The array is the classic ripple-carry ("Braun-style") array multiplier.
// Cell (i, j) sits at row i (0 = north edge) and column j (0 = west edge),
// and contains exactly one AND gate and one full adder -- every cell in
// the grid is identical; only the wiring at the boundary differs.
//
//   * Factor A enters from the NORTH: bit a_j is driven into column j and
//     flows south, so every cell in column j sees a_j.
//   * Factor B enters from the EAST: bit b_i is driven into row i and
//     flows west, so every cell in row i sees b_i.
//
//   North and east are perpendicular, so the two operands enter from two
//   perpendicular sides, and the remaining two sides (west and south)
//   carry the product out.
//
// Cell (i, j) forms the partial product pp = a_j AND b_i, which has
// arithmetic weight 2^(i+j) -- so cells on the same anti-diagonal
// (i + j = const) all carry the same weight. The full adder then folds pp
// into a running sum:
//
//     pp + sum_in + carry_in  ==  sum_out + 2 * carry_out           (*)
//
// The wiring is chosen so that (*) is weight-consistent everywhere:
//
//     carry_in (i, j)  <-  carry_out of cell (i, j-1)     [west neighbour]
//     sum_in   (i, j)  <-  sum_out   of cell (i-1, j+1)   [north-east]
//
//   - carry_out of (i,j) has weight 2^(i+j+1), which is exactly the weight
//     of cell (i, j+1) -- so carries ripple EAST along a row.
//   - sum_out of (i,j) has weight 2^(i+j), which is exactly the weight of
//     cell (i+1, j-1) -- so sums fall SOUTH-WEST to the next row.
//
// Boundary conditions:
//
//   - Row 0 has no north neighbour        -> sum_in   = constant 0
//   - Column 0 has no west neighbour      -> carry_in = constant 0
//   - Column n-1 has no north-east neighbour, so its otherwise-unused
//     sum_in port is reused to catch the carry that ripples off the east
//     end of the previous row:
//         sum_in (i, n-1)  <-  carry_out of cell (i-1, n-1)
//     This is weight-correct. Cell (i, n-1) has weight 2^(i + n - 1), and
//     every port of a cell carries that cell's own weight, so its sum_in
//     port expects weight 2^(i + n - 1). The carry arriving there has
//     weight 2^((i-1) + (n-1) + 1) = 2^(i + n - 1). They match exactly.
//     Without this wrap those n carries would be dropped and the array
//     would compute the wrong product.
//
// -------------------------------------------------------------------------
// 2. Where the product comes out
// -------------------------------------------------------------------------
// Summing (*) over all cells, weighted by 2^(i+j), makes every internally
// routed sum_out/carry_out cancel against the sum_in/carry_in that
// consumes it. What is left is
//
//     A * B  =  sum_i 2^i * sum_out(i, 0)                  [WEST edge]
//             + sum_{j>=1} 2^(n-1+j) * sum_out(n-1, j)     [SOUTH edge]
//             + 2^(2n-1) * carry_out(n-1, n-1)             [SE corner]
//
// i.e. exactly 2n product bits:
//
//     p_k       = sum_out(k, 0)        for k in [0, n)      (west edge)
//     p_{n-1+j} = sum_out(n-1, j)      for j in [1, n)      (south edge)
//     p_{2n-1}  = carry_out(n-1, n-1)                       (south-east)
//
// So: operands in from north and east, product out to west and south.
//
// -------------------------------------------------------------------------
// 3. State variables and the energy (Ising) formulation
// -------------------------------------------------------------------------
// The state of the machine is one bit per wire that is not a boundary
// constant:
//
//     a_0 .. a_{n-1}          n bits    (factor A)
//     b_0 .. b_{n-1}          n bits    (factor B)
//     sum_out(i, j)           n^2 bits
//     carry_out(i, j)         n^2 bits
//                            ---------
//     total                   2n + 2n^2 bits
//
// Every other wire (pp, sum_in, carry_in) is *not* an independent
// variable: it is either one of the above, read through the wiring above,
// or a hard-wired 0.
//
// Each cell contributes one local energy term -- the squared arithmetic
// residual of its full-adder equation (*):
//
//     E(i,j) = ( pp + sum_in + carry_in  -  sum_out - 2*carry_out )^2
//
// which takes values in {0, 1, 4, 9} and is 0 exactly when the cell is
// locally consistent. The total energy is the sum over all cells:
//
//     E_total = sum_{i,j} E(i,j)     >= 0,   = 0 iff every cell is correct
//
// This is deliberately the same shape as an Ising / QUBO penalty: a sum of
// local terms over a fixed interaction graph, minimized at 0 on exactly the
// set of valid configurations. Two honest notes about the analogy:
//
//   * Degree. Because pp = a_j * b_i is itself quadratic, squaring the
//     residual makes E(i,j) a degree-4 polynomial in the binary variables,
//     so this is strictly a higher-order binary model (HOBO), not a
//     2-local Ising Hamiltonian. It becomes a genuine QUBO/Ising if you
//     introduce one auxiliary variable m(i,j) for the AND output and add
//     the standard AND penalty  a_j*b_i - 2*m*(a_j + b_i) + 3*m  (zero iff
//     m = a_j AND b_i, positive otherwise). We keep the direct quartic form
//     because it has the identical zero-energy set with n^2 fewer variables
//     and a cheaper local update; the reduction is only needed if you want
//     to hand the model to hardware that insists on 2-body couplings.
//
//   * Squared vs. absolute residual. |residual| would also be zero exactly
//     when consistent. Squaring is the conventional QUBO form and gives a
//     steeper penalty for badly-broken cells (9 vs 3 for a residual of 3),
//     which makes the local search prefer repairing gross violations first.
//
// Clamping the 2n product bits to the bits of a target N and searching for
// a zero-energy state is therefore *exactly* a local search for a
// factorization of N: E_total = 0 with the product bits pinned to N implies
// A * B = N by the weighted-sum identity above, and conversely every
// factorization yields a zero-energy state (just evaluate the grid
// forwards). See invmul/solver.hpp for the search itself.

namespace invmul {

// Index of a state variable in the flat variable array. Signed so that the
// kConstZero sentinel below can share the type.
using VarIndex = std::int32_t;

// Sentinel used by the wiring tables for a port that is tied to a constant
// 0 by the grid boundary rather than being a state variable. Reading it
// always yields 0 and nothing may write to it.
constexpr VarIndex kConstZero = -1;

// The six wires incident on one cell, as variable indices. `sum_in_var`
// and `carry_in_var` may be kConstZero at the north/west boundary; the
// other four are always real variables.
struct CellWiring {
    VarIndex a_var = kConstZero;         // a_j  (shared down column j)
    VarIndex b_var = kConstZero;         // b_i  (shared across row i)
    VarIndex sum_in_var = kConstZero;    // from north-east (or wrapped carry)
    VarIndex carry_in_var = kConstZero;  // from west neighbour
    VarIndex sum_out_var = kConstZero;   // this cell's sum output
    VarIndex carry_out_var = kConstZero; // this cell's carry output
};

// A flat array of one-bit state variables.
//
// The bits are stored as std::atomic<std::uint8_t> and accessed with
// memory_order_relaxed. This is *not* for synchronization -- it is so that
// the optimistic parallel solver (invmul/solver.hpp), in which several
// threads repair different regions of the grid between synchronization
// barriers and may occasionally touch the same shared variable, is
// well-defined C++ rather than a data race. Relaxed byte-sized atomics
// compile to ordinary byte loads and stores on the platforms we target, so
// this costs nothing in the single-threaded path.
class BitState {
public:
    BitState() = default;
    explicit BitState(std::size_t num_vars) : m_bits(num_vars) {
        for (auto& bit : m_bits) bit.store(0, std::memory_order_relaxed);
    }

    std::size_t size() const { return m_bits.size(); }

    // Reading kConstZero yields 0, which is what makes boundary ports work
    // without a special case at every call site.
    std::uint8_t get(VarIndex v) const {
        return v == kConstZero ? std::uint8_t{0} : m_bits[static_cast<std::size_t>(v)].load(std::memory_order_relaxed);
    }

    void set(VarIndex v, std::uint8_t value) {
        m_bits[static_cast<std::size_t>(v)].store(value, std::memory_order_relaxed);
    }

    // Plain-bytes copy in/out, used to keep the best-so-far configuration
    // and to take the read-only snapshot the synchronous update mode needs.
    std::vector<std::uint8_t> snapshot() const;
    void restore(const std::vector<std::uint8_t>& values);

private:
    std::vector<std::atomic<std::uint8_t>> m_bits;
};

// Reads a wire out of a plain snapshot buffer, honouring kConstZero the
// same way BitState::get does.
inline std::uint8_t snapshot_get(const std::vector<std::uint8_t>& snap, VarIndex v) {
    return v == kConstZero ? std::uint8_t{0} : snap[static_cast<std::size_t>(v)];
}

// The n x n grid: variable numbering, wiring, forward evaluation and the
// energy function. Immutable once constructed; a single MultiplierGrid can
// be shared by every thread of a solve.
class MultiplierGrid {
public:
    // Throws std::invalid_argument unless 1 <= n <= 31 (n = 31 already
    // implies a 62-bit product, at the edge of what std::uint64_t can hold
    // for the convenience accessors below).
    explicit MultiplierGrid(int n);

    int n() const { return m_n; }
    int num_cells() const { return m_n * m_n; }
    int num_vars() const { return m_num_vars; }
    int num_product_bits() const { return 2 * m_n; }

    int cell_id(int row, int col) const { return row * m_n + col; }
    int cell_row(int cell) const { return cell / m_n; }
    int cell_col(int cell) const { return cell % m_n; }

    const CellWiring& wiring(int cell) const { return m_wiring[static_cast<std::size_t>(cell)]; }

    // --- variable numbering (see the header comment for the layout) ---
    VarIndex a_var(int j) const { return static_cast<VarIndex>(j); }
    VarIndex b_var(int i) const { return static_cast<VarIndex>(m_n + i); }
    VarIndex sum_var(int i, int j) const { return static_cast<VarIndex>(2 * m_n + i * m_n + j); }
    VarIndex carry_var(int i, int j) const {
        return static_cast<VarIndex>(2 * m_n + m_n * m_n + i * m_n + j);
    }

    // Variable holding product bit k, for k in [0, 2n). These are the bits
    // the solver clamps to the target.
    VarIndex product_var(int k) const { return m_product_vars[static_cast<std::size_t>(k)]; }

    // Every cell whose energy depends on variable `v`. This is the
    // interaction graph of the Ising model: flipping v can only change the
    // energy of these cells, which is what makes an incremental delta-E
    // cheap (O(n) at worst, for the row/column-shared operand bits).
    const std::vector<int>& cells_touching(VarIndex v) const {
        return m_cells_touching[static_cast<std::size_t>(v)];
    }

    // --- forward (normal) operation ---------------------------------------
    // Drives a and b into the north/east edges and evaluates every cell in
    // row-major order, which is a valid topological order for the wiring
    // above (a cell's carry_in comes from its own row's previous column and
    // its sum_in from the previous row). Fills every variable in `state`.
    // Throws std::invalid_argument if a or b does not fit in n bits.
    void forward(std::uint64_t a, std::uint64_t b, BitState& state) const;

    std::uint64_t read_a(const BitState& state) const;
    std::uint64_t read_b(const BitState& state) const;
    std::uint64_t read_product(const BitState& state) const;

    // Writes the low 2n bits of `product` into the product-bit variables.
    // Throws std::invalid_argument if product does not fit in 2n bits.
    void write_product(std::uint64_t product, BitState& state) const;

    // --- energy -----------------------------------------------------------
    // Local energy of one cell: the squared residual of its full-adder
    // equation. 0 iff the cell is consistent.
    int cell_energy(const BitState& state, int cell) const;
    int cell_energy(const std::vector<std::uint8_t>& snap, int cell) const;

    // Sum of cell_energy over the whole grid.
    long long total_energy(const BitState& state) const;

    // Human-readable dump: operands, product, energy and the per-cell
    // energy map (used by the demo and by failing tests).
    std::string to_string(const BitState& state) const;

private:
    int m_n = 0;
    int m_num_vars = 0;
    std::vector<CellWiring> m_wiring;
    std::vector<VarIndex> m_product_vars;
    std::vector<std::vector<int>> m_cells_touching;
};

// The full-adder residual used by cell_energy, exposed for the solver's
// candidate-move evaluation (it scores hypothetical input assignments
// without writing them into the state first).
//
//   residual = (pp + sum_in + carry_in) - (sum_out + 2*carry_out)
//
// with pp = a AND b. Energy is residual^2.
inline int cell_residual(std::uint8_t a, std::uint8_t b, std::uint8_t sum_in, std::uint8_t carry_in,
                         std::uint8_t sum_out, std::uint8_t carry_out) {
    const int pp = (a & b) ? 1 : 0;
    const int lhs = pp + static_cast<int>(sum_in) + static_cast<int>(carry_in);
    const int rhs = static_cast<int>(sum_out) + 2 * static_cast<int>(carry_out);
    return lhs - rhs;
}

inline int cell_energy_from_wires(std::uint8_t a, std::uint8_t b, std::uint8_t sum_in,
                                  std::uint8_t carry_in, std::uint8_t sum_out,
                                  std::uint8_t carry_out) {
    const int r = cell_residual(a, b, sum_in, carry_in, sum_out, carry_out);
    return r * r;
}

}
