#pragma once

#include <complex>
#include <cstdint>
#include <vector>
#include <array>
#include <string>

namespace qcs {

using Complex = std::complex<double>;

// A 2x2 complex matrix, row-major: {{a00, a01}, {a10, a11}}.
using Mat2x2 = std::array<Complex, 4>;

// StateVector holds the 2^n complex amplitudes of an n-qubit register and
// applies gates to it in place.
//
// -------------------------------------------------------------------------
// Index / bit-masking convention (read this before touching gate code)
// -------------------------------------------------------------------------
// An amplitude index `i` is an n-bit integer in [0, 2^n). Bit `t` of `i`
// (counting from bit 0 = least significant) tells you the basis-state value
// of qubit `t` in that term of the superposition: bit 0 -> qubit is |0>,
// bit 1 -> qubit is |1>. So qubit `t` maps to the bitmask `1u << t`.
//
// Applying a single-qubit gate to qubit `t` only ever mixes amplitudes
// whose indices differ *exactly* in bit `t` and agree on every other bit
// (every other qubit's value is a spectator, unchanged by the gate). That
// means the 2^n amplitudes split into 2^(n-1) independent pairs:
//
//   (i0, i1)  where i0 has bit t = 0, i1 = i0 | (1<<t) has bit t = 1,
//             and i0, i1 agree on all bits other than t.
//
// To enumerate exactly these pairs without duplicates, we iterate `i0` over
// all indices with bit `t` cleared. That set has size 2^(n-1). One way to
// generate it directly: split the index into a "low" part (bits below t)
// and a "high" part (bits at/above t), and reinsert a 0 at position t:
//
//   low  = g & ((1<<t) - 1)          // bits below t, kept as-is
//   high = g >> t                    // bits at/above t, shifted up by
//                                     // one position to make room for bit t
//   i0   = (high << (t+1)) | low     // low bits unchanged, high bits
//                                     // shifted left by 1, bit t = 0
//   i1   = i0 | (1 << t)             // same index with bit t set to 1
//
// where `g` ranges over [0, 2^(n-1)). This is a bijection between g and
// the (i0, i1) pairs: every pair is visited once, none are skipped, none
// are duplicated. Because each pair only touches amplitudes[i0] and
// amplitudes[i1], and different pairs never share an index, the 2^(n-1)
// iterations are fully independent -- which is exactly what lets us hand
// the loop to OpenMP with a plain `#pragma omp parallel for` and no
// synchronization between iterations.
//
// For each pair, applying gate matrix U = [[a,b],[c,d]] means treating
// (amplitudes[i0], amplitudes[i1]) as a 2-vector (a0, a1) and replacing it
// with U * (a0, a1):
//
//   new_a0 = a*a0 + b*a1
//   new_a1 = c*a0 + d*a1
//
// Two-qubit gates (CNOT) generalize this: we iterate over quadruples of
// indices that agree on every bit except the control and target bits, and
// only move amplitude between the two indices where control = 1 (the
// target-flip only happens when the control qubit is |1>); the two
// control=0 indices are left untouched.
class StateVector {
public:
    explicit StateVector(int num_qubits);

    int num_qubits() const { return m_num_qubits; }
    std::size_t size() const { return m_amplitudes.size(); }

    std::vector<Complex>& amplitudes() { return m_amplitudes; }
    const std::vector<Complex>& amplitudes() const { return m_amplitudes; }

    // Reset to the |00...0> basis state.
    void reset_to_zero_state();

    // Apply an arbitrary single-qubit gate matrix to qubit `target`.
    void apply_single_qubit_gate(int target, const Mat2x2& gate);

    // Named single-qubit gates.
    void h(int target);
    void x(int target);
    void z(int target);
    void rz(int target, double theta);
    void ry(int target, double theta);

    // Two-qubit controlled-NOT: flips `target` iff `control` is |1>.
    void cnot(int control, int target);

    // Multiplies amplitude[index] by -1, leaving every other amplitude
    // untouched. This is a diagonal n-qubit unitary (all +1 on the
    // diagonal except a single -1), i.e. exactly what a multi-controlled-Z
    // gate realizes when its controls are wired (via X-sandwiching where
    // needed) to fire only on one specific basis state. It's the one
    // primitive both pieces of Grover's algorithm reduce to: the oracle is
    // phase_flip(marked_state), and the diffusion operator's reflection
    // step is H^{\otimes n}, phase_flip(0), H^{\otimes n}. Unlike the
    // gates above, this touches exactly one amplitude, so there is no
    // pair/quadruple decomposition to parallelize -- it's already O(1).
    void phase_flip(std::size_t index);

    // Probability of measuring basis state `i` (|amplitude[i]|^2).
    double probability(std::size_t index) const;

    // All 2^n basis-state probabilities.
    std::vector<double> probabilities() const;

    // Human-readable dump of nonzero-probability basis states, e.g.
    // "|00>: 0.5000   |11>: 0.5000". Entries below `threshold` are skipped.
    std::string to_string(double threshold = 1e-9) const;

private:
    int m_num_qubits;
    std::vector<Complex> m_amplitudes;
};

}
