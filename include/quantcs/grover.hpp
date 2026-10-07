#pragma once

#include "quantcs/circuit.hpp"

#include <cstddef>

namespace qcs {

// Number of Grover iterations that maximizes the marked state's
// measurement probability, for a single marked item among 2^num_qubits
// basis states. Closed form: with theta = asin(1/sqrt(N)), each iteration
// rotates the state by 2*theta towards the marked state (in the 2D plane
// spanned by the marked and unmarked subspaces), starting pi/2 - theta
// away from it; the iteration count nearest a total rotation of pi/2 is
// round(pi/(4*theta) - 1/2).
int grover_optimal_iterations(int num_qubits);

// Builds a Grover's-algorithm circuit over `num_qubits` qubits searching
// for `marked_state` (an index in [0, 2^num_qubits)):
//
//   1. Uniform superposition: H on every qubit.
//   2. `num_iterations` repetitions of:
//        a. Oracle: flip the sign of the marked state's amplitude.
//        b. Diffusion: reflect the state about the uniform superposition,
//           which amplifies the marked state's amplitude at the expense
//           of every other basis state's.
//
// Pass -1 (the default) for `num_iterations` to use
// grover_optimal_iterations(num_qubits).
//
// Throws std::invalid_argument if marked_state is out of range -- this is
// the one input to the algorithm a caller supplies directly (as opposed to
// values only ever produced internally), so it's the one place in the
// Grover construction worth validating.
Circuit build_grover_circuit(int num_qubits, std::size_t marked_state, int num_iterations = -1);

}
