#pragma once

#include "quantcs/state_vector.hpp"

#ifdef QUANTCS_WITH_CUDA
#include "quantcs/state_vector_gpu.hpp"
#endif

#include <vector>

namespace qcs {

enum class GateType {
    H,
    X,
    Z,
    RZ,
    RY,
    CNOT,
    PHASE_FLIP
};

// A single scheduled gate operation. For single-qubit gates, `qubit` is the
// target and `control` is unused; `theta` is only used by RZ/RY;
// `basis_index` is only used by PHASE_FLIP.
struct GateOp {
    GateType type;
    int qubit = -1;
    int control = -1;
    double theta = 0.0;
    std::size_t basis_index = 0;
};

// Circuit records a sequence of gates and applies them, in order, to a
// state vector initialized to |0...0>.
class Circuit {
public:
    explicit Circuit(int num_qubits);

    int num_qubits() const { return m_num_qubits; }

    Circuit& h(int target);
    Circuit& x(int target);
    Circuit& z(int target);
    Circuit& rz(int target, double theta);
    Circuit& ry(int target, double theta);
    Circuit& cnot(int control, int target);

    // Multiplies the amplitude of basis state `index` by -1. See
    // StateVector::phase_flip for what this represents and why it's the
    // primitive Grover's algorithm (include/quantcs/grover.hpp) is
    // built from.
    Circuit& phase_flip(std::size_t index);

    // Applies all recorded gates in order to a fresh |0...0> state and
    // returns the resulting state vector. Runs on the CPU (OpenMP-parallel
    // loop over amplitude pairs).
    StateVector run() const;

#ifdef QUANTCS_WITH_CUDA
    // Same gate sequence, same semantics as run() -- applied to a
    // GPU-resident state vector instead. The circuit itself is backend-
    // agnostic: it just records GateOp entries, and each backend's run()
    // replays them against its own StateVector implementation.
    StateVectorGpu run_gpu() const;
#endif

    std::size_t gate_count() const { return m_ops.size(); }

private:
    int m_num_qubits;
    std::vector<GateOp> m_ops;
};

}
