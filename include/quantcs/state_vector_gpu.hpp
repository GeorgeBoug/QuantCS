#pragma once

#include "quantcs/state_vector.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace qcs {

// {free_bytes, total_bytes} currently reported by the CUDA driver for the
// active device. Lets a caller (e.g. a benchmark sweeping qubit counts)
// check whether an allocation will fit before attempting it, rather than
// only finding out via a failed cudaMalloc.
std::pair<std::size_t, std::size_t> gpu_memory_info();

// GPU-backed state vector. Same flat representation as StateVector --
// 2^n complex amplitudes, one per basis state -- except the array lives in
// CUDA device global memory instead of host RAM, and gates are applied by
// CUDA kernels instead of an OpenMP-parallelized host loop.
//
// This header is plain C++ (no CUDA syntax), so it can be included from
// .cpp translation units compiled by a regular host compiler (e.g.
// circuit.cpp). The device pointer is stored as `void*` here and only
// reinterpreted as `cuDoubleComplex*` inside state_vector_gpu.cu, which is
// the one file actually compiled by nvcc. This mirrors how the rest of the
// library never has to know CUDA exists unless QUANTCS_WITH_CUDA is
// defined by CMake.
//
// Threading model: each gate call launches one CUDA kernel that maps
// exactly one GPU thread to one independent amplitude pair (single-qubit
// gates) or quadruple (CNOT) -- the same index pairs/quadruples the CPU
// backend's OpenMP loop iterates over, described in state_vector.hpp.
// Where the CPU version hands loop iterations to OpenMP threads, the GPU
// version hands them to CUDA threads; the index math is identical.
class StateVectorGpu {
public:
    explicit StateVectorGpu(int num_qubits);
    ~StateVectorGpu();

    // Device memory is owned exclusively by this object; copying would
    // require either a deep device-to-device copy or shared ownership,
    // neither of which any current caller needs.
    StateVectorGpu(const StateVectorGpu&) = delete;
    StateVectorGpu& operator=(const StateVectorGpu&) = delete;
    StateVectorGpu(StateVectorGpu&& other) noexcept;
    StateVectorGpu& operator=(StateVectorGpu&& other) noexcept;

    int num_qubits() const { return m_num_qubits; }
    std::size_t size() const { return m_size; }

    void reset_to_zero_state();

    void apply_single_qubit_gate(int target, const Mat2x2& gate);
    void h(int target);
    void x(int target);
    void z(int target);
    void rz(int target, double theta);
    void ry(int target, double theta);
    void cnot(int control, int target);

    // Same diagonal phase-flip primitive as StateVector::phase_flip -- see
    // that declaration for why this is the operation Grover's oracle and
    // diffusion step are both built from. Implemented as a single-thread
    // CUDA kernel: only one amplitude changes, so there's no work to
    // spread across threads.
    void phase_flip(std::size_t index);

    // Copies the full state vector from device to host. Every
    // measurement/printing entry point below funnels through this, so the
    // device-to-host transfer happens exactly once per call, not once per
    // amplitude.
    std::vector<Complex> to_host() const;

    double probability(std::size_t index) const;
    std::vector<double> probabilities() const;
    std::string to_string(double threshold = 1e-9) const;

private:
    int m_num_qubits;
    std::size_t m_size;
    void* m_d_amplitudes; // cuDoubleComplex*, see state_vector_gpu.cu
};

}
