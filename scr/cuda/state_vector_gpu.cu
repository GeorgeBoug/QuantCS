#include "quantcs/state_vector_gpu.hpp"

#include <cuComplex.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace qcs {

namespace {

void cuda_check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

#define QCS_CUDA_CHECK(expr) cuda_check((expr), #expr)

inline cuDoubleComplex to_cu(Complex c) {
    return make_cuDoubleComplex(c.real(), c.imag());
}

inline Complex from_cu(cuDoubleComplex c) {
    return Complex(cuCreal(c), cuCimag(c));
}


constexpr int kBlockSize = 256;

inline int grid_size_for(std::int64_t n_threads) {
    return static_cast<int>((n_threads + kBlockSize - 1) / kBlockSize);
}


__global__ void single_qubit_gate_kernel(
    cuDoubleComplex* amp,
    std::int64_t half,
    std::size_t low_mask,
    int target,
    cuDoubleComplex a, cuDoubleComplex b, cuDoubleComplex c, cuDoubleComplex d) {
    const std::int64_t g = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= half) return;

    const std::size_t gg = static_cast<std::size_t>(g);
    const std::size_t low  = gg & low_mask;
    const std::size_t high = gg >> target;
    const std::size_t i0 = (high << (target + 1)) | low;
    const std::size_t i1 = i0 | (std::size_t{1} << target);

    const cuDoubleComplex a0 = amp[i0];
    const cuDoubleComplex a1 = amp[i1];
    amp[i0] = cuCadd(cuCmul(a, a0), cuCmul(b, a1));
    amp[i1] = cuCadd(cuCmul(c, a0), cuCmul(d, a1));
}

// One CUDA thread per independent index quadruple, mirroring
// StateVector::cnot's (lo, mid, top) bit-insertion. Each thread swaps the
// two amplitudes where control=1; control=0 amplitudes are never touched.
__global__ void cnot_kernel(
    cuDoubleComplex* amp,
    std::int64_t quarter,
    std::size_t lo_mask,
    std::size_t mid_mask,
    int lo, int hi,
    std::size_t control_bit,
    std::size_t target_bit) {
    const std::int64_t g = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (g >= quarter) return;

    const std::size_t gg = static_cast<std::size_t>(g);
    const std::size_t low = gg & lo_mask;
    const std::size_t rest = gg >> lo;
    const std::size_t mid = rest & mid_mask;
    const std::size_t top = rest >> (hi - lo - 1);

    const std::size_t i00 = (top << (hi + 1)) | (mid << (lo + 1)) | low;
    const std::size_t i_ctrl1_tgt0 = i00 | control_bit;
    const std::size_t i_ctrl1_tgt1 = i00 | control_bit | target_bit;

    const cuDoubleComplex tmp = amp[i_ctrl1_tgt0];
    amp[i_ctrl1_tgt0] = amp[i_ctrl1_tgt1];
    amp[i_ctrl1_tgt1] = tmp;
}

// Negates a single amplitude. Launched as a single thread: unlike the
// gate kernels above, there are no independent pairs/quadruples to spread
// across threads here -- exactly one array element changes.
__global__ void phase_flip_kernel(cuDoubleComplex* amp, std::size_t index) {
    amp[index] = make_cuDoubleComplex(-amp[index].x, -amp[index].y);
}

inline cuDoubleComplex* as_cu(void* p) { return reinterpret_cast<cuDoubleComplex*>(p); }

} // namespace

std::pair<std::size_t, std::size_t> gpu_memory_info() {
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    QCS_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    return {free_bytes, total_bytes};
}

StateVectorGpu::StateVectorGpu(int num_qubits)
    : num_qubits_(num_qubits), size_(std::size_t{1} << num_qubits), d_amplitudes_(nullptr) {
    if (num_qubits <= 0) {
        throw std::invalid_argument("num_qubits must be positive");
    }
    void* raw = nullptr;
    QCS_CUDA_CHECK(cudaMalloc(&raw, size_ * sizeof(cuDoubleComplex)));
    d_amplitudes_ = raw;
    reset_to_zero_state();
}

StateVectorGpu::~StateVectorGpu() {
    if (d_amplitudes_ != nullptr) {
        cudaFree(d_amplitudes_); // destructors must not throw; ignore the status
    }
}

StateVectorGpu::StateVectorGpu(StateVectorGpu&& other) noexcept
    : num_qubits_(other.num_qubits_), size_(other.size_), d_amplitudes_(other.d_amplitudes_) {
    other.d_amplitudes_ = nullptr;
    other.size_ = 0;
}

StateVectorGpu& StateVectorGpu::operator=(StateVectorGpu&& other) noexcept {
    if (this != &other) {
        if (d_amplitudes_ != nullptr) {
            cudaFree(d_amplitudes_);
        }
        num_qubits_ = other.num_qubits_;
        size_ = other.size_;
        d_amplitudes_ = other.d_amplitudes_;
        other.d_amplitudes_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

void StateVectorGpu::reset_to_zero_state() {
    // The IEEE-754 bit pattern for 0.0 is all-zero bits, so zeroing the
    // raw bytes of the whole buffer is a valid (and much faster than a
    // kernel launch) way to set every amplitude to 0+0i.
    QCS_CUDA_CHECK(cudaMemset(d_amplitudes_, 0, size_ * sizeof(cuDoubleComplex)));
    const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
    QCS_CUDA_CHECK(cudaMemcpy(d_amplitudes_, &one, sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
}

void StateVectorGpu::apply_single_qubit_gate(int target, const Mat2x2& gate) {
    const std::int64_t half = static_cast<std::int64_t>(size_ >> 1);
    const std::size_t low_mask = (std::size_t{1} << target) - 1;

    single_qubit_gate_kernel<<<grid_size_for(half), kBlockSize>>>(
        as_cu(d_amplitudes_), half, low_mask, target,
        to_cu(gate[0]), to_cu(gate[1]), to_cu(gate[2]), to_cu(gate[3]));
    QCS_CUDA_CHECK(cudaGetLastError());
    QCS_CUDA_CHECK(cudaDeviceSynchronize());
}

void StateVectorGpu::h(int target) {
    static const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    apply_single_qubit_gate(target, {Complex(inv_sqrt2, 0), Complex(inv_sqrt2, 0),
                                      Complex(inv_sqrt2, 0), Complex(-inv_sqrt2, 0)});
}

void StateVectorGpu::x(int target) {
    apply_single_qubit_gate(target, {Complex(0, 0), Complex(1, 0),
                                      Complex(1, 0), Complex(0, 0)});
}

void StateVectorGpu::z(int target) {
    apply_single_qubit_gate(target, {Complex(1, 0), Complex(0, 0),
                                      Complex(0, 0), Complex(-1, 0)});
}

void StateVectorGpu::rz(int target, double theta) {
    const Complex e_minus = std::polar(1.0, -theta / 2.0);
    const Complex e_plus  = std::polar(1.0,  theta / 2.0);
    apply_single_qubit_gate(target, {e_minus, Complex(0, 0),
                                      Complex(0, 0), e_plus});
}

void StateVectorGpu::ry(int target, double theta) {
    const double c = std::cos(theta / 2.0);
    const double s = std::sin(theta / 2.0);
    apply_single_qubit_gate(target, {Complex(c, 0), Complex(-s, 0),
                                      Complex(s, 0), Complex(c, 0)});
}

void StateVectorGpu::cnot(int control, int target) {
    const std::int64_t quarter = static_cast<std::int64_t>(size_ >> 2);

    const int lo = std::min(control, target);
    const int hi = std::max(control, target);
    const std::size_t lo_mask = (std::size_t{1} << lo) - 1;
    const std::size_t mid_mask = (std::size_t{1} << (hi - lo - 1)) - 1;
    const std::size_t control_bit = std::size_t{1} << control;
    const std::size_t target_bit = std::size_t{1} << target;

    cnot_kernel<<<grid_size_for(quarter), kBlockSize>>>(
        as_cu(d_amplitudes_), quarter, lo_mask, mid_mask, lo, hi, control_bit, target_bit);
    QCS_CUDA_CHECK(cudaGetLastError());
    QCS_CUDA_CHECK(cudaDeviceSynchronize());
}

void StateVectorGpu::phase_flip(std::size_t index) {
    phase_flip_kernel<<<1, 1>>>(as_cu(d_amplitudes_), index);
    QCS_CUDA_CHECK(cudaGetLastError());
    QCS_CUDA_CHECK(cudaDeviceSynchronize());
}

std::vector<Complex> StateVectorGpu::to_host() const {
    std::vector<cuDoubleComplex> host_buf(size_);
    QCS_CUDA_CHECK(cudaMemcpy(host_buf.data(), d_amplitudes_,
                              size_ * sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    std::vector<Complex> result(size_);
    for (std::size_t i = 0; i < size_; ++i) {
        result[i] = from_cu(host_buf[i]);
    }
    return result;
}

double StateVectorGpu::probability(std::size_t index) const {
    cuDoubleComplex value;
    QCS_CUDA_CHECK(cudaMemcpy(&value, as_cu(d_amplitudes_) + index,
                              sizeof(cuDoubleComplex), cudaMemcpyDeviceToHost));
    const double re = cuCreal(value);
    const double im = cuCimag(value);
    return re * re + im * im;
}

std::vector<double> StateVectorGpu::probabilities() const {
    const std::vector<Complex> host = to_host();
    std::vector<double> probs(host.size());
    for (std::size_t i = 0; i < host.size(); ++i) {
        probs[i] = host[i].real() * host[i].real() + host[i].imag() * host[i].imag();
    }
    return probs;
}

std::string StateVectorGpu::to_string(double threshold) const {
    const std::vector<double> probs = probabilities();
    std::ostringstream oss;
    for (std::size_t i = 0; i < probs.size(); ++i) {
        if (probs[i] < threshold) continue;
        oss << "|";
        for (int b = num_qubits_ - 1; b >= 0; --b) {
            oss << ((i >> b) & 1);
        }
        oss << ">: " << probs[i] << "\n";
    }
    return oss.str();
}

}
