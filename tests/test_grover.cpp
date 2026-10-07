// Grover's search algorithm test/demo: builds a marked-item search circuit
// via qcs::build_grover_circuit (oracle = phase_flip(marked), diffusion =
// H^n, phase_flip(0), H^n), runs it on the CPU backend and -- when this
// build has CUDA enabled - the GPU backend too, and checks that the
// marked state's probability has been amplified close to 1. Prints the
// full probability distribution before (uniform superposition only) and
// after (post Grover iterations) so the amplification is visible, not
// just asserted.

#include "quantcs/circuit.hpp"
#include "quantcs/grover.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
constexpr double kAmplificationThreshold = 0.9; // "close to 1"
constexpr double kCrossBackendTolerance = 1e-6;

std::string basis_label(std::size_t index, int num_qubits) {
    std::string s(num_qubits, '0');
    for (int b = 0; b < num_qubits; ++b) {
        if ((index >> b) & 1) s[num_qubits - 1 - b] = '1';
    }
    return "|" + s + ">";
}

void print_distribution(const std::vector<double>& probs, int num_qubits) {
    for (std::size_t i = 0; i < probs.size(); ++i) {
        std::cout << "    " << basis_label(i, num_qubits) << ": "
                  << std::fixed << std::setprecision(4) << probs[i] << "\n";
    }
}

void check(bool ok, const std::string& label) {
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << label << "\n";
    if (!ok) ++g_failures;
}

void run_grover_case(int num_qubits, std::size_t marked_state) {
    const std::string marked_label = basis_label(marked_state, num_qubits);
    std::cout << "Test: Grover search, " << num_qubits << " qubits, marked state "
              << marked_label << "\n";

    // "Before": uniform superposition only, no oracle/diffusion yet --
    // every basis state should have probability ~1/2^n.
    qcs::Circuit uniform(num_qubits);
    for (int q = 0; q < num_qubits; ++q) uniform.h(q);
    const std::vector<double> before = uniform.run().probabilities();

    std::cout << "  Before Grover iterations (uniform superposition):\n";
    print_distribution(before, num_qubits);

    const int iterations = qcs::grover_optimal_iterations(num_qubits);
    std::cout << "  Running " << iterations << " Grover iteration(s) (optimal for N="
              << (std::size_t{1} << num_qubits) << ")\n";

    qcs::Circuit grover = qcs::build_grover_circuit(num_qubits, marked_state, iterations);

    const qcs::StateVector cpu_state = grover.run();
    const std::vector<double> after_cpu = cpu_state.probabilities();

    std::cout << "  After Grover iterations (CPU backend):\n";
    print_distribution(after_cpu, num_qubits);

    check(after_cpu[marked_state] >= kAmplificationThreshold,
          "CPU: P(" + marked_label + ") = " + std::to_string(after_cpu[marked_state]) +
              " >= " + std::to_string(kAmplificationThreshold));

    double cpu_total = 0.0;
    for (double p : after_cpu) cpu_total += p;
    check(std::fabs(cpu_total - 1.0) <= 1e-6, "CPU: total probability sums to 1");

#ifdef QUANTCS_WITH_CUDA
    const qcs::StateVectorGpu gpu_state = grover.run_gpu();
    const std::vector<double> after_gpu = gpu_state.probabilities();

    std::cout << "  After Grover iterations (GPU backend):\n";
    print_distribution(after_gpu, num_qubits);

    check(after_gpu[marked_state] >= kAmplificationThreshold,
          "GPU: P(" + marked_label + ") = " + std::to_string(after_gpu[marked_state]) +
              " >= " + std::to_string(kAmplificationThreshold));

    bool backends_agree = after_cpu.size() == after_gpu.size();
    if (backends_agree) {
        for (std::size_t i = 0; i < after_cpu.size(); ++i) {
            if (std::fabs(after_cpu[i] - after_gpu[i]) > kCrossBackendTolerance) {
                backends_agree = false;
                break;
            }
        }
    }
    check(backends_agree, "CPU and GPU distributions match within " +
                               std::to_string(kCrossBackendTolerance));
#else
    std::cout << "  (GPU backend skipped -- built without CUDA support)\n";
#endif

    std::cout << "\n";
}

} // namespace

int main() {
    run_grover_case(3, 0b101); // 3 qubits, marked state |101>
    run_grover_case(4, 0b1011); // 4 qubits, marked state |1011>

    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    } else {
        std::cout << g_failures << " CHECK(S) FAILED\n";
        return 1;
    }
}
