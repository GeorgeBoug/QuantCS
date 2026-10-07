// Cross-backend correctness test: runs the same Bell-state and GHZ-state
// circuits from test_bell_ghz.cpp on both the CPU (OpenMP) and GPU (CUDA)
// backends, and checks that per-basis-state probabilities agree within
// floating-point tolerance. This only builds when QUANTCS_WITH_CUDA is
// defined (see tests/CMakeLists.txt) - Circuit::run_gpu() doesn't exist
// otherwise.

#include "quantcs/circuit.hpp"

#include <cmath>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;
constexpr double kTolerance = 1e-6;

void compare_circuit(const std::string& label, qcs::Circuit& circuit) {
    std::cout << "Test: " << label << " (CPU vs GPU)\n";

    const qcs::StateVector cpu_state = circuit.run();
    const qcs::StateVectorGpu gpu_state = circuit.run_gpu();

    const std::vector<double> cpu_probs = cpu_state.probabilities();
    const std::vector<double> gpu_probs = gpu_state.probabilities();

    if (cpu_probs.size() != gpu_probs.size()) {
        std::cout << "  [FAIL] probability vector size mismatch (cpu=" << cpu_probs.size()
                  << ", gpu=" << gpu_probs.size() << ")\n";
        ++g_failures;
        return;
    }

    int local_failures = 0;
    for (std::size_t i = 0; i < cpu_probs.size(); ++i) {
        const double diff = std::fabs(cpu_probs[i] - gpu_probs[i]);
        const bool ok = diff <= kTolerance;
        if (!ok) {
            std::cout << "  [FAIL] basis state " << i << ": cpu=" << cpu_probs[i]
                      << " gpu=" << gpu_probs[i] << " diff=" << diff << "\n";
            ++local_failures;
        }
    }
    if (local_failures == 0) {
        std::cout << "  [PASS] all " << cpu_probs.size() << " basis-state probabilities match within "
                  << kTolerance << "\n";
    }
    g_failures += local_failures;
}

}

int main() {
    qcs::Circuit bell(2);
    bell.h(0).cnot(0, 1);
    compare_circuit("Bell state (H + CNOT, 2 qubits)", bell);

    qcs::Circuit ghz(3);
    ghz.h(0).cnot(0, 1).cnot(0, 2);
    compare_circuit("GHZ state (H + 2x CNOT, 3 qubits)", ghz);

    std::cout << "\n";
    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    } else {
        std::cout << g_failures << " CHECK(S) FAILED\n";
        return 1;
    }
}
