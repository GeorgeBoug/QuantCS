// Correctness tests for QuantCS: Bell state (2 qubits) and GHZ state
// (3 qubits). No external test framework - plain checks with clear
// pass/fail printing, process exit code communicates overall result to
// CTest.

#include "quantcs/circuit.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

namespace {

int g_failures = 0;

void check_close(const std::string& label, double actual, double expected, double tol = 1e-6) {
    const bool ok = std::fabs(actual - expected) <= tol;
    std::cout << "  [" << (ok ? "PASS" : "FAIL") << "] " << label
              << " (expected " << expected << ", got " << actual << ")\n";
    if (!ok) ++g_failures;
}

// Bell state |Phi+> = (|00> + |11>) / sqrt(2): H(0) then CNOT(0 -> 1).
// Every other basis state should have ~0 probability.
void test_bell_state() {
    std::cout << "Test: Bell state (H + CNOT, 2 qubits)\n";

    qcs::Circuit circuit(2);
    circuit.h(0).cnot(0, 1);
    qcs::StateVector state = circuit.run();

    // Index bit b corresponds to qubit b (bit 0 = qubit 0, bit 1 = qubit 1).
    check_close("P(|00>)", state.probability(0b00), 0.5);
    check_close("P(|01>)", state.probability(0b01), 0.0);
    check_close("P(|10>)", state.probability(0b10), 0.0);
    check_close("P(|11>)", state.probability(0b11), 0.5);

    double total = 0.0;
    for (double p : state.probabilities()) total += p;
    check_close("total probability sums to 1", total, 1.0);
}

// GHZ state |GHZ> = (|000> + |111>) / sqrt(2): H(0), CNOT(0->1), CNOT(0->2).
void test_ghz_state() {
    std::cout << "Test: GHZ state (H + 2x CNOT, 3 qubits)\n";

    qcs::Circuit circuit(3);
    circuit.h(0).cnot(0, 1).cnot(0, 2);
    qcs::StateVector state = circuit.run();

    check_close("P(|000>)", state.probability(0b000), 0.5);
    check_close("P(|111>)", state.probability(0b111), 0.5);

    // Every other basis state should be ~0.
    for (std::size_t i = 0; i < state.size(); ++i) {
        if (i == 0b000 || i == 0b111) continue;
        check_close("P(other basis state " + std::to_string(i) + ")", state.probability(i), 0.0);
    }

    double total = 0.0;
    for (double p : state.probabilities()) total += p;
    check_close("total probability sums to 1", total, 1.0);
}

}

int main() {
    test_bell_state();
    test_ghz_state();

    std::cout << "\n";
    if (g_failures == 0) {
        std::cout << "ALL TESTS PASSED\n";
        return 0;
    } else {
        std::cout << g_failures << " CHECK(S) FAILED\n";
        return 1;
    }
}
