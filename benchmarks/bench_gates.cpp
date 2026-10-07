// Timing harness for gate application: builds a circuit of alternating
// H/CNOT layers over N qubits and times how long a full run() takes,
// optionally sweeping the OpenMP thread count via the OMP_NUM_THREADS
// environment variable (set it before invoking this binary).

#include "quantcs/circuit.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>

#ifdef _OPENMP
#include <omp.h>
#endif

int main(int argc, char** argv) {
    const int num_qubits = argc > 1 ? std::atoi(argv[1]) : 24;
    const int num_layers = argc > 2 ? std::atoi(argv[2]) : 10;

    qcs::Circuit circuit(num_qubits);
    for (int layer = 0; layer < num_layers; ++layer) {
        for (int q = 0; q < num_qubits; ++q) {
            circuit.h(q);
        }
        for (int q = 0; q + 1 < num_qubits; ++q) {
            circuit.cnot(q, q + 1);
        }
    }

#ifdef _OPENMP
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP not enabled in this build\n";
#endif
    std::cout << "Qubits: " << num_qubits << "  Layers: " << num_layers
              << "  Gates: " << circuit.gate_count() << "\n";

    {
        const auto start = std::chrono::steady_clock::now();
        qcs::StateVector state = circuit.run();
        const auto end = std::chrono::steady_clock::now();

        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[CPU/OpenMP] Elapsed: " << elapsed_ms << " ms\n";

        // Touch the result so the compiler can't optimize the whole run away.
        double total = 0.0;
        for (double p : state.probabilities()) total += p;
        std::cout << "[CPU/OpenMP] (sanity) total probability: " << total << "\n";
    }

#ifdef QUANTCS_WITH_CUDA
    {
        // Timed region intentionally excludes the one-time CUDA context
        // initialization that happens on the first CUDA call in a process
        // (device memory allocators, driver handshake, etc.) -- that cost
        // is amortized across a whole program's lifetime in any real
        // workload, not per-circuit, so a fair CPU/GPU comparison warms
        // the GPU up first.
        { qcs::StateVectorGpu warmup(1); }

        const auto start = std::chrono::steady_clock::now();
        qcs::StateVectorGpu state = circuit.run_gpu();
        const auto end = std::chrono::steady_clock::now();

        const double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        std::cout << "[GPU/CUDA]   Elapsed: " << elapsed_ms << " ms\n";

        double total = 0.0;
        for (double p : state.probabilities()) total += p;
        std::cout << "[GPU/CUDA]   (sanity) total probability: " << total << "\n";
    }
#else
    std::cout << "[GPU/CUDA]   skipped (built without CUDA support)\n";
#endif

    return 0;
}
