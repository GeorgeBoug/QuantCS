#include "quantcs/circuit.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

qcs::Circuit build_benchmark_circuit(int num_qubits, int num_layers) {
    qcs::Circuit circuit(num_qubits);
    for (int layer = 0; layer < num_layers; ++layer) {
        for (int q = 0; q < num_qubits; ++q) {
            circuit.h(q);
        }
        for (int q = 0; q + 1 < num_qubits; ++q) {
            circuit.cnot(q, q + 1);
        }
    }
    return circuit;
}

// A conservative safety margin below the CUDA driver's reported free
// memory -- other allocations (CUDA context, driver bookkeeping) already
// eat into it, and we'd rather report "oom" cleanly than get a genuine
// cudaMalloc failure mid-sweep.
constexpr double kGpuMemorySafetyFactor = 0.9;

std::string format_mib(double bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1) << (bytes / (1024.0 * 1024.0));
    return oss.str();
}

// Status strings can contain commas (e.g. "oom (needs 4096.0 MiB, 3962.0
// MiB free)") or, via an exception's what(), arbitrary text -- quote them
// properly so the CSV stays well-formed instead of silently gaining extra
// columns when read back by pandas/Excel/etc.
std::string csv_field(const std::string& s) {
    std::string escaped;
    escaped.reserve(s.size() + 2);
    escaped += '"';
    for (char c : s) {
        if (c == '"') escaped += '"';
        escaped += c;
    }
    escaped += '"';
    return escaped;
}


template <typename Fn>
double time_with_adaptive_repeats(Fn&& fn) {
    auto once = [&]() {
        const auto start = std::chrono::steady_clock::now();
        fn();
        const auto end = std::chrono::steady_clock::now();
        return std::chrono::duration<double, std::milli>(end - start).count();
    };

    double best = once();
    if (best < 200.0) {
        for (int i = 0; i < 2; ++i) {
            best = std::min(best, once());
        }
    }
    return best;
}

}

int main(int argc, char** argv) {
    const int min_qubits = argc > 1 ? std::atoi(argv[1]) : 4;
    const int max_qubits = argc > 2 ? std::atoi(argv[2]) : 26;
    const int num_layers = argc > 3 ? std::atoi(argv[3]) : 5;
    const std::string csv_path = argc > 4 ? argv[4] : "scaling_results.csv";

    std::ofstream csv(csv_path);
    if (!csv) {
        std::cerr << "Failed to open " << csv_path << " for writing\n";
        return 1;
    }
    csv << "qubits,gate_count,memory_bytes,memory_mib,cpu_ms,cpu_status,gpu_ms,gpu_status,speedup\n";

#ifdef _OPENMP
    std::cout << "OpenMP max threads: " << omp_get_max_threads() << "\n";
#else
    std::cout << "OpenMP not enabled in this build\n";
#endif
#ifdef QUANTCS_WITH_CUDA
    std::cout << "GPU backend: enabled\n";
#else
    std::cout << "GPU backend: not built (CPU-only)\n";
#endif
    std::cout << "Circuit: " << num_layers << " layer(s) of [H on every qubit; CNOT chain]\n";
    std::cout << "Sweeping qubits " << min_qubits << ".." << max_qubits << ", writing to "
              << csv_path << "\n\n";

    std::cout << std::left
              << std::setw(8) << "Qubits"
              << std::setw(14) << "Memory(MiB)"
              << std::setw(14) << "CPU(ms)"
              << std::setw(14) << "GPU(ms)"
              << std::setw(10) << "Speedup" << "\n";

#ifdef QUANTCS_WITH_CUDA
    // Pay CUDA context initialization cost here, before any row is timed,
    // rather than have it silently inflate the first GPU measurement.
    { qcs::StateVectorGpu warmup(1); }
#endif

    bool cpu_alive = true;
    bool gpu_alive = true;

    for (int n = min_qubits; n <= max_qubits; ++n) {
        const std::size_t amplitudes = std::size_t{1} << n;
        const double memory_bytes = static_cast<double>(amplitudes) * sizeof(qcs::Complex); // 16 bytes/amplitude

        const qcs::Circuit circuit = build_benchmark_circuit(n, num_layers);
        const std::size_t gate_count = circuit.gate_count();

        double cpu_ms = -1.0;
        std::string cpu_status = "skipped";
        if (cpu_alive) {
            try {
                double sanity = 0.0;
                cpu_ms = time_with_adaptive_repeats([&]() {
                    const qcs::StateVector state = circuit.run();
                    sanity = state.probability(0); // touch the result; prevents dead-code elimination
                });
                cpu_status = "ok";
                if (sanity < -1.0) std::cout << "unreachable\n";
            } catch (const std::bad_alloc&) {
                cpu_status = "oom";
                cpu_alive = false; // larger n will only need more memory; stop trying
            } catch (const std::exception& e) {
                cpu_status = std::string("error:") + e.what();
                cpu_alive = false;
            }
        }

        double gpu_ms = -1.0;
        std::string gpu_status = "skipped";
#ifdef QUANTCS_WITH_CUDA
        if (gpu_alive) {
            const auto [free_bytes, total_bytes] = qcs::gpu_memory_info();
            if (memory_bytes > static_cast<double>(free_bytes) * kGpuMemorySafetyFactor) {
                gpu_status = "oom (needs " + format_mib(memory_bytes) + " MiB, " +
                             format_mib(static_cast<double>(free_bytes)) + " MiB free)";
                gpu_alive = false;
            } else {
                try {
                    double sanity = 0.0;
                    gpu_ms = time_with_adaptive_repeats([&]() {
                        const qcs::StateVectorGpu state = circuit.run_gpu();
                        sanity = state.probability(0);
                    });
                    gpu_status = "ok";
                    if (sanity < -1.0) std::cout << "unreachable\n";
                } catch (const std::exception& e) {
                    gpu_status = std::string("error:") + e.what();
                    gpu_alive = false;
                }
            }
        }
#endif

        const double speedup = (cpu_status == "ok" && gpu_status == "ok" && gpu_ms > 0.0)
                                    ? cpu_ms / gpu_ms
                                    : -1.0;

        std::cout << std::left << std::setw(8) << n
                  << std::setw(14) << format_mib(memory_bytes)
                  << std::setw(14) << (cpu_status == "ok" ? std::to_string(cpu_ms) : cpu_status)
                  << std::setw(14) << (gpu_status == "ok" ? std::to_string(gpu_ms) : gpu_status)
                  << std::setw(10) << (speedup > 0 ? std::to_string(speedup) + "x" : "-")
                  << "\n";

        csv << n << ',' << gate_count << ',' << std::fixed << std::setprecision(0) << memory_bytes
            << ',' << std::setprecision(4) << (memory_bytes / (1024.0 * 1024.0)) << ','
            << std::setprecision(6) << cpu_ms << ',' << csv_field(cpu_status) << ',' << gpu_ms << ','
            << csv_field(gpu_status) << ',' << speedup << '\n';
        csv.flush();

        if (!cpu_alive && !gpu_alive) {
            std::cout << "\nBoth backends have hit their limits; stopping sweep early.\n";
            break;
        }
    }

    std::cout << "\nResults written to " << csv_path << "\n";
    return 0;
}
