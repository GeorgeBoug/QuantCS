#include "quantcs/grover.hpp"
#include <cmath>
#include <stdexcept>

namespace qcs{
    
    namespace{
        constexpr double kPi=3.14159265358979323846;
    }

    int grover_optimal_iterations(int num_qubits){
        const double n=static_cast<double>(std::size_t{1}<<num_qubits);
        const double theta=std::asin(1.0/std::sqrt(n));
        return static_cast<int>(std::lround(kPi/(4.0*theta)-0.5));
    }

    Circuit build_grover_circuit(int num_qubits, std::size_t marked_state, int num_iterations){
        if(marked_state>=(std::size_t{1}<<num_qubits)){
            throw std::invalid_argument("marked_state out of range for num_qubits");
        }
        
        if(num_iterations<0){
            num_iterations=grover_optimal_iterations(num_qubits);
        }

        Circuit circuit(num_qubits);

        //Uniform superposition |s>
        for(int q=0; q<num_qubits; ++q){
            circuit.h(q);
        }

        for (int iter=0; iter<num_iterations; ++iter){
            circuit.phase_flip(marked_state);
            for(int q=0; q<num_qubits; ++q){
                circuit.h(q);
            }
            for(int q=0; q<num_qubits; ++q){
                circuit.h(q);
            }
        }

        return circuit;
    }

}