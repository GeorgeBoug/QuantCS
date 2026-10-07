#include "QuantCS/circuit.hpp"
#include <iostream>

int main(){
    qf::Circuit circuit(2);
    circuit.h(0).cnot(0, 1);

    qf::StateVector state=circuit.run();

    std::cout<<"Bell state probabilities:\n" << state.to_string();
    return 0;
}