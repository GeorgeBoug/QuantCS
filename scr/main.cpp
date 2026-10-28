#include "quantcs/circuit.hpp"
#include <iostream>

int main(){
    qcs::Circuit circuit(2);
    circuit.h(0).cnot(0, 1);

    qcs::StateVector state=circuit.run();

    std::cout<<"Bell state probabilities:\n" << state.to_string();
    return 0;
}
