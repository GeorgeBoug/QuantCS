#include "quantcs/circuit.hpp"
#include <stdexcept>

namespace qf{

    Circuit::Circuit(int num_qubits):num_qubits_(num_qubits){
        if(num_qubits<=0){
            throw std::invalid_argument("invalid num_qubits");
        }
    }

    Circuit& Circuit::h(int target){
        ops_.push_back({GateType::H, target});
        return *this;
    }

    Circuit& Circuit::x(int target){
        ops_.push_back({GateType::X, target});
        return *this;
    }

    Circuit& Circuit::z(int target){
        ops_.push_back({GateType::Z, target});
        return *this;
    }

    Circuit& Circuit::rz(int target, double theta){
        ops_.push_back({GateType::RZ, target, -1, theta});
        return *this;
    }

    Circuit& Circuit::ry(int target, double theta){
        ops_.push_back({GateType::RY, target, -1, theta});
        return *this;
    }

    Circuit& Circuit::cnot(int control, int target){
        ops_.push_back({GateType::CNOT, target, control});
        return *this;
    }

    Circuit& Circuit::phase_flip(std::size_t index){
        GateOp op;
        op.type=GateType::PHASE_FLIP;
        op.basis_index=index;
        ops_.push_back(op);
        return *this;
    }

    StateVector Circuit::run() const{
        StateVector state(num_qubits_);
        for(const GateOp& op : ops_){
            switch(op.type){
                case GateType::H: state.h(op.qubit); break;
                case GateType::X: state.x(op.qubit); break;
                case GateType::Z: state.z(op.qubit); break;
                case GateType::RZ: state.rz(op.qubit, op.theta); break;
                case GateType::RY: state.ry(op.qubit, op.theta); break;
                case GateType::CNOT: state.cnot(op.control, op.qubit); break;
                case GateType::PHASE_FLIP: state.phase_flip(op.basis_index); break;
            }
        }
        return state;
    }

    //GPU
    #ifdef QUANTCS_WITH_CUDA
    StateVectorGpu Circuit::run_gpu() const{
        StateVector state(num_qubits_);
        for(const GateOp& op : ops_){
            switch(op.type){
                case GateType::H: state.h(op.qubit); break;
                case GateType::X: state.x(op.qubit); break;
                case GateType::Z: state.z(op.qubit); break;
                case GateType::RZ: state.rz(op.qubit, op.theta); break;
                case GateType::RY: state.ry(op.qubit, op.theta); break;
                case GateType::CNOT: state.cnot(op.control, op.qubit); break;
                case GateType::PHASE_FLIP: state.phase_flip(op.basis_index); break;
            }
        }
        return state;
    }
    #endif

}