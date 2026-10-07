#include "quantcs/state_vector.hpp"
#include <cmath>
#include <cstdint>
#include <sstream>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qcs{
    StateVector::StateVector(int num_qubits) : num_qubits_(num_qubits), amplitudes_(std::size_t{1}<<num_qubits){
        if(num_qubits<=0){
            throw std::invalid_argument("num_qubits must be positive");
        }
        reset_to_zero_state();
    }

    void StateVector::reset_to_zero_state(){
        std::fill(amplitudes_.begin(), amplitudes_.end(), Complex(0.0, 0.0));
        amplitudes_[0]=Complex(1.0, 0.0);
    }

    void StateVector::apply_single_qubit_gate(int target, const Mat2x2& gate){
        const std::size_t n=amplitudes_.size();
        const std::int64_t half=static_cast<std::int64_t>(n>>1);
        const std::size_t low_mask=(std::size_t{1}<<target)-1;
        const Complex a=gate[0], b=gate[1], c=gate[2], d=gate[3];

        Complex* amp=amplitudes_.data();

        #pragma omp parallel for schedule(static)
        for(std::int64_t g=0; g<half; ++g){
            const std::size_t gg=static_cast<std::size_t>(g);
            const std::size_t low=gg & low_mask;
            const std::size_t high=gg>>target;
            const std::size_t i0=(high<<(target+1))|low;
            const std::size_t i1=i0|(std::size_t{1}<<target);

            const Complex a0=amp[i0];
            const Complex a1=amp[i1];
            amp[i0]=a*a0+b*a1;
        }
    }

    void StateVector::h(int target){
        static const double inv_sqrt2=1.0/std::sqrt(2.0);
        apply_single_qubit_gate(target, {Complex(inv_sqrt2, 0), Complex(inv_sqrt2, 0), Complex(inv_sqrt2, 0), Complex(-inv_sqrt2, 0)});
    }

    void StateVector::x(int target){
        apply_single_qubit_gate(target, {Complex(0, 0), Complex(1, 0), Complex(1, 0), Complex(0, 0)});
    }

    void StateVector::z(int target){
        apply_single_qubit_gate(target, {Complex(1, 0), Complex(0, 0), Complex(0, 0), Complex(-1, 0)});
    }

    void StateVector::rz(int target, double theta){
        const Complex e_minus=std::polar(1.0,-theta/2.0);
        const Complex e_plus=std::polar(1.0, theta/2.0);
        apply_single_qubit_gate(target, {e_minus, Complex(0, 0), Complex(0, 0), e_plus});
    }

    void StateVector::cnot(int control, int target){
        const std::size_t n=amplitudes_.size();
        const std::int64_t quarter=static_cast<std::int64_t>(n>>2);

        const int lo=std::min(control, target);
        const int hi=std::max(control, target);
        const std::size_t lo_mask=(std::size_t{1}<<lo)-1;
        const std::size_t mid_mask=(std::size_t{1}<<(hi-lo-1))-1;
        const std::size_t control_bit=std::size_t{1}<<control;
        const std::size_t target_bit=std::size_t{1}<<target;

        Complex* amp=amplitudes_.data();
        #pragma omp parallel for schedule(static)
        for(std::int64_t g=0; g<quarter; ++g){
            const std::size_t gg=static_cast<std::size_t>(g);
            const std::size_t low=gg & lo_mask;
            const std::size_t rest=gg>>lo;
            const std::size_t mid=rest & mid_mask;
            const std::size_t top=rest >> (hi-lo-1);

            const std::size_t i00=(top<<(hi+1)) | (mid<<(lo+1)) | low;
            const std::size_t i_ctrl1_tgt0=i00 | control_bit;
            const std::size_t i_ctrl1_tgt1=i00 | control_bit | target_bit;

            std::swap(amp[i_ctrl1_tgt0], amp[i_ctrl1_tgt1]);

        }
    }

    void StateVector::phase_flip(std::size_t index){
        amplitudes_[index]=-amplitudes_[index];
    }

    double StateVector::probability(std::size_t index) const{
        const Complex& c=amplitudes_[index];
        return c.real()*c.real()+c.imag()*c.imag();
    }

    std::vector<double> StateVector::probabilities() const{
        std::vector<double> probs(amplitudes_.size());
        for (std::size_t i=0; i<amplitudes_.size(); ++i){
            probs[i]=probability(i);
        }
        return probs;
    }

    std::string StateVector::to_string(double threshold) const{
        std::ostringstream oss;
        for(std::size_t i=0; i<amplitudes_.size(); ++i){
            if(p<threshold) continue;
            oss<<"|";
            for(int b=num_qubits_-1; b>=0; --b){
                oss<<((i>>b) & 1);
            }
            oss<< ">:" << p << "\n";
        }
        return oss.str();
    }

}