#include <iostream>
#include <cppsim/state_gpu.hpp>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/utility.hpp>
#include <cppsim/pauli_operator.hpp>
#include <complex>
#include <time.h>
#include <vector>
#include <chrono>
#include <iomanip>

using Complex = std::complex<double>;
using UINT = uint64_t;

void test(int, int, std::vector<double>&);

double single_qubit_bench(int);
double single_qubit_rotation_bench(int);
double cnot_bench(int);
double single_target_matrix_bench(int);
double double_target_matrix_bench(int);
double double_control_matrix_bench(int);

clock_t start, end;

int main(int argc, char** argv){
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <circuit_id> <n_qubits> <n_repeats>" << std::endl;
        return 1;
    }
    UINT circuit_id = std::strtoul(argv[1], nullptr, 10);
    UINT qubit = std::strtoul(argv[2], nullptr, 10);
    UINT repeat = std::strtoul(argv[3], nullptr, 10);

    std::vector<double> time_list;
    test(qubit, repeat, time_list);

    std::ofstream ofs("duration.txt");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }
    for(int i=0;i<repeat;i++){
        double t;
        switch(circuit_id){
            case 0:
                t = single_qubit_bench(qubit);
                break;
            case 1:
                t = single_qubit_rotation_bench(qubit);
                break;
            case 2:
                t = cnot_bench(qubit);
                break;
            case 3:
                t = single_target_matrix_bench(qubit);
                break;
            case 4:
                t = double_target_matrix_bench(qubit);
                break;
            case 5:
                t = double_control_matrix_bench(qubit);
                break;
        }
        ofs << t << " ";
    }
    ofs << std::endl;
    return 0;
}

double single_qubit_bench(UINT qubit){
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);

    QuantumState state(qubit);
    std::vector<Complex> state_vector(1ULL << qubit);
    for(int i=0;i<1<<qubit;i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }

    std::vector<UINT> gate(10);
    std::vector<UINT> target(10);
    for(UINT i=0;i<10;i++){
        gate[i] = mt() % 4;
        target[i] = mt() % qubit;
    }
    state.load(state_vector);
    start = clock();
    for(UINT i=0;i<10;i++){
        switch(gate[i]){
            case 0:
                auto gateX = gate::X(target[i]);
                gateX->update_quantum_state(&state);
                break;
            case 1:
                auto gateY = gate::Y(target[i]);
                gateY->update_quantum_state(&state);
                break;
            case 2:
                auto gateZ = gate::Z(target[i]);
                gateZ->update_quantum_state(&state);
                break;
            case 3:
                auto gateH = gate::H(target[i]);
                gateH->update_quantum_state(&state);
                break;
        }
    }
    end = clock();
    return (double)(end-start)/CLOCKS_PER_SEC;
}

double single_qubit_rotation_bench(UINT qubit){
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> gate_gen(0, 2);
    std::uniform_int_distribution<> target_gen(0, qubit-1);
    std::uniform_real_distribution<> angle_gen(0., M_PI*2);

    QuantumState state(qubit);
    std::vector<Complex> state_vector(1ULL << qubit);
    for(int i=0;i<10;i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }

    std::vector<UINT> gate(10);
    std::vector<UINT> target(10);
    std::vector<double> angle(10);
    for(UINT i=0;i<10;i++){
        gate[i] = gate_gen(mt);
        target[i] = target_gen(mt);
        angle[i] = angle_gen(mt);
    }

    state.load(state_vector);
    start = clock();
    for(UINT i=0;i<10;i++){
        switch(gate[i]){
            case 0:
                auto gateRX = gate::RX(target[i], angle[i]);
                gateRX->update_quantum_state(&state);
                break;
            case 1:
                auto gateRY = gate::RY(target[i], angle[i]);
                gateRY->update_quantum_state(&state);
                break;
            case 2:
                auto gateRZ = gate::RZ(target[i], angle[i]);
                gateRZ->update_quantum_state(&state);
                break;
        }
    }
    end = clock();
    return (double)(end-start)/CLOCKS_PER_SEC;
}

double cnot_bench(UINT qubit){
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, qubit-1);
    std::uniform_real_distribution<> target_gen_1(0., qubit-2);

    QuantumState state(qubit);
    std::vector<Complex> state_vector(1ULL << qubit);
    for(int i=0;i<1<<qubit;i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }

    std::vector<UINT> target(10);
    std::vector<UINT> control(10);
    for(UINT i=0;i<10;i++){
        target[i] = target_gen(mt);
        control[i] = target_gen_1(mt);
        if(control[i] == target[i]) control[i] = qubit-1;
    }

    state.load(state_vector);
    start = clock();
    for(UINT i=0;i<10;i++){
        auto gateCNOT = gate::CNOT(control[i], target[i]);
        gateCNOT->update_quantum_state(&state);
    }
    end = clock();
    return (double)(end-start)/CLOCKS_PER_SEC;
}

void test(int qubit_num, int repeat, std::vector<double>& time_list){
    for(int i=0;i<repeat;i++){
        QuantumState state(qubit_num);
        state.set_Haar_random_state();
        auto gateX = gate::X(0);
        auto gateH = gate::H(0);
        auto gateCNOT = gate::CNOT(0,1);
        auto gateRX = gate::RX(0,0.5);
        auto gateRZ = gate::RZ(0,0.5);
        auto gateRY = gate::RY(0,1);
        // auto gateMatrix = gate::DenseMatrix(0,SparseComplexMatrix::random(2,2));
        gateX->update_quantum_state(&state);
        gateH->update_quantum_state(&state);
        gateCNOT->update_quantum_state(&state);
        gateRX->update_quantum_state(&state);
        gateRZ->update_quantum_state(&state);
        gateRY->update_quantum_state(&state);
        // gateMatrix->update_quantum_state(&state);
    }
}

