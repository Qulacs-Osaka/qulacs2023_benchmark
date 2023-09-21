#define _USE_MATH_DEFINES
#include <iostream>
#include <cppsim/state_gpu.hpp>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/utility.hpp>
#include <cppsim/pauli_operator.hpp>
#include <vector>
#include <cuda.h>
#include <iomanip>

using UINT = unsigned int;
using Complex = std::complex<double>;
using LL = long long;

LL single_qubit_bench(UINT);
LL single_qubit_rotation_bench(UINT);
LL cnot_bench(UINT);
LL single_target_matrix_bench(UINT);
LL double_target_matrix_bench(UINT);
LL double_control_matrix_bench(UINT);

cudaEvent_t start, stop;

int main(int argc, char** argv){
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <circuit_number> <n_qubits> <n_repeats>" << std::endl;
        return 1;
    }
    UINT circuit_id = std::strtoul(argv[1], nullptr, 10);
    UINT qubit = std::strtoul(argv[2], nullptr, 10);
    UINT repeat = std::strtoul(argv[3], nullptr, 10);

    std::ofstream ofs("durations.txt");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    for(int i=0;i<repeat;i++){
        LL t;
        switch(circuit_id){
            case 0:{
                t = single_qubit_bench(qubit);
                break;
            }
            case 1:{
                t = single_qubit_rotation_bench(qubit);
                break;
            }
            case 2:{
                t = cnot_bench(qubit);
                break;
            }
            case 3:{
                t = single_target_matrix_bench(qubit);
                break;
            }
            case 4:{
                t = double_target_matrix_bench(qubit);
                break;
            }
            case 5:{
                t = double_control_matrix_bench(qubit);
                break;
            }
        }
        ofs << t << " ";
    }
    ofs << std::endl;
    return 0;
}

LL single_qubit_bench(UINT qubit){
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, qubit-1), gate_gen(0, 3);

    QuantumStateGpu state(qubit);
    std::vector<Complex> state_vector(1ULL << qubit);
    for(int i=0;i<1<<qubit;i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }
    state.load(state_vector);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    LL loopcnt = 0;

    while(1){
        loopcnt++;
        switch(gate_gen(mt)){
            case 0:{
                auto gateX = gate::X(target_gen(mt));
                gateX->update_quantum_state(&state);
                break;
            }
            case 1:{
                auto gateY = gate::Y(target_gen(mt));
                gateY->update_quantum_state(&state);
                break;
            }
            case 2:{
                auto gateZ = gate::Z(target_gen(mt));
                gateZ->update_quantum_state(&state);
                break;
            }
            case 3:{
                auto gateH = gate::H(target_gen(mt));
                gateH->update_quantum_state(&state);
                break;
            }
        }
        cudaEventRecord(stop); 
        float time = 0;
        cudaEventElapsedTime(&time, start, stop); // msで計測
        if(time > 1000) break;
    }
    cudaEventSynchronize(stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return loopcnt;
}

LL single_qubit_rotation_bench(UINT qubit){
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> gate_gen(0, 2), target_gen(0, qubit-1);
    std::uniform_real_distribution<> angle_gen(0., M_PI*2);

    QuantumStateGpu state(qubit);
    std::vector<Complex> state_vector(1ULL << qubit);
    for(int i=0;i<(1<<qubit);i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }

    state.load(state_vector);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    LL loopcnt = 0;

    while(1){
        loopcnt++;
        switch(gate_gen(mt)){
            case 0:{
                auto gateRX = gate::RX(target_gen(mt), angle_gen(mt));
                gateRX->update_quantum_state(&state);
                break;
            }
            case 1:{
                auto gateRY = gate::RY(target_gen(mt), angle_gen(mt));
                gateRY->update_quantum_state(&state);
                break;
            }
            case 2:{
                auto gateRZ = gate::RZ(target_gen(mt), angle_gen(mt));
                gateRZ->update_quantum_state(&state);
                break;
            }
        }
        cudaEventRecord(stop);
        float time = 0;
        cudaEventElapsedTime(&time, start, stop); // msで計測
        if(time > 1000) break;
    }
    cudaEventSynchronize(stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return loopcnt;
}

LL cnot_bench(UINT qubit){
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> gen(0, qubit-1);

    QuantumStateGpu state(qubit);
    std::vector<Complex> state_vector(1ULL << qubit);
    for(int i=0;i<1<<qubit;i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }

    state.load(state_vector);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    LL loopcnt = 0;

    while(1){
        loopcnt++;
        UINT tar = gen(mt);
        UINT ctrl = gen(mt);
        while(tar == ctrl) ctrl = gen(mt);
        auto gateCNOT = gate::CNOT(ctrl, tar);
        gateCNOT->update_quantum_state(&state);
        cudaEventRecord(stop);
        float time = 0;
        cudaEventElapsedTime(&time, start, stop); // msで計測
        if(time > 1000) break;
    }

    cudaEventSynchronize(stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return loopcnt;
}

LL single_target_matrix_bench(UINT qubit){
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, qubit - 1);

    QuantumStateGpu state(qubit);
    std::vector<Complex> state_vector(1ULL << qubit);
    for(int i=0;i<(1<<qubit);i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }

    std::vector<std::vector<UINT>> targets(10, std::vector<UINT>(1));

    state.load(state_vector);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    std::vector<UINT> target(1);
    std::vector<Complex> matrix(4);

    LL loopcnt = 0;

    while(1){
        loopcnt++;
        ComplexMatrix mat(2, 2);
        target[0] = target_gen(mt);
        for(int i=0;i<4;i++) matrix[i] = {normal(mt), normal(mt)};
        mat << matrix[0], matrix[1], matrix[2], matrix[3];
        auto gateMatrix = gate::DenseMatrix(target, mat);
        gateMatrix->update_quantum_state(&state);
        cudaEventRecord(stop);
        float time = 0;
        cudaEventElapsedTime(&time, start, stop); // msで計測
        if(time > 1000) break;
    }

    cudaEventSynchronize(stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return loopcnt;
}

LL double_target_matrix_bench(UINT qubit){
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, qubit - 1);
    std::uniform_int_distribution<> target_gen_1(0, qubit - 2);

    QuantumStateGpu state(qubit);
    std::vector<Complex> state_vector(1ULL << qubit);
    for(int i=0;i<(1<<qubit);i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }

    state.load(state_vector);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    std::vector<UINT> target(2);
    std::vector<Complex> matrix(16);

    LL loopcnt = 0;

    while(1){
        loopcnt++;
        target[0] = target_gen(mt);
        target[1] = target_gen_1(mt);
        if(target[0] == target[1]) target[1] = qubit-1;
        ComplexMatrix mat(4, 4);
        for(int i=0;i<16;i++) matrix[i] = {normal(mt), normal(mt)};
        mat << matrix[0], matrix[1], matrix[2], matrix[3],
               matrix[4], matrix[5], matrix[6], matrix[7],
               matrix[8], matrix[9], matrix[10], matrix[11],
               matrix[12], matrix[13], matrix[14], matrix[15];
        auto gateMatrix = gate::DenseMatrix(target, mat);
        gateMatrix->update_quantum_state(&state);
        cudaEventRecord(stop);
        float time = 0;
        cudaEventElapsedTime(&time, start, stop); // msで計測
        if(time > 1000) break;
    }

    cudaEventSynchronize(stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return loopcnt;
}

LL double_control_matrix_bench(UINT qubit){
    assert(qubit >= 3);
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, qubit - 1);
    std::uniform_int_distribution<> target_gen_1(0, qubit - 2);
    std::uniform_int_distribution<> target_gen_2(0, qubit - 3);
    std::uniform_int_distribution<> binary_gen(0, 1);

    QuantumStateGpu state(qubit);
    std::vector<Complex> state_vector(1ULL << qubit);
    for(int i=0;i<(1<<qubit);i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }

    state.load(state_vector);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    std::vector<UINT> target(1);
    std::vector<UINT> control(2);
    std::vector<UINT> control_value(2);
    std::vector<Complex> matrix(4);
    LL loopcnt = 0;

    while(1){
        loopcnt++;
        ComplexMatrix mat(2, 2);
        for(int i=0;i<4;i++) matrix[i] = {normal(mt), normal(mt)};
        mat << matrix[0], matrix[1], matrix[2], matrix[3];
        target[0] = target_gen(mt);
        control[0] = target_gen_1(mt); 
        if(target[0] == control[0]) control[0] = qubit-1;
        control[1] = target_gen_2(mt);
        if(control[1] == target[0]) control[1] = qubit-2;
        if(control[0] == control[1]) {
            if(qubit - 1 == target[0]) control[1] = qubit-2;
            else control[1] = qubit-1;
        }
        for(int i=0;i<2;i++) control_value[i] = binary_gen(mt);
        auto gateMatrix = gate::DenseMatrix(target[0], mat);
        gateMatrix->add_control_qubit(control[0], control_value[0]);
        gateMatrix->add_control_qubit(control[1], control_value[1]);
        gateMatrix->update_quantum_state(&state);
        
        cudaEventRecord(stop);
        float time = 0;
        cudaEventElapsedTime(&time, start, stop); // msで計測
        if(time > 1000) break;
    }

    cudaEventSynchronize(stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return loopcnt;
}


