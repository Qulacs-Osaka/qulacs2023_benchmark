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

UINT single_qubit_bench(UINT);
UINT single_qubit_rotation_bench(UINT);
UINT cnot_bench(UINT);
UINT single_target_matrix_bench(UINT);
UINT double_target_matrix_bench(UINT);
UINT double_control_matrix_bench(UINT);

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
        UINT t;
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

UINT single_qubit_bench(UINT qubit){
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

    UINT loopcnt = 0;

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

double single_qubit_rotation_bench(UINT qubit){
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

    for(UINT i=0;i<10;i++){
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
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop); // msで計測
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time;
}

double cnot_bench(UINT qubit){
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

    for(UINT i=0;i<10;i++){
        UINT tar = gen(mt);
        UINT ctrl = gen(mt);
        while(tar == ctrl) ctrl = gen(mt);
        auto gateCNOT = gate::CNOT(ctrl, tar);
        gateCNOT->update_quantum_state(&state);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop); // msで計測
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time;
}

double single_target_matrix_bench(UINT qubit){
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, qubit - 1);

    QuantumStateGpu state(qubit);
    std::vector<Complex> state_vector(1ULL << qubit);
    for(int i=0;i<(1<<qubit);i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }

    std::vector<std::vector<UINT>> targets(10, std::vector<UINT>(1));
    std::vector<std::vector<Complex>> matrix(10, std::vector<Complex>(4));
    for(UINT i=0;i<10;i++){
        targets[i][0] = target_gen(mt);
        for(int j=0;j<4;j++){
            matrix[i][j] = {normal(mt), normal(mt)};
        }
    }

    state.load(state_vector);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    std::vector<UINT> target(1);

    for(UINT i=0;i<10;i++){
        ComplexMatrix mat(2, 2);
        target[0] = targets[i][0];
        mat << matrix[i][0], matrix[i][1],
               matrix[i][2], matrix[i][3];
        auto gateMatrix = gate::DenseMatrix(target, mat);
        gateMatrix->update_quantum_state(&state);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop); // msで計測
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time;
}

double double_target_matrix_bench(UINT qubit){
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, qubit - 1);
    std::uniform_int_distribution<> target_gen_1(0, qubit - 2);

    QuantumStateGpu state(qubit);
    std::vector<Complex> state_vector(1ULL << qubit);
    for(int i=0;i<(1<<qubit);i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }

    std::vector<std::vector<UINT>> target_list(10, std::vector<UINT>(2));
    std::vector<std::vector<Complex>> matrix(10, std::vector<Complex>(16));
    for(UINT i=0;i<10;i++){
        target_list[i][0] = target_gen(mt);
        target_list[i][1] = target_gen_1(mt);
        if(target_list[i][0] == target_list[i][1]) target_list[i][1] = qubit-1;
        for(int j=0;j<16;j++){
            matrix[i][j] = {normal(mt), normal(mt)};
        }
    }

    state.load(state_vector);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    std::vector<UINT> target(2);
    for(UINT i=0;i<10;i++){
        target[0] = target_list[i][0];
        target[1] = target_list[i][1];
        ComplexMatrix mat(4, 4);   
        mat << matrix[i][0], matrix[i][1], matrix[i][2], matrix[i][3],
               matrix[i][4], matrix[i][5], matrix[i][6], matrix[i][7],
               matrix[i][8], matrix[i][9], matrix[i][10], matrix[i][11],
               matrix[i][12], matrix[i][13], matrix[i][14], matrix[i][15];
        auto gateMatrix = gate::DenseMatrix(target, mat);
        gateMatrix->update_quantum_state(&state);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop); // msで計測
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time;
}

double double_control_matrix_bench(UINT qubit){
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

    std::vector<UINT> target(10);
    std::vector<std::vector<UINT>> control_list(10, std::vector<UINT>(2));
    std::vector<std::vector<UINT>> control_value(10, std::vector<UINT>(2));
    std::vector<std::vector<Complex>> matrix(10, std::vector<Complex>(4)); 
    for(UINT i=0;i<10;i++){
        target[i] = target_gen(mt);
        control_list[i][0] = target_gen_1(mt); if(target[i] == control_list[i][0]) control_list[i][0] = qubit - 1;
        control_list[i][1] = target_gen_2(mt);
        if(control_list[i][1] == target[i]) control_list[i][1] = qubit-2;
        if(control_list[i][0] == control_list[i][1]) {
            if(qubit - 1 == target[i]) control_list[i][1] = qubit-2;
            else control_list[i][1] = qubit-1;
        }
        for(int j=0;j<2;j++) control_value[i][j] = binary_gen(mt);
        for(int j=0;j<4;j++) matrix[i][j] = {normal(mt), normal(mt)};
    }

    state.load(state_vector);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    std::vector<UINT> target_(1);
    std::vector<UINT> control_(2);
    std::vector<UINT> control_value_(2);
    ComplexMatrix mat(2, 2);

    for(UINT i=0;i<10;i++){
        mat << matrix[i][0], matrix[i][1],
               matrix[i][2], matrix[i][3];
        target_[0] = target[i];
        control_[0] = control_list[i][0];
        control_[1] = control_list[i][1];
        control_value_[0] = control_value[i][0];
        control_value_[1] = control_value[i][1];
        auto gateMatrix = gate::DenseMatrix(target_[0], mat);
        gateMatrix->add_control_qubit(control_list_[0], control_value_[0]);
        gateMatrix->add_control_qubit(control_list_[1], control_value_[1]);
        gateMatrix->update_quantum_state(&state);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    cudaEventElapsedTime(&time, start, stop); // msで計測
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return time;
}


