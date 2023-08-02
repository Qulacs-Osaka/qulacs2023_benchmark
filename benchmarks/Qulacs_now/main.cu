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

using Complex = std::complex<double>;

void test(int, int, std::vector<double>&);

double single_qubit_bench(UINT);
double single_qubit_rotation_bench(UINT);
double cnot_bench(UINT);
double single_target_matrix_bench(UINT);
double double_target_matrix_bench(UINT);
double double_control_matrix_bench(UINT);

cudaEvent_t start, stop;

int main(int argc, char** argv){
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <circuit_number> <n_qubits> <n_repeats>" << std::endl;
        return 1;
    }
    UINT circuit_id = std::strtoul(argv[1], nullptr, 10);
    UINT qubit = std::strtoul(argv[2], nullptr, 10);
    UINT repeat = std::strtoul(argv[3], nullptr, 10);

    std::ofstream ofs("duration.txt");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    for(int i=0;i<repeat;i++){
        double t;
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
            // case 4:
            //     t = double_target_matrix_bench(qubit);
            //     break;
            // case 5:
            //     t = double_control_matrix_bench(qubit);
            //     break;
        }
        ofs << t << " ";
    }
    ofs << std::endl;
    return 0;
}

void test(int qubit_num, int repeat, std::vector<double>& time_list){
    for(int i=0;i<repeat;i++){
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        QuantumStateGpu state(qubit_num);
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
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float time = 0;
        cudaEventElapsedTime(&time, start, stop); // msで計測
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        time_list.push_back(time);
    }
}

double single_qubit_bench(UINT qubit){
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);

    QuantumStateGpu state(qubit);
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

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(UINT i=0;i<10;i++){
        switch(gate[i]){
            case 0:{
                auto gateX = gate::X(target[i]);
                gateX->update_quantum_state(&state);
                break;
            }
            case 1:{
                auto gateY = gate::Y(target[i]);
                gateY->update_quantum_state(&state);
                break;
            }
            case 2:{
                auto gateZ = gate::Z(target[i]);
                gateZ->update_quantum_state(&state);
                break;
            }
            case 3:{
                auto gateH = gate::H(target[i]);
                gateH->update_quantum_state(&state);
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

double single_qubit_rotation_bench(UINT qubit){
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> gate_gen(0, 2);
    std::uniform_int_distribution<> target_gen(0, qubit-1);
    std::uniform_real_distribution<> angle_gen(0., M_PI*2);

    QuantumStateGpu state(qubit);
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

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(UINT i=0;i<10;i++){
        switch(gate[i]){
            case 0:{
                auto gateRX = gate::RX(target[i], angle[i]);
                gateRX->update_quantum_state(&state);
                break;
            }
            case 1:{
                auto gateRY = gate::RY(target[i], angle[i]);
                gateRY->update_quantum_state(&state);
                break;
            }
            case 2:{
                auto gateRZ = gate::RZ(target[i], angle[i]);
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
    std::uniform_int_distribution<> target_gen(0, qubit-1);
    std::uniform_real_distribution<> target_gen_1(0., qubit-2);

    QuantumStateGpu state(qubit);
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

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(UINT i=0;i<10;i++){
        auto gateCNOT = gate::CNOT(control[i], target[i]);
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
    for(int i=0;i<10;i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }

    std::vector<UINT> target(10);
    std::vector<std::vector<Complex>> matrix(10, std::vector<Complex>(4));
    for(UINT i=0;i<10;i++){
        target[i] = target_gen(mt);
        for(int j=0;j<4;j++){
            matrix[i][j] = {normal(mt), normal(mt)};
        }
    }

    state.load(state_vector);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for(UINT i=0;i<10;i++){
        ComplexMatrix mat(2, 2);
        mat << matrix[i][0], matrix[i][1],
               matrix[i][2], matrix[i][3];
        auto gateMatrix = gate::DenseMatrix(target[i], mat);
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
    for(int i=0;i<10;i++){
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

    for(UINT i=0;i<10;i++){
        ComplexMatrix mat(4, 4);
        mat << matrix[i][0], matrix[i][1], matrix[i][2], matrix[i][3],
               matrix[i][4], matrix[i][5], matrix[i][6], matrix[i][7],
               matrix[i][8], matrix[i][9], matrix[i][10], matrix[i][11],
               matrix[i][12], matrix[i][13], matrix[i][14], matrix[i][15];
        auto gateMatrix = gate::DenseMatrix(target_list[i], mat);
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
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, qubit - 1);
    std::uniform_int_distribution<> target_gen_1(0, qubit - 2);
    std::uniform_int_distribution<> target_gen_2(0, qubit - 3);
    std::uniform_int_distribution<> binary_gen(0, 1);

    QuantumStateGpu state(qubit);
    std::vector<Complex> state_vector(1ULL << qubit);
    for(int i=0;i<10;i++){
        state_vector[i] = {normal(mt), normal(mt)};
    }

    std::vector<UINT> target(10);
    std::vector<std::vector<UINT>> control_list(10, std::vector<UINT>(2));
    std::vector<std::vector<UINT>> control_value(10, std::vector<UINT>(2));
    std::vector<std::vector<Complex>> matrix(10, std::vector<Complex>(4)); 
    for(UINT i=0;i<10;i++){
        target[i] = target_gen(mt);
        control_list[i][0] = target_gen_1(mt);
        control_list[i][1] = target_gen_2(mt);
        if(control_list[i][0] == target[i]) control_list[i][0] = qubit-1;
        if(control_list[i][1] == target[i]) control_list[i][1] = qubit-2;
        if(control_list[i][0] == control_list[i][1]) control_list[i][1] = qubit-1;
        for(int j=0;j<2;j++) control_value[i][j] = binary_gen(mt);
        for(int j=0;j<4;j++) matrix[i][j] = {normal(mt), normal(mt)};
    }

    state.load(state_vector);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for(UINT i=0;i<10;i++){
        ComplexMatrix mat(2, 2);
        mat << matrix[i][0], matrix[i][1],
               matrix[i][2], matrix[i][3];
        auto gateMatrix = gate::DenseMatrix(control_list[i], control_value[i], target[i], mat);
        gateMatrix->update_quantum_state(&state);
    }
}


