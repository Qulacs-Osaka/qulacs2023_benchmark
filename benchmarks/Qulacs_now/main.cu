#include <iostream>
#include <cppsim/state_gpu.hpp>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/utility.hpp>
#include <cppsim/pauli_operator.hpp>
#include <time.h>
#include <vector>
#include <cuda.h>
#include <chrono>
#include <iomanip>

void test(int, int, std::vector<double>&);

cudaEvent_t start, stop;

int main(int argc, char** argv){
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << "<circuit_number> <n_qubits> <n_repeats>" << std::endl;
        return 1;
    }
    int qubit = atoi(argv[1]);
    int repeat = atoi(argv[2]);

    std::vector<double> time_list;
    test(qubit, repeat, time_list);

    std::ofstream ofs("duration.txt");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }
    for(int i=0;i<time_list.size();i++){
        ofs << std::scientific << std::setprecision(10) << time_list[i] << ' ';
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

