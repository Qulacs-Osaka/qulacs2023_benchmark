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

using namespace std;

void test(int, int, vector<double>&);
void dbg(vector<double>);

cudaEvent_t start, stop;
// chrono::system_clock::time_point s_time, e_time;

int main(){
    int qubit_start = 4;
    int qubit_end = 20;
    int repeat = 100;
    cin >> qubit_end >> repeat;
    qubit_end = qubit_end >= qubit_start ? qubit_end : qubit_start;

    vector<double> time_list;
    for(int i=qubit_start;i<=qubit_end;i++){
        test(i, repeat, time_list);
    }
    dbg(time_list);
}

void test(int qubit_num, int repeat, vector<double>& time_list){
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    // s_time = chrono::system_clock::now();
    for(int i=0;i<repeat;i++){
        QuantumStateGpu state(qubit_num);
        state.set_Haar_random_state();
        auto gate = gate::X(0);
        gate->update_quantum_state(&state);
    }
    // e_time = chrono::system_clock::now();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float time = 0;
    // time = chrono::duration_cast<chrono::microseconds>(e_time - s_time).count();
    cudaEventElapsedTime(&time, start, stop); // msで計測
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    time_list.push_back(time / repeat);
}

void dbg(vector<double> time_list){
    for(int i=0;i<time_list.size();i++){
        cout << scientific << setprecision(1) << time_list[i] << " ";
    }
    cout << endl;
}
