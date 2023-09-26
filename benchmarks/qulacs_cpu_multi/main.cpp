#include <iostream>
#include <cppsim/circuit.hpp>
#include <cppsim/state.hpp>
#include <cppsim/gate_factory.hpp>

using Complex = std::complex<double>;

double single_qubit_bench(UINT n_qubits) {
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);

    std::vector<Complex> state_original(1 << n_qubits);
    for(int i = 0; i < 1 << n_qubits; i++) {
        state_original[i] = {normal(mt), normal(mt)};
    }
    
    QuantumCircuit* circ = new QuantumCircuit(n_qubits);
    for(UINT i = 0; i < 10; i++) {
        UINT gate = mt() % 4;
        UINT target = mt() % n_qubits;
        switch(gate) {
            case 0:
            circ->add_X_gate(target);
            case 1:
            circ->add_Y_gate(target);
            case 2:
            circ->add_Z_gate(target);
            case 3:
            circ->add_H_gate(target);
        }
    }

    QuantumState* state_test = new QuantumState(n_qubits);
    state_test->load(state_original);
    auto st_time =std::chrono::high_resolution_clock::now();
    circ->update_quantum_state(state_test);
    auto ed_time = std::chrono::high_resolution_clock::now();
    
    delete state_test;
    delete circ;

    return std::chrono::duration_cast<std::chrono::nanoseconds>(ed_time - st_time).count();
}

double single_qubit_rotation_bench(UINT n_qubits) {
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> gate_gen(0, 2);
    std::uniform_int_distribution<> target_gen(0, n_qubits - 1);
    std::uniform_real_distribution<> angle_gen(0., M_PI * 2);

    std::vector<Complex> state_original(1 << n_qubits);
    for(int i = 0; i < 1 << n_qubits; i++) {
        state_original[i] = {normal(mt), normal(mt)};
    }
    
    QuantumCircuit* circ = new QuantumCircuit(n_qubits);
    for(UINT i = 0; i < 10; i++) {
        UINT gate = gate_gen(mt);
        UINT target = target_gen(mt);
        UINT angle = angle_gen(mt);
        switch(gate) {
            case 0:
            circ->add_RotX_gate(target, angle);
            case 1:
            circ->add_RotY_gate(target, angle);
            case 2:
            circ->add_RotZ_gate(target, angle);
        }
    }

    QuantumState* state_test = new QuantumState(n_qubits);
    state_test->load(state_original);
    auto st_time =std::chrono::high_resolution_clock::now();
    circ->update_quantum_state(state_test);
    auto ed_time = std::chrono::high_resolution_clock::now();
    
    delete circ;
    delete state_test;

    return std::chrono::duration_cast<std::chrono::nanoseconds>(ed_time - st_time).count();
}

double cnot_bench(UINT n_qubits) {
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, n_qubits - 1);
    std::uniform_int_distribution<> target_gen_1(0., n_qubits - 2);

    std::vector<Complex> state_original(1 << n_qubits);
    for(int i = 0; i < 1 << n_qubits; i++) {
        state_original[i] = {normal(mt), normal(mt)};
    }
    
    QuantumCircuit* circ = new QuantumCircuit(n_qubits);
    for(UINT i = 0; i < 10; i++) {
        UINT target = target_gen(mt);
        UINT control = target_gen_1(mt); if(target == control) control = n_qubits - 1;
        circ->add_CNOT_gate(control, target);
    }

    QuantumState* state_test = new QuantumState(n_qubits);
    state_test->load(state_original);
    auto st_time =std::chrono::high_resolution_clock::now();
    circ->update_quantum_state(state_test);
    auto ed_time = std::chrono::high_resolution_clock::now();
    
    delete circ;
    delete state_test;

    return std::chrono::duration_cast<std::chrono::nanoseconds>(ed_time - st_time).count();
}

double single_target_matrix_bench(UINT n_qubits) {
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, n_qubits - 1);

    std::vector<Complex> state_original(1 << n_qubits);
    for(int i = 0; i < 1 << n_qubits; i++) {
        state_original[i] = {normal(mt), normal(mt)};
    }
    
    QuantumCircuit* circ = new QuantumCircuit(n_qubits);
    for(UINT i = 0; i < 10; i++) {
        UINT target = target_gen(mt);
        ComplexMatrix mat(2, 2);
        for(int j = 0; j < 2; j++) for(int k = 0; k < 2; k++) {
            mat(j, k) = {normal(mt), normal(mt)};
        }
        circ->add_dense_matrix_gate({target}, mat);
    }

    QuantumState* state_test = new QuantumState(n_qubits);
    state_test->load(state_original);
    auto st_time =std::chrono::high_resolution_clock::now();
    circ->update_quantum_state(state_test);
    auto ed_time = std::chrono::high_resolution_clock::now();
    
    delete circ;
    delete state_test;

    return std::chrono::duration_cast<std::chrono::nanoseconds>(ed_time - st_time).count();
}

double double_target_matrix_bench(UINT n_qubits) {
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, n_qubits - 1);
    std::uniform_int_distribution<> target_gen_1(0, n_qubits - 2);

    std::vector<Complex> state_original(1 << n_qubits);
    for(int i = 0; i < 1 << n_qubits; i++) {
        state_original[i] = {normal(mt), normal(mt)};
    }
    
    QuantumCircuit* circ = new QuantumCircuit(n_qubits);
    for(UINT i = 0; i < 10; i++) {
        std::vector<UINT> target_list(2);
        target_list[0] = target_gen(mt);
        target_list[1] = target_gen_1(mt); if(target_list[0] == target_list[1]) target_list[1] = n_qubits - 1;
        ComplexMatrix mat(4, 4);
        for(int j = 0; j < 4; j++) for(int k = 0; k < 4; k++) {
            mat(j, k) = {normal(mt), normal(mt)};
        }
        circ->add_dense_matrix_gate(target_list, mat);
    }

    QuantumState* state_test = new QuantumState(n_qubits);
    state_test->load(state_original);
    auto st_time =std::chrono::high_resolution_clock::now();
    circ->update_quantum_state(state_test);
    auto ed_time = std::chrono::high_resolution_clock::now();
    
    delete circ;
    delete state_test;

    return std::chrono::duration_cast<std::chrono::nanoseconds>(ed_time - st_time).count();
}

double double_control_matrix_bench(UINT n_qubits) {
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, n_qubits - 1);
    std::uniform_int_distribution<> target_gen_1(0, n_qubits - 2);
    std::uniform_int_distribution<> target_gen_2(0, n_qubits - 3);
    std::uniform_int_distribution<> binary_gen(0, 1);

    std::vector<Complex> state_original(1 << n_qubits);
    for(int i = 0; i < 1 << n_qubits; i++) {
        state_original[i] = {normal(mt), normal(mt)};
    }
    
    QuantumCircuit* circ = new QuantumCircuit(n_qubits);
    for(UINT i = 0; i < 10; i++) {
        UINT target = target_gen(mt);
        std::vector<UINT> control_list(2);
        control_list[0] = target_gen_1(mt); if(target == control_list[0]) control_list[0] = n_qubits - 1;
        control_list[1] = target_gen_2(mt);
        if(target == control_list[1]) control_list[1] = n_qubits - 2;
        if(control_list[0] == control_list[1]) {
            if(n_qubits - 1 == target) control_list[1] = n_qubits - 2;
            else control_list[1] = n_qubits - 1;
        }
        std::vector<UINT> control_value(2);
        for(int j = 0; j < 2; j++) control_value[j] = binary_gen(mt);
        ComplexMatrix mat(2, 2);
        for(int j = 0; j < 2; j++) for(int k = 0; k < 2; k++) {
            mat(j, k) = {normal(mt), normal(mt)};
        }
        std::cerr << target << " " << control_list[0] << " " << control_list[1] << std::endl;
        auto matrix_gate = gate::DenseMatrix(target, mat);
        matrix_gate->add_control_qubit(control_list[0], control_value[0]);
        matrix_gate->add_control_qubit(control_list[1], control_value[1]);
        circ->add_gate(matrix_gate);
    }

    QuantumState* state_test = new QuantumState(n_qubits);
    state_test->load(state_original);
    auto st_time =std::chrono::high_resolution_clock::now();
    circ->update_quantum_state(state_test);
    auto ed_time = std::chrono::high_resolution_clock::now();
    
    delete circ;
    delete state_test;

    return std::chrono::duration_cast<std::chrono::nanoseconds>(ed_time - st_time).count();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <circuit_id> <n_qubits> <n_repeats>" << std::endl;
        return 1;
    }

    putenv(const_cast<char*>("QULACS_PARALLEL_NQUBIT_THRESHOLD=64"));

    const auto circuit_id = std::strtoul(argv[1], nullptr, 10);
    const auto n_qubits = std::strtoul(argv[2], nullptr, 10);
    const auto n_repeats = std::strtoul(argv[3], nullptr, 10);

    std::ofstream ofs("durations.txt");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }
    single_qubit_bench(3); // warmup
    for (int i = 0; i < n_repeats; i++) {
        double t;
        switch(circuit_id) {
            case 0:
            t = single_qubit_bench(n_qubits);
            break;
            case 1:
            t = single_qubit_rotation_bench(n_qubits);
            break;
            case 2:
            t = cnot_bench(n_qubits);
            break;
            case 3:
            t = single_target_matrix_bench(n_qubits);
            break;
            case 4:
            t = double_target_matrix_bench(n_qubits);
            break;
            case 5:
            t = double_control_matrix_bench(n_qubits);
            break;
        }
        ofs << t / 1000000. << " ";
    }
    ofs << std::endl;
    return 0;
}
