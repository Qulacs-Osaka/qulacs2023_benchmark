#include <sycl/sycl.hpp>

#include <cmath>
#include <complex>
#include <iostream>
#include <random>

using UINT = uint64_t;
using Complex = std::complex<double>;

class SingleQubitUpdaterX {
public:
    void operator()(Complex& a0, Complex& a1) const {
        std::swap(a0, a1);
    }
};
class SingleQubitUpdaterY {
public:
    void operator()(Complex& a0, Complex& a1) const {
        Complex tmp = a0;
        a0 = {a1.imag(), -a1.real()};
        a1 = {-tmp.imag(), tmp.real()};
    }
};
class SingleQubitUpdaterZ {
public:
    void operator()(Complex& a0, Complex& a1) const {
        a1 = -a1;
    }
};
class SingleQubitUpdaterH {
public:
    void operator()(Complex& a0, Complex& a1) const {
        constexpr double sqrt1_2 = M_SQRT1_2;
        Complex tmp = a0;
        a0 = (a0 + a1) * sqrt1_2;
        a1 = (tmp - a1) * sqrt1_2;
    }
};
class SingleQubitUpdaterRX {
    double angle;
    double cos_angle_2;
    double sin_angle_2;
    
public:
    SingleQubitUpdaterRX(double angle) : angle{angle}, cos_angle_2{cos(angle / 2)}, sin_angle_2{sin(angle / 2)} {}
    
    void operator()(Complex& a0, Complex& a1) const {
        Complex tmp = a0;
        a0 = {
            a0.real() * cos_angle_2 + a1.imag() * sin_angle_2,
            a0.imag() * cos_angle_2 - a1.real() * sin_angle_2
        };
        a1 = {
            tmp.imag() * sin_angle_2 + a1.real() * cos_angle_2,
            -tmp.real() * sin_angle_2 + a1.imag() * cos_angle_2
        };
    }
};
class SingleQubitUpdaterRY {
    double angle;
    double cos_angle_2;
    double sin_angle_2;
    
public:
    SingleQubitUpdaterRY(double angle) : angle{angle}, cos_angle_2{cos(angle / 2)}, sin_angle_2{sin(angle / 2)} {}
    
    void operator()(Complex& a0, Complex& a1) const {
        Complex tmp = a0;
        a0 = {
            a0.real() * cos_angle_2 - a1.real() * sin_angle_2,
            a0.imag() * cos_angle_2 - a1.imag() * sin_angle_2
        };
        a1 = {
            tmp.real() * sin_angle_2 + a1.real() * cos_angle_2,
            tmp.imag() * sin_angle_2 + a1.imag() * cos_angle_2
        };
    }
};
class SingleQubitUpdaterRZ {
    double angle;
    double cos_angle_2;
    double sin_angle_2;
    
public:
    SingleQubitUpdaterRZ(double angle) : angle{angle}, cos_angle_2{cos(angle / 2)}, sin_angle_2{sin(angle / 2)} {}
    
    void operator()(Complex& a0, Complex& a1) const {
        Complex tmp = a0;
        a0 = {
            a0.real() * cos_angle_2 + a0.real() * sin_angle_2,
            a0.imag() * cos_angle_2 - a0.real() * sin_angle_2
        };
        a1 = {
            a1.real() * cos_angle_2 - a1.real() * sin_angle_2,
            a1.imag() * cos_angle_2 + a1.real() * sin_angle_2
        };
    }
};

template <class SingleQubitUpdater>
void update_with_single_target_gate(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, UINT target, const SingleQubitUpdater& updater) {
    int higher_mask = (1 << (n-1)) - (1 << target);
    int lower_mask = (1 << target) - 1;
    q.submit([&](sycl::handler& h) {
        auto state_acc = state_sycl.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::range<1>(1 << (n - 1)), [=](sycl::id<1> it) {
            int i = ((it[0] & higher_mask) << 1) | (it[0] & lower_mask);
            int j = i | (1 << target);
            updater(state_acc[i], state_acc[j]);
        });
    });
}

template <class SingleQubitUpdater>
void update_with_single_control_single_target_gate(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, UINT control, UINT target, const SingleQubitUpdater& updater) {
    int higher_mask = 0, middle_mask = 0, lower_mask;
    if(target > control) {
        higher_mask = (1 << (n-2)) - (1 << (target-1));
        middle_mask = (1 << (target-1)) - (1 << control);
        lower_mask = (1 << control) - 1;
    } else {
        higher_mask = (1 << (n-2)) - (1 << (control-1));
        middle_mask = (1 << (control-1)) - (1 << target);
        lower_mask = (1 << target) - 1;
    }
    q.submit([&](sycl::handler& h) {
        auto state_acc = state_sycl.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::range<1>(1 << (n - 2)), [=](sycl::id<1> it) {
            int i = ((it[0] & higher_mask) << 2) | ((it[0] & middle_mask) << 1) | (it[0] & lower_mask) | (1 << control);
            int j = i | (1 << target);
            updater(state_acc[i], state_acc[j]);
        });
    });
}

void update_with_x(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, UINT target) {
    update_with_single_target_gate(q, state_sycl, n, target, SingleQubitUpdaterX{});
}
void update_with_y(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, UINT target) {
    update_with_single_target_gate(q, state_sycl, n, target, SingleQubitUpdaterY{});
}
void update_with_z(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, UINT target) {
    update_with_single_target_gate(q, state_sycl, n, target, SingleQubitUpdaterZ{});
}
void update_with_h(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, UINT target) {
    update_with_single_target_gate(q, state_sycl, n, target, SingleQubitUpdaterH{});
}
void update_with_rx(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, UINT target, double angle) {
    update_with_single_target_gate(q, state_sycl, n, target, SingleQubitUpdaterRX{angle});
}
void update_with_ry(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, UINT target, double angle) {
    update_with_single_target_gate(q, state_sycl, n, target, SingleQubitUpdaterRY{angle});
}
void update_with_rz(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, UINT target, double angle) {
    update_with_single_target_gate(q, state_sycl, n, target, SingleQubitUpdaterRZ{angle});
}
void update_with_cnot(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, UINT control, UINT target) {
    update_with_single_control_single_target_gate(q, state_sycl, n, control, target, SingleQubitUpdaterX{});
}

void update_with_dense_matrix(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, const std::vector<UINT>& target_list, sycl::buffer<Complex, 2>& matrix_sycl) {
    int num_target = target_list.size(), num_outer = n - num_target;
    sycl::buffer<UINT, 1> target_list_sycl(target_list);
    sycl::buffer<int, 1> outer_bits_expanded_sycl(sycl::range<1>(1 << num_outer));
    sycl::buffer<int, 1> target_bits_expanded_sycl(sycl::range<1>(1 << num_target));
    q.submit([&](sycl::handler& h) {
        auto target_list_acc = target_list_sycl.get_access<sycl::access::mode::read>(h);
        auto outer_bits_expanded_acc = outer_bits_expanded_sycl.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(1 << num_outer), [=](sycl::id<1> it) {
            int bits = it[0];
            for(UINT target_idx = 0; target_idx < num_target; target_idx++) {
                UINT target = target_list_acc[target_idx];
                int upper_mask = ((1 << (n - target)) - 1) << target;
                int lower_mask = (1 << target) - 1;
                bits = (bits & upper_mask) << 1 | (bits & lower_mask);
            }
            outer_bits_expanded_acc[it[0]] = bits;
        });
    });
    q.submit([&](sycl::handler& h) {
        auto target_list_acc = target_list_sycl.get_access<sycl::access::mode::read>(h);
        auto target_bits_expanded_acc = target_bits_expanded_sycl.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(1 << num_target), [=](sycl::id<1> it) {
            int bits = 0;
            for(UINT target_idx = 0; target_idx < num_target; target_idx++) {
                UINT target = target_list_acc[target_idx];
                bits |= 1 << target;
            }
            target_bits_expanded_acc[it[0]] = bits;
        });
    });
    sycl::buffer<Complex, 1> state_updated_sycl(sycl::range<1>(1 << (num_outer + num_target)));
    q.submit([&](sycl::handler& h) {
        auto outer_bits_expanded_acc = outer_bits_expanded_sycl.get_access<sycl::access::mode::read>(h);
        auto target_bits_expanded_acc = target_bits_expanded_sycl.get_access<sycl::access::mode::read>(h);
        auto matrix_acc = matrix_sycl.get_access<sycl::access::mode::read>(h);
        auto state_acc = state_sycl.get_access<sycl::access::mode::read>(h);
        auto state_updated_acc = state_updated_sycl.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(1 << (num_outer + num_target + num_target)), [=](sycl::id<1> it) {
            int outer_bits = it[0] >> (num_target + num_target);
            int target_bits_1 = it[0] >> (num_target) & ((1 << num_target) - 1);
            int target_bits_2 = it[0] & ((1 << num_target) - 1);
            int source_idx = outer_bits_expanded_acc[outer_bits] | target_bits_expanded_acc[target_bits_2];
            state_updated_acc[outer_bits << num_target | target_bits_1] += matrix_acc[target_bits_1][target_bits_2] * state_acc[source_idx];
        });
    });
    q.submit([&](sycl::handler& h) {
        auto outer_bits_expanded_acc = outer_bits_expanded_sycl.get_access<sycl::access::mode::read>(h);
        auto target_bits_expanded_acc = target_bits_expanded_sycl.get_access<sycl::access::mode::read>(h);
        auto state_updated_acc = state_updated_sycl.get_access<sycl::access::mode::read>(h);
        auto state_acc = state_sycl.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(1 << (num_outer + num_target)), [=](sycl::id<1> it) {
            int outer_bits = it[0] >> num_target;
            int target_bits = it[0] & ((1 << num_target) - 1);
            int dest_idx = outer_bits_expanded_acc[outer_bits] | target_bits_expanded_acc[target_bits];
            state_acc[dest_idx] = state_updated_acc[it[0]];
        });
    });
}

void update_with_dense_matrix_controlled(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, const std::vector<UINT>& control_list, const std::vector<UINT>& control_value, const std::vector<UINT>& target_list, sycl::buffer<Complex, 2>& matrix_sycl) {
    int num_control = control_list.size(), num_target = target_list.size(), num_outer = n - num_control - num_target;
    if(num_control == 0) update_with_dense_matrix(q, state_sycl, n, target_list, matrix_sycl);
    int control_value_mask = 0;
    for(int i = 0; i < num_control; i++) {
        if(control_value[i]) control_value_mask |= control_list[i];
    }
    sycl::buffer<UINT, 1> control_list_sycl(control_list);
    sycl::buffer<UINT, 1> target_list_sycl(target_list);
    sycl::buffer<int, 1> outer_bits_expanded_sycl(sycl::range<1>(1 << num_outer));
    sycl::buffer<int, 1> target_bits_expanded_sycl(sycl::range<1>(1 << num_target));
    q.submit([&](sycl::handler& h) {
        auto target_list_acc = target_list_sycl.get_access<sycl::access::mode::read>(h);
        auto control_list_acc = control_list_sycl.get_access<sycl::access::mode::read>(h);
        auto outer_bits_expanded_acc = outer_bits_expanded_sycl.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(1 << num_outer), [=](sycl::id<1> it) {
            int bits = it[0];
            for(UINT target_idx = 0, control_idx = 0; target_idx < num_target || control_idx < num_control;) {
                UINT target = target_idx == num_target ? n : target_list_acc[target_idx];
                UINT control = control_idx == num_control ? n : control_list_acc[control_idx];
                if(target < control) {
                    int upper_mask = ((1 << (n - target)) - 1) << target;
                    int lower_mask = (1 << target) - 1;
                    bits = (bits & upper_mask) << 1 | (bits & lower_mask);
                    target_idx++;
                } else {
                    int upper_mask = ((1 << (n - control)) - 1) << control;
                    int lower_mask = (1 << control) - 1;
                    bits = (bits & upper_mask) << 1 | (bits & lower_mask);
                    control_idx++;
                }
            }
            outer_bits_expanded_acc[it[0]] = bits;
        });
    });
    q.submit([&](sycl::handler& h) {
        auto target_list_acc = target_list_sycl.get_access<sycl::access::mode::read>(h);
        auto target_bits_expanded_acc = target_bits_expanded_sycl.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(1 << num_target), [=](sycl::id<1> it) {
            int bits = 0;
            for(UINT target_idx = 0; target_idx < num_target; target_idx++) {
                UINT target = target_list_acc[target_idx];
                bits |= 1 << target;
            }
            target_bits_expanded_acc[it[0]] = bits;
        });
    });
    sycl::buffer<Complex, 1> state_updated_sycl(sycl::range<1>(1 << (num_outer + num_target)));
    q.submit([&](sycl::handler& h) {
        auto outer_bits_expanded_acc = outer_bits_expanded_sycl.get_access<sycl::access::mode::read>(h);
        auto target_bits_expanded_acc = target_bits_expanded_sycl.get_access<sycl::access::mode::read>(h);
        auto matrix_acc = matrix_sycl.get_access<sycl::access::mode::read>(h);
        auto state_acc = state_sycl.get_access<sycl::access::mode::read>(h);
        auto state_updated_acc = state_updated_sycl.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<1>(1 << (num_outer + num_target + num_target)), [=](sycl::id<1> it) {
            int outer_bits = it[0] >> (num_target + num_target);
            int target_bits_1 = it[0] >> (num_target) & ((1 << num_target) - 1);
            int target_bits_2 = it[0] & ((1 << num_target) - 1);
            int source_idx = outer_bits_expanded_acc[outer_bits] | target_bits_expanded_acc[target_bits_2] | control_value_mask;
            state_updated_acc[outer_bits << num_target | target_bits_1] += matrix_acc[target_bits_1][target_bits_2] * state_acc[source_idx];
        });
    });
    q.submit([&](sycl::handler& h) {
        auto outer_bits_expanded_acc = outer_bits_expanded_sycl.get_access<sycl::access::mode::read>(h);
        auto target_bits_expanded_acc = target_bits_expanded_sycl.get_access<sycl::access::mode::read>(h);
        auto state_acc = state_sycl.get_access<sycl::access::mode::write>(h);
        auto state_updated_acc = state_updated_sycl.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<1>(1 << (num_outer + num_target)), [=](sycl::id<1> it) {
            int outer_bits = it[0] >> num_target;
            int target_bits = it[0] & ((1 << num_target) - 1);
            int dest_idx = outer_bits_expanded_acc[outer_bits] | target_bits_expanded_acc[target_bits] | control_value_mask;
            state_acc[dest_idx] = state_updated_acc[it[0]];
        });
    });
}

double single_qubit_bench(UINT n_qubits) {
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);

    std::vector<Complex> state_original(1 << n_qubits);
    for(int i = 0; i < 1 << n_qubits; i++) {
        state_original[i] = {normal(mt), normal(mt)};
    }
    std::vector<Complex> state_test(state_original.begin(), state_original.end());
    
    std::vector<UINT> gate(10);
    std::vector<UINT> target(10);
    for(UINT i = 0; i < 10; i++) {
        gate[i] = mt() % 4;
        target[i] = mt() % n_qubits;
    }

    sycl::queue q(sycl::gpu_selector_v);
    auto state_sycl = sycl::buffer(state_test.data(), sycl::range<1>(1 << n_qubits));
    auto st_time =std::chrono::high_resolution_clock::now();
    for(UINT i = 0; i < 10; i++) {
        switch(gate[i]) {
            case 0:
            update_with_x(q, state_sycl, n_qubits, target[i]);
            break;
            case 1:
            update_with_y(q, state_sycl, n_qubits, target[i]);
            break;
            case 2:
            update_with_z(q, state_sycl, n_qubits, target[i]);
            break;
            case 3:
            update_with_h(q, state_sycl, n_qubits, target[i]);
            break;
        }
    }
    q.wait();
    auto ed_time = std::chrono::high_resolution_clock::now();

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
    std::vector<Complex> state_test(state_original.begin(), state_original.end());
    
    std::vector<UINT> gate(10);
    std::vector<UINT> target(10);
    std::vector<double> angle(10);
    for(UINT i = 0; i < 10; i++) {
        gate[i] = gate_gen(mt);
        target[i] = target_gen(mt);
        angle[i] = angle_gen(mt);
    }

    sycl::queue q(sycl::gpu_selector_v);
    auto state_sycl = sycl::buffer(state_test.data(), sycl::range<1>(1 << n_qubits));
    auto st_time =std::chrono::high_resolution_clock::now();
    for(UINT i = 0; i < 10; i++) {
        switch(gate[i]) {
            case 0:
            update_with_rx(q, state_sycl, n_qubits, target[i], angle[i]);
            break;
            case 1:
            update_with_ry(q, state_sycl, n_qubits, target[i], angle[i]);
            break;
            case 2:
            update_with_rz(q, state_sycl, n_qubits, target[i], angle[i]);
            break;
        }
    }
    q.wait();
    auto ed_time = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(ed_time - st_time).count();
}

double cnot_bench(UINT n_qubits) {
    std::mt19937 mt(std::random_device{}());
    std::normal_distribution<> normal(0., 1.);
    std::uniform_int_distribution<> target_gen(0, n_qubits - 1);
    std::uniform_real_distribution<> target_gen_1(0., n_qubits - 2);

    std::vector<Complex> state_original(1 << n_qubits);
    for(int i = 0; i < 1 << n_qubits; i++) {
        state_original[i] = {normal(mt), normal(mt)};
    }
    std::vector<Complex> state_test(state_original.begin(), state_original.end());
    
    std::vector<UINT> target(10);
    std::vector<UINT> control(10);
    for(UINT i = 0; i < 10; i++) {
        target[i] = target_gen(mt);
        control[i] = target_gen_1(mt); if(target[i] == control[i]) control[i] = n_qubits - 1;
    }

    sycl::queue q(sycl::gpu_selector_v);
    auto state_sycl = sycl::buffer(state_test.data(), sycl::range<1>(1 << n_qubits));
    auto st_time =std::chrono::high_resolution_clock::now();
    for(UINT i = 0; i < 10; i++) {
        update_with_cnot(q, state_sycl, n_qubits, control[i], target[i]);
    }
    q.wait();
    auto ed_time = std::chrono::high_resolution_clock::now();

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
    std::vector<Complex> state_test(state_original.begin(), state_original.end());
    
    std::vector<UINT> target(10);
    std::vector<std::vector<Complex>> matrix(10, std::vector<Complex>(4));
    for(UINT i = 0; i < 10; i++) {
        target[i] = target_gen(mt);
        for(int j = 0; j < 4; j++) matrix[i][j] = {normal(mt), normal(mt)};
    }

    sycl::queue q(sycl::gpu_selector_v);
    auto state_sycl = sycl::buffer(state_test.data(), sycl::range<1>(1 << n_qubits));
    auto st_time =std::chrono::high_resolution_clock::now();
    for(UINT i = 0; i < 10; i++) {
        auto matrix_sycl = sycl::buffer(matrix[i].data(), sycl::range<2>(2, 2));
        update_with_dense_matrix(q, state_sycl, n_qubits, {target[i]}, matrix_sycl);
    }
    q.wait();
    auto ed_time = std::chrono::high_resolution_clock::now();

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
    std::vector<Complex> state_test(state_original.begin(), state_original.end());
    
    std::vector<std::vector<UINT>> target_list(10, std::vector<UINT>(2));
    std::vector<std::vector<Complex>> matrix(10, std::vector<Complex>(16));
    for(UINT i = 0; i < 10; i++) {
        target_list[i][0] = target_gen(mt);
        target_list[i][1] = target_gen_1(mt); if(target_list[i][0] == target_list[i][1]) target_list[i][1] = n_qubits - 1;
        for(int j = 0; j < 16; j++) matrix[i][j] = {normal(mt), normal(mt)};
    }

    sycl::queue q(sycl::gpu_selector_v);
    auto state_sycl = sycl::buffer(state_test.data(), sycl::range<1>(1 << n_qubits));
    auto st_time =std::chrono::high_resolution_clock::now();
    for(UINT i = 0; i < 10; i++) {
        auto matrix_sycl = sycl::buffer(matrix[i].data(), sycl::range<2>(4, 4));
        update_with_dense_matrix(q, state_sycl, n_qubits, target_list[i], matrix_sycl);
    }
    q.wait();
    auto ed_time = std::chrono::high_resolution_clock::now();

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
    std::vector<Complex> state_test(state_original.begin(), state_original.end());
    
    std::vector<UINT> target(10);
    std::vector<std::vector<UINT>> control_list(10, std::vector<UINT>(2));
    std::vector<std::vector<UINT>> control_value(10, std::vector<UINT>(2));
    std::vector<std::vector<Complex>> matrix(10, std::vector<Complex>(4));
    for(UINT i = 0; i < 10; i++) {
        target[i] = target_gen(mt);
        control_list[i][0] = target_gen_1(mt); if(target[i] == control_list[i][0]) control_list[i][0] = n_qubits - 1;
        control_list[i][1] = target_gen_2(mt);
        if(target[i] == control_list[i][1]) control_list[i][1] = n_qubits - 2;
        if(control_list[i][0] == control_list[i][1]) {
            if(n_qubits - 1 == target[i]) control_list[i][1] = n_qubits - 2;
            else control_list[i][1] = n_qubits - 1;
        }
        for(int j = 0; j < 2; j++) control_value[i][j] = binary_gen(mt);
        for(int j = 0; j < 4; j++) matrix[i][j] = {normal(mt), normal(mt)};
    }

    sycl::queue q(sycl::gpu_selector_v);
    auto state_sycl = sycl::buffer(state_test.data(), sycl::range<1>(1 << n_qubits));
    auto st_time =std::chrono::high_resolution_clock::now();
    for(UINT i = 0; i < 10; i++) {
        auto matrix_sycl = sycl::buffer(matrix[i].data(), sycl::range<2>(4, 4));
        update_with_dense_matrix_controlled(q, state_sycl, n_qubits, control_list[i], control_value[i], {target[i]}, matrix_sycl);
    }
    q.wait();
    auto ed_time = std::chrono::high_resolution_clock::now();

    return std::chrono::duration_cast<std::chrono::nanoseconds>(ed_time - st_time).count();
}

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <circuit_id> <n_qubits> <n_repeats>" << std::endl;
        return 1;
    }

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
