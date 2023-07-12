#include <sycl/sycl.hpp>
#include <cmath>
#include <complex>
#include <iostream>

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
            int i = ((it[0] & higher_mask) << 1) | (it[1] & lower_mask);
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
void update_with_x_single_loop(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, UINT target) {
    update_with_single_target_gate_single_loop(q, state_sycl, n, target, SingleQubitUpdaterX{});
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
    int target_mask = 0;
    for(int idx : target_list) target_mask |= 1 << idx;
    sycl::buffer<UINT, 1> target_list_sycl(target_list);
    sycl::buffer<int, 2> state_idx_sycl(sycl::range<2>(1 << num_outer, 1 << num_target));
    sycl::buffer<Complex, 2> state_updated_sycl(sycl::range<2>(1 << num_outer, 1 << num_target));
    q.submit([&](sycl::handler& h) {
        auto target_list_acc = target_list_sycl.get_access<sycl::access::mode::read>(h);
        auto state_idx_acc = state_idx_sycl.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<2>(1 << num_outer, 1 << num_target), [=](sycl::id<2> it) {
            int idx = it[0];
            int target_idx = 0;
            while(target_idx < num_target) {
                UINT target = target_list_acc[target_idx];
                UINT value = it[1] >> target_idx;
                target_idx++;
                int upper_mask = ((1 << (n - target)) - 1) << target;
                int lower_mask = (1 << target) - 1;
                idx = ((idx & upper_mask) << 1) | (value << target) | (idx & lower_mask);
            }
            state_idx_acc[it[0]][it[1]] = idx;
        });
    });
    q.submit([&](sycl::handler& h) {
        auto state_acc = state_sycl.get_access<sycl::access::mode::read>(h);
        auto state_idx_acc = state_idx_sycl.get_access<sycl::access::mode::read>(h);
        auto state_updated_acc = state_updated_sycl.get_access<sycl::access::mode::write>(h);
        auto matrix_acc = matrix_sycl.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<3>(1 << num_outer, 1 << num_target, 1 << num_target), [=](sycl::id<3> it) {
            state_updated_acc[it[0]][it[1]] += matrix_acc[it[1]][it[2]] * state_acc[state_idx_acc[it[0]][it[2]]];
        });
    });
    q.submit([&](sycl::handler& h) {
        auto state_acc = state_sycl.get_access<sycl::access::mode::write>(h);
        auto state_idx_acc = state_idx_sycl.get_access<sycl::access::mode::read>(h);
        auto state_updated_acc = state_updated_sycl.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<2>(1 << num_outer, 1 << num_target), [=](sycl::id<2> it) {
            state_acc[state_idx_acc[it[0]][it[1]]] = state_updated_acc[it[0]][it[1]];
        });
    });
}

void update_with_dense_matrix_controlled(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, const std::vector<UINT>& control_list, const std::vector<UINT>& control_value, const std::vector<UINT>& target_list, sycl::buffer<Complex, 2>& matrix_sycl) {
    int num_control = control_list.size(), num_target = target_list.size(), num_outer = n - num_control - num_target;
    if(num_control == 0) update_with_dense_matrix(q, state_sycl, n, target_list, matrix_sycl);
    int control_mask = 0, control_value_mask = 0, target_mask = 0;
    for(int i = 0; i < num_control; i++) {
        control_mask |= 1 << control_list[i];
        if(control_value[i]) control_value_mask |= 1 << control_list[i];
    }
    for(int idx : target_list) target_mask |= 1 << idx;
    sycl::buffer<UINT, 1> control_list_sycl(control_list);
    sycl::buffer<UINT, 1> control_value_sycl(control_value);
    sycl::buffer<UINT, 1> target_list_sycl(target_list);
    sycl::buffer<int, 2> controlled_state_idx_sycl(sycl::range<2>(1 << num_outer, 1 << num_target));
    sycl::buffer<Complex, 2> controlled_state_updated_sycl(sycl::range<2>(1 << num_outer, 1 << num_target));
    q.submit([&](sycl::handler& h) {
        auto control_list_acc = control_list_sycl.get_access<sycl::access::mode::read>(h);
        auto control_value_acc = control_value_sycl.get_access<sycl::access::mode::read>(h);
        auto target_list_acc = target_list_sycl.get_access<sycl::access::mode::read>(h);
        auto controlled_state_idx_acc = controlled_state_idx_sycl.get_access<sycl::access::mode::write>(h);
        h.parallel_for(sycl::range<2>(1 << num_outer, 1 << num_target), [=](sycl::id<2> it) {
            int idx = it[0];
            int control_idx = 0, target_idx = 0;
            while(control_idx < num_control || target_idx < num_target) {
                if(target_idx == num_target || (control_idx < num_control && control_list_acc[control_idx] < target_list_acc[target_idx])) {
                    UINT control = control_list_acc[control_idx];
                    UINT value = control_value_acc[control_idx];
                    control_idx++;
                    int upper_mask = ((1 << (n - control)) - 1) << control;
                    int lower_mask = (1 << control) - 1;
                    idx = ((idx & upper_mask) << 1) | (value << control) | (idx & lower_mask);
                } else {
                    UINT target = target_list_acc[target_idx];
                    UINT value = it[1] >> target_idx;
                    target_idx++;
                    int upper_mask = ((1 << (n - target)) - 1) << target;
                    int lower_mask = (1 << target) - 1;
                    idx = ((idx & upper_mask) << 1) | (value << target) | (idx & lower_mask);
                }
            }
            controlled_state_idx_acc[it[0]][it[1]] = idx;
        });
    });
    q.submit([&](sycl::handler& h) {
        auto state_acc = state_sycl.get_access<sycl::access::mode::read>(h);
        auto controlled_state_idx_acc = controlled_state_idx_sycl.get_access<sycl::access::mode::read>(h);
        auto controlled_state_updated_acc = controlled_state_updated_sycl.get_access<sycl::access::mode::write>(h);
        auto matrix_acc = matrix_sycl.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<3>(1 << num_outer, 1 << num_target, 1 << num_target), [=](sycl::id<3> it) {
            controlled_state_updated_acc[it[0]][it[1]] += matrix_acc[it[1]][it[2]] * state_acc[controlled_state_idx_acc[it[0]][it[2]]];
        });
    });
    q.submit([&](sycl::handler& h) {
        auto state_acc = state_sycl.get_access<sycl::access::mode::write>(h);
        auto controlled_state_idx_acc = controlled_state_idx_sycl.get_access<sycl::access::mode::read>(h);
        auto controlled_state_updated_acc = controlled_state_updated_sycl.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<2>(1 << num_outer, 1 << num_target), [=](sycl::id<2> it) {
            state_acc[controlled_state_idx_acc[it[0]][it[1]]] = controlled_state_updated_acc[it[0]][it[1]];
        });
    });
}

void x_test() {
    int nqubits = 28;
    std::cout << "X gate\n";
    std::cout << "q = " << nqubits << "\n\n";
    
    for(UINT target = 4; target <= nqubits; target += 5) {
        std::cout << "target = " << target << "\n";

        {
            std::cout << "double :\n";
            std::vector<double> results;
            for(int iter = 0; iter < 10; iter++) {
                std::vector<Complex> state(1 << nqubits);
                for(int i = 0; i < 1 << nqubits; i++) state[i] = i;

                auto start_time = std::chrono::high_resolution_clock::now();
                sycl::queue q(sycl::gpu_selector_v);
                auto state_sycl = sycl::buffer(state.data(), sycl::range<1>(1 << nqubits));

                update_with_x(q, state_sycl, nqubits, 3);

                q.wait();
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                double duration_sec = duration.count() / 1e6;
                results.push_back(duration_sec);
            }
            for(auto x : results) std::cout << x << ' ';
            std::cout << std::endl;
        }
        {
            std::cout << "single :\n";
            std::vector<double> results;
            for(int iter = 0; iter < 10; iter++) {
                std::vector<Complex> state(1 << nqubits);
                for(int i = 0; i < 1 << nqubits; i++) state[i] = i;

                auto start_time = std::chrono::high_resolution_clock::now();
                sycl::queue q(sycl::gpu_selector_v);
                auto state_sycl = sycl::buffer(state.data(), sycl::range<1>(1 << nqubits));

                update_with_x_single_loop(q, state_sycl, nqubits, 3);

                q.wait();
                auto end_time = std::chrono::high_resolution_clock::now();
                auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
                double duration_sec = duration.count() / 1e6;
                results.push_back(duration_sec);
            }
            for(auto x : results) std::cout << x << ' ';
            std::cout << std::endl;
        }
    }
}

int main(int argc, char** argv) {
    x_test();
    /**
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <n_qubits> <n_repeats>" << std::endl;
        return 1;
    }

    const auto n_qubits = std::strtoul(argv[1], nullptr, 10);
    const auto n_repeats = std::strtoul(argv[2], nullptr, 10);
    

    std::vector<unsigned long long> execution_time(n_repeats);
    for(int repeat_itr = 0; repeat_itr < n_repeats; repeat_itr++) {
        auto st_time =std::chrono::system_clock::now();

        sycl::queue q(sycl::gpu_selector_v);
        std::vector<Complex> state(1 << n_qubits);
        for(int i = 0; i < 1 << n_qubits; i++) state[i] = i;
        auto state_sycl = sycl::buffer(state.data(), sycl::range<1>(1 << n_qubits));
        update_with_x(q, state_sycl, n_qubits, 0 % n_qubits);
        update_with_h(q, state_sycl, n_qubits, 1 % n_qubits);
        update_with_rx(q, state_sycl, n_qubits, 2 % n_qubits, M_PI / 4);
        update_with_ry(q, state_sycl, n_qubits, 3 % n_qubits, -M_PI * 5 / 6);
        update_with_rz(q, state_sycl, n_qubits, 4 % n_qubits, M_PI * 2 / 3);
        update_with_cnot(q, state_sycl, n_qubits, 5 % n_qubits, 6 % n_qubits);
        {
            // sqrtX Gate Apply
            std::vector<std::vector<Complex>> matrix = {{.5+.5j, .5-.5j}, {.5-.5j, .5+.5j}};
            std::vector<Complex> matrix_1;
            for(auto& row_elements : matrix) {
                std::copy(row_elements.begin(), row_elements.end(), std::back_inserter(matrix_1));
            }
            sycl::buffer<Complex, 2> matrix_sycl(matrix_1.data(), sycl::range<2>(2, 2));
            update_with_dense_matrix(q, state_sycl, n_qubits, {7 % n_qubits}, matrix_sycl);
        }
        {
            // SWAP Gate Apply
            std::vector<std::vector<Complex>> matrix = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}};
            std::vector<Complex> matrix_1;
            for(auto& row_elements : matrix) {
                std::copy(row_elements.begin(), row_elements.end(), std::back_inserter(matrix_1));
            }
            sycl::buffer<Complex, 2> matrix_sycl(matrix_1.data(), sycl::range<2>(4, 4));
            update_with_dense_matrix(q, state_sycl, n_qubits, {8 % n_qubits, 9 % n_qubits}, matrix_sycl);
        }
        {
            // CRZ(PI/3) Gate Apply        
            std::vector<std::vector<Complex>> matrix = {{std::polar(1., -M_PI/6), 0}, {0, std::polar(1., M_PI/6)}};
            std::vector<Complex> matrix_1;
            for(auto& row_elements : matrix) {
                std::copy(row_elements.begin(), row_elements.end(), std::back_inserter(matrix_1));
            }
            sycl::buffer<Complex, 2> matrix_sycl(matrix_1.data(), sycl::range<2>(2, 2));
            update_with_dense_matrix_controlled(q, state_sycl, n_qubits, {10 % n_qubits}, {1}, {11 % n_qubits}, matrix_sycl);
        }
        q.wait();
        
        auto ed_time = std::chrono::system_clock::now();
        execution_time[repeat_itr] = std::chrono::duration_cast<std::chrono::milliseconds>(ed_time - st_time).count();
    }
    
    std::ofstream ofs("durations.txt");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    for (int i = 0; i < n_repeats; i++) {
        ofs << execution_time[i] << " ";
    }
    ofs << std::endl;

    return 0;
    */
}
