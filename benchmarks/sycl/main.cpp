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
    SingleQubitUpdaterRX(double angle) : angle{angle}, cos_angle_2{cos(angle) / 2}, sin_angle_2{sin(angle) / 2} {}
    
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
    SingleQubitUpdaterRY(double angle) : angle{angle}, cos_angle_2{cos(angle) / 2}, sin_angle_2{sin(angle) / 2} {}
    
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
    SingleQubitUpdaterRZ(double angle) : angle{angle}, cos_angle_2{cos(angle) / 2}, sin_angle_2{sin(angle) / 2} {}
    
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
    q.submit([&](sycl::handler& h) {
        auto state_acc = state_sycl.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::range<2>(1 << (n - target - 1), 1 << target), [=](sycl::id<2> it) {
            int i = (it[0] << (target + 1)) | it[1];
            int j = i | (1 << target);
            updater(state_acc[i], state_acc[j]);
        });
    });
}

template <class SingleQubitUpdater>
void update_with_single_control_single_target_gate(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, UINT control, UINT target, const SingleQubitUpdater& updater) {
    q.submit([&](sycl::handler& h) {
        auto state_acc = state_sycl.get_access<sycl::access::mode::read_write>(h);
        if(target > control) {
            h.parallel_for(sycl::range<3>(1 << (n-target-1), 1 << (target-control-1), 1 << control), [=](sycl::id<3> it) {
                int i = (it[0] << (target+1)) | (it[1] << (control+1)) | (1 << control) | it[2];
                int j = i | (1 << target);
                updater(state_acc[i], state_acc[j]);
            });
        } else {
            h.parallel_for(sycl::range<3>(1 << (n-control-1), 1 << (control-target-1), 1 << target), [=](sycl::id<3> it) {
                int i = (it[0] << (control+1)) | (1 << control) | (it[1] << (target+1)) | it[2];
                int j = i | (1 << target);
                updater(state_acc[i], state_acc[j]);
            });
        }
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

void update_with_dense_matrix(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, const std::vector<UINT>& control_list, const std::vector<UINT>& control_value, const std::vector<UINT>& target_list, sycl::buffer<Complex, 2>& matrix_sycl) {
    int num_control = control_list.size(), num_target = target_list.size(), num_outer = n - num_control - num_target;
    int control_mask = 0, control_value_mask = 0, target_mask = 0;
    for(int idx : control_list) control_mask |= 1 << idx;
    for(int i = 0; i < num_control; i++) {
        control_mask |= 1 << control_list[i];
        if(control_value[i]) control_value_mask |= 1 << control_list[i];
    }
    for(int idx : target_list) target_mask |= 1 << idx;
    int outer_mask = (~(1 << n) - 1) & ~control_mask & ~target_mask;
    sycl::buffer<int, 2> controlled_state_idx_sycl(sycl::range<2>(1 << num_outer, 1 << num_target));
    sycl::buffer<Complex, 2> controlled_state_updated_sycl(sycl::range<2>(1 << num_outer, 1 << num_target));
    q.submit([&](sycl::handler& h) {
        auto state_acc = state_sycl.get_access<sycl::access::mode::read_write>(h);
        auto controlled_state_idx_acc = controlled_state_idx_sycl.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for(sycl::range<2>(1 << num_outer, 1 << num_target), [=](sycl::id<2> it) {
            int idx = control_value_mask;
            int outer_idx = 0, target_idx = 0;
            for(int i = 0; i < n; i++) {
                if(outer_mask >> i & 1) idx |= (it[0] >> outer_idx++ & 1) << i;
                else if(target_mask >> i & 1) idx |= (it[1] >> target_idx++ & 1) << i;
            }
            controlled_state_idx_acc[it[0]][it[1]] = idx;
        });

        auto controlled_state_updated_acc = controlled_state_updated_sycl.get_access<sycl::access::mode::read_write>(h);
        auto matrix_acc = matrix_sycl.get_access<sycl::access::mode::read>(h);
        h.parallel_for(sycl::range<3>(1 << num_outer, 1 << num_target, 1 << num_target), [=](sycl::id<3> it) {
            controlled_state_updated_acc[it[0]][it[1]] += matrix_acc[it[1]][it[2]] * state_acc[controlled_state_idx_acc[it[0]][it[2]]];
        });

        h.parallel_for(sycl::range<2>(1 << num_outer, 1 << num_target), [=](sycl::id<2> it) {
            state_acc[controlled_state_idx_acc[it[0]][it[1]]] = controlled_state_updated_acc[it[0]][it[1]];
        });
    });
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <n_qubits> <n_repeats>" << std::endl;
        return 1;
    }

    const auto n_qubits = std::strtoul(argv[1], nullptr, 10);
    const auto n_repeats = std::strtoul(argv[2], nullptr, 10);
    
    sycl::queue q(sycl::default_selector_v);

    std::vector<Complex> init_state(1 << n_qubits);
    for(int i = 0; i < 1 << n_qubits; i++) init_state[i] = i;
    auto state_sycl = sycl::buffer(init_state.data(), sycl::range<1>(1 << n_qubits));

    q.wait();
}
