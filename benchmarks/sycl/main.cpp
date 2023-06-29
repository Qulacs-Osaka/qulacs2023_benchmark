#include <sycl/sycl.hpp>
#include <cmath>
#include <complex>
#include <iostream>

using UINT = uint64_t;
using Complex = std::complex<double>;

class SingleQubitUpdaterX {
public:
    void operator()(Complex& a0, Complex& a1) {
        std::swap(a0, a1);
    }
};
class SingleQubitUpdaterY {
public:
    void operator()(Complex& a0, Complex& a1) {
        Complex tmp = a0;
        a0 = {a1.imag(), -a1.real()};
        a1 = {-tmp.imag(), tmp.real()};
    }
};
class SingleQubitUpdaterZ {
public:
    void operator()(Complex& a0, Complex& a1) {
        a1 = -a1;
    }
};
class SingleQubitUpdaterH {
public:
    void operator()(Complex& a0, Complex& a1) {
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
    
    void operator()(Complex& a0, Complex& a1) {
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
    
    void operator()(Complex& a0, Complex& a1) {
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
    
    void operator()(Complex& a0, Complex& a1) {
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
        h.parallel_for<class nstream>(sycl::range<2>(1 << (n - target - 1), 1 << target), [=](sycl::id<2> it) {
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
            h.parallel_for<class nstream>(sycl::range<3>(1 << (n-target-1), 1 << (target-control-1), 1 << control), [=](sycl::id<3> it) {
                int i = (it[0] << (target+1)) | (it[1] << (control+1)) | (1 << control) | it[2];
                int j = i | (1 << target);
                updater(state_acc[i], state_acc[j]);
            });
        } else {
            h.parallel_for<class nstream>(sycl::range<3>(1 << (n-control-1), 1 << (control-target-1), 1 << target), [=](sycl::id<3> it) {
                int i = (it[0] << (control+1)) (1 << control) | (it[1] << (target+1)) | it[2];
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

// 注意: ソートされたtargetに対するmatrixの並びであることを前提
void update_with_dense_matrix(sycl::queue& q, sycl::buffer<Complex, 1>& state_sycl, UINT n, const std::vector<UINT>& control_list, const std::vector<UINT>& control_value, const std::vector<UINT>& target_list, const std::vector<std::vector<Complex>>& matrix) {
    sycl::buffer<Complex, 1> new_state_sycl(sycl::range<1>(1 << n));
    int num_control = control_list.size(), num_target = target_list.size();
    int control_mask = 0, target_mask = 0;
    for(int idx : control_list) control_mask |= 1 << idx;
    for(int idx : target_list) target_mask |= 1 << idx;
    q.submit([&](sycl::handler& h) {
        auto state_acc = state_sycl.get_access<sycl::access::mode::read_write>(h);
        auto new_state_acc = new_state_sycl.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for<class nstream>(sycl::range<3>(1 << (n-num_control-num_target), 1 << num_target, 1 << num_target), [=](sycl::id<3> it) {
            int iter_raw = 0, iter_col = 0;
            int outer_idx = 0, target_idx = 0;
            for(int i = 0; i < n; i++) {
                if(control_mask >> i & 1) {
                    iter_raw |= 1 << i;
                    iter_col |= 1 << i;
                } else if(target_mask >> i & 1) {
                    iter_raw |= (it[1] >> target_idx & 1) << i;
                    iter_col |= (it[2] >> target_idx & 1) << i;
                    ++target_idx;
                } else {
                    iter_raw |= (it[0] >> outer_idx & 1) << i;
                    iter_col |= (it[0] >> outer_idx & 1) << i;
                    ++outer_idx;
                }
            }
            new_state_acc[iter_raw] += matrix[iter_raw][iter_col] * state_acc[iter_col];
        });
        h.parallel_for<class nstream>(sycl::range<1>(1 << n), [=](sycl::id<1> it) {
            state_acc[it[0]] = new_state_acc[it[0]];
        });
    });
}

int main() {
    constexpr int n = 4;
    
    sycl::queue q(sycl::default_selector_v);

    std::vector<Complex> init_state(1 << n);
    for(int i = 0; i < 1 << n; i++) init_state[i] = i;
    auto state_sycl = sycl::buffer(init_state.data(), sycl::range(1 << n));

    update_with_x(q, state_sycl, n, 1);

    q.wait();
    for(int i = 0; i < 1 << n; i++) std::cout << ' ' << init_state[i];
    std::cout << std::endl;
}
