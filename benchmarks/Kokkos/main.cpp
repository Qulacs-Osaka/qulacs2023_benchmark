#include <complex>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <iostream>
#include <algorithm>

#ifdef KOKKOS_ENABLE_CUDA
#include <thrust/complex.h>
#endif


using UINT = unsigned int;
using ITYPE = unsigned long long;
#ifdef KOKKOS_ENABLE_CUDA
    using CTYPE = thrust::complex<double>;
#else
    using CTYPE = std::complex<double>;
#endif

void update_with_x_double_loop(Kokkos::View<CTYPE*> &state_kokkos, UINT n, UINT target) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        Kokkos::Experimental::swap(state_kokkos[i], state_kokkos[j]);
    });
    Kokkos::fence();
}

void update_with_x_single_loop(Kokkos::View<CTYPE*> &state_kokkos, UINT n, UINT target) {
    const ITYPE low_mask = (1ULL << target) - 1;
    const ITYPE high_mask = ~low_mask;
    Kokkos::parallel_for(1ULL << (n - 1), KOKKOS_LAMBDA(const ITYPE& it) {
        ITYPE low = it & low_mask;
        ITYPE high = (it & high_mask) << 1;      
        ITYPE i = low | high;
        ITYPE j = i | (1ULL << target);
        Kokkos::Experimental::swap(state_kokkos[i], state_kokkos[j]);
    });
    Kokkos::fence();
}

void update_with_h(Kokkos::View<CTYPE*> &state_kokkos, UINT n, UINT target) {
    const double inv_sqrt_2 = 1. / sqrt(2.);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        CTYPE temp_i = state_kokkos[i];
        CTYPE temp_j = state_kokkos[j];
        state_kokkos[i] = inv_sqrt_2 * (temp_i + temp_j);
        state_kokkos[j] = inv_sqrt_2 * (temp_i - temp_j);
    });
    Kokkos::fence();
}

void update_with_Rx(Kokkos::View<CTYPE*> &state_kokkos, UINT n, double angle, UINT target) {
    const double angle_half = angle / 2, sin_half = std::sin(angle_half), cos_half = std::cos(angle_half);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        CTYPE temp_i = state_kokkos[i];
        CTYPE temp_j = state_kokkos[j];
        state_kokkos[i] = cos_half * temp_i - CTYPE(0, 1) * sin_half * temp_j;
        state_kokkos[j] = cos_half * temp_j - CTYPE(0, 1) * sin_half * temp_i;
    });
    Kokkos::fence();
}

void update_with_Ry(Kokkos::View<CTYPE*> &state_kokkos, UINT n, double angle, UINT target) {
    const double angle_half = angle / 2, sin_half = std::sin(angle_half), cos_half = std::cos(angle_half);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        CTYPE temp_i = state_kokkos[i];
        CTYPE temp_j = state_kokkos[j];
        state_kokkos[i] = cos_half * temp_i + sin_half * temp_j;
        state_kokkos[j] = cos_half * temp_j - sin_half * temp_i;
    });
    Kokkos::fence();
}

void update_with_Rz(Kokkos::View<CTYPE*> &state_kokkos, UINT n, double angle, UINT target) {
    const double angle_half = angle / 2;
    const CTYPE phase0 = std::exp(std::complex<double>(0, -angle_half)), 
                               phase1 = std::exp(std::complex<double>(0, angle_half));
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        state_kokkos[i] *= phase0;
        state_kokkos[j] *= phase1;
    });
    Kokkos::fence();
}

void update_with_SWAP(Kokkos::View<CTYPE*> state_kokkos, UINT n, UINT target0, UINT target1) {
    if (target0 > target1) Kokkos::Experimental::swap(target0, target1);
    const ITYPE mask0 = 1ULL << target0;
    const ITYPE mask1 = 1ULL << target1;
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {1ULL << (n - target1 - 1), 1ULL << (target1 - target0 - 1), 1ULL << target0});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE &upper_bit_it, const ITYPE &middle_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target1 + 1)) | (middle_bit_it << (target0 + 1)) | lower_bit_it;
        Kokkos::Experimental::swap(state_kokkos[i | mask0], state_kokkos[i | mask1]);
    });
    Kokkos::fence();
}

void update_with_CNOT(Kokkos::View<CTYPE*> &state_kokkos, UINT n, UINT control, UINT target) {
    const ITYPE mask_control = 1ULL << control;
    const ITYPE mask_target = 1ULL << target;
    ITYPE ub = std::max(target, control);
    ITYPE lb = std::min(target, control);
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {1ULL << (n - ub - 1), 1ULL << (ub - lb - 1), 1ULL << lb});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE &upper_bit_it, const ITYPE &middle_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (ub + 1)) | (middle_bit_it << (lb + 1)) | lower_bit_it | mask_control;
        Kokkos::Experimental::swap(state_kokkos[i], state_kokkos[i | mask_target]);
    });
    Kokkos::fence();
}

void update_with_dense_matrix(Kokkos::View<CTYPE*> &state_kokkos, UINT n, const Kokkos::View<UINT*>& control_list, const Kokkos::View<UINT*>& control_value, const Kokkos::View<UINT*>& target_list, const Kokkos::View<CTYPE**>& matrix) {
    Kokkos::View<CTYPE*> new_state_kokkos("new_state_kokkos", 1ULL << n);
    int num_control = control_list.size(), num_target = target_list.size();
    int control_mask = 0, target_mask = 0;
    for(int i = 0; i < (int)control_list.size(); ++i) control_mask |= 1 << control_list[i];
    for(int i = 0; i < (int)target_list.size(); ++i) target_mask |= 1 << target_list[i];
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {1ULL << (n - num_control - num_target), 1ULL << num_target, 1ULL << num_target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE &outer_bit_it, const ITYPE &target_bit_it1, const ITYPE &target_bit_it2) {
        ITYPE iter_raw = 0, iter_col = 0;
        int outer_idx = 0, control_idx = 0, target_idx = 0;
        for(int i = 0; i < n; i++) {
            if(control_mask >> i & 1) {
                iter_raw |= control_value[control_idx] << i;
                iter_col |= control_value[control_idx] << i;
                ++control_idx;
            } else if(target_mask >> i & 1) {
                iter_raw |= (target_bit_it1 >> target_idx & 1) << i;
                iter_col |= (target_bit_it2 >> target_idx & 1) << i;
                ++target_idx;
            } else {
                iter_raw |= (outer_bit_it >> outer_idx & 1) << i;
                iter_col |= (outer_bit_it >> outer_idx & 1) << i;
                ++outer_idx;
            }
        }
        Kokkos::atomic_add(&new_state_kokkos(iter_raw), matrix(iter_raw, iter_col) * state_kokkos(iter_col));
    });
    Kokkos::fence();
    Kokkos::deep_copy(state_kokkos, new_state_kokkos);
}

int main() {
Kokkos::initialize();
{    
    int n = 28;
    std::vector<double> results;  // change to double to store seconds

    for (int qubit = 4; qubit <= n; ++qubit) {
        auto start_time = std::chrono::high_resolution_clock::now();

        Kokkos::View<CTYPE*> init_state("init_state", 1ULL << qubit);
        Kokkos::parallel_for(1ULL << qubit, KOKKOS_LAMBDA(int i) {
            init_state(i) = CTYPE(i, 0);
        });

        Kokkos::View<CTYPE*> state(init_state);

        update_with_x_single_loop(state, qubit, 3);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        double duration_sec = duration.count() / 1e6;

        results.push_back(duration_sec);
    }
    for(auto x : results) std::cout << x << ' ';
    std::cout << std::endl;

}
Kokkos::finalize();
}
