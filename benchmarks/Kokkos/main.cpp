#include <complex>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <iostream>
#include <algorithm>

using UINT = unsigned int;
using ITYPE = unsigned long long;
using CTYPE = std::complex<double>;

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
    const std::complex<double> phase0 = std::exp(std::complex<double>(0, -angle_half)), 
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

int main() {
Kokkos::initialize();
{    

    int n = 28;
    std::vector<int64_t> results;

    for (int qubit = 4; qubit <= n; ++qubit) {

        Kokkos::View<CTYPE*> init_state("init_state", 1ULL << qubit);
        for(int i = 0; i < 1ULL << qubit; i++) init_state[i] = CTYPE(i, 0);

        Kokkos::View<CTYPE*> state(init_state);

        auto start_time = std::chrono::high_resolution_clock::now();

        update_with_SWAP(state, qubit, 1, 3);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);

        results.push_back(duration.count());

        std::cout << duration.count() << std::endl;

    }
    for(auto x : results) std::cout << x << ' ';
    std::cout << std::endl;

}
Kokkos::finalize();
}
