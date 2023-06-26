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
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& it1, const ITYPE &it2) {
        ITYPE i = (it1 << (target + 1)) | it2;
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
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& it1, const ITYPE &it2) {
        ITYPE i = (it1 << (target + 1)) | it2;
        ITYPE j = i | (1ULL << target);
        CTYPE temp_i = state_kokkos[i];
        CTYPE temp_j = state_kokkos[j];
        state_kokkos[i] = inv_sqrt_2 * (temp_i + temp_j);
        state_kokkos[j] = inv_sqrt_2 * (temp_i - temp_j);
    });
    Kokkos::fence();
}

int main() {
    Kokkos::initialize();
    {
        constexpr int n = 4;

        Kokkos::View<CTYPE*> init_state("init_state", 1ULL << n);
        for(int i = 0; i < 1ULL << n; i++) init_state[i] = CTYPE(i, 0);

        for(int i = 0; i < 1ULL << n; i++) std::cout << ' ' << init_state[i];
        std::cout << std::endl;

        Kokkos::View<CTYPE*> state_kokkos(init_state);
        update_with_h(state_kokkos, n, 1);
        Kokkos::Experimental::swap(init_state, state_kokkos);

        for(int i = 0; i < 1ULL << n; i++) std::cout << ' ' << init_state[i];
        std::cout << std::endl;
    } 
    Kokkos::finalize();
}
