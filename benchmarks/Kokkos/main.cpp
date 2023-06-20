#include <complex>
#include <Kokkos_Core.hpp>
#include <iostream>

using UINT = unsigned int;
using ITYPE = unsigned long long;
using CTYPE = std::complex<double>;

void update_with_x(Kokkos::View<CTYPE*> state_kokkos, UINT n, UINT target) {
    ITYPE range1 = 1ULL << (n - target - 1);
    ITYPE range2 = 1ULL << target;
    Kokkos::parallel_for("update_with_x", range1 * range2, KOKKOS_LAMBDA(const ITYPE& index) {
        ITYPE it1 = index / range2;
        ITYPE it2 = index % range2;
        ITYPE i = (it1 << (target + 1)) | it2;
        ITYPE j = i | (1ULL << target);
        CTYPE tmp = state_kokkos[i];
        state_kokkos[i] = state_kokkos[j];
        state_kokkos[j] = tmp;
    });
    Kokkos::fence();
}

int main() {
    Kokkos::initialize();
    {
        constexpr int n = 4;

        Kokkos::View<CTYPE*> init_state("init_state", 1ULL << n);
        for(int i = 0; i < 1ULL << n; i++) init_state[i] = CTYPE(i, 0);

        Kokkos::View<CTYPE*> state_kokkos(init_state);
        update_with_x(state_kokkos, n, 1);
        Kokkos::deep_copy(init_state, state_kokkos);

        for(int i = 0; i < 1ULL << n; i++) std::cout << ' ' << init_state[i];
        std::cout << std::endl;
    }
    Kokkos::finalize();
}
