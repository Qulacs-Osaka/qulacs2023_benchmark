#include <complex>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

void update_with_x_single_loop(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, UINT target) {
    const ITYPE low_mask = (1ULL << target) - 1;
    const ITYPE high_mask = ~low_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const ITYPE& it) {
        ITYPE low = it & low_mask;
        ITYPE high = (it & high_mask) << 1;      
        ITYPE i = low | high;
        ITYPE j = i | (1ULL << target);
        Kokkos::Experimental::swap(state_kokkos[i], state_kokkos[j]);
    });
}

void update_with_h_single_loop(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, UINT target) {
    const double inv_sqrt_2 = 1. / sqrt(2.);
    const ITYPE low_mask = (1ULL << target) - 1;
    const ITYPE high_mask = ~low_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const ITYPE& it) {
        ITYPE low = it & low_mask;
        ITYPE high = (it & high_mask) << 1;      
        ITYPE i = low | high;
        ITYPE j = i | (1ULL << target);
        CTYPE temp_i = state_kokkos[i];
        CTYPE temp_j = state_kokkos[j];
        state_kokkos[i] = inv_sqrt_2 * (temp_i + temp_j);
        state_kokkos[j] = inv_sqrt_2 * (temp_i - temp_j);
    });
}

void update_with_Rx_shuffle(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, double angle, UINT target) {
    const double angle_half = angle / 2, sin_half = Kokkos::sin(angle_half), cos_half = Kokkos::cos(angle_half);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n_qubits - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        CTYPE temp_i = state_kokkos[i];
#ifdef __CUDA_ARCH__
        CTYPE temp_j = 
            Kokkos::complex(__shfl_xor_sync(0xffffffff, temp_i.real(), 1 << target), 
                            __shfl_xor_sync(0xffffffff, temp_i.imag(), 1 << target));
#else
        CTYPE temp_j = state_kokkos[j];
#endif
        state_kokkos[i] = cos_half * temp_i - CTYPE(0, 1) * sin_half * temp_j;
        state_kokkos[j] = cos_half * temp_j - CTYPE(0, 1) * sin_half * temp_i;
    });
}

void update_with_RX_batched_shuffle(Kokkos::View<CTYPE **> state, UINT n, double angle, UINT target)
{
    double angle_half = angle / 2, sin_half = std::sin(angle_half), cos_half = std::cos(angle_half);

    Kokkos::parallel_for(
        "RX_gate_batched", Kokkos::TeamPolicy(state.extent(1), 1 << 6),
        KOKKOS_LAMBDA(const TeamHandle &member) {
            int sample = member.league_rank();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 1ULL << n), [=](ITYPE i) {
                ITYPE j = i ^ (1ULL << target);

                CTYPE tmp_i = state(i, sample);
#ifdef __CUDA_ARCH__
                CTYPE tmp_j =
                    Kokkos::complex(__shfl_xor_sync(0xffffffff, tmp_i.real(), 1 << target),
                                    __shfl_xor_sync(0xffffffff, tmp_i.imag(), 1 << target));
#else
                CTYPE tmp_j = state(j, sample);
#endif

                if (j > i) {
                    state(i, sample) = cos_half * tmp_i - CTYPE(0, 1) * sin_half * tmp_j;
                } else {
                    state(i, sample) = CTYPE(0, 1) * sin_half * tmp_i - cos_half * tmp_j;
                }
            });
        });
}

void update_with_Rx_single_loop(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, double angle, UINT target) {
    const double angle_half = angle / 2, sin_half = Kokkos::sin(angle_half), cos_half = Kokkos::cos(angle_half);
    const ITYPE lower_mask = (1ULL << target) - 1;
    const ITYPE upper_mask = ~lower_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const ITYPE& it) {
        ITYPE low = it & lower_mask;
        ITYPE high = (it & upper_mask) << 1;    
        ITYPE i = low | high;
        ITYPE j = i | (1ULL << target); 
        CTYPE temp_i = state_kokkos[i];
        CTYPE temp_j = state_kokkos[j];
        state_kokkos[i] = cos_half * temp_i - CTYPE(0, 1) * sin_half * temp_j;
        state_kokkos[j] = cos_half * temp_j - CTYPE(0, 1) * sin_half * temp_i;
    });
}

void update_with_Ry_single_loop(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, double angle, UINT target) {
    const double angle_half = angle / 2, sin_half = Kokkos::sin(angle_half), cos_half = Kokkos::cos(angle_half);
    const ITYPE low_mask = (1ULL << target) - 1;
    const ITYPE high_mask = ~low_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const ITYPE& it) {
        ITYPE low = it & low_mask;
        ITYPE high = (it & high_mask) << 1;    
        ITYPE i = low | high;
        ITYPE j = i | (1ULL << target); 
        CTYPE temp_i = state_kokkos[i];
        CTYPE temp_j = state_kokkos[j];
        state_kokkos[i] = cos_half * temp_i + sin_half * temp_j;
        state_kokkos[j] = cos_half * temp_j - sin_half * temp_i;
    });
}

void update_with_Rz_single_loop(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, double angle, UINT target) {
    const double angle_half = angle / 2;
    const CTYPE phase0 = Kokkos::exp(CTYPE(0, -angle_half)), 
                phase1 = Kokkos::exp(CTYPE(0, angle_half));
    const ITYPE low_mask = (1ULL << target) - 1;
    const ITYPE high_mask = ~low_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const ITYPE& it) {
        ITYPE low = it & low_mask;
        ITYPE high = (it & high_mask) << 1;    
        ITYPE i = low | high;
        ITYPE j = i | (1ULL << target); 
        state_kokkos[i] *= phase0;
        state_kokkos[j] *= phase1;
    });
}

void update_with_CNOT_single_loop(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, UINT control, UINT target) {
    ITYPE ub = Kokkos::max(target, control);
    ITYPE lb = Kokkos::min(target, control);
    const ITYPE lower_mask = (1ULL << lb) - 1;
    const ITYPE middle_mask = ((1ULL << (ub - lb - 1)) - 1) << lb;
    const ITYPE upper_mask = ~(lower_mask | middle_mask);
    Kokkos::parallel_for(1ULL << (n_qubits - 2), KOKKOS_LAMBDA(const ITYPE &it) {
        ITYPE lower_idx = it & lower_mask;
        ITYPE middle_idx = (it & middle_mask) << 1;
        ITYPE upper_idx = (it & upper_mask) << 2;
        ITYPE i = upper_idx | middle_idx | lower_idx | (1ULL << control);
        Kokkos::Experimental::swap(state_kokkos[i], state_kokkos[i | (1ULL << target)]);
    });
}


