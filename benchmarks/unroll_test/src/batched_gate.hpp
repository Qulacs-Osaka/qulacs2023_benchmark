#pragma once

#include <complex>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <assert.h>
#include <unistd.h>

void update_with_Rx_batched(Kokkos::View<CTYPE **> &state, UINT n, double angle, UINT target)
{
    double sin_half = Kokkos::sin(angle / 2), cos_half = Kokkos::cos(angle / 2);
    const ITYPE lower_mask = (1ULL << target) - 1;
    const ITYPE upper_mask = ~lower_mask;

    Kokkos::parallel_for(
        "RX_gate_batched_simple", Kokkos::TeamPolicy(state.extent(1), 1 << 5),
        KOKKOS_LAMBDA(const TeamHandle &member) {
            int sample = member.league_rank();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 1ULL << (n - 1)), [=](ITYPE it) {
                ITYPE low = it & lower_mask;
                ITYPE high = (it & upper_mask) << 1;    
                ITYPE i = low | high;
                ITYPE j = i | (1ULL << target); 
                CTYPE temp_i = state(i, sample);
                CTYPE temp_j = state(j, sample);
                state(i, sample) = cos_half * temp_i - CTYPE(0, 1) * sin_half * temp_j;
                state(j, sample) = cos_half * temp_j - CTYPE(0, 1) * sin_half * temp_i;
            });
        });
}

#ifdef KOKKOS_ENABLE_CUDA
void update_with_Rx_batched_shuffle(Kokkos::View<CTYPE **> &state, UINT n, double angle, UINT target)
{
    double sin_half = Kokkos::sin(angle / 2), cos_half = Kokkos::cos(angle / 2);

    Kokkos::parallel_for(
        "RX_gate_batched", Kokkos::TeamPolicy(state.extent(1), 1 << 5),
        [=] __device__ (const TeamHandle &member) {
            int sample = member.league_rank();
            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 1ULL << n), [=] __device__ (ITYPE i) {
                CTYPE tmp_i = state(i, sample);
                CTYPE tmp_j =
                    Kokkos::complex(__shfl_xor_sync(0xffffffff, tmp_i.real(), 1 << target),
                                    __shfl_xor_sync(0xffffffff, tmp_i.imag(), 1 << target));
                state(i, sample) = cos_half * tmp_i - CTYPE(0, 1) * sin_half * tmp_j;
            });
        });
}
#endif
