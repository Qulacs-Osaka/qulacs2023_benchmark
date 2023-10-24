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

#include "util.hpp"

void update_with_Rx(Kokkos::View<CTYPE*> &state, UINT n_qubits, double angle, UINT target) {
    const double angle_half = angle / 2, sin_half = Kokkos::sin(angle_half), cos_half = Kokkos::cos(angle_half);
    const ITYPE lo_mask = (1ULL << target) - 1;
    const ITYPE hi_mask = ~lo_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const ITYPE& it) {
        ITYPE i = ((it & hi_mask) << 1) | (it & lo_mask);
        ITYPE j = i | (1ULL << target);
        CTYPE temp_i = state[i];
        CTYPE temp_j = state[j];
        state[i] = cos_half * temp_i - CTYPE(0, 1) * sin_half * temp_j;
        state[j] = cos_half * temp_j - CTYPE(0, 1) * sin_half * temp_i;
    });
}

#ifdef KOKKOS_ENABLE_CUDA
void update_with_Rx_shuffle(Kokkos::View<CTYPE *> &state, UINT n_qubits, double angle, UINT target)
{
    if (target >= 5) {
        update_with_Rx(state, n_qubits, angle, target);
    } else {
        
        double angle_half = angle / 2, sin_half = std::sin(angle_half), cos_half = std::cos(angle_half);
        Kokkos::parallel_for(1ULL << n_qubits, [=] __device__ (ITYPE i) {

            CTYPE tmp_i = state(i);
            CTYPE tmp_j =
                Kokkos::complex(__shfl_xor_sync(0xffffffff, tmp_i.real(), 1 << target),
                                __shfl_xor_sync(0xffffffff, tmp_i.imag(), 1 << target));

            state(i) = cos_half * tmp_i - CTYPE(0, 1) * sin_half * tmp_j;
        });
    }
}
#endif

void update_with_Rx_unroll(Kokkos::View<CTYPE*> &state, UINT n_qubits, double angle, UINT target) {
    const double sin_half = Kokkos::sin(angle / 2), cos_half = Kokkos::cos(angle / 2);
    if (target == 0) {
        Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const ITYPE& it) {
            ITYPE i = (it << 1);
            CTYPE tmp_i = state[i];
            CTYPE tmp_j = state[i + 1];
            state[i] = cos_half * tmp_i - CTYPE(0, 1) * sin_half * tmp_j;
            state[i + 1] = cos_half * tmp_j - CTYPE(0, 1) * sin_half * tmp_i;
        });       
    } else {
        const ITYPE lo_mask = (1ULL << target) - 1;
        const ITYPE hi_mask = ~lo_mask;
        Kokkos::parallel_for(1ULL << (n_qubits - 2), KOKKOS_LAMBDA(const ITYPE& _it) {
            const ITYPE it = _it << 1;
            ITYPE i = ((it & hi_mask) << 1) | (it & lo_mask);
            ITYPE j = i | (1ULL << target);
            CTYPE tmp_i0 = state[i], tmp_i1 = state[i + 1];
            CTYPE tmp_j0 = state[j], tmp_j1 = state[j + 1];
            state[i] = cos_half * tmp_i0 - CTYPE(0, 1) * sin_half * tmp_j0;
            state[i + 1] = cos_half * tmp_i1 - CTYPE(0, 1) * sin_half * tmp_j1;
            state[j] = cos_half * tmp_j0 - CTYPE(0, 1) * sin_half * tmp_i0;
            state[j + 1] = cos_half * tmp_j1 - CTYPE(0, 1) * sin_half * tmp_i1;
        });        
    }
}

void update_with_x(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, UINT target) {
    const ITYPE lo_mask = (1ULL << target) - 1;
    const ITYPE hi_mask = ~lo_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const ITYPE& it) {  
        ITYPE i = ((it & hi_mask) << 1) | (it & lo_mask);
        ITYPE j = i | (1ULL << target);
        Kokkos::Experimental::swap(state_kokkos[i], state_kokkos[j]);
    });
}

void update_with_x_unroll(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, UINT target) {
    if (target == 0) {
        Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const ITYPE& i) {
            Kokkos::Experimental::swap(state_kokkos[i << 1], state_kokkos[(i << 1) + 1]);
        });     
    } else {
        const ITYPE lo_mask = (1ULL << target) - 1;
        const ITYPE hi_mask = ~lo_mask;
        Kokkos::parallel_for(1ULL << (n_qubits - 2), KOKKOS_LAMBDA(const ITYPE& _it) {
            ITYPE it = _it << 1; 
            ITYPE i = ((it & hi_mask) << 1) | (it & lo_mask);
            ITYPE j = i | (1ULL << target);
            Kokkos::Experimental::swap(state_kokkos[i], state_kokkos[j]);
            Kokkos::Experimental::swap(state_kokkos[i + 1], state_kokkos[j + 1]);
        });        
    }
}
