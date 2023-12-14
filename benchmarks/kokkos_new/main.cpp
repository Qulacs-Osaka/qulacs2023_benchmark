#include <complex>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>
#include <assert.h>

using UINT = unsigned int;
using ITYPE = unsigned long long;
using CTYPE = Kokkos::complex<double>;
using TeamHandle = Kokkos::TeamPolicy<>::member_type;

const int warp_size = 32;

void apply_x_shfl(Kokkos::View<CTYPE*> &state, UINT n_qubits, UINT target) {
    assert((1ULL << target) < warp_size);
    Kokkos::parallel_for(1ULL << n_qubits, [=] __device__ (const ITYPE& i) {
        state(i) = CTYPE{
            __shfl_xor_sync(0xffffffff, state(i).real(), 1ULL << target),
            __shfl_xor_sync(0xffffffff, state(i).imag(), 1ULL << target)
        };
    });
}

void apply_x_nrml(Kokkos::View<CTYPE*> &state, UINT n_qubits, UINT target) {
    const ITYPE low_mask = (1ULL << target) - 1;
    const ITYPE high_mask = ~low_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA (const ITYPE& it) {
        ITYPE i = (it & high_mask) << 1 | (it & low_mask);
        Kokkos::Experimental::swap(state[i], state[i | (1ULL << target)]);
    });
}

void update_with_x(Kokkos::View<CTYPE*> &state, UINT n_qubits, UINT target) {
#ifdef KOKKOS_ENABLE_CUDA
    if ((1ULL << target) < warp_size) {
        apply_x_shfl(state, n_qubits, target);
    } else {
        apply_x_nrml(state, n_qubits, target);
    }
#else 
    apply_x_nrml(state, n_qubits, target);
#endif
}


void apply_y_shfl(Kokkos::View<CTYPE*> &state, UINT n_qubits, UINT target) {
    assert((1ULL << target) < warp_size);
    Kokkos::parallel_for(1ULL << n_qubits, [=] __device__ (const ITYPE& i) {
        CTYPE tmp_i = state(i);
        state(i) =
        CTYPE(
            -__shfl_xor_sync(0xffffffff, tmp_i.imag(), 1ULL << target),
            __shfl_xor_sync(0xffffffff, tmp_i.real(), 1ULL << target)
        ) * (i >> target & 1 ? 1 : -1);
    });
}

void apply_y_nrml(Kokkos::View<CTYPE*> &state, UINT n_qubits, UINT target) {
    const ITYPE low_mask = (1ULL << target) - 1;
    const ITYPE high_mask = ~low_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA (const ITYPE& it) {
        ITYPE i = (it & high_mask) << 1 | (it & low_mask);
        ITYPE j = i | 1ULL << target;
        Kokkos::Experimental::swap(state[i], state[j]);
        state[i] *= CTYPE(0, -1);
        state[j] *= CTYPE(0, 1);
    });
}

void update_with_y(Kokkos::View<CTYPE*> &state, UINT n_qubits, UINT target) {
#ifdef KOKKOS_ENABLE_CUDA
    if ((1ULL << target) < warp_size) {
        apply_y_shfl(state, n_qubits, target);
    } else {
        apply_y_nrml(state, n_qubits, target);
    }
#else 
    apply_y_nrml(state, n_qubits, target);
#endif
}

void update_with_z(Kokkos::View<CTYPE*> &state, UINT n_qubits, UINT target) {
    const ITYPE low_mask = (1ULL << target) - 1;
    const ITYPE upper_mask = ~low_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA (const ITYPE& it) {
        state[(it & upper_mask) << 1 | (it & low_mask) | 1ULL << target] *= -1;
    });
}

void apply_h_shfl(Kokkos::View<CTYPE*> &state, UINT n_qubits, UINT target) {
    const double inv_sqrt_2 = 1. / sqrt(2.);
    Kokkos::parallel_for(1ULL << n_qubits, [=] __device__ (const ITYPE& i) {
        CTYPE tmp_i = state(i);
        state(i) = inv_sqrt_2 * (tmp_i * (i >> target & 1 ? -1 : 1) +
        CTYPE(
            __shfl_xor_sync(0xffffffff, tmp_i.real(), 1ULL << target),
            __shfl_xor_sync(0xffffffff, tmp_i.imag(), 1ULL << target)
        ));
    });
}

void apply_h_nrml(Kokkos::View<CTYPE*> &state, UINT n_qubits, UINT target) {
    const ITYPE lower_mask = (1ULL << target) - 1;
    const ITYPE upper_mask = ~lower_mask;
    const double inv_sqrt_2 = 1. / sqrt(2.);
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA (const ITYPE& it) {
        ITYPE i = (it & upper_mask) << 1 | (it & lower_mask);
        ITYPE j = i | 1ULL << target;
        CTYPE tmp_i = state[i];
        CTYPE tmp_j = state[j];
        state[i] = inv_sqrt_2 * (tmp_i + tmp_j);
        state[j] = inv_sqrt_2 * (tmp_i - tmp_j);
    });
}

void update_with_h(Kokkos::View<CTYPE*> &state, UINT n_qubits, UINT target) {
#ifdef KOKKOS_ENABLE_CUDA
    if ((1ULL << target) < warp_size) {
        apply_h_shfl(state, n_qubits, target);
    } else {
        apply_h_nrml(state, n_qubits, target);
    }
#else 
    apply_h_nrml(state, n_qubits, target);
#endif
}

void apply_Rx_shfl(Kokkos::View<CTYPE*> &state, UINT n_qubits, double angle, UINT target) {
    assert((1ULL << target) < warp_size);
    double sin_half = Kokkos::sin(angle / 2), cos_half = Kokkos::cos(angle / 2);
    Kokkos::parallel_for(1 << n_qubits, [=] __device__ (const ITYPE& i) {
        CTYPE tmp_i = state(i);
        CTYPE tmp_j(
            __shfl_xor_sync(0xffffffff, tmp_i.real(), 1 << target),
            __shfl_xor_sync(0xffffffff, tmp_i.imag(), 1 << target)
        );
        state(i) = cos_half * tmp_i - CTYPE(0, 1) * sin_half * tmp_j;
    });
}

void apply_Rx_nrml(Kokkos::View<CTYPE*> &state, UINT n_qubits, double angle, UINT target) {
    double sin_half = Kokkos::sin(angle / 2), cos_half = Kokkos::cos(angle / 2);
    const ITYPE lower_mask = (1ULL << target) - 1;
    const ITYPE upper_mask = ~lower_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), [=] __device__ (const ITYPE& it) {
        ITYPE i = (it & upper_mask) << 1 | (it & lower_mask);
        ITYPE j = i | (1ULL << target); 
        CTYPE tmp_i = state[i];
        CTYPE tmp_j = state[j];
        state[i] = cos_half * tmp_i - CTYPE(0, 1) * sin_half * tmp_j;
        state[j] = cos_half * tmp_j - CTYPE(0, 1) * sin_half * tmp_i;
    });
}

void update_with_Rx(Kokkos::View<CTYPE*> &state, UINT n_qubits, double angle, UINT target) {
#ifdef KOKKOS_ENABLE_CUDA
    if ((1ULL << target) < warp_size) {
        apply_Rx_shfl(state, n_qubits, angle, target);
    } else {
        apply_Rx_nrml(state, n_qubits, angle, target);
    }
#else 
    apply_Rx_nrml(state, n_qubits, angle, target);
#endif
}

void apply_Ry_shfl(Kokkos::View<CTYPE*> &state, UINT n_qubits, double angle, UINT target) {
    const double sin_half = Kokkos::sin(angle / 2), cos_half = Kokkos::cos(angle / 2);
    Kokkos::parallel_for(1ULL << n_qubits, [=] __device__ (const ITYPE& i) {
        CTYPE tmp_i = state(i);
        state(i) = cos_half * tmp_i - sin_half * CTYPE(
            __shfl_xor_sync(0xffffffff, tmp_i.real(), 1ULL << target),
            __shfl_xor_sync(0xffffffff, tmp_i.imag(), 1ULL << target)
        ) * ((i >> target) & 1 ? -1 : 1);
    });
}

void apply_Ry_nrml(Kokkos::View<CTYPE*> &state, UINT n_qubits, double angle, UINT target) {
    const double sin_half = Kokkos::sin(angle / 2), cos_half = Kokkos::cos(angle / 2);
    const ITYPE lower_mask = (1ULL << target) - 1;
    const ITYPE upper_mask = ~lower_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA(const ITYPE& it) {
        ITYPE i = ((it & upper_mask) << 1) | (it & lower_mask);
        ITYPE j = i | (1ULL << target); 
        CTYPE tmp_i = state[i];
        CTYPE tmp_j = state[j];
        state[i] = cos_half * tmp_i - sin_half * tmp_j;
        state[j] = cos_half * tmp_j + sin_half * tmp_i;
    });
}

void update_with_Ry(Kokkos::View<CTYPE*> &state, UINT n_qubits, double angle, UINT target) {
#ifdef KOKKOS_ENABLE_CUDA
    if ((1ULL << target) < warp_size) {
        apply_Ry_shfl(state, n_qubits, angle, target);
    } else {
        apply_Ry_nrml(state, n_qubits, angle, target);
    }
#else 
    apply_Ry_nrml(state, n_qubits, angle, target);
#endif
}

void update_with_Rz(Kokkos::View<CTYPE*> &state, UINT n_qubits, double angle, UINT target) {
    const CTYPE phase0 = Kokkos::exp(CTYPE(0, - angle / 2)), 
                phase1 = Kokkos::exp(CTYPE(0, angle / 2));
    const ITYPE lower_mask = (1ULL << target) - 1;
    const ITYPE upper_mask = ~lower_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA (const ITYPE& it) {
        ITYPE i = (it & upper_mask) << 1 | (it & lower_mask);
        ITYPE j = i | 1ULL << target;
        state[i] *= phase0;
        state[j] *= phase1;
    });
}

void update_with_SWAP(Kokkos::View<CTYPE*> state_kokkos, UINT n_qubits, UINT target0, UINT target1) {
    if (target0 > target1) std::swap(target0, target1);
    const ITYPE mask0 = 1ULL << target0;
    const ITYPE mask1 = 1ULL << target1;
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {1ULL << (n_qubits - target1 - 1), 1ULL << (target1 - target0 - 1), 1ULL << target0});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE &upper_bit_it, const ITYPE &middle_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target1 + 1)) | (middle_bit_it << (target0 + 1)) | lower_bit_it;
        Kokkos::Experimental::swap(state_kokkos[i | mask0], state_kokkos[i | mask1]);
    });
}

void update_with_SWAP_single_loop(Kokkos::View<CTYPE*> state_kokkos, UINT n_qubits, UINT target0, UINT target1) {
    if (target0 > target1) std::swap(target0, target1);
    const ITYPE lower_mask = (1ULL << target0) - 1;
    const ITYPE middle_mask = ((1ULL << (target1 - target0 - 1)) - 1) << target0;
    const ITYPE upper_mask = ~(lower_mask | middle_mask);
    Kokkos::parallel_for(1ULL << (n_qubits - 2), KOKKOS_LAMBDA(const ITYPE &it) {
        ITYPE lower_idx = it & lower_mask;
        ITYPE middle_idx = (it & middle_mask) << 1;
        ITYPE upper_idx = (it & upper_mask) << 2;
        ITYPE i = upper_idx | middle_idx | lower_idx;
        Kokkos::Experimental::swap(state_kokkos[i | (1ULL << target0)],
            state_kokkos[i | (1ULL << target1)]);
    });
}

void update_with_CNOT(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, UINT control, UINT target) {
    const ITYPE mask_control = 1ULL << control;
    const ITYPE mask_target = 1ULL << target;
    ITYPE ub = Kokkos::max(target, control);
    ITYPE lb = Kokkos::min(target, control);
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {1ULL << (n_qubits - ub - 1), 1ULL << (ub - lb - 1), 1ULL << lb});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE &upper_bit_it, const ITYPE &middle_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (ub + 1)) | (middle_bit_it << (lb + 1)) | lower_bit_it | mask_control;
        Kokkos::Experimental::swap(state_kokkos[i], state_kokkos[i | mask_target]);
    });    
}

void update_with_dense_matrix_single_target(Kokkos::View<CTYPE*> state_kokkos, UINT n_qubits, UINT target, CTYPE matrix[4]) {
    const ITYPE lower_mask = (1ULL << target) - 1;
    const ITYPE upper_mask = ~lower_mask;
    CTYPE matrix0 = matrix[0], matrix1 = matrix[1], matrix2 = matrix[2], matrix3 = matrix[3];
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA (const ITYPE& it) {
        ITYPE i = (it & upper_mask) << 1 | (it & lower_mask);
        ITYPE j = i | 1ULL << target;
        auto state_updated_i = matrix0 * state_kokkos[i] + matrix1 * state_kokkos[j];
        auto state_updated_j = matrix2 * state_kokkos[i] + matrix3 * state_kokkos[j];
        state_kokkos[i] = state_updated_i;
        state_kokkos[j] = state_updated_j;
    });
}

void update_with_dense_matrix_double_target(Kokkos::View<CTYPE*> state_kokkos, UINT n_qubits, UINT target0, UINT target1, CTYPE matrix[16]) {

    auto [target_low, target_high] = Kokkos::minmax(target0, target1);
    const ITYPE lower_mask = (1ULL << target_low) - 1;
    const ITYPE middle_mask = ((1ULL << (target_high-1)) - 1) & ~lower_mask;
    const ITYPE upper_mask = ~(lower_mask | middle_mask);
    CTYPE matrix0 = matrix[0], matrix1 = matrix[1], matrix2 = matrix[2], matrix3 = matrix[3], matrix4 = matrix[4], matrix5 = matrix[5], matrix6 = matrix[6], matrix7 = matrix[7], matrix8 = matrix[8], matrix9 = matrix[9], matrix10 = matrix[10], matrix11 = matrix[11], matrix12 = matrix[12], matrix13 = matrix[13], matrix14 = matrix[14], matrix15 = matrix[15];
    Kokkos::parallel_for(1ULL << (n_qubits - 2), KOKKOS_LAMBDA (const ITYPE& it) {
        ITYPE i = (it & upper_mask) << 2 | (it & middle_mask) << 1 | (it & lower_mask);
        ITYPE base[4] = {i, i | 1 << target0, i | 1 << target1, i | 1 << target0 | 1 << target1};
        CTYPE new_state[4] = {
            matrix0 * state_kokkos[base[0]] + matrix1 * state_kokkos[base[1]] + matrix2 * state_kokkos[base[2]] + matrix3 * state_kokkos[base[3]],
            matrix4 * state_kokkos[base[0]] + matrix5 * state_kokkos[base[1]] + matrix6 * state_kokkos[base[2]] + matrix7 * state_kokkos[base[3]],
            matrix8 * state_kokkos[base[0]] + matrix9 * state_kokkos[base[1]] + matrix10 * state_kokkos[base[2]] + matrix11 * state_kokkos[base[3]],
            matrix12 * state_kokkos[base[0]] + matrix13 * state_kokkos[base[1]] + matrix14 * state_kokkos[base[2]] + matrix15 * state_kokkos[base[3]]
        };
        for(UINT i = 0; i < 4; i++) {
            state_kokkos[base[i]] = new_state[i];
        }
    });
}

void update_with_dense_matrix_single_target_double_control(Kokkos::View<CTYPE*> state_kokkos, UINT n_qubits, UINT target, UINT control0, UINT value0, UINT control1, UINT value1, CTYPE matrix[4]) {
    UINT qubit2 = target, qubit1 = control0, qubit0 = control1;
    if(qubit2 < qubit1) std::swap(qubit2, qubit1);
    if(qubit1 < qubit0) std::swap(qubit1, qubit0);
    if(qubit2 < qubit1) std::swap(qubit2, qubit1);
    const ITYPE mask0 = (1ULL << qubit0) - 1;
    const ITYPE mask1 = ((1ULL << (qubit1-1)) - 1) & ~mask0;
    const ITYPE mask2 = ((1ULL << (qubit2-2)) - 1) & ~(mask0 | mask1);
    const ITYPE mask3 = ~(mask0 | mask1 | mask2);
    const ITYPE control_mask = value0 << control0 | value1 << control1;
    CTYPE matrix0 = matrix[0], matrix1 = matrix[1], matrix2 = matrix[2], matrix3 = matrix[3];
    Kokkos::parallel_for(1ULL << (n_qubits - 3), KOKKOS_LAMBDA (const ITYPE& it) {
        ITYPE i = (it & mask3) << 3 | (it & mask2) << 2 | (it & mask1) << 1 | (it & mask1) | control_mask;
        ITYPE j = i | 1ULL << target;
        auto state_updated_i = matrix0 * state_kokkos[i] + matrix1 * state_kokkos[j];
        auto state_updated_j = matrix2 * state_kokkos[i] + matrix3 * state_kokkos[j];
        state_kokkos[i] = state_updated_i;
        state_kokkos[j] = state_updated_j;
    });
}

void update_with_dense_matrix(Kokkos::View<CTYPE*> state_kokkos, UINT n, Kokkos::View<UINT*> target_list, Kokkos::View<CTYPE**> matrix_kokkos) {

    int num_target = target_list.size(), num_outer = n - num_target;
    int target_mask = 0;
    Kokkos::parallel_reduce(n, KOKKOS_LAMBDA(int i, int& lcl_target_mask) {
        lcl_target_mask |= 1 << target_list[i];
    }, target_mask);


    Kokkos::View<int**> state_idx_kokkos("state_idx", 1 << num_outer, 1 << num_target);
    Kokkos::View<CTYPE**> state_updated_kokkos("state_updated", 1 << num_outer, 1 << num_target);

    Kokkos::parallel_for("calculate_state_indices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {1 << num_outer, 1 << num_target}),
        KOKKOS_LAMBDA (int idx_outer, int idx_target) {
            int idx = idx_outer;
            int target_idx = 0;
            while(target_idx < num_target) {
                UINT target = target_list[target_idx];
                UINT value = idx_target >> target_idx;
                target_idx++;
                int upper_mask = ((1 << (n - target)) - 1) << target;
                int lower_mask = (1 << target) - 1;
                idx = ((idx & upper_mask) << 1) | (value << target) | (idx & lower_mask);
            }
            state_idx_kokkos(idx_outer, idx_target) = idx;
        });

    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {1 << num_outer, 1 << num_target, 1 << num_target}),
        KOKKOS_LAMBDA (const int idx_outer, const int idx_target1, const int idx_target2) {
            state_updated_kokkos(idx_outer, idx_target1) += matrix_kokkos(idx_target1, idx_target2) * state_kokkos(state_idx_kokkos(idx_outer, idx_target2));
        });

    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {1 << num_outer, 1 << num_target}),
        KOKKOS_LAMBDA (const int idx_outer, const int idx_target) {
            state_kokkos(state_idx_kokkos(idx_outer, idx_target)) = state_updated_kokkos(idx_outer, idx_target);
        });
}

void update_with_dense_matrix_controlled(Kokkos::View<CTYPE*> state_kokkos, UINT n, Kokkos::View<UINT*> control_list, Kokkos::View<UINT*> control_value, Kokkos::View<UINT*> target_list, Kokkos::View<CTYPE**> matrix_kokkos) {
    int num_control = control_list.extent(0), num_target = target_list.extent(0), num_outer = n - num_control - num_target;
    if(num_control == 0) {
        update_with_dense_matrix(state_kokkos, n, target_list, matrix_kokkos);
        return;
    }
    int control_value_mask = 0;
    Kokkos::parallel_reduce("set_mask", num_control, KOKKOS_LAMBDA(int i, int& lcl_control_value_mask) {
        if(control_value(i)) lcl_control_value_mask |= control_list(i);
    }, control_value_mask);

    Kokkos::View<int*> outer_bits_expanded_kokkos("outer_bits_expanded", 1 << num_outer);
    Kokkos::View<int*> target_bits_expanded_kokkos("target_bits_expanded", 1 << num_target);

    Kokkos::parallel_for("expand_outer_bits", 1 << num_outer, KOKKOS_LAMBDA(int it) {
        int bits = it;
        for(UINT target_idx = 0, control_idx = 0; target_idx < num_target || control_idx < num_control;) {
            UINT target = target_idx == num_target ? n : target_list(target_idx);
            UINT control = control_idx == num_control ? n : control_list(control_idx);
            if(target < control) {
                int upper_mask = ((1 << (n - target)) - 1) << target;
                int lower_mask = (1 << target) - 1;
                bits = (bits & upper_mask) << 1 | (bits & lower_mask);
                target_idx++;
            } else {
                int upper_mask = ((1 << (n - control)) - 1) << control;
                int lower_mask = (1 << control) - 1;
                bits = (bits & upper_mask) << 1 | (bits & lower_mask);
                control_idx++;
            }
        }
        outer_bits_expanded_kokkos(it) = bits;
    });

    Kokkos::parallel_for("expand_target_bits", 1 << num_target, KOKKOS_LAMBDA(int it) {
        int bits = 0;
        for(UINT target_idx = 0; target_idx < num_target; target_idx++) {
            UINT target = target_list(target_idx);
            bits |= 1 << target;
        }
        target_bits_expanded_kokkos(it) = bits;
    });

    Kokkos::View<CTYPE*> state_updated_kokkos("state_updated", 1 << (num_outer + num_target));
    Kokkos::parallel_for("update_state", 1 << (num_outer + num_target + num_target), KOKKOS_LAMBDA(int it) {
        int outer_bits = it >> (num_target + num_target);
        int target_bits_1 = it >> num_target & ((1 << num_target) - 1);
        int target_bits_2 = it & ((1 << num_target) - 1);
        int source_idx = outer_bits_expanded_kokkos(outer_bits) | target_bits_expanded_kokkos(target_bits_2) | control_value_mask;
        state_updated_kokkos(outer_bits << num_target | target_bits_1) += matrix_kokkos(target_bits_1, target_bits_2) * state_kokkos(source_idx);
    });

    Kokkos::parallel_for("copy_to_state", 1 << (num_outer + num_target), KOKKOS_LAMBDA(int it) {
        int outer_bits = it >> num_target;
        int target_bits = it & ((1 << num_target) - 1);
        int dest_idx = outer_bits_expanded_kokkos(outer_bits) | target_bits_expanded_kokkos(target_bits) | control_value_mask;
        state_kokkos(dest_idx) = state_updated_kokkos(it);
    });
}

Kokkos::View<CTYPE*> make_random_state(int n_qubits) {
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    Kokkos::Random_XorShift64_Pool<> random_pool(seed);

    Kokkos::View<CTYPE*> state("state", 1ULL << n_qubits);
    Kokkos::parallel_for(1ULL << n_qubits, KOKKOS_LAMBDA(int i) {
        auto random_generator = random_pool.get_state();
        double real_part = Kokkos::rand<decltype(random_generator), double>::draw(random_generator);
        double imag_part = Kokkos::rand<decltype(random_generator), double>::draw(random_generator);
        state(i) = CTYPE(real_part, imag_part);
        random_pool.free_state(random_generator);
    });
    return state;
}

double single_target_bench(int n_qubits) {
    std::uniform_int_distribution<> target_gen(0, n_qubits - 1), circuit_gen(0, 3);
    std::mt19937 mt(std::random_device{}());

    auto state(make_random_state(n_qubits));
    Kokkos::fence();

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        switch(circuit_gen(mt)) {
            case 0:
            update_with_x(state, n_qubits, target_gen(mt));
            break;
            case 1:
            update_with_y(state, n_qubits, target_gen(mt));
            break;
            case 2:
            update_with_z(state, n_qubits, target_gen(mt));
            break;
            case 3:
            update_with_h(state, n_qubits, target_gen(mt));
            break;
        }
    }
    Kokkos::fence();
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
}

double single_qubit_rotation_bench(int n_qubits) {
    std::uniform_int_distribution<> target_gen(0, n_qubits - 1), circuit_gen(0, 2);
    std::uniform_real_distribution<> angle_gen(0, 2 * M_PI);
    std::mt19937 mt(std::random_device{}());

    auto state(make_random_state(n_qubits));
    Kokkos::fence();

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        switch(circuit_gen(mt)) {
            case 0:
            update_with_Rx(state, n_qubits, angle_gen(mt), target_gen(mt));
            break;
            case 1:
            update_with_Ry(state, n_qubits, angle_gen(mt), target_gen(mt));
            break;
            case 2:
            update_with_Rz(state, n_qubits, angle_gen(mt), target_gen(mt));
            break;
        }
    }
    Kokkos::fence();
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
}

double cnot_bench(int n_qubits) {
    std::uniform_int_distribution<> gen(0, n_qubits - 1);
    std::mt19937 mt(std::random_device{}());

    auto state(make_random_state(n_qubits));
    Kokkos::fence();

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        int tar = gen(mt);
        int ctrl = gen(mt);
        while (tar == ctrl) ctrl = gen(mt);
        update_with_CNOT(state, n_qubits, ctrl, tar);
    }
    Kokkos::fence();
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
}

double single_target_matrix_bench(UINT n_qubits) {

    Kokkos::View<UINT*> targets("targets", 10);
    Kokkos::View<CTYPE***> matrixes("matrixes", 10, 2, 2);
    auto state(make_random_state(n_qubits));
    Kokkos::fence();

    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    Kokkos::Random_XorShift64_Pool<> random_pool(seed);

    Kokkos::parallel_for(10, KOKKOS_LAMBDA(int i) {
        auto random_generator = random_pool.get_state();
        targets(i) = Kokkos::rand<decltype(random_generator), UINT>::draw(random_generator, 0, n_qubits - 1);
        random_pool.free_state(random_generator);
    });

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {10, 2, 2});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int i, int j, int k) {
        auto random_generator = random_pool.get_state();
        double real_part = Kokkos::rand<decltype(random_generator), double>::draw(random_generator);
        double imag_part = Kokkos::rand<decltype(random_generator), double>::draw(random_generator);
        matrixes(i, j, k) = CTYPE(real_part, imag_part);
        random_pool.free_state(random_generator);
    });
        
    decltype(targets)::HostMirror targets_host = Kokkos::create_mirror_view(targets);
    Kokkos::deep_copy(targets_host, targets);
    decltype(matrixes)::HostMirror matrixes_host = Kokkos::create_mirror_view(matrixes);
    Kokkos::deep_copy(matrixes_host, matrixes);
    auto start_time = std::chrono::system_clock::now();
    // Kokkos::View<CTYPE*> matrix("matrix", 4);
    for (int i = 0; i < 10; ++i) {
        UINT target = targets_host(i);
        CTYPE matrix[4] = {matrixes_host(i, 0, 0), matrixes_host(i, 0, 1), matrixes_host(i, 1, 0), matrixes_host(i, 1, 1)};
        update_with_dense_matrix_single_target(state, n_qubits, target, matrix);
    }
    Kokkos::fence();
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
}

double double_target_matrix_bench(UINT n_qubits) {
    assert(n_qubits >= 3);
    Kokkos::View<UINT**> targets("targets", 10, 2);
    Kokkos::View<CTYPE***> matrixes("matrixes", 10, 4, 4);
    auto state(make_random_state(n_qubits));
    Kokkos::fence();

    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    Kokkos::Random_XorShift64_Pool<> random_pool(seed);

    Kokkos::parallel_for(10, KOKKOS_LAMBDA(int i) {
        auto random_generator = random_pool.get_state();
        targets(i, 0) = Kokkos::rand<decltype(random_generator), UINT>::draw(random_generator, 0, n_qubits - 2);
        targets(i, 1) = Kokkos::rand<decltype(random_generator), UINT>::draw(random_generator, 0, n_qubits - 1);
        if(targets(i, 0) == targets(i, 1)) targets(i, 1) = n_qubits - 1;
        random_pool.free_state(random_generator);
    });

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {10, 4, 4});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int i, int j, int k) {
        auto random_generator = random_pool.get_state();
        double real_part = Kokkos::rand<decltype(random_generator), double>::draw(random_generator);
        double imag_part = Kokkos::rand<decltype(random_generator), double>::draw(random_generator);
        matrixes(i, j, k) = CTYPE(real_part, imag_part);
        random_pool.free_state(random_generator);
    });
        
    decltype(targets)::HostMirror targets_host = Kokkos::create_mirror_view(targets);
    Kokkos::deep_copy(targets_host, targets);
    decltype(matrixes)::HostMirror matrixes_host = Kokkos::create_mirror_view(matrixes);
    Kokkos::deep_copy(matrixes_host, matrixes);
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        UINT target0 = targets_host(i, 0), target1 = targets_host(i, 1);
        CTYPE matrix[16];
        for(int j = 0; j < 4; j++) for(int k = 0; k < 4; k++) {
            matrix[j*4+k] = matrixes_host(i, j, k);
        }
        update_with_dense_matrix_double_target(state, n_qubits, target0, target1, matrix);
    }
    Kokkos::fence();
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
}

double double_control_matrix_bench(UINT n_qubits) {
    assert(n_qubits >= 3);
    Kokkos::View<UINT**> targets("targets", 10, 1);
    Kokkos::View<UINT**> control_list("control_list", 10, 2);
    Kokkos::View<UINT**> control_values("control_values", 10, 2);
    Kokkos::View<CTYPE***> matrixes("matrixes", 10, 2, 2);
    auto state(make_random_state(n_qubits));
    Kokkos::fence();

    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    Kokkos::Random_XorShift64_Pool<> random_pool(seed);

    Kokkos::parallel_for(10, KOKKOS_LAMBDA(int i) {
        auto random_generator = random_pool.get_state();
        targets(i, 0) = Kokkos::rand<decltype(random_generator), UINT>::draw(random_generator, 0, n_qubits - 1);
        control_list(i, 0) = Kokkos::rand<decltype(random_generator), UINT>::draw(random_generator, 0, n_qubits - 1);
        control_list(i, 1) = Kokkos::rand<decltype(random_generator), UINT>::draw(random_generator, 0, n_qubits - 2);
        if(targets(i, 0) == control_list(i, 0)) control_list(i, 0) = n_qubits - 1;
        if(targets(i, 0) == control_list(i, 1)) control_list(i, 1) = n_qubits - 2;
        if(control_list(i, 0) == control_list(i, 1)) {
            if(n_qubits - 1 == targets(i, 0)) control_list(i, 1) = n_qubits - 2;
            else control_list(i, 1) = n_qubits - 1;
        }
        control_values(i, 0) = Kokkos::rand<decltype(random_generator), UINT>::draw(random_generator, 0, 1);
        control_values(i, 1) = Kokkos::rand<decltype(random_generator), UINT>::draw(random_generator, 0, 1);
        random_pool.free_state(random_generator);
    });

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy({0, 0, 0}, {10, 2, 2});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int i, int j, int k) {
        auto random_generator = random_pool.get_state();
        double real_part = Kokkos::rand<decltype(random_generator), double>::draw(random_generator);
        double imag_part = Kokkos::rand<decltype(random_generator), double>::draw(random_generator);
        matrixes(i, j, k) = CTYPE(real_part, imag_part);
        random_pool.free_state(random_generator);
    });
        
    decltype(targets)::HostMirror targets_host = Kokkos::create_mirror_view(targets);
    Kokkos::deep_copy(targets_host, targets);
    decltype(control_list)::HostMirror control_list_host = Kokkos::create_mirror_view(control_list);
    Kokkos::deep_copy(control_list_host, control_list);
    decltype(control_values)::HostMirror control_values_host = Kokkos::create_mirror_view(control_values);
    Kokkos::deep_copy(control_values_host, control_values);
    decltype(matrixes)::HostMirror matrixes_host = Kokkos::create_mirror_view(matrixes);
    Kokkos::deep_copy(matrixes_host, matrixes);
    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        UINT target = targets_host(i, 0);
        UINT control0 = control_list_host(i, 0), control1 = control_list_host(i, 1);
        UINT value0 = control_values_host(i, 0), value1 = control_values_host(i, 1);
        CTYPE matrix[4] = {matrixes_host(i, 0, 0), matrixes_host(i, 0, 1), matrixes_host(i, 1, 0), matrixes_host(i, 1, 1)};
        update_with_dense_matrix_single_target_double_control(state, n_qubits, target, control0, value0, control1, value1, matrix);
    }
    Kokkos::fence();
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
}

int main(int argc, char *argv[]) {
Kokkos::initialize();
{    
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <circuit_id> <n_qubits> <n_repeats>" << std::endl;
        return 1;
    }

    const auto circuit_id = std::strtoul(argv[1], nullptr, 10);
    const auto n_qubits = std::strtoul(argv[2], nullptr, 10);
    const auto n_repeats = std::strtoul(argv[3], nullptr, 10);

    std::ofstream ofs("durations.txt");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }
    for (int i = 0; i < n_repeats; i++) {
        double t;
        switch(circuit_id) {
            case 0:
            t = single_target_bench(n_qubits);
            break;
            case 1:
            t = single_qubit_rotation_bench(n_qubits);
            break;
            case 2:
            t = cnot_bench(n_qubits);
            break;
            case 3:
            t = single_target_matrix_bench(n_qubits);
            break;
            case 4:
            t = double_target_matrix_bench(n_qubits);
            break;
            case 5:
            t = double_control_matrix_bench(n_qubits);
            break;
            default:
            std::cerr << "Usage: " << "0 <= circuit_id <= 5" << std::endl;
            return 1;
        }
        ofs << t / 1000000. << " ";
    }
    ofs << std::endl;
}
Kokkos::finalize();
}
