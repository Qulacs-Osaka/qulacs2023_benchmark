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

#ifdef KOKKOS_ENABLE_CUDA
void apply_x_shfl(Kokkos::View<CTYPE*> &state, UINT n_qubits, UINT target) {
    assert((1ULL << target) < warp_size);
    Kokkos::parallel_for(1ULL << n_qubits, [=] __device__ (const ITYPE& i) {
        state(i) = CTYPE{
            __shfl_xor_sync(0xffffffff, state(i).real(), 1ULL << target),
            __shfl_xor_sync(0xffffffff, state(i).imag(), 1ULL << target)
        };
    });
}
#endif

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

#ifdef KOKKOS_ENABLE_CUDA
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
#endif


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

#ifdef KOKKOS_ENABLE_CUDA
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
#endif

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

#ifdef KOKKOS_ENABLE_CUDA
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
#endif


void apply_Rx_nrml(Kokkos::View<CTYPE*> &state, UINT n_qubits, double angle, UINT target) {
    double sin_half = Kokkos::sin(angle / 2), cos_half = Kokkos::cos(angle / 2);
    const ITYPE lower_mask = (1ULL << target) - 1;
    const ITYPE upper_mask = ~lower_mask;
    Kokkos::parallel_for(1ULL << (n_qubits - 1), KOKKOS_LAMBDA (const ITYPE& it) {
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

#ifdef KOKKOS_ENABLE_CUDA
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
#endif

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
    std::vector<UINT> gate(10);
    std::vector<UINT> target(10);
    for(UINT i = 0; i < 10; i++) {
        gate[i] = circuit_gen(mt);
        target[i] = target_gen(mt);
    }
    Kokkos::fence();

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        switch(gate[i]) {
            case 0:
            update_with_x(state, n_qubits, target[i]);
            break;
            case 1:
            update_with_y(state, n_qubits, target[i]);
            break;
            case 2:
            update_with_z(state, n_qubits, target[i]);
            break;
            case 3:
            update_with_h(state, n_qubits, target[i]);
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
    std::vector<UINT> gate(10);
    std::vector<UINT> target(10);
    std::vector<double> angle(10);
    for(UINT i = 0; i < 10; i++) {
        gate[i] = circuit_gen(mt);
        target[i] = target_gen(mt);
        angle[i] = angle_gen(mt);
    }
    Kokkos::fence();

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        switch(gate[i]) {
            case 0:
            update_with_Rx(state, n_qubits, angle[i], target[i]);
            break;
            case 1:
            update_with_Ry(state, n_qubits, angle[i], target[i]);
            break;
            case 2:
            update_with_Rz(state, n_qubits, angle[i], target[i]);
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
    std::vector<UINT> target(10), control(10);
    for (int i = 0; i < 10; ++i) {
        target[i] = gen(mt);
        control[i] = gen(mt);
        while (target[i] == control[i]) control[i] = gen(mt);
    }
    Kokkos::fence();

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        update_with_CNOT(state, n_qubits, control[i], target[i]);
    }
    Kokkos::fence();
    auto end_time = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
}

double single_target_matrix_bench(UINT n_qubits) {

    Kokkos::View<UINT**> targets("targets", 10, 1);
    Kokkos::View<CTYPE***> matrixes("matrixes", 10, 2, 2);
    auto state(make_random_state(n_qubits));
    Kokkos::fence();

    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    Kokkos::Random_XorShift64_Pool<> random_pool(seed);

    Kokkos::parallel_for(10, KOKKOS_LAMBDA(int i) {
        auto random_generator = random_pool.get_state();
        targets(i, 0) = Kokkos::rand<decltype(random_generator), UINT>::draw(random_generator, 0, n_qubits - 1);
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
        
    auto start_time = std::chrono::system_clock::now();
    Kokkos::View<UINT*> target("target", 1);
    Kokkos::View<CTYPE**> matrix("matrix", 2, 2);
    for (int i = 0; i < 10; ++i) {
        Kokkos::deep_copy(target, Kokkos::subview(targets, i, Kokkos::ALL));
        Kokkos::deep_copy(matrix, Kokkos::subview(matrixes, i, Kokkos::ALL, Kokkos::ALL));
        update_with_dense_matrix(state, n_qubits, target, matrix);
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
        
    auto start_time = std::chrono::high_resolution_clock::now();
    Kokkos::View<UINT*> target("target", 2);
    Kokkos::View<CTYPE**> matrix("matrix", 4, 4);
    for (int i = 0; i < 10; ++i) {
        Kokkos::deep_copy(target, Kokkos::subview(targets, i, Kokkos::ALL));
        Kokkos::deep_copy(matrix, Kokkos::subview(matrixes, i, Kokkos::ALL, Kokkos::ALL));
        update_with_dense_matrix(state, n_qubits, target, matrix);
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
        if(control_list(i, 0) == control_list(i, 1)) control_list(i, 1) = n_qubits - 1;
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
        
    auto start_time = std::chrono::high_resolution_clock::now();
    Kokkos::View<UINT*> target("target", 1);
    Kokkos::View<UINT*> control("controls", 2);
    Kokkos::View<UINT*> control_val("control_vals", 2);
    Kokkos::View<CTYPE**> matrix("matrix", 2, 2);
    for (int i = 0; i < 10; ++i) {
        Kokkos::deep_copy(target, Kokkos::subview(targets, i, Kokkos::ALL));
        Kokkos::deep_copy(control, Kokkos::subview(control_list, i, Kokkos::ALL));
        Kokkos::deep_copy(control_val, Kokkos::subview(control_values, i, Kokkos::ALL));
        Kokkos::deep_copy(matrix, Kokkos::subview(matrixes, i, Kokkos::ALL, Kokkos::ALL));
        update_with_dense_matrix_controlled(state, n_qubits, control, control_val, target, matrix);
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
