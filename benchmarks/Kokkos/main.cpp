#include <complex>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <random>

using UINT = unsigned int;
using ITYPE = unsigned long long;
using CTYPE = Kokkos::complex<double>;
using TeamHandle = Kokkos::TeamPolicy<>::member_type;

void update_with_x(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, UINT target) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n_qubits - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        Kokkos::Experimental::swap(state_kokkos[i], state_kokkos[j]);
    });
}

void update_with_y(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, UINT target) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n_qubits - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        Kokkos::Experimental::swap(state_kokkos[i], state_kokkos[j]);
        state_kokkos[i] *= CTYPE(0, 1);
        state_kokkos[j] *= CTYPE(0, -1);
    });
}

void update_with_z(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, UINT target) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n_qubits - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it | (1ULL << target);
        state_kokkos[i] *= -1;
    });
}

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

void update_with_h(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, UINT target) {
    const double inv_sqrt_2 = 1. / sqrt(2.);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n_qubits - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        CTYPE temp_i = state_kokkos[i];
        CTYPE temp_j = state_kokkos[j];
        state_kokkos[i] = inv_sqrt_2 * (temp_i + temp_j);
        state_kokkos[j] = inv_sqrt_2 * (temp_i - temp_j);
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

void update_with_Rx(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, double angle, UINT target) {
    const double angle_half = angle / 2, sin_half = Kokkos::sin(angle_half), cos_half = Kokkos::cos(angle_half);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n_qubits - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        CTYPE temp_i = state_kokkos[i];
        CTYPE temp_j = state_kokkos[j];
        state_kokkos[i] = cos_half * temp_i - CTYPE(0, 1) * sin_half * temp_j;
        state_kokkos[j] = cos_half * temp_j - CTYPE(0, 1) * sin_half * temp_i;
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

void update_with_Ry(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, double angle, UINT target) {
    const double angle_half = angle / 2, sin_half = Kokkos::sin(angle_half), cos_half = Kokkos::cos(angle_half);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n_qubits - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        CTYPE temp_i = state_kokkos[i];
        CTYPE temp_j = state_kokkos[j];
        state_kokkos[i] = cos_half * temp_i + sin_half * temp_j;
        state_kokkos[j] = cos_half * temp_j - sin_half * temp_i;
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

void update_with_Rz(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, double angle, UINT target) {
    const double angle_half = angle / 2;
    const CTYPE phase0 = Kokkos::exp(CTYPE(0, -angle_half)), 
                phase1 = Kokkos::exp(CTYPE(0, angle_half));
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n_qubits - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        state_kokkos[i] *= phase0;
        state_kokkos[j] *= phase1;
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

void update_with_dense_matrix(Kokkos::View<CTYPE*> state_kokkos, UINT n, const Kokkos::View<UINT*>& target_list, Kokkos::View<CTYPE**> matrix_kokkos) {

    int num_target = target_list.size(), num_outer = n - num_target;
    int target_mask = 0;
    for(int i = 0; i < num_target; ++i) target_mask |= 1 << target_list[i];

    

    Kokkos::View<int**> state_idx_kokkos("state_idx", 1 << num_outer, 1 << num_target);
    Kokkos::View<CTYPE**> state_updated_kokkos("state_updated", 1 << num_outer, 1 << num_target);

    Kokkos::parallel_for("calculate_state_indices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {1 << num_outer, 1 << num_target}),
        KOKKOS_LAMBDA (const int idx_outer, const int idx_target) {
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

void update_with_dense_matrix_controlled(Kokkos::View<CTYPE*> state_kokkos, UINT n, const Kokkos::View<UINT*>& control_list, const Kokkos::View<UINT*>& control_value, const Kokkos::View<UINT*>& target_list, Kokkos::View<CTYPE**> matrix_kokkos) {
    int num_control = control_list.size(), num_target = target_list.size(), num_outer = n - num_control - num_target;
    if(num_control == 0) {
        update_with_dense_matrix(state_kokkos, n, target_list, matrix_kokkos);
        return;
    }

    Kokkos::View<int**> controlled_state_idx_kokkos("controlled_state_idx", 1 << num_outer, 1 << num_target);
    Kokkos::View<CTYPE**> controlled_state_updated_kokkos("controlled_state_updated", 1 << num_outer, 1 << num_target);

    Kokkos::parallel_for("calculate_controlled_state_indices", Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {1 << num_outer, 1 << num_target}),
        KOKKOS_LAMBDA (const int idx_outer, const int idx_target) {
            int idx = idx_outer;
            int control_idx = 0, target_idx = 0;
            while(control_idx < num_control || target_idx < num_target) {
                if(target_idx == num_target || (control_idx < num_control && control_list(control_idx) < target_list(target_idx))) {
                    UINT control = control_list(control_idx);
                    UINT value = control_value(control_idx);
                    control_idx++;
                    int upper_mask = ((1 << (n - control)) - 1) << control;
                    int lower_mask = (1 << control) - 1;
                    idx = ((idx & upper_mask) << 1) | (value << control) | (idx & lower_mask);
                } else {
                    UINT target = target_list(target_idx);
                    UINT value = idx_target >> target_idx;
                    target_idx++;
                    int upper_mask = ((1 << (n - target)) - 1) << target;
                    int lower_mask = (1 << target) - 1;
                    idx = ((idx & upper_mask) << 1) | (value << target) | (idx & lower_mask);
                }
            }
            controlled_state_idx_kokkos(idx_outer, idx_target) = idx;
        });

    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0, 0, 0}, {1 << num_outer, 1 << num_target, 1 << num_target}),
        KOKKOS_LAMBDA (const int idx_outer, const int idx_target1, const int idx_target2) {
            controlled_state_updated_kokkos(idx_outer, idx_target1) += matrix_kokkos(idx_target1, idx_target2) * state_kokkos(controlled_state_idx_kokkos(idx_outer, idx_target2));
        });

    Kokkos::parallel_for(Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {1 << num_outer, 1 << num_target}),
        KOKKOS_LAMBDA (const int idx_outer, const int idx_target) {
            state_kokkos(controlled_state_idx_kokkos(idx_outer, idx_target)) = controlled_state_updated_kokkos(idx_outer, idx_target);
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
    std::chrono::duration<double> elapsed = end_time - start_time;
    return elapsed.count();
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
    std::chrono::duration<double> elapsed = end_time - start_time;
    return elapsed.count();
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
    std::chrono::duration<double> elapsed = end_time - start_time;
    return elapsed.count();
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
        
    auto st_time = std::chrono::system_clock::now();
    for (int i = 0; i < 10; ++i) {
        //std::cout << "b" << std::endl;
        update_with_dense_matrix(state, n_qubits, Kokkos::subview(targets, i, Kokkos::ALL()), Kokkos::subview(matrixes, i, Kokkos::ALL(), Kokkos::ALL()));
    }
    auto ed_time = std::chrono::system_clock::now();

    return std::chrono::duration_cast<std::chrono::milliseconds>(ed_time - st_time).count();

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
            /*case 4:
            t = double_target_matrix_bench(n_qubits);
            break;
            case 5:
            t = double_control_matrix_bench(n_qubits);
            break;*/
        }
        ofs << t << " ";
        std::cout << t << " ";
    }
    ofs << std::endl;
    std::cout << std::endl;

}
Kokkos::finalize();
}