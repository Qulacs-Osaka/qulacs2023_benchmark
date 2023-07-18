#include <complex>
#include <Kokkos_Core.hpp>
#include <Kokkos_StdAlgorithms.hpp>
#include <iostream>
#include <fstream>
#include <algorithm>

using UINT = unsigned int;
using ITYPE = unsigned long long;
using CTYPE = Kokkos::complex<double>;

void update_with_x(Kokkos::View<CTYPE*> &state_kokkos, UINT n_qubits, UINT target) {
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << (n_qubits - target - 1), 1ULL << target});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const ITYPE& upper_bit_it, const ITYPE &lower_bit_it) {
        ITYPE i = (upper_bit_it << (target + 1)) | lower_bit_it;
        ITYPE j = i | (1ULL << target);
        Kokkos::Experimental::swap(state_kokkos[i], state_kokkos[j]);
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


int main(int argc, char *argv[]) {
Kokkos::initialize();
{    
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <n_qubits> <n_repeats>" << std::endl;
        Kokkos::finalize();
        return 1;
    }
    const auto n_qubits = std::strtoul(argv[1], nullptr, 10);
    const auto n_repeats = std::strtoul(argv[2], nullptr, 10);

    std::vector<double> results;  // change to double to store seconds

    for (int i = 0; i < n_repeats; ++i) {
        auto start_time = std::chrono::high_resolution_clock::now();

        Kokkos::View<CTYPE*> init_state("init_state", 1ULL << n_qubits);
        Kokkos::parallel_for(1ULL << n_qubits, KOKKOS_LAMBDA(int i) {
            init_state(i) = CTYPE(i, 0);
        });

        update_with_CNOT_single_loop(init_state, n_qubits, n_qubits / 3, n_qubits * 2 / 3);
        Kokkos::fence();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        results.push_back(duration.count() / 1e6);         
    }

    std::ofstream ofs("durations.txt");
    if (!ofs.is_open()) {
        std::cerr << "Failed to open file" << std::endl;
        Kokkos::finalize();
        return 1;
    }

    for (int i = 0; i < n_repeats; i++) {
        std::cout << results[i] << " ";
    }
    std::cout << std::endl;

}
Kokkos::finalize();
}
