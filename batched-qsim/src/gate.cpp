#define _USE_MATH_DEFINES
#include <cmath>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <Kokkos_StdAlgorithms.hpp>

#include "gate.hpp"

using TeamHandle = Kokkos::TeamPolicy<>::member_type;

void set_zero_state(Kokkos::View<CTYPE *> state, UINT n)
{
    Kokkos::parallel_for(
        "set_zero_state", 1ULL << n, KOKKOS_LAMBDA(ITYPE i) { state(i) = i == 0 ? 1.0 : 0.0; });
}

void set_zero_state(Kokkos::View<CTYPE **> state, UINT n)
{
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {state.extent(0), state.extent(1)});

    Kokkos::parallel_for(
        "set_zero_state_batch", policy,
        KOKKOS_LAMBDA(ITYPE i, ITYPE sample) { state(i, sample) = i == 0 ? 1.0 : 0.0; });
}

void update_with_X(Kokkos::View<CTYPE *> state, UINT n, UINT target)
{
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0},
                                                  {1ULL << (n - target - 1), 1ULL << target});

    Kokkos::parallel_for(
        "X_gate", policy, KOKKOS_LAMBDA(ITYPE upper_bit, ITYPE lower_bit) {
            ITYPE i = (upper_bit << (target + 1)) | lower_bit;
            ITYPE j = i | (1ULL << target);
            Kokkos::Experimental::swap(state[i], state[j]);
        });
}

void update_with_H(Kokkos::View<CTYPE *> state, UINT n, UINT target)
{
    double inv_sqrt_2 = 1. / std::sqrt(2.);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0},
                                                  {1ULL << (n - target - 1), 1ULL << target});

    Kokkos::parallel_for(
        "H_gate", policy, KOKKOS_LAMBDA(ITYPE upper_bit, ITYPE lower_bit) {
            ITYPE i = (upper_bit << (target + 1)) | lower_bit;
            ITYPE j = i | (1ULL << target);
            CTYPE temp_i = state[i];
            CTYPE temp_j = state[j];
            state[i] = inv_sqrt_2 * (temp_i + temp_j);
            state[j] = inv_sqrt_2 * (temp_i - temp_j);
        });
}

void update_with_H_batched(Kokkos::View<CTYPE **> state, UINT n, UINT target)
{
    double inv_sqrt_2 = 1. / std::sqrt(2.);
    Kokkos::TeamPolicy outer_policy(state.extent(1), Kokkos::AUTO);

    Kokkos::parallel_for(
        "H_gate_batched", outer_policy, KOKKOS_LAMBDA(const TeamHandle &member) {
            int sample = member.league_rank();

            Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle> inner_policy(
                member, 1ULL << (n - target - 1), 1ULL << target);

            Kokkos::parallel_for(inner_policy, [=](ITYPE upper_bit, ITYPE lower_bit) {
                ITYPE i = (upper_bit << (target + 1)) | lower_bit;
                ITYPE j = i | (1ULL << target);
                CTYPE temp_i = state(i, sample);
                CTYPE temp_j = state(j, sample);
                state(i, sample) = inv_sqrt_2 * (temp_i + temp_j);
                state(j, sample) = inv_sqrt_2 * (temp_i - temp_j);
            });
        });
}

void update_with_RX(Kokkos::View<CTYPE *> state, UINT n, double angle, UINT target)
{
    double angle_half = angle / 2, sin_half = std::sin(angle_half), cos_half = std::cos(angle_half);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0},
                                                  {1ULL << (n - target - 1), 1ULL << target});

    Kokkos::parallel_for(
        "RX_gate", policy, KOKKOS_LAMBDA(ITYPE upper_bit, ITYPE lower_bit) {
            ITYPE i = (upper_bit << (target + 1)) | lower_bit;
            ITYPE j = i | (1ULL << target);
            CTYPE temp_i = state[i];
            CTYPE temp_j = state[j];
            state[i] = cos_half * temp_i + CTYPE(0, 1) * sin_half * temp_j;
            state[j] = cos_half * temp_j + CTYPE(0, 1) * sin_half * temp_i;
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
                    state(i, sample) = cos_half * tmp_i + CTYPE(0, 1) * sin_half * tmp_j;
                } else {
                    state(i, sample) = CTYPE(0, 1) * sin_half * tmp_j + cos_half * tmp_i;
                }
            });
        });
}

void update_with_RX_batched_simple(Kokkos::View<CTYPE **> state, UINT n, double angle, UINT target)
{
    double angle_half = angle / 2, sin_half = std::sin(angle_half), cos_half = std::cos(angle_half);

    Kokkos::parallel_for(
        "RX_gate_batched", Kokkos::TeamPolicy(state.extent(1), 1 << 6),
        KOKKOS_LAMBDA(const TeamHandle &member) {
            int sample = member.league_rank();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(member, 1ULL << (n - 1)), [=](ITYPE i) {
                ITYPE j = i | (1ULL << target);

                CTYPE tmp_i = state(i, sample);
                CTYPE tmp_j = state(j, sample);

                state(i, sample) = cos_half * tmp_i + CTYPE(0, 1) * sin_half * tmp_j;
                state(j, sample) = cos_half * tmp_j + CTYPE(0, 1) * sin_half * tmp_i;
            });
        });
}

void update_with_RX_batched(Kokkos::View<CTYPE **> state, UINT n, double angle, UINT target)
{
    if (target > 5) {
        update_with_RX_batched_simple(state, n, angle, target);
    } else {
        update_with_RX_batched_shuffle(state, n, angle, target);
    }
}

void update_with_RY(Kokkos::View<CTYPE *> state, UINT n, double angle, UINT target)
{
    double angle_half = angle / 2, sin_half = std::sin(angle_half), cos_half = std::cos(angle_half);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0},
                                                  {1ULL << (n - target - 1), 1ULL << target});

    Kokkos::parallel_for(
        "RY_gate", policy, KOKKOS_LAMBDA(ITYPE upper_bit, ITYPE lower_bit) {
            ITYPE i = (upper_bit << (target + 1)) | lower_bit;
            ITYPE j = i | (1ULL << target);
            CTYPE temp_i = state[i];
            CTYPE temp_j = state[j];
            state[i] = cos_half * temp_i + sin_half * temp_j;
            state[j] = cos_half * temp_j - sin_half * temp_i;
        });
}

void update_with_RY_batched(Kokkos::View<CTYPE **> state, UINT n, double angle, UINT target)
{
    double angle_half = angle / 2, sin_half = std::sin(angle_half), cos_half = std::cos(angle_half);

    Kokkos::TeamPolicy outer_policy(state.extent(1), Kokkos::AUTO);

    Kokkos::parallel_for(
        "RY_gate_batched", outer_policy, KOKKOS_LAMBDA(const TeamHandle &member) {
            int sample = member.league_rank();

            Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle> inner_policy(
                member, 1ULL << (n - target - 1), 1ULL << target);

            Kokkos::parallel_for(inner_policy, [=](ITYPE upper_bit, ITYPE lower_bit) {
                ITYPE i = (upper_bit << (target + 1)) | lower_bit;
                ITYPE j = i | (1ULL << target);
                CTYPE temp_i = state(i, sample);
                CTYPE temp_j = state(j, sample);
                state(i, sample) = cos_half * temp_i + sin_half * temp_j;
                state(j, sample) = cos_half * temp_j - sin_half * temp_i;
            });
        });
}

void update_with_RZ(Kokkos::View<CTYPE *> state, UINT n, double angle, UINT target)
{
    double angle_half = angle / 2;
    CTYPE phase0 = std::exp(std::complex<double>(0, angle_half)),
          phase1 = std::exp(std::complex<double>(0, -angle_half));
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0},
                                                  {1ULL << (n - target - 1), 1ULL << target});

    Kokkos::parallel_for(
        "RZ_gate", policy, KOKKOS_LAMBDA(ITYPE upper_bit, ITYPE lower_bit) {
            ITYPE i = (upper_bit << (target + 1)) | lower_bit;
            ITYPE j = i | (1ULL << target);
            state[i] *= phase0;
            state[j] *= phase1;
        });
}

void update_with_RZ_batched(Kokkos::View<CTYPE **> state, UINT n, double angle, UINT target)
{
    double angle_half = angle / 2;
    CTYPE phase0 = std::exp(std::complex<double>(0, angle_half)),
          phase1 = std::exp(std::complex<double>(0, -angle_half));
    Kokkos::TeamPolicy outer_policy(state.extent(1), Kokkos::AUTO);

    Kokkos::parallel_for(
        "RZ_gate_batched", outer_policy, KOKKOS_LAMBDA(const TeamHandle &member) {
            int sample = member.league_rank();

            Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle> inner_policy(
                member, 1ULL << (n - target - 1), 1ULL << target);

            Kokkos::parallel_for(inner_policy, [=](ITYPE upper_bit, ITYPE lower_bit) {
                ITYPE i = (upper_bit << (target + 1)) | lower_bit;
                ITYPE j = i | (1ULL << target);
                state(i, sample) *= phase0;
                state(j, sample) *= phase1;
            });
        });
}

void update_with_SWAP(Kokkos::View<CTYPE *> state, UINT n, UINT target0, UINT target1)
{
    if (target0 > target1) Kokkos::Experimental::swap(target0, target1);
    ITYPE mask0 = 1ULL << target0;
    ITYPE mask1 = 1ULL << target1;
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
        {0, 0, 0}, {1ULL << (n - target1 - 1), 1ULL << (target1 - target0 - 1), 1ULL << target0});

    Kokkos::parallel_for(
        "SWAP_gate", policy, KOKKOS_LAMBDA(ITYPE upper_bit, ITYPE middle_bit, ITYPE lower_bit) {
            ITYPE i = (upper_bit << (target1 + 1)) | (middle_bit << (target0 + 1)) | lower_bit;
            Kokkos::Experimental::swap(state[i | mask0], state[i | mask1]);
        });
}

void update_with_CNOT(Kokkos::View<CTYPE *> state, UINT n, UINT control, UINT target)
{
    ITYPE mask_control = 1ULL << control;
    ITYPE mask_target = 1ULL << target;
    ITYPE ub = std::max(target, control);
    ITYPE lb = std::min(target, control);
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
        {0, 0, 0}, {1ULL << (n - ub - 1), 1ULL << (ub - lb - 1), 1ULL << lb});

    Kokkos::parallel_for(
        "CNOT_gate", policy, KOKKOS_LAMBDA(ITYPE upper_bit, ITYPE middle_bit, ITYPE lower_bit) {
            ITYPE i = (upper_bit << (ub + 1)) | (middle_bit << (lb + 1)) | lower_bit | mask_control;
            Kokkos::Experimental::swap(state[i], state[i | mask_target]);
        });
}

void update_with_CNOT_batched(Kokkos::View<CTYPE **> state, UINT n, UINT control, UINT target)
{
    ITYPE mask_control = 1ULL << control;
    ITYPE mask_target = 1ULL << target;
    ITYPE ub = std::max(target, control);
    ITYPE lb = std::min(target, control);

    Kokkos::TeamPolicy outer_policy(state.extent(1), Kokkos::AUTO);

    Kokkos::parallel_for(
        "CNOT_gate_batched", outer_policy, KOKKOS_LAMBDA(const TeamHandle &member) {
            int sample = member.league_rank();

            Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle> inner_policy(
                member, 1ULL << (n - ub - 1), 1ULL << (ub - lb - 1), 1ULL << lb);
            Kokkos::parallel_for(inner_policy, [=](ITYPE upper_bit, ITYPE middle_bit,
                                                   ITYPE lower_bit) {
                ITYPE i =
                    (upper_bit << (ub + 1)) | (middle_bit << (lb + 1)) | lower_bit | mask_control;
                Kokkos::Experimental::swap(state(i, sample), state(i | mask_target, sample));
            });
        });
}

void update_with_CZ(Kokkos::View<CTYPE *> state, UINT n, UINT control, UINT target)
{
    ITYPE mask_control = 1ULL << control;
    ITYPE mask_target = 1ULL << target;
    ITYPE ub = std::max(target, control);
    ITYPE lb = std::min(target, control);
    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
        {0, 0, 0}, {1ULL << (n - ub - 1), 1ULL << (ub - lb - 1), 1ULL << lb});

    Kokkos::parallel_for(
        "CZ_gate", policy, KOKKOS_LAMBDA(ITYPE upper_bit, ITYPE middle_bit, ITYPE lower_bit) {
            ITYPE i = (upper_bit << (ub + 1)) | (middle_bit << (lb + 1)) | lower_bit |
                      mask_control | mask_target;
            state[i] *= -1;
        });
}

void update_with_CZ_batched(Kokkos::View<CTYPE **> state, UINT n, UINT control, UINT target)
{
    ITYPE mask_control = 1ULL << control;
    ITYPE mask_target = 1ULL << target;
    ITYPE ub = std::max(target, control);
    ITYPE lb = std::min(target, control);

    Kokkos::TeamPolicy outer_policy(state.extent(1), Kokkos::AUTO);

    Kokkos::parallel_for(
        "CZ_gate_batched", outer_policy, KOKKOS_LAMBDA(const TeamHandle &member) {
            int sample = member.league_rank();

            Kokkos::TeamThreadMDRange<Kokkos::Rank<3>, TeamHandle> inner_policy(
                member, 1ULL << (n - ub - 1), 1ULL << (ub - lb - 1), 1ULL << lb);
            Kokkos::parallel_for(inner_policy,
                                 [=](ITYPE upper_bit, ITYPE middle_bit, ITYPE lower_bit) {
                                     ITYPE i = (upper_bit << (ub + 1)) | (middle_bit << (lb + 1)) |
                                               lower_bit | mask_control | mask_target;
                                     state(i, sample) *= -1;
                                 });
        });
}

void update_with_dense_matrix(Kokkos::View<CTYPE *> state, UINT n,
                              Kokkos::View<const UINT *> control_list,
                              Kokkos::View<const UINT *> control_value,
                              Kokkos::View<const UINT *> target_list,
                              Kokkos::View<const CTYPE **> matrix)
{
    Kokkos::View<CTYPE *> new_state("new_state", 1ULL << n);

    int num_control = control_list.size(), num_target = target_list.size();
    int control_mask = 0, target_mask = 0;
    for (int i = 0; i < (int)control_list.size(); ++i) control_mask |= 1 << control_list[i];
    for (int i = 0; i < (int)target_list.size(); ++i) target_mask |= 1 << target_list[i];

    Kokkos::MDRangePolicy<Kokkos::Rank<3>> policy(
        {0, 0, 0},
        {1ULL << (n - num_control - num_target), 1ULL << num_target, 1ULL << num_target});

    Kokkos::parallel_for(
        "dense_gate", policy, KOKKOS_LAMBDA(ITYPE outer_bit, ITYPE target_bit1, ITYPE target_bit2) {
            ITYPE iter_row = 0, iter_col = 0;
            int outer_idx = 0, control_idx = 0, target_idx = 0;
            for (int i = 0; i < n; i++) {
                if (control_mask >> i & 1) {
                    iter_row |= control_value[control_idx] << i;
                    iter_col |= control_value[control_idx] << i;
                    ++control_idx;
                } else if (target_mask >> i & 1) {
                    iter_row |= (target_bit1 >> target_idx & 1) << i;
                    iter_col |= (target_bit2 >> target_idx & 1) << i;
                    ++target_idx;
                } else {
                    iter_row |= (outer_bit >> outer_idx & 1) << i;
                    iter_col |= (outer_bit >> outer_idx & 1) << i;
                    ++outer_idx;
                }
            }
            Kokkos::atomic_add(&new_state(iter_row), matrix(iter_row, iter_col) * state(iter_col));
        });
    Kokkos::deep_copy(state, new_state);
}

void update_with_depoloarizing_noise(Kokkos::View<CTYPE **> state, UINT n, UINT target, double prob,
                                     Kokkos::Random_XorShift64_Pool<> random_pool)
{
    Kokkos::TeamPolicy outer_policy(state.extent(1), Kokkos::AUTO);

    Kokkos::parallel_for(
        "depolarizing_noize_gate_batched", outer_policy, KOKKOS_LAMBDA(const TeamHandle &member) {
            int sample = member.league_rank();

            double r;
            Kokkos::single(
                Kokkos::PerTeam(member),
                [=](double &r) {
                    auto generator = random_pool.get_state();
                    r = generator.drand();
                    random_pool.free_state(generator);
                },
                r);

            if (r >= prob) return;

            Kokkos::TeamThreadMDRange<Kokkos::Rank<2>, TeamHandle> inner_policy(
                member, 1ULL << (n - target - 1), 1ULL << target);

            Kokkos::parallel_for(inner_policy, [=](ITYPE upper_bit, ITYPE lower_bit) {
                ITYPE i = (upper_bit << (target + 1)) | lower_bit;
                ITYPE j = i | (1ULL << target);

                if (r < prob / 3) {
                    // Apply X gate
                    Kokkos::Experimental::swap(state(i, sample), state(j, sample));
                } else if (r < prob * 2 / 3) {
                    // Apply Y gate
                    CTYPE temp_i = state(i, sample);
                    CTYPE temp_j = state(j, sample);
                    state(i, sample) = CTYPE(0, -1) * temp_j;
                    state(j, sample) = CTYPE(0, 1) * temp_i;
                } else if (r < prob) {
                    // Apply Z gate
                    state(j, sample) = -state(j, sample);
                } else {
                    // No noise
                }
            });
        });
}
