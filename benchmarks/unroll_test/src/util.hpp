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

using UINT = unsigned int;
using ITYPE = unsigned long long;
using CTYPE = Kokkos::complex<double>;
using TeamHandle = Kokkos::TeamPolicy<>::member_type;

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

Kokkos::View<CTYPE **> make_random_batched_state(int n_qubits, int batch_size) {
    unsigned seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    Kokkos::Random_XorShift64_Pool<> random_pool(seed);

    Kokkos::View<CTYPE **> state("state", 1ULL << n_qubits, batch_size);
    Kokkos::MDRangePolicy<Kokkos::Rank<2>> policy({0, 0}, {1ULL << n_qubits, batch_size});
    Kokkos::parallel_for(policy, KOKKOS_LAMBDA(int i, int j) {
        auto random_generator = random_pool.get_state();
        double real_part = Kokkos::rand<decltype(random_generator), double>::draw(random_generator);
        double imag_part = Kokkos::rand<decltype(random_generator), double>::draw(random_generator);
        state(i, j) = CTYPE(real_part, imag_part);
        random_pool.free_state(random_generator);
    });
    return state;
}

bool is_equal(const Kokkos::View<CTYPE*> &state1, const Kokkos::View<CTYPE*> &state2, int n_qubits) {
    if (n_qubits <= 12) {
        auto host_mirror1 = Kokkos::create_mirror_view(state1);
        auto host_mirror2 = Kokkos::create_mirror_view(state2);
        Kokkos::deep_copy(host_mirror1, state1);
        Kokkos::deep_copy(host_mirror2, state2);
        int cnt = 0;
        for (int i = 0; i < (1 << n_qubits); ++i) {
            if (std::abs(host_mirror1[i].real() - host_mirror2[i].real()) > 1e-5
                || std::abs(host_mirror1[i].imag() - host_mirror2[i].imag()) > 1e-5) {
                std::cout << "index(" << i << "): " << host_mirror1[i] << ", " << host_mirror2[i] << std::endl;
                cnt++;
            }
        }
        std::cerr << "THERE WERE " << cnt << " VALUE ERRORS." << std::endl;
        return cnt == 0;
    } else {
        std::cerr << "TEST WAS SKIPPED." << std::endl;
        return true;
    }
}

bool is_equal_batched(const Kokkos::View<CTYPE **> &state1, const Kokkos::View<CTYPE **> &state2, int n_qubits, int batch_size) {
    auto host_mirror1 = Kokkos::create_mirror_view(state1);
    auto host_mirror2 = Kokkos::create_mirror_view(state2);
    Kokkos::deep_copy(host_mirror1, state1);
    Kokkos::deep_copy(host_mirror2, state2);
    int cnt = 0;
    for (int i = 0; i < (1 << n_qubits); ++i) {
        for (int j = 0; j < batch_size; ++j) {
            if (std::abs(host_mirror1(i, j).real() - host_mirror2(i, j).real()) > 1e-5
                || std::abs(host_mirror1(i, j).imag() - host_mirror2(i, j).imag()) > 1e-5) {
                std::cout << "index(" << i << "," << j << "): " << host_mirror1(i, j) << ", " << host_mirror2(i, j) << std::endl;
                cnt++;
            }
        }
    }
    std::cerr << "THERE WERE " << cnt << " VALUE ERRORS." << std::endl;
    return cnt == 0;
}

std::pair<double, double> x_bench(int n_qubits, int rep, auto f1, auto f2) {
    std::uniform_int_distribution<> target_gen(0, n_qubits - 1);
    std::mt19937 mt(std::random_device{}());

    auto state(make_random_state(n_qubits));
    Kokkos::View<CTYPE *> state_clone("state_clone", 1 << n_qubits);
    Kokkos::deep_copy(state_clone, state);
    Kokkos::fence();

    std::vector<int> tars;
    std::vector<double> agls;
    for (int i = 0; i < rep; ++i) {
        tars.push_back(target_gen(mt));
    }
    
    auto start_time1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rep; ++i)
        f1(state, n_qubits, tars[i]);
    Kokkos::fence();
    auto end_time1 = std::chrono::high_resolution_clock::now();

    auto start_time2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rep; ++i)
        f2(state_clone, n_qubits, tars[i]);
    Kokkos::fence();
    auto end_time2 = std::chrono::high_resolution_clock::now();

    if(!is_equal(state, state_clone, n_qubits)) {
        exit(1);
    }

    return {std::chrono::duration_cast<std::chrono::nanoseconds>(end_time1 - start_time1).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time2 - start_time2).count()};
}

std::pair<double, double> Rx_bench(int n_qubits, int rep, auto f1, auto f2) {
    std::uniform_int_distribution<> target_gen(0, n_qubits - 1);
    std::uniform_real_distribution<> angle_gen(0, 2 * M_PI);
    std::mt19937 mt(std::random_device{}());

    auto state(make_random_state(n_qubits));
    Kokkos::View<CTYPE *> state_clone("state_clone", 1 << n_qubits);
    Kokkos::deep_copy(state_clone, state);
    Kokkos::fence();

    std::vector<int> tars;
    std::vector<double> agls;
    for (int i = 0; i < rep; ++i) {
        tars.push_back(target_gen(mt));
        agls.push_back(angle_gen(mt));
    }
    
    auto start_time1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rep; ++i)
        f1(state, n_qubits, agls[i], tars[i]);
    Kokkos::fence();
    auto end_time1 = std::chrono::high_resolution_clock::now();

    auto start_time2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rep; ++i)
        f2(state_clone, n_qubits, agls[i], tars[i]);
    Kokkos::fence();
    auto end_time2 = std::chrono::high_resolution_clock::now();

    if(!is_equal(state, state_clone, n_qubits)) {
        exit(1);
    }

    return {std::chrono::duration_cast<std::chrono::nanoseconds>(end_time1 - start_time1).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time2 - start_time2).count()};
}

std::pair<double, double> target_fixed_bench(int n_qubits, int rep, auto f1, auto f2, int target) {
    std::uniform_real_distribution<> angle_gen(0, 2 * M_PI);
    std::mt19937 mt(std::random_device{}());

    auto state(make_random_state(n_qubits));
    Kokkos::View<CTYPE *> state_clone("state_clone", 1 << n_qubits);
    Kokkos::deep_copy(state_clone, state);
    Kokkos::fence();

    std::vector<double> agls;
    for (int i = 0; i < rep; ++i) {
        agls.push_back(angle_gen(mt));
    }
    
    auto start_time1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rep; ++i)
        f1(state, n_qubits, agls[i], target);
    Kokkos::fence();
    auto end_time1 = std::chrono::high_resolution_clock::now();

    auto start_time2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rep; ++i)
        f2(state_clone, n_qubits, agls[i], target);
    Kokkos::fence();
    auto end_time2 = std::chrono::high_resolution_clock::now();

    if(!is_equal(state, state_clone, n_qubits)) {
        exit(1);
    }

    return {std::chrono::duration_cast<std::chrono::nanoseconds>(end_time1 - start_time1).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time2 - start_time2).count()};
}


std::pair<double, double> batched_bench(int n_qubits, int batch_size, int rep, auto f1, auto f2) {
    std::uniform_int_distribution<> target_gen(0, Kokkos::min(4, n_qubits - 1)), circuit_gen(0, 2);
    std::uniform_real_distribution<> angle_gen(0, 2 * M_PI);
    std::mt19937 mt(std::random_device{}());

    auto state(make_random_batched_state(n_qubits, batch_size));
    Kokkos::View<CTYPE **> state_clone("state_clone", 1ULL << n_qubits, batch_size);
    Kokkos::deep_copy(state_clone, state);
    Kokkos::fence();

    std::vector<int> tars;
    std::vector<double> agls;
    for (int i = 0; i < rep; ++i) {
        tars.push_back(target_gen(mt));
        agls.push_back(angle_gen(mt));
    }

    auto start_time1 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rep; ++i)
        f1(state, n_qubits, agls[i], tars[i]);
    Kokkos::fence();
    auto end_time1 = std::chrono::high_resolution_clock::now();

    auto start_time2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < rep; ++i)
        f2(state_clone, n_qubits, agls[i], tars[i]);
    Kokkos::fence();
    auto end_time2 = std::chrono::high_resolution_clock::now();

    if(!is_equal_batched(state, state_clone, n_qubits, 1000)) {
        exit(1);
    }

    return {std::chrono::duration_cast<std::chrono::nanoseconds>(end_time1 - start_time1).count(),
        std::chrono::duration_cast<std::chrono::nanoseconds>(end_time2 - start_time2).count()};
}