#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include "gate.hpp"

int main(int argc, char **argv)
{
    Kokkos::ScopeGuard kokkos(argc, argv);

    return doctest::Context(argc, argv).run();
}

TEST_CASE("Test H gate")
{
    const int N = 4;
    Kokkos::View<CTYPE *> state("state", 1ULL << N);

    set_zero_state(state, N);

    for (int i = 0; i < N; i++) {
        update_with_H(state, N, i);
    }

    auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);

    for (int i = 0; i < 1ULL << N; i++) {
        REQUIRE(mirror(i).real() == doctest::Approx(0.25));
        REQUIRE(mirror(i).imag() == doctest::Approx(0.0));
    }
}

TEST_CASE("Test batched H gate")
{
    const int N = 4;
    const int BATCH_SIZE = 10;
    Kokkos::View<CTYPE **> state("state", 1ULL << N, BATCH_SIZE);

    set_zero_state_batched(state, N);

    for (int i = 0; i < N; i++) {
        update_with_H_batched(state, N, i);
    }

    auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        for (int i = 0; i < 1ULL << N; i++) {
            REQUIRE(mirror(i, sample).real() == doctest::Approx(0.25));
            REQUIRE(mirror(i, sample).imag() == doctest::Approx(0.0));
        }
    }
}

TEST_CASE("Test RX gate")
{
    const int N = 2;
    Kokkos::View<CTYPE *> state("state", 1ULL << N);

    set_zero_state(state, N);

    for (int i = 0; i < N; i++) {
        update_with_RX(state, N, 1.0, i);
    }

    auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);

    REQUIRE(mirror(0).real() == doctest::Approx(0.7701511529340699));
    REQUIRE(mirror(0).imag() == doctest::Approx(0.0));
    REQUIRE(mirror(1).real() == doctest::Approx(0.0));
    REQUIRE(mirror(1).imag() == doctest::Approx(0.42073549240394825));
    REQUIRE(mirror(2).real() == doctest::Approx(0.0));
    REQUIRE(mirror(2).imag() == doctest::Approx(0.42073549240394825));
    REQUIRE(mirror(3).real() == doctest::Approx(-0.22984884706593015));
    REQUIRE(mirror(3).imag() == doctest::Approx(0.0));
}

TEST_CASE("Test batched RX gate")
{
    const int N = 2;
    const int BATCH_SIZE = 10;
    Kokkos::View<CTYPE **> state("state", 1ULL << N, BATCH_SIZE);

    set_zero_state_batched(state, N);

    for (int i = 0; i < N; i++) {
        update_with_RX_batched(state, N, 1.0, i);
    }

    auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        REQUIRE(mirror(0, sample).real() == doctest::Approx(0.7701511529340699));
        REQUIRE(mirror(0, sample).imag() == doctest::Approx(0.0));
        REQUIRE(mirror(1, sample).real() == doctest::Approx(0.0));
        REQUIRE(mirror(1, sample).imag() == doctest::Approx(0.42073549240394825));
        REQUIRE(mirror(2, sample).real() == doctest::Approx(0.0));
        REQUIRE(mirror(2, sample).imag() == doctest::Approx(0.42073549240394825));
        REQUIRE(mirror(3, sample).real() == doctest::Approx(-0.22984884706593015));
        REQUIRE(mirror(3, sample).imag() == doctest::Approx(0.0));
    }
}

TEST_CASE("Test RY gate")
{
    const int N = 2;
    Kokkos::View<CTYPE *> state("state", 1ULL << N);

    set_zero_state(state, N);

    for (int i = 0; i < N; i++) {
        update_with_RY(state, N, 1.0, i);
    }

    auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);

    REQUIRE(mirror(0).real() == doctest::Approx(0.7701511529340699));
    REQUIRE(mirror(0).imag() == doctest::Approx(0.0));
    REQUIRE(mirror(1).real() == doctest::Approx(-0.42073549240394825));
    REQUIRE(mirror(1).imag() == doctest::Approx(0.0));
    REQUIRE(mirror(2).real() == doctest::Approx(-0.42073549240394825));
    REQUIRE(mirror(2).imag() == doctest::Approx(0.0));
    REQUIRE(mirror(3).real() == doctest::Approx(0.22984884706593015));
    REQUIRE(mirror(3).imag() == doctest::Approx(0.0));
}

TEST_CASE("Test batched RY gate")
{
    const int N = 2;
    const int BATCH_SIZE = 10;
    Kokkos::View<CTYPE **> state("state", 1ULL << N, BATCH_SIZE);

    set_zero_state_batched(state, N);

    for (int i = 0; i < N; i++) {
        update_with_RY_batched(state, N, 1.0, i);
    }

    auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        REQUIRE(mirror(0, sample).real() == doctest::Approx(0.7701511529340699));
        REQUIRE(mirror(0, sample).imag() == doctest::Approx(0.0));
        REQUIRE(mirror(1, sample).real() == doctest::Approx(-0.42073549240394825));
        REQUIRE(mirror(1, sample).imag() == doctest::Approx(0.0));
        REQUIRE(mirror(2, sample).real() == doctest::Approx(-0.42073549240394825));
        REQUIRE(mirror(2, sample).imag() == doctest::Approx(0.0));
        REQUIRE(mirror(3, sample).real() == doctest::Approx(0.22984884706593015));
        REQUIRE(mirror(3, sample).imag() == doctest::Approx(0.0));
    }
}

TEST_CASE("Test RZ gate")
{
    const int N = 2;
    Kokkos::View<CTYPE *> state("state", 1ULL << N);

    set_zero_state(state, N);

    for (int i = 0; i < N; i++) {
        update_with_H(state, N, i);
    }

    for (int i = 0; i < N; i++) {
        update_with_RZ(state, N, 1.0, i);
    }

    auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);

    REQUIRE(mirror(0).real() == doctest::Approx(0.2701511529340698));
    REQUIRE(mirror(0).imag() == doctest::Approx(0.4207354924039482));
    REQUIRE(mirror(1).real() == doctest::Approx(0.49999999999999994));
    REQUIRE(mirror(1).imag() == doctest::Approx(0.0));
    REQUIRE(mirror(2).real() == doctest::Approx(0.49999999999999994));
    REQUIRE(mirror(2).imag() == doctest::Approx(0.0));
    REQUIRE(mirror(3).real() == doctest::Approx(0.2701511529340698));
    REQUIRE(mirror(3).imag() == doctest::Approx(-0.4207354924039482));
}

TEST_CASE("Test batched RZ gate")
{
    const int N = 2;
    const int BATCH_SIZE = 10;
    Kokkos::View<CTYPE **> state("state", 1ULL << N, BATCH_SIZE);

    set_zero_state_batched(state, N);

    for (int i = 0; i < N; i++) {
        update_with_H_batched(state, N, i);
    }

    for (int i = 0; i < N; i++) {
        update_with_RZ_batched(state, N, 1.0, i);
    }

    auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        REQUIRE(mirror(0, sample).real() == doctest::Approx(0.2701511529340698));
        REQUIRE(mirror(0, sample).imag() == doctest::Approx(0.4207354924039482));
        REQUIRE(mirror(1, sample).real() == doctest::Approx(0.49999999999999994));
        REQUIRE(mirror(1, sample).imag() == doctest::Approx(0.0));
        REQUIRE(mirror(2, sample).real() == doctest::Approx(0.49999999999999994));
        REQUIRE(mirror(2, sample).imag() == doctest::Approx(0.0));
        REQUIRE(mirror(3, sample).real() == doctest::Approx(0.2701511529340698));
        REQUIRE(mirror(3, sample).imag() == doctest::Approx(-0.4207354924039482));
    }
}

TEST_CASE("Test CZ gate")
{
    const int N = 3;
    Kokkos::View<CTYPE *> state("state", 1ULL << N);

    set_zero_state(state, N);

    for (int i = 0; i < N; i++) {
        update_with_H(state, N, i);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            update_with_CZ(state, N, i, j);
        }
    }

    auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);

    REQUIRE(mirror(0).real() == doctest::Approx(0.3535533905932737));
    REQUIRE(mirror(0).imag() == doctest::Approx(0.0));
    REQUIRE(mirror(1).real() == doctest::Approx(0.3535533905932737));
    REQUIRE(mirror(1).imag() == doctest::Approx(0.0));
    REQUIRE(mirror(2).real() == doctest::Approx(0.3535533905932737));
    REQUIRE(mirror(2).imag() == doctest::Approx(0.0));
    REQUIRE(mirror(3).real() == doctest::Approx(-0.3535533905932737));
    REQUIRE(mirror(3).imag() == doctest::Approx(-0.0));
    REQUIRE(mirror(4).real() == doctest::Approx(0.3535533905932737));
    REQUIRE(mirror(4).imag() == doctest::Approx(0.0));
    REQUIRE(mirror(5).real() == doctest::Approx(-0.3535533905932737));
    REQUIRE(mirror(5).imag() == doctest::Approx(-0.0));
    REQUIRE(mirror(6).real() == doctest::Approx(-0.3535533905932737));
    REQUIRE(mirror(6).imag() == doctest::Approx(-0.0));
    REQUIRE(mirror(7).real() == doctest::Approx(-0.3535533905932737));
    REQUIRE(mirror(7).imag() == doctest::Approx(-0.0));
}

TEST_CASE("Test batched CZ gate")
{
    const int N = 3;
    const int BATCH_SIZE = 10;
    Kokkos::View<CTYPE **> state("state", 1ULL << N, BATCH_SIZE);

    set_zero_state_batched(state, N);

    for (int i = 0; i < N; i++) {
        update_with_H_batched(state, N, i);
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            update_with_CZ_batched(state, N, i, j);
        }
    }

    auto mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), state);

    for (int sample = 0; sample < BATCH_SIZE; sample++) {
        REQUIRE(mirror(0, sample).real() == doctest::Approx(0.3535533905932737));
        REQUIRE(mirror(0, sample).imag() == doctest::Approx(0.0));
        REQUIRE(mirror(1, sample).real() == doctest::Approx(0.3535533905932737));
        REQUIRE(mirror(1, sample).imag() == doctest::Approx(0.0));
        REQUIRE(mirror(2, sample).real() == doctest::Approx(0.3535533905932737));
        REQUIRE(mirror(2, sample).imag() == doctest::Approx(0.0));
        REQUIRE(mirror(3, sample).real() == doctest::Approx(-0.3535533905932737));
        REQUIRE(mirror(3, sample).imag() == doctest::Approx(-0.0));
        REQUIRE(mirror(4, sample).real() == doctest::Approx(0.3535533905932737));
        REQUIRE(mirror(4, sample).imag() == doctest::Approx(0.0));
        REQUIRE(mirror(5, sample).real() == doctest::Approx(-0.3535533905932737));
        REQUIRE(mirror(5, sample).imag() == doctest::Approx(-0.0));
        REQUIRE(mirror(6, sample).real() == doctest::Approx(-0.3535533905932737));
        REQUIRE(mirror(6, sample).imag() == doctest::Approx(-0.0));
        REQUIRE(mirror(7, sample).real() == doctest::Approx(-0.3535533905932737));
        REQUIRE(mirror(7, sample).imag() == doctest::Approx(-0.0));
    }
}
