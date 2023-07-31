#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

using UINT = unsigned int;
using ITYPE = unsigned long long;
using CTYPE = Kokkos::complex<double>;

void set_zero_state(Kokkos::View<CTYPE *> state, UINT n);
void set_zero_state_batched(Kokkos::View<CTYPE **> state, UINT n);
void update_with_X(Kokkos::View<CTYPE *> state, UINT n, UINT target);
void update_with_H(Kokkos::View<CTYPE *> state, UINT n, UINT target);
void update_with_H_batched(Kokkos::View<CTYPE **> state, UINT n, UINT target);
void update_with_RX(Kokkos::View<CTYPE *> state, UINT n, double angle, UINT target);
void update_with_RX_batched(Kokkos::View<CTYPE **> state, UINT n, double angle, UINT target);
void update_with_RY(Kokkos::View<CTYPE *> state, UINT n, double angle, UINT target);
void update_with_RY_batched(Kokkos::View<CTYPE **> state, UINT n, double angle, UINT target);
void update_with_RZ(Kokkos::View<CTYPE *> state, UINT n, double angle, UINT target);
void update_with_RZ_batched(Kokkos::View<CTYPE **> state, UINT n, double angle, UINT target);
void update_with_SWAP(Kokkos::View<CTYPE *> state, UINT n, UINT target0, UINT target1);
void update_with_CNOT(Kokkos::View<CTYPE *> state, UINT n, UINT control, UINT target);
void update_with_CNOT_batched(Kokkos::View<CTYPE **> state, UINT n, UINT control, UINT target);
void update_with_CZ(Kokkos::View<CTYPE *> state, UINT n, UINT control, UINT target);
void update_with_CZ_batched(Kokkos::View<CTYPE **> state, UINT n, UINT control, UINT target);
void update_with_dense_matrix(Kokkos::View<CTYPE *> state, UINT n,
                              Kokkos::View<const UINT *> control_list,
                              Kokkos::View<const UINT *> control_value,
                              Kokkos::View<const UINT *> target_list,
                              Kokkos::View<const CTYPE **> matrix);
void update_with_depoloarizing_noise(Kokkos::View<CTYPE **> state, UINT n, UINT target, double prob,
                                     Kokkos::Random_XorShift64_Pool<> random_pool);
