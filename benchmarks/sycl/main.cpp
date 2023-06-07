#include <sycl/sycl.hpp>
#include <complex>
#include <iostream>

using UINT = unsigned int;
using ITYPE = unsigned long long;
using CTYPE = std::complex<double>;

void update_with_x(sycl::queue& q, sycl::buffer<CTYPE, 1>& state_sycl, UINT n, UINT target) {
    std::vector<CTYPE> v(10);
    q.submit([&](sycl::handler& h) {
        auto state_acc = state_sycl.get_access<sycl::access::mode::read_write>(h);
        h.parallel_for<class nstream>(sycl::range<2>(1ULL << (n - target - 1), 1ULL << target), [=](sycl::id<2> it) {
            ITYPE i = (it[0] << (target + 1)) | it[1];
            ITYPE j = i | (1ULL << target);
            CTYPE tmp = state_acc[i];
            state_acc[i] = state_acc[j];
            state_acc[j] = tmp;
        });
    });
}

int main() {
    constexpr int n = 4;
    
    sycl::queue q(sycl::default_selector_v);

    std::vector<CTYPE> init_state(1ULL << n);
    for(int i = 0; i < 1ULL << n; i++) init_state[i] = i;
    auto state_sycl = sycl::buffer(init_state.data(), sycl::range(1ULL << n));

    update_with_x(q, state_sycl, n, 1);

    q.wait();
    for(int i = 0; i < 1ULL << n; i++) std::cout << ' ' << init_state[i];
    std::cout << std::endl;
}
