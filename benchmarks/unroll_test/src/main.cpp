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
#include "gate.hpp"

int main(int argc, char *argv[]) {
Kokkos::initialize();
{    

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <n_qubits>" << std::endl;
        return 1;
    }

    const auto n_qubits = std::strtoul(argv[1], nullptr, 10);
    warmup(n_qubits, 1000, update_with_x);
    auto [t1, t2] = x_bench(n_qubits, 1000, update_with_x, update_with_x_unroll);
    //auto [t1, t2] = Rx_bench(n_qubits, 1000, update_with_Rx, update_with_Rx_unroll);
    std::cout << t1 / 1000000. << " " << t2 / 1000000. << std::endl;

}
Kokkos::finalize();
}
