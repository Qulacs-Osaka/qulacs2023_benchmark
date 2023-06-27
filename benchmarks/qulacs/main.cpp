#include <chrono>
#include <thread>
#include <iostream>

double measure() {
    const auto start = std::chrono::system_clock::now();
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    const auto end = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <n_qubits> <n_repeats>" << std::endl;
        return 1;
    }

    const auto n_qubits = std::strtoul(argv[1], nullptr, 10);
    const auto n_repeats = std::strtoul(argv[2], nullptr, 10);

    for (int i = 0; i < n_repeats; i++) {
        std::cout << measure()<< " ";
    }
    std::cout << std::endl;

    return 0;
}
