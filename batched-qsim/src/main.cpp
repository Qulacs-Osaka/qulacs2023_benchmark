#include <random>

#include <Kokkos_Core.hpp>
#include <argparse/argparse.hpp>

#include "gate.hpp"

void run_single_batch(Kokkos::View<CTYPE **> state, int N_QUBITS, int DEPTH, std::mt19937 &engine,
                      std::uniform_real_distribution<double> &dist)
{
    for (int d = 0; d < DEPTH; d++) {
        for (int i = 0; i < N_QUBITS; i++) {
            update_with_RX_batched(state, N_QUBITS, dist(engine), i);
            // update_with_depoloarizing_noise(state, N_QUBITS, i, 0.1, random_pool);
        }
    }
}

int main(int argc, char *argv[])
{
    Kokkos::ScopeGuard kokkos(argc, argv);

    argparse::ArgumentParser program("qsim-bench");

    program.add_argument("--samples").help("# of samples").default_value(1000).scan<'i', int>();
    program.add_argument("--depth").help("Depth of the circuit").default_value(10).scan<'i', int>();
    program.add_argument("--qubits").help("# of qubits").default_value(10).scan<'i', int>();
    program.add_argument("--batch-size").help("Batch size").default_value(1000).scan<'i', int>();
    program.add_argument("--trials").help("# of trials").default_value(10).scan<'i', int>();

    try {
        program.parse_args(argc, argv);
    } catch (const std::runtime_error &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const int N_SAMPLES = program.get<int>("--samples");
    const int DEPTH = program.get<int>("--depth");
    const int N_QUBITS = program.get<int>("--qubits");
    const int BATCH_SIZE = program.get<int>("--batch-size");
    const int N_TRIALS = program.get<int>("--trials");

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<double> dist(0.0, M_PI);
    std::vector<double> durations;
    Kokkos::View<CTYPE **> state("state", 1ULL << N_QUBITS, BATCH_SIZE);

    // Warmup run
    for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
        run_single_batch(state, N_QUBITS, DEPTH, engine, dist);
    }

    for (int trial = 0; trial < N_TRIALS; trial++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int batch = 0; batch < N_SAMPLES; batch += BATCH_SIZE) {
            run_single_batch(state, N_QUBITS, DEPTH, engine, dist);
        }
        Kokkos::fence();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration =
            std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        durations.push_back(duration.count() / 1e6);
    }

    double average = 0.0;
    for (int trial = 0; trial < N_TRIALS; trial++) {
        average += durations[trial];
    }
    average /= N_TRIALS;

    double variance = 0.0;
    for (int trial = 0; trial < N_TRIALS; trial++) {
        variance += (durations[trial] - average) * (durations[trial] - average);
    }
    variance = variance / N_TRIALS;

    std::cout << average << "±" << std::sqrt(variance) << std::endl;

    return 0;
}
