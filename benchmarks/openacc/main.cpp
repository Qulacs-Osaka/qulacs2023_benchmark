#include <complex>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>

using UINT = uint64_t;
using Complex = std::complex<double>;

void update_with_x(Complex &a0, Complex &a1)
{
    std::swap(a0, a1);
}

void update_with_y(Complex &a0, Complex &a1)
{
    Complex i(0, 1);
    std::swap(a0, a1);
    a0 *= -i;
    a1 *= i;
}

void update_with_z(Complex &a0, Complex &a1)
{
    a1 *= -1;
}

void update_with_h(Complex &a0, Complex &a1)
{
    Complex a0_old = a0;
    Complex a1_old = a1;
    a0 = (a0_old + a1_old) / sqrt(2);
    a1 = (a0_old - a1_old) / sqrt(2);
}

void update_with_single_control_single_target_gate(std::vector<Complex> &state, int n, int control, int target, void (*gate)(Complex &, Complex &))
{
    int range1 = 1 << (n - target - 1);
    int range2 = 1 << target;

    Complex *state_ptr = state.data();
#pragma acc data copy(state_ptr[0 : 1 << n])
    {
#pragma acc parallel loop
        for (int idx1 = 0; idx1 < range1; idx1++)
        {
            for (int idx2 = 0; idx2 < range2; idx2++)
            {
                int i = (idx1 << (target + 1)) | idx2;
                int j = i | (1 << target);
                gate(state_ptr[i], state_ptr[j]);
            }
        }
    }
}

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0] << " <n_qubits> <n_repeats>" << std::endl;
        return 1;
    }
    const auto n_qubits = std::strtoul(argv[1], nullptr, 10);
    const auto n_repeats = std::strtoul(argv[2], nullptr, 10);
    std::vector<unsigned long long> execution_time(n_repeats);

    for (int repeat_itr = 0; repeat_itr < n_repeats; repeat_itr++)
    {
        auto st_time = std::chrono::system_clock::now();

        std::vector<Complex> state(1ULL << n_qubits, 0);
        for (int i = 0; i < 1 << n_qubits; i++)
            state[i] = i;

        update_with_single_control_single_target_gate(state, n_qubits, 0, 1, update_with_x);
        update_with_single_control_single_target_gate(state, n_qubits, 0, 2, update_with_x);
        update_with_single_control_single_target_gate(state, n_qubits, 0, 3, update_with_x);
        update_with_single_control_single_target_gate(state, n_qubits, 0, 4, update_with_x);

        auto ed_time = std::chrono::system_clock::now();
        execution_time[repeat_itr] = std::chrono::duration_cast<std::chrono::nanoseconds>(ed_time - st_time).count();
    }

    std::ofstream ofs("durations.txt");
    if (!ofs.is_open())
    {
        std::cerr << "Failed to open file" << std::endl;
        return 1;
    }

    for (int i = 0; i < n_repeats; i++)
    {
        ofs << execution_time[i] << " ";
    }
    ofs << std::endl;

    return 0;
}