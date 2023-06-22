#include <complex>
#include <iostream>
#include <vector>

using UINT = unsigned int;
using ITYPE = unsigned long long;
using CTYPE = std::complex<double>;

void update_with_x(std::vector<CTYPE> &state, int n, int target)
{
    int range1 = 1 << (n - target - 1);
    int range2 = 1 << target;

    CTYPE *state_ptr = state.data();
#pragma acc data copy(state_ptr[0 : 1 << n])
    {
#pragma acc parallel loop
        for (int idx1 = 0; idx1 < range1; idx1++)
        {
            for (int idx2 = 0; idx2 < range2; idx2++)
            {
                int i = (idx1 << (target + 1)) | idx2;
                int j = i | (1 << target);
                CTYPE tmp = state_ptr[i];
                state_ptr[i] = state_ptr[j];
                state_ptr[j] = tmp;
            }
        }
    }
}

int main()
{
    int n = 4;
    std::cout << n << std::endl;

    std::vector<CTYPE> init_state(1 << n);

    for (int i = 0; i < (1 << n); i++)
        init_state[i] = i;

    update_with_x(init_state, n, 1);

    for (int i = 0; i < (1 << n); i++)
        std::cout << ' ' << init_state[i];
    return 0;
}