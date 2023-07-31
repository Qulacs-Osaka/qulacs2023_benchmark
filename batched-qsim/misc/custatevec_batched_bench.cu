#include <iostream>
#include <chrono>
#include <random>
#include <vector>

#include <cuComplex.h>        // cuDoubleComplex
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <custatevec.h>       // custatevecApplyMatrix

int main(int argc, char *argv[])
{
    const int N_TRIALS = 10;
    const int DEPTH = 10;
    const int nIndexBits = std::atoi(argv[1]);
    const int nSvSize = (1 << nIndexBits);
    const int nSVs = std::atoi(argv[2]);
    const int nTargets = 1;
    const int nControls = 0;
    const int adjoint = 0;

    int targets[] = {0};

    cuDoubleComplex *h_sv =
        (cuDoubleComplex *)malloc(nSVs * nSvSize * sizeof(cuDoubleComplex));
    cuDoubleComplex *matrix =
        (cuDoubleComplex *)malloc(2 * 2 * sizeof(cuDoubleComplex));

    cuDoubleComplex *d_sv;
    cudaMalloc((void **)&d_sv, nSVs * nSvSize * sizeof(cuDoubleComplex));

    cudaMemcpy(d_sv, h_sv, nSVs * nSvSize * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    //--------------------------------------------------------------------------

    // custatevec handle initialization
    custatevecHandle_t handle;

    custatevecCreate(&handle);

    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    // check the size of external workspace
    custatevecApplyMatrixBatchedGetWorkspaceSize(
        handle, CUDA_C_64F, nIndexBits, nSVs, nSvSize,
        CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, nullptr, matrix, CUDA_C_64F,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, 1, nTargets, nControls,
        CUSTATEVEC_COMPUTE_64F, &extraWorkspaceSizeInBytes);

    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
        cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes);

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    std::uniform_real_distribution<double> dist(0.0, M_PI);
    std::vector<double> durations;

    for (int trial = 0; trial < N_TRIALS + 1; trial++) {
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int d = 0; d < DEPTH; d++) {
            for (int i = 0; i < nIndexBits; i++) {
                double theta = dist(engine);
                matrix[0] = make_cuDoubleComplex(cos(theta / 2), 0);
                matrix[1] = make_cuDoubleComplex(0, -cos(theta / 2));
                matrix[2] = make_cuDoubleComplex(0, -sin(theta / 2));
                matrix[3] = make_cuDoubleComplex(cos(theta / 2), 0);

                targets[0] = i;

                // Apply RX gate
                custatevecApplyMatrixBatched(
                    handle, d_sv, CUDA_C_64F, nIndexBits, nSVs, nSvSize,
                    CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, NULL, matrix,
                    CUDA_C_64F, CUSTATEVEC_MATRIX_LAYOUT_ROW, adjoint, 1,
                    targets, nTargets, nullptr, nullptr, nControls,
                    CUSTATEVEC_COMPUTE_64F, extraWorkspace,
                    extraWorkspaceSizeInBytes);
            }
        }

        cudaDeviceSynchronize();

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time);

        if (trial > 0) {
            durations.push_back(duration.count() / 1e6);
        }
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

    // destroy handle
    custatevecDestroy(handle);

    //--------------------------------------------------------------------------

    cudaFree(d_sv);
    if (extraWorkspaceSizeInBytes) cudaFree(extraWorkspace);

    return EXIT_SUCCESS;
}
