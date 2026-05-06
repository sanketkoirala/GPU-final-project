#include <cstdlib>
#include <cstdio>
#include <chrono>

#include "gputk.h"
#include "kmeans_cpu.hpp"

int main(int argc, char **argv) {
    gpuTKArg_t args;
    float *hostPoints, *hostInit, *hostOutput;
    int N, D, initRows, initCols, K;

    args = gpuTKArg_read(argc, argv);

    // input.raw: N x D points
    hostPoints = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &N, &D);
    // init.raw: K x D initial centroids
    hostInit = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1),
                                    &initRows, &initCols);
    K = initRows;

    if (initCols != D) {
        fprintf(stderr, "init.raw cols (%d) != input.raw cols (%d)\n",
                initCols, D);
        return 1;
    }

    hostOutput = (float *)malloc((size_t)K * D * sizeof(float));

    auto t0 = std::chrono::high_resolution_clock::now();
    kmeansCPU(hostPoints, hostInit, hostOutput, N, D, K,
              KMEANS_NUM_ITERATIONS);
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsedMs =
        std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("[time] N=%d D=%d K=%d %.3f ms\n", N, D, K, elapsedMs);

    gpuTKSolution(args, hostOutput, K, D);

    free(hostOutput);
    return 0;
}
