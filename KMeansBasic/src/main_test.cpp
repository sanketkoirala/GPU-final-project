#include <cstdlib>
#include <cstdio>
#include <cuda_runtime.h>

#include "gputk.h"
#include "kmeans_gpu.h"

int main(int argc, char **argv) {
    gpuTKArg_t args;
    float *hostPoints, *hostInit, *hostOutput;
    int N, D, initRows, initCols, K;

    args = gpuTKArg_read(argc, argv);

    hostPoints = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 0), &N, &D);
    hostInit = (float *)gpuTKImport(gpuTKArg_getInputFile(args, 1),
                                    &initRows, &initCols);
    K = initRows;

    if (initCols != D) {
        fprintf(stderr, "init.raw cols (%d) != input.raw cols (%d)\n",
                initCols, D);
        return 1;
    }

    hostOutput = (float *)malloc((size_t)K * D * sizeof(float));

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);

    kmeansGPU(hostPoints, hostInit, hostOutput, N, D, K,
              KMEANS_NUM_ITERATIONS);

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    float elapsedMs = 0.0f;
    cudaEventElapsedTime(&elapsedMs, t0, t1);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    printf("[time] N=%d D=%d K=%d %.3f ms\n", N, D, K, elapsedMs);

    gpuTKSolution(args, hostOutput, K, D);

    free(hostOutput);
    return 0;
}
