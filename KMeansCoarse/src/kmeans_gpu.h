#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define KMEANS_NUM_ITERATIONS 10

// Each thread processes COARSE_FACTOR data points.
#define COARSE_FACTOR 4

// Maximum dimensions for constant memory buffer.
#define MAX_K 64
#define MAX_D 128

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

// Run T iterations of K-Means on the GPU using thread coarsening + constant-memory centroids.
void kmeansGPU(const float *points, const float *initCentroids,
               float *outCentroids, int N, int D, int K, int T);
