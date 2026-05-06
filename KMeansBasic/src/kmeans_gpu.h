#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define KMEANS_NUM_ITERATIONS 10

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

// Run T iterations of K-Means on the GPU.
// points: N x D row-major host array
// initCentroids: K x D row-major host array
// outCentroids: K x D row-major host array (output)
void kmeansGPU(const float *points, const float *initCentroids,
               float *outCentroids, int N, int D, int K, int T);
