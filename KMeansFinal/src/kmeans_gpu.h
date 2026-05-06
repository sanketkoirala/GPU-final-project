#pragma once

#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define KMEANS_NUM_ITERATIONS 10

// Thread coarsening factor: each thread handles COARSE_FACTOR points.
#define COARSE_FACTOR 4

// Tile width along dimension axis for shared-memory tiling in assign kernel.
#define D_TILE 32

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

// Run T iterations of K-Means on the GPU using all optimizations combined:
// constant-memory centroids, dim-major coalesced data, shared-memory tiling,
// thread coarsening, and per-block reduction for the update step.
void kmeansGPU(const float *points, const float *initCentroids,
               float *outCentroids, int N, int D, int K, int T);
