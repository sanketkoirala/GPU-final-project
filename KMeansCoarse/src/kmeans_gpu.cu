#include "kmeans_gpu.h"

#include <cfloat>

// Phase 4: Thread Coarsening.
// Each thread processes COARSE_FACTOR data points instead of 1.
// This amortizes per-thread setup, and the centroid value cv is loaded once
// from constant memory and reused across all COARSE_FACTOR points, improving
// instruction-level parallelism and reducing centroid read traffic.
// Builds on Phase 2 (constant memory centroids) only — no shared memory,
// so we can cleanly isolate the coarsening benefit.

__constant__ float kCentroids_d[MAX_K * MAX_D];

__global__ void assign_coarse_kernel(const float *data, int *assignments,
                                     int N, int D, int K) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;

    // Per-point distance accumulators
    float dist[COARSE_FACTOR * MAX_K];
    for (int c = 0; c < COARSE_FACTOR; ++c)
        for (int k = 0; k < K; ++k)
            dist[c * MAX_K + k] = 0.0f;

    // Outer loop over centroids, inner over dimensions.
    // centroid value cv is loaded once per (k,d) and reused across COARSE_FACTOR points.
    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < D; ++d) {
            float cv = kCentroids_d[k * D + d];  // load once from constant mem
            for (int c = 0; c < COARSE_FACTOR; ++c) {
                int i = base + c;
                if (i < N) {
                    float diff = data[i * D + d] - cv;
                    dist[c * MAX_K + k] += diff * diff;
                }
            }
        }
    }

    // Find nearest centroid for each coarsened point
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int i = base + c;
        if (i < N) {
            float minDist = dist[c * MAX_K + 0];
            int best = 0;
            for (int k = 1; k < K; ++k) {
                if (dist[c * MAX_K + k] < minDist) {
                    minDist = dist[c * MAX_K + k];
                    best = k;
                }
            }
            assignments[i] = best;
        }
    }
}

// Update kernel: each thread also processes COARSE_FACTOR points.
__global__ void update_coarse_kernel(const float *data, const int *assignments,
                                     float *sums, int *counts, int N, int D) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;

    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int i = base + c;
        if (i < N) {
            int k = assignments[i];
            for (int d = 0; d < D; ++d) {
                atomicAdd(&sums[k * D + d], data[i * D + d]);
            }
            atomicAdd(&counts[k], 1);
        }
    }
}

__global__ void finalize_kernel(const float *sums, const int *counts,
                                float *centroids, int D, int K) {
    int k = blockIdx.x;
    int d = threadIdx.x;
    if (k >= K || d >= D) return;

    int c = counts[k];
    if (c > 0) {
        centroids[k * D + d] = sums[k * D + d] / (float)c;
    }
}

void kmeansGPU(const float *points, const float *initCentroids,
               float *outCentroids, int N, int D, int K, int T) {
    float *data_d, *centroids_d, *sums_d;
    int *assignments_d, *counts_d;

    size_t pointsBytes = (size_t)N * D * sizeof(float);
    size_t centroidsBytes = (size_t)K * D * sizeof(float);

    gpuErrchk(cudaMalloc((void **)&data_d, pointsBytes));
    gpuErrchk(cudaMalloc((void **)&centroids_d, centroidsBytes));
    gpuErrchk(cudaMalloc((void **)&sums_d, centroidsBytes));
    gpuErrchk(cudaMalloc((void **)&assignments_d, (size_t)N * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&counts_d, (size_t)K * sizeof(int)));

    gpuErrchk(cudaMemcpy(data_d, points, pointsBytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(centroids_d, initCentroids, centroidsBytes,
                         cudaMemcpyHostToDevice));

    // Grid is divided by COARSE_FACTOR since each thread handles multiple points
    int totalThreads = (N + COARSE_FACTOR - 1) / COARSE_FACTOR;
    dim3 assignBlock(BLOCK_SIZE);
    dim3 assignGrid((totalThreads + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int t = 0; t < T; ++t) {
        // Copy current centroids to constant memory
        gpuErrchk(cudaMemcpyToSymbol(kCentroids_d, centroids_d,
                                     centroidsBytes, 0,
                                     cudaMemcpyDeviceToDevice));

        // Coarsened assign
        assign_coarse_kernel<<<assignGrid, assignBlock>>>(
            data_d, assignments_d, N, D, K);
        gpuErrchk(cudaGetLastError());

        // Reset accumulators
        gpuErrchk(cudaMemsetAsync(sums_d, 0, centroidsBytes));
        gpuErrchk(cudaMemsetAsync(counts_d, 0, (size_t)K * sizeof(int)));

        // Coarsened update
        update_coarse_kernel<<<assignGrid, assignBlock>>>(
            data_d, assignments_d, sums_d, counts_d, N, D);
        gpuErrchk(cudaGetLastError());

        // Finalize centroids
        finalize_kernel<<<dim3(K), dim3(D)>>>(sums_d, counts_d, centroids_d, D, K);
        gpuErrchk(cudaGetLastError());
    }

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(outCentroids, centroids_d, centroidsBytes,
                         cudaMemcpyDeviceToHost));

    cudaFree(data_d);
    cudaFree(centroids_d);
    cudaFree(sums_d);
    cudaFree(assignments_d);
    cudaFree(counts_d);
}
