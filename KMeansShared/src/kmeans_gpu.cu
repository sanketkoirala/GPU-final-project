#include "kmeans_gpu.h"

#include <cfloat>

// Phase 3: Shared-Memory Tiling along the Dimension Axis.
// Each block cooperatively loads a BLOCK_SIZE × D_TILE tile of data points
// into shared memory, then each thread accumulates partial squared-distances
// against all K centroids (still in __constant__ memory from Phase 2).
// This reduces per-thread global loads and improves data reuse for high-D datasets.

__constant__ float kCentroids_d[MAX_K * MAX_D];

__global__ void assign_tiled_kernel(const float *data, int *assignments,
                                    int N, int D, int K) {
    __shared__ float dataTile[BLOCK_SIZE * D_TILE];
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Per-thread distance accumulators (one per centroid, in registers).
    // Using a fixed-size local array; for K up to MAX_K this stays in registers
    // or spills gracefully.
    float dist[MAX_K];
    for (int k = 0; k < K; ++k) dist[k] = 0.0f;

    // Tile along the D dimension
    for (int d0 = 0; d0 < D; d0 += D_TILE) {
        int tileWidth = D_TILE;
        if (d0 + D_TILE > D) tileWidth = D - d0;

        // Cooperative load: each thread loads its own row's chunk into shared memory
        for (int dd = 0; dd < tileWidth; ++dd) {
            dataTile[threadIdx.x * D_TILE + dd] =
                (i < N) ? data[i * D + d0 + dd] : 0.0f;
        }
        // Zero remaining tile entries if tileWidth < D_TILE
        for (int dd = tileWidth; dd < D_TILE; ++dd) {
            dataTile[threadIdx.x * D_TILE + dd] = 0.0f;
        }
        __syncthreads();

        // Accumulate partial distances for this tile
        if (i < N) {
            for (int k = 0; k < K; ++k) {
                for (int dd = 0; dd < tileWidth; ++dd) {
                    float diff = dataTile[threadIdx.x * D_TILE + dd]
                               - kCentroids_d[k * D + d0 + dd];
                    dist[k] += diff * diff;
                }
            }
        }
        __syncthreads();
    }

    // Find nearest centroid
    if (i < N) {
        float minDist = dist[0];
        int best = 0;
        for (int k = 1; k < K; ++k) {
            if (dist[k] < minDist) {
                minDist = dist[k];
                best = k;
            }
        }
        assignments[i] = best;
    }
}

// Update kernel unchanged from Phase 1/2 (atomicAdd to global sums/counts).
__global__ void update_kernel(const float *data, const int *assignments,
                              float *sums, int *counts, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int k = assignments[i];
    for (int d = 0; d < D; ++d) {
        atomicAdd(&sums[k * D + d], data[i * D + d]);
    }
    atomicAdd(&counts[k], 1);
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

    dim3 assignBlock(BLOCK_SIZE);
    dim3 assignGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int t = 0; t < T; ++t) {
        // Copy current centroids to constant memory
        gpuErrchk(cudaMemcpyToSymbol(kCentroids_d, centroids_d,
                                     centroidsBytes, 0,
                                     cudaMemcpyDeviceToDevice));

        // Tiled assign kernel
        assign_tiled_kernel<<<assignGrid, assignBlock>>>(
            data_d, assignments_d, N, D, K);
        gpuErrchk(cudaGetLastError());

        // Reset accumulators
        gpuErrchk(cudaMemsetAsync(sums_d, 0, centroidsBytes));
        gpuErrchk(cudaMemsetAsync(counts_d, 0, (size_t)K * sizeof(int)));

        // Accumulate sums/counts
        update_kernel<<<assignGrid, assignBlock>>>(
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
