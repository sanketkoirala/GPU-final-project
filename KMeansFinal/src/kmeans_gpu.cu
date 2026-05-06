#include "kmeans_gpu.h"

#include <cfloat>
#include <cstring>

// Phase 6: Final Combined Kernel.
// Combines ALL optimizations from previous phases:
//   - Constant-memory centroids (Phase 2)
//   - Dim-major coalesced data layout (Phase 5)
//   - Thread coarsening (Phase 4) — each thread processes COARSE_FACTOR points
//   - Per-block partial-sum reduction (replaces atomicAdd contention from Phase 1)
//
// The update kernel uses shared-memory reduction (lab-4 pattern) to accumulate
// per-block partial sums in shared memory, then writes K×D partial sums and K
// counts to global memory per block. A second finalize kernel reduces across
// blocks and divides to get new centroids.

__constant__ float kCentroids_d[MAX_K * MAX_D];

// -----------------------------------------------------------------------
// Transpose kernel: row-major [N×D] → dim-major [D×N]
// -----------------------------------------------------------------------
__global__ void transpose_kernel(const float *src, float *dst, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    for (int d = 0; d < D; ++d) {
        dst[d * N + i] = src[i * D + d];
    }
}

// -----------------------------------------------------------------------
// Assign kernel: coalesced dim-major reads + coarsening + constant-mem centroids
// -----------------------------------------------------------------------
__global__ void assign_final_kernel(const float *data_dm, int *assignments,
                                    int N, int D, int K) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;

    // Distance accumulators for COARSE_FACTOR points × K centroids
    float dist[COARSE_FACTOR * MAX_K];
    for (int c = 0; c < COARSE_FACTOR; ++c)
        for (int k = 0; k < K; ++k)
            dist[c * MAX_K + k] = 0.0f;

    // Outer loop over centroids, inner over dimensions.
    // Centroid value cv loaded once from constant mem, reused across COARSE_FACTOR points.
    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < D; ++d) {
            float cv = kCentroids_d[k * D + d];
            for (int c = 0; c < COARSE_FACTOR; ++c) {
                int i = base + c;
                if (i < N) {
                    float diff = data_dm[d * N + i] - cv;  // coalesced read
                    dist[c * MAX_K + k] += diff * diff;
                }
            }
        }
    }

    // Find nearest centroid for each point
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

// -----------------------------------------------------------------------
// Update kernel with per-block reduction.
// Each block accumulates partial sums in shared memory, then one thread
// per cluster/dim atomicAdd's the block partial to global buffers.
// This drastically reduces atomic contention vs. Phase 1's per-thread atomicAdd.
//
// We process points in [blockStart .. blockEnd) where each thread handles
// COARSE_FACTOR points. The block writes K×D partial sums and K counts.
// -----------------------------------------------------------------------
__global__ void update_reduction_kernel(const float *data_dm, const int *assignments,
                                        float *globalSums, int *globalCounts,
                                        int N, int D, int K) {
    // Shared memory for per-block partial sums and counts.
    // Layout: sums[K * D] followed by counts[K].
    extern __shared__ float sharedMem[];
    float *blockSums = sharedMem;                       // K * D floats
    int   *blockCounts = (int *)&sharedMem[K * D];      // K ints

    // Initialize shared memory (each thread zeros a portion)
    int sharedSize = K * D;
    for (int idx = threadIdx.x; idx < sharedSize; idx += blockDim.x) {
        blockSums[idx] = 0.0f;
    }
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        blockCounts[idx] = 0;
    }
    __syncthreads();

    // Each thread accumulates its COARSE_FACTOR points into shared memory
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * COARSE_FACTOR;
    for (int c = 0; c < COARSE_FACTOR; ++c) {
        int i = base + c;
        if (i < N) {
            int cluster = assignments[i];
            atomicAdd(&blockCounts[cluster], 1);
            for (int d = 0; d < D; ++d) {
                atomicAdd(&blockSums[cluster * D + d], data_dm[d * N + i]);
            }
        }
    }
    __syncthreads();

    // Reduce block results to global with one atomicAdd per cluster/dim per block
    // (much less contention than one per thread per point)
    for (int idx = threadIdx.x; idx < sharedSize; idx += blockDim.x) {
        if (blockSums[idx] != 0.0f) {
            atomicAdd(&globalSums[idx], blockSums[idx]);
        }
    }
    for (int idx = threadIdx.x; idx < K; idx += blockDim.x) {
        if (blockCounts[idx] != 0) {
            atomicAdd(&globalCounts[idx], blockCounts[idx]);
        }
    }
}

// -----------------------------------------------------------------------
// Finalize kernel: divide sums by counts to produce new centroids.
// -----------------------------------------------------------------------
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

// -----------------------------------------------------------------------
// Host wrapper
// -----------------------------------------------------------------------
void kmeansGPU(const float *points, const float *initCentroids,
               float *outCentroids, int N, int D, int K, int T) {
    float *data_d, *data_dm_d, *centroids_d, *sums_d;
    int *assignments_d, *counts_d;

    size_t pointsBytes = (size_t)N * D * sizeof(float);
    size_t centroidsBytes = (size_t)K * D * sizeof(float);

    gpuErrchk(cudaMalloc((void **)&data_d, pointsBytes));
    gpuErrchk(cudaMalloc((void **)&data_dm_d, pointsBytes));
    gpuErrchk(cudaMalloc((void **)&centroids_d, centroidsBytes));
    gpuErrchk(cudaMalloc((void **)&sums_d, centroidsBytes));
    gpuErrchk(cudaMalloc((void **)&assignments_d, (size_t)N * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&counts_d, (size_t)K * sizeof(int)));

    gpuErrchk(cudaMemcpy(data_d, points, pointsBytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(centroids_d, initCentroids, centroidsBytes,
                         cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 transposeGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // One-time transpose: row-major → dim-major
    transpose_kernel<<<transposeGrid, block>>>(data_d, data_dm_d, N, D);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // Assign grid: coarsened
    int totalThreads = (N + COARSE_FACTOR - 1) / COARSE_FACTOR;
    dim3 assignGrid((totalThreads + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Shared memory size for update reduction kernel
    size_t sharedBytes = (size_t)K * D * sizeof(float) + (size_t)K * sizeof(int);

    for (int t = 0; t < T; ++t) {
        // Copy current centroids to constant memory
        gpuErrchk(cudaMemcpyToSymbol(kCentroids_d, centroids_d,
                                     centroidsBytes, 0,
                                     cudaMemcpyDeviceToDevice));

        // Coarsened assign with coalesced reads
        assign_final_kernel<<<assignGrid, block>>>(
            data_dm_d, assignments_d, N, D, K);
        gpuErrchk(cudaGetLastError());

        // Reset accumulators
        gpuErrchk(cudaMemsetAsync(sums_d, 0, centroidsBytes));
        gpuErrchk(cudaMemsetAsync(counts_d, 0, (size_t)K * sizeof(int)));

        // Update with per-block shared-memory reduction
        update_reduction_kernel<<<assignGrid, block, sharedBytes>>>(
            data_dm_d, assignments_d, sums_d, counts_d, N, D, K);
        gpuErrchk(cudaGetLastError());

        // Finalize centroids
        finalize_kernel<<<dim3(K), dim3(D)>>>(sums_d, counts_d, centroids_d, D, K);
        gpuErrchk(cudaGetLastError());
    }

    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(outCentroids, centroids_d, centroidsBytes,
                         cudaMemcpyDeviceToHost));

    cudaFree(data_d);
    cudaFree(data_dm_d);
    cudaFree(centroids_d);
    cudaFree(sums_d);
    cudaFree(assignments_d);
    cudaFree(counts_d);
}
