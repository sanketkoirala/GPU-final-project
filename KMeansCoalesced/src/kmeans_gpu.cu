#include "kmeans_gpu.h"

#include <cfloat>
#include <cstring>

// Phase 5: Memory Coalescing via Dim-Major (Column-Major) Layout.
// Data is transposed from row-major [N×D] to dim-major [D×N] on the host
// before uploading. Now when threads i=0..31 in a warp access dim d of their
// respective points, they read data_dm[d*N + i..i+31] — consecutive addresses,
// which coalesce into a single 128-byte transaction instead of 32 scattered ones.
//
// Centroids remain in __constant__ memory (Phase 2 carryover).
// The transpose is a one-time cost amortized over T=10 iterations.

__constant__ float kCentroids_d[MAX_K * MAX_D];

// Transpose kernel: convert row-major [N×D] to dim-major [D×N] on GPU
__global__ void transpose_kernel(const float *src, float *dst, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // point index
    if (i >= N) return;
    for (int d = 0; d < D; ++d) {
        dst[d * N + i] = src[i * D + d];
    }
}

// Assign kernel using dim-major data layout — coalesced global reads.
__global__ void assign_coalesced_kernel(const float *data_dm, int *assignments,
                                        int N, int D, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float dist[MAX_K];
    for (int k = 0; k < K; ++k) dist[k] = 0.0f;

    for (int d = 0; d < D; ++d) {
        float x = data_dm[d * N + i];  // coalesced load: warp reads consecutive addrs
        for (int k = 0; k < K; ++k) {
            float diff = x - kCentroids_d[k * D + d];
            dist[k] += diff * diff;
        }
    }

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

// Update kernel using dim-major layout — also coalesced reads.
__global__ void update_coalesced_kernel(const float *data_dm, const int *assignments,
                                        float *sums, int *counts, int N, int D) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int k = assignments[i];
    for (int d = 0; d < D; ++d) {
        atomicAdd(&sums[k * D + d], data_dm[d * N + i]);  // coalesced read
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
    float *data_d, *data_dm_d, *centroids_d, *sums_d;
    int *assignments_d, *counts_d;

    size_t pointsBytes = (size_t)N * D * sizeof(float);
    size_t centroidsBytes = (size_t)K * D * sizeof(float);

    gpuErrchk(cudaMalloc((void **)&data_d, pointsBytes));
    gpuErrchk(cudaMalloc((void **)&data_dm_d, pointsBytes));  // transposed copy
    gpuErrchk(cudaMalloc((void **)&centroids_d, centroidsBytes));
    gpuErrchk(cudaMalloc((void **)&sums_d, centroidsBytes));
    gpuErrchk(cudaMalloc((void **)&assignments_d, (size_t)N * sizeof(int)));
    gpuErrchk(cudaMalloc((void **)&counts_d, (size_t)K * sizeof(int)));

    gpuErrchk(cudaMemcpy(data_d, points, pointsBytes, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(centroids_d, initCentroids, centroidsBytes,
                         cudaMemcpyHostToDevice));

    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // One-time transpose: row-major [N×D] → dim-major [D×N]
    transpose_kernel<<<grid, block>>>(data_d, data_dm_d, N, D);
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    for (int t = 0; t < T; ++t) {
        // Copy current centroids to constant memory
        gpuErrchk(cudaMemcpyToSymbol(kCentroids_d, centroids_d,
                                     centroidsBytes, 0,
                                     cudaMemcpyDeviceToDevice));

        // Assign with coalesced reads
        assign_coalesced_kernel<<<grid, block>>>(
            data_dm_d, assignments_d, N, D, K);
        gpuErrchk(cudaGetLastError());

        // Reset accumulators
        gpuErrchk(cudaMemsetAsync(sums_d, 0, centroidsBytes));
        gpuErrchk(cudaMemsetAsync(counts_d, 0, (size_t)K * sizeof(int)));

        // Update with coalesced reads
        update_coalesced_kernel<<<grid, block>>>(
            data_dm_d, assignments_d, sums_d, counts_d, N, D);
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
