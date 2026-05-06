#include "kmeans_gpu.h"

#include <cfloat>

// Phase 1: Naive GPU K-Means.
// One thread per data point. Distances and centroid updates both use global memory.
// Centroid updates use atomicAdd, which serializes contributions to the same cluster.

__global__ void assign_kernel(const float *data, const float *centroids,
                              int *assignments, int N, int D, int K) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float minDist = FLT_MAX;
    int best = 0;
    for (int k = 0; k < K; ++k) {
        float dist = 0.0f;
        for (int d = 0; d < D; ++d) {
            float diff = data[i * D + d] - centroids[k * D + d];
            dist += diff * diff;
        }
        if (dist < minDist) {
            minDist = dist;
            best = k;
        }
    }
    assignments[i] = best;
}

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
    // If a cluster has no points, leave the centroid as-is (matches CPU reference).
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
        // Assign each point to nearest centroid
        assign_kernel<<<assignGrid, assignBlock>>>(
            data_d, centroids_d, assignments_d, N, D, K);
        gpuErrchk(cudaGetLastError());

        // Reset accumulators
        gpuErrchk(cudaMemsetAsync(sums_d, 0, centroidsBytes));
        gpuErrchk(cudaMemsetAsync(counts_d, 0, (size_t)K * sizeof(int)));

        // Accumulate per-cluster sums and counts via atomicAdd
        update_kernel<<<assignGrid, assignBlock>>>(
            data_d, assignments_d, sums_d, counts_d, N, D);
        gpuErrchk(cudaGetLastError());

        // Divide sums by counts to produce new centroids
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
