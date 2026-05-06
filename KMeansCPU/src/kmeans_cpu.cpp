#include "kmeans_cpu.hpp"
#include <cstdlib>
#include <cstring>
#include <cfloat>

void assignStepCPU(const float *points, const float *centroids,
                   int *assignments, int N, int D, int K) {
    for (int i = 0; i < N; ++i) {
        float minDist = FLT_MAX;
        int best = 0;
        for (int k = 0; k < K; ++k) {
            float dist = 0.0f;
            for (int d = 0; d < D; ++d) {
                float diff = points[i * D + d] - centroids[k * D + d];
                dist += diff * diff;
            }
            if (dist < minDist) {
                minDist = dist;
                best = k;
            }
        }
        assignments[i] = best;
    }
}

void updateStepCPU(const float *points, const int *assignments,
                   float *centroids, int N, int D, int K) {
    float *sums = (float *)calloc(K * D, sizeof(float));
    int *counts = (int *)calloc(K, sizeof(int));

    for (int i = 0; i < N; ++i) {
        int k = assignments[i];
        counts[k] += 1;
        for (int d = 0; d < D; ++d) {
            sums[k * D + d] += points[i * D + d];
        }
    }

    for (int k = 0; k < K; ++k) {
        if (counts[k] > 0) {
            float invCount = 1.0f / (float)counts[k];
            for (int d = 0; d < D; ++d) {
                centroids[k * D + d] = sums[k * D + d] * invCount;
            }
        }
    }

    free(sums);
    free(counts);
}

void kmeansCPU(const float *points, const float *initCentroids,
               float *outCentroids, int N, int D, int K, int T) {
    if (outCentroids != initCentroids) {
        memcpy(outCentroids, initCentroids, K * D * sizeof(float));
    }

    int *assignments = (int *)malloc(N * sizeof(int));

    for (int t = 0; t < T; ++t) {
        assignStepCPU(points, outCentroids, assignments, N, D, K);
        updateStepCPU(points, assignments, outCentroids, N, D, K);
    }

    free(assignments);
}
