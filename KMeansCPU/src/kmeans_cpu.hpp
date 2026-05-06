#pragma once

// CPU reference K-Means implementation.
// Used by both the datagen (to produce ground-truth output.raw)
// and the CPU test executable.

// Number of K-Means iterations performed by every phase (datagen and tests).
// All phases run exactly this many iterations from the same init centroids,
// so output.raw is comparable across phases.
#define KMEANS_NUM_ITERATIONS 10

// Single assignment step: for each point, write argmin centroid index.
void assignStepCPU(const float *points, const float *centroids,
                   int *assignments, int N, int D, int K);

// Single update step: recompute centroids as mean of assigned points.
// If a cluster has zero points its centroid is left unchanged.
void updateStepCPU(const float *points, const int *assignments,
                   float *centroids, int N, int D, int K);

// Run T iterations of K-Means. outCentroids is filled with the final centroids.
// initCentroids and outCentroids may alias.
void kmeansCPU(const float *points, const float *initCentroids,
               float *outCentroids, int N, int D, int K, int T);
