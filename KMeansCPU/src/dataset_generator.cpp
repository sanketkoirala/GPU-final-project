#include "gputk.h"

#include "kmeans_cpu.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static char *base_dir;

// Box-Muller transform: standard normal sample.
static float rand_normal() {
    float u1 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    float u2 = (rand() + 1.0f) / ((float)RAND_MAX + 2.0f);
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
}

// Generate N points in D dimensions sampled from K Gaussian blobs.
// blob_centers is K*D floats; spread controls within-cluster stddev.
static float *generate_blobs(int N, int D, int K,
                             const float *blob_centers, float spread) {
    float *data = (float *)malloc((size_t)N * D * sizeof(float));
    for (int i = 0; i < N; ++i) {
        int b = rand() % K;
        for (int d = 0; d < D; ++d) {
            data[i * D + d] = blob_centers[b * D + d] + spread * rand_normal();
        }
    }
    return data;
}

// Generate K blob centers uniformly in [-10, 10]^D.
static float *generate_centers(int K, int D) {
    float *centers = (float *)malloc((size_t)K * D * sizeof(float));
    for (int k = 0; k < K; ++k) {
        for (int d = 0; d < D; ++d) {
            float u = rand() / (float)RAND_MAX;
            centers[k * D + d] = (u * 20.0f) - 10.0f;
        }
    }
    return centers;
}

// Generate K initial centroids by picking K random points from the dataset.
static float *generate_init_centroids(const float *points, int N, int D, int K) {
    float *init = (float *)malloc((size_t)K * D * sizeof(float));
    for (int k = 0; k < K; ++k) {
        int idx = rand() % N;
        for (int d = 0; d < D; ++d) {
            init[k * D + d] = points[idx * D + d];
        }
    }
    return init;
}

// Write a rows x cols float matrix in libgputk's text matrix format.
static void write_data(const char *file_name, const float *data,
                       int rows, int cols) {
    FILE *handle = fopen(file_name, "w");
    fprintf(handle, "%d %d\n", rows, cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            fprintf(handle, "%f", data[r * cols + c]);
            if (c != cols - 1) fprintf(handle, " ");
        }
        if (r != rows - 1) fprintf(handle, "\n");
    }
    fflush(handle);
    fclose(handle);
}

static void create_dataset(int datasetNum, int N, int D, int K) {
    const char *dir_name =
        gpuTKDirectory_create(gpuTKPath_join(base_dir, datasetNum));

    char *input_file = gpuTKPath_join(dir_name, "input.raw");
    char *init_file = gpuTKPath_join(dir_name, "init.raw");
    char *output_file = gpuTKPath_join(dir_name, "output.raw");

    float *centers = generate_centers(K, D);
    float *points = generate_blobs(N, D, K, centers, 0.5f);
    float *init = generate_init_centroids(points, N, D, K);

    float *finalCentroids = (float *)malloc((size_t)K * D * sizeof(float));
    kmeansCPU(points, init, finalCentroids, N, D, K, KMEANS_NUM_ITERATIONS);

    write_data(input_file, points, N, D);
    write_data(init_file, init, K, D);
    write_data(output_file, finalCentroids, K, D);

    printf("Dataset %d: N=%d D=%d K=%d -> %s\n", datasetNum, N, D, K, dir_name);

    free(centers);
    free(points);
    free(init);
    free(finalCentroids);
}

int main() {
    srand(42);
    base_dir = gpuTKPath_join(gpuTKDirectory_current(), "data");

    //                #     N       D    K
    create_dataset(   0,    1024,    2,   4);
    create_dataset(   1,    4096,    4,   8);
    create_dataset(   2,   16384,    8,   8);
    create_dataset(   3,   65536,   16,  16);
    create_dataset(   4,  262144,   32,  16);
    create_dataset(   5,  524288,   32,  32);
    create_dataset(   6, 1048576,   64,  32);
    create_dataset(   7, 1048576,  128,  32);
    create_dataset(   8, 2097152,   64,  64);
    create_dataset(   9, 2097152,  128,  64);

    return 0;
}
