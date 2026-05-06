#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

# Build (idempotent)
cmake -B build
cmake --build build -j

# Generate datasets if data/ does not exist
if [ ! -d data ]; then
    ./build/release/kmeans_datagen
fi

# Phase 0: CPU
echo "**********************************"
echo "* Running KMeansCPU tests (Phase 0) *"
echo "**********************************"
for dir in $(seq 0 9); do
    echo "Test data/$dir"
    ./build/release/kmeans_cpu_test \
        -e data/$dir/output.raw \
        -i data/$dir/input.raw,data/$dir/init.raw \
        -t matrix
done

# Phase 1: Basic GPU
echo ""
echo "**********************************"
echo "* Running KMeansBasic tests (Phase 1) *"
echo "**********************************"
for dir in $(seq 0 9); do
    echo "Test data/$dir"
    ./build/release/kmeans_basic_test \
        -e data/$dir/output.raw \
        -i data/$dir/input.raw,data/$dir/init.raw \
        -t matrix
done

# Phase 2: Constant Memory
echo ""
echo "**********************************"
echo "* Running KMeansCMem tests (Phase 2) *"
echo "**********************************"
for dir in $(seq 0 9); do
    echo "Test data/$dir"
    ./build/release/kmeans_cmem_test \
        -e data/$dir/output.raw \
        -i data/$dir/input.raw,data/$dir/init.raw \
        -t matrix
done

# Phase 3: Shared Memory Tiling
echo ""
echo "**********************************"
echo "* Running KMeansShared tests (Phase 3) *"
echo "**********************************"
for dir in $(seq 0 9); do
    echo "Test data/$dir"
    ./build/release/kmeans_shared_test \
        -e data/$dir/output.raw \
        -i data/$dir/input.raw,data/$dir/init.raw \
        -t matrix
done

# Phase 4: Thread Coarsening
echo ""
echo "**********************************"
echo "* Running KMeansCoarse tests (Phase 4) *"
echo "**********************************"
for dir in $(seq 0 9); do
    echo "Test data/$dir"
    ./build/release/kmeans_coarse_test \
        -e data/$dir/output.raw \
        -i data/$dir/input.raw,data/$dir/init.raw \
        -t matrix
done

# Phase 5: Memory Coalescing
echo ""
echo "**********************************"
echo "* Running KMeansCoalesced tests (Phase 5) *"
echo "**********************************"
for dir in $(seq 0 9); do
    echo "Test data/$dir"
    ./build/release/kmeans_coalesced_test \
        -e data/$dir/output.raw \
        -i data/$dir/input.raw,data/$dir/init.raw \
        -t matrix
done

# Phase 6: Final Combined
echo ""
echo "**********************************"
echo "* Running KMeansFinal tests (Phase 6) *"
echo "**********************************"
for dir in $(seq 0 9); do
    echo "Test data/$dir"
    ./build/release/kmeans_final_test \
        -e data/$dir/output.raw \
        -i data/$dir/input.raw,data/$dir/init.raw \
        -t matrix
done
