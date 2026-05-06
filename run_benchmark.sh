#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

export TMPDIR=$HOME/tmp/ncu-lock
mkdir -p $TMPDIR

# Benchmarks profile against the largest dataset (data/9: N=2M, D=128, K=64)
# --set basic = fast (~1-2 min per phase), enough for phase-vs-phase comparison.
# --launch-count 3 = profile only the first iteration's 3 kernel launches.
# Use run_benchmark_full.sh for the final writeup pass with --set full.

# Phase 1: Naive GPU
echo "Benchmarking Phase 1: KMeansBasic"
ncu --set basic --launch-count 3 --log-file benchmark1.txt -f \
    ./build/release/kmeans_basic_test \
    -e data/9/output.raw \
    -i data/9/input.raw,data/9/init.raw -t matrix

# Phase 2: Constant memory
echo "Benchmarking Phase 2: KMeansCMem"
ncu --set basic --launch-count 3 --log-file benchmark2.txt -f \
    ./build/release/kmeans_cmem_test \
    -e data/9/output.raw \
    -i data/9/input.raw,data/9/init.raw -t matrix

# Phase 3: Shared memory tiling
echo "Benchmarking Phase 3: KMeansShared"
ncu --set basic --launch-count 3 --log-file benchmark3.txt -f \
    ./build/release/kmeans_shared_test \
    -e data/9/output.raw \
    -i data/9/input.raw,data/9/init.raw -t matrix

# Phase 4: Thread coarsening
echo "Benchmarking Phase 4: KMeansCoarse"
ncu --set basic --launch-count 3 --log-file benchmark4.txt -f \
    ./build/release/kmeans_coarse_test \
    -e data/9/output.raw \
    -i data/9/input.raw,data/9/init.raw -t matrix

# Phase 5: Memory coalescing
echo "Benchmarking Phase 5: KMeansCoalesced"
ncu --set basic --launch-count 3 --log-file benchmark5.txt -f \
    ./build/release/kmeans_coalesced_test \
    -e data/9/output.raw \
    -i data/9/input.raw,data/9/init.raw -t matrix

# Phase 6: Final combined
echo "Benchmarking Phase 6: KMeansFinal"
ncu --set basic --launch-count 3 --log-file benchmark6.txt -f \
    ./build/release/kmeans_final_test \
    -e data/9/output.raw \
    -i data/9/input.raw,data/9/init.raw -t matrix
