#!/bin/bash
#SBATCH --export=/usr/local/cuda/bin
#SBATCH --gres=gpu:1

# FULL benchmark - use this once for the final writeup.
# Slow (~5-10 min per phase with --launch-count 3) because --set full replays
# each kernel many times to collect every metric (sectors-per-request,
# source-correlated stalls, full memory pipeline breakdown, etc.).
# For day-to-day phase comparison use run_benchmark.sh (--set basic).

export TMPDIR=$HOME/tmp/ncu-lock
mkdir -p $TMPDIR

# Phase 1: Naive GPU
echo "Benchmarking Phase 1: KMeansBasic"
ncu --set full --launch-count 3 --log-file benchmark1_full.txt -f \
    ./build/release/kmeans_basic_test \
    -e data/9/output.raw \
    -i data/9/input.raw,data/9/init.raw -t matrix

# Phase 2: Constant memory
echo "Benchmarking Phase 2: KMeansCMem"
ncu --set full --launch-count 3 --log-file benchmark2_full.txt -f \
    ./build/release/kmeans_cmem_test \
    -e data/9/output.raw \
    -i data/9/input.raw,data/9/init.raw -t matrix

# Phase 3: Shared memory tiling
echo "Benchmarking Phase 3: KMeansShared"
ncu --set full --launch-count 3 --log-file benchmark3_full.txt -f \
    ./build/release/kmeans_shared_test \
    -e data/9/output.raw \
    -i data/9/input.raw,data/9/init.raw -t matrix

# Phase 4: Thread coarsening
echo "Benchmarking Phase 4: KMeansCoarse"
ncu --set full --launch-count 3 --log-file benchmark4_full.txt -f \
    ./build/release/kmeans_coarse_test \
    -e data/9/output.raw \
    -i data/9/input.raw,data/9/init.raw -t matrix

# Phase 5: Memory coalescing
echo "Benchmarking Phase 5: KMeansCoalesced"
ncu --set full --launch-count 3 --log-file benchmark5_full.txt -f \
    ./build/release/kmeans_coalesced_test \
    -e data/9/output.raw \
    -i data/9/input.raw,data/9/init.raw -t matrix

# Phase 6: Final combined
echo "Benchmarking Phase 6: KMeansFinal"
ncu --set full --launch-count 3 --log-file benchmark6_full.txt -f \
    ./build/release/kmeans_final_test \
    -e data/9/output.raw \
    -i data/9/input.raw,data/9/init.raw -t matrix
