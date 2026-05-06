# GPU K-Means Clustering

CUDA implementation of K-Means clustering with iterative GPU optimizations, benchmarked against a CPU baseline.

## Implementations

| Phase | Directory | Optimization |
|-------|-----------|--------------|
| CPU | `KMeansCPU/` | Baseline CPU reference |
| 1 | `KMeansBasic/` | Naive GPU kernel |
| 2 | `KMeansCMem/` | Constant memory for centroids |
| 3 | `KMeansShared/` | Shared memory tiling |
| 4 | `KMeansCoarse/` | Thread coarsening |
| 5 | `KMeansCoalesced/` | Memory coalescing |
| 6 | `KMeansFinal/` | Combined optimizations |

## Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

Requires CUDA and CMake 3.18+. Automatically fetches [libgputk](https://github.com/ajdillhoff/libgputk) if not found on the system.

## Run

```bash
# Run correctness tests across all phases
./run_tests.sh

# Run Nsight Compute benchmarks (requires GPU + ncu)
./run_benchmark.sh       # Quick pass (--set basic)
./run_benchmark_full.sh  # Full profiling pass
```

Test data lives in `data/` (gitignored). Benchmark results are written to `benchmark*.txt`.
