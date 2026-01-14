# Multi-Scale ANN Benchmarks

We test `jin` against datasets of increasing size and difficulty to understand its scaling behavior and failure modes.

## Scales

| Scale | Vectors | Use Case |
|-------|---------|----------|
| **S** | 10,000 | CI, quick validation (<1 min) |
| **M** | 100,000 | Development, parameter tuning (~5 min) |
| **L** | 1,000,000 | Production baseline (~30 min) |
| **XL** | 10,000,000 | Stress test (requires >32GB RAM) |

## Methodology

### 1. Data Generation (`scripts/generate_multiscale_data.py`)
We don't use random uniform data (which is too easy). We generate **clustered** data with:
- **Zipfian cluster sizes**: Some topics are popular, others rare.
- **Query Perturbations**: Queries are training vectors + noise, ensuring ground truth exists.
- **Concept Drift**: We shift cluster centers for the query set to simulate "semantic drift" (e.g. user intent vs indexed content).

### 2. Difficulty Stratification
We compute **Local Intrinsic Dimensionality (LID)** for every query.
- **Easy**: Dense regions, low LID. HNSW finds these easily.
- **Hard**: Sparse regions, high LID. These usually require higher `ef` search parameters.

### 3. Evaluation (`examples/04_rigorous_benchmark.rs`)
- **5 Runs**: We average results to smooth out OS scheduler noise.
- **95% Confidence Intervals**: We report error bars, not just means.
- **Brute Force Baseline**: We measure exact speedup over optimized linear scan.

## How to Run

**Prerequisites:** `cargo`, `uvx` (python)

```bash
# Run everything (S, M, L)
just all

# Or run specific steps
just gen S
just bench S
just plot
```

## Output
Results are saved to `data/multiscale/results_{SCALE}.json` and visualized in `data/multiscale/plots/benchmark_report.html`.
