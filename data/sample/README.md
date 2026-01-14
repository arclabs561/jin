# Bundled Sample Datasets

Pre-generated datasets for benchmarking without external downloads.

## Datasets

| Name | Train | Test | Dims | Size | Difficulty | Relative Contrast (measured) |
|------|-------|------|------|------|------------|-------------------|
| quick | 2,000 | 200 | 128 | ~1MB | Easy | 1.416 |
| bench | 10,000 | 500 | 384 | ~16MB | Medium | 2.833 |
| hard | 10,000 | 500 | 768 | ~31MB | Hard | 1.130 |

## What Makes "hard" Hard (and realistic)?

This dataset aims to resemble *real embedding corpora* rather than being purely adversarial.

1. **Anisotropy + topic mixture**
   - Vectors live mostly in a low-rank subspace (rank≈64 inside 768d).
   - Topics follow a Zipf-like long tail (few large topics, many small).

2. **Near-duplicates**
   - We inject near-duplicate vectors to mimic repeated/templated content.
   - Controlled by `JIN_HARD_DUP_FRAC` (default: 0.10).

3. **Hard-tail queries**
   - Most queries are in-distribution.
   - A smaller slice is selected for tiny top1–top2 similarity margins.

4. **High dimensionality**
   - 768 dims (matches many transformer embedding models).

## Measuring Recall

These datasets are synthetic and we occasionally retune them. Treat any “expected recall”
numbers as stale unless they come from a fresh run.

```sh
cargo run --example 03_quick_benchmark --release
JIN_DATASET=hard cargo run --example 03_quick_benchmark --release

# Scenarios
JIN_DATASET=hard JIN_TEST_VARIANT=drift cargo run --example 03_quick_benchmark --release
JIN_DATASET=hard JIN_TEST_VARIANT=filter cargo run --example 03_quick_benchmark --release
```

## Usage

```sh
# Easy (CI)
JIN_DATASET=quick cargo run --example 03_quick_benchmark --release

# Medium (default)
cargo run --example 03_quick_benchmark --release

# Hard (stress test)
JIN_DATASET=hard cargo run --example 03_quick_benchmark --release
```

## File Format

```
Vectors: VEC1 (4 bytes) + n (u32) + dim (u32) + data (f32 * n * dim)
Neighbors: NBR1 (4 bytes) + n (u32) + k (u32) + data (i32 * n * k)
Labels: LBL1 (4 bytes) + n (u32) + labels (u32 * n)
```

## Regenerating

```sh
uvx --with numpy python scripts/generate_sample_data.py
```

## References

- He, Kumar, Chang. "On the Difficulty of Nearest Neighbor Search" (ICML 2012)
- Radovanovic et al. "Hubs in Space" (JMLR 2010)
- Patel et al. "ACORN" (arXiv 2403.04871)
- Jaiswal et al. "OOD-DiskANN" (arXiv 2211.12850)
- Iff et al. "Benchmarking Filtered ANN on transformer embeddings" (arXiv 2507.21989)
