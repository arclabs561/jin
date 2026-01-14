# Documentation Audit Results

Generated from testing and improving all documentation.

## Summary (After Jan 2026 Polish)

| Metric | Before | After |
|--------|--------|-------|
| Doc tests passing | 1/17 | 2/21 |
| Decision tables | 1 | 8 |
| Memory formulas | 0 | 4 |
| Parameter rec tables | 0 | 6 |
| Grade C → A modules | 3 | 0 (ALL UPGRADED) |

## Line Counts (Documentation Only)

| Module | Before | After | Change |
|--------|--------|-------|--------|
| hnsw | 213 | 185 | -28 (tighter) |
| ivf_pq | 105 | 126 | +21 (memory calc) |
| nsw | 49 | 78 | +29 (decision guide) |
| diskann | 23 | 104 | +81 |
| sng | 26 | 103 | +77 |
| scann | 29 | 114 | +85 |
| quantization | 123 | 140 | +17 (decision table) |
| **Total** | 568 | 850 | +282 |

## Key Additions

1. **Decision Tables**: "Which X Should I Use?" in every major module
2. **Memory Formulas**: HNSW, IVF-PQ, NSW, DiskANN
3. **Parameter Recommendations**: By dataset size (100K, 1M, 10M, 100M, 1B)
4. **Performance Expectations**: Real numbers from papers/benchmarks
5. **When NOT to Use**: Explicit counter-recommendations

## Test Results

```
jin: 2 passed, 19 ignored (feature-gated)
stratify: 3 passed, 4 ignored (need external data)
innr: compiles
hyperball: compiles
```

## Detailed Findings

### 1. HNSW Example: WORKS

```rust
use jin::hnsw::HNSWIndex;
let mut index = HNSWIndex::new(128, 16, 16)?;
index.add(0, vec![0.1; 128])?;
index.build()?;
let results = index.search(&vec![0.15; 128], 10, 50)?;
```

**Status**: Compiles and runs. Should change `rust,ignore` to `rust,no_run` or actual doctest.

### 2. IVF-PQ Example: WRONG FIELD NAMES

**Documentation says**:
```rust
IVFPQParams {
    num_centroids: 1024,    // WRONG
    num_codebooks: 8,
    codebook_bits: 8,       // WRONG
    nprobe: 10,
}
```

**Actual struct**:
```rust
IVFPQParams {
    num_clusters: 1024,     // CORRECT
    num_codebooks: 8,
    codebook_size: 256,     // CORRECT (and semantic: 256 codewords, not 8 bits)
    nprobe: 10,
}
```

**Fix**: Update `ivf_pq/mod.rs` lines 71-74.

### 3. Vamana Example: MISSING FEATURE GATE

Example uses `jin::vamana::{VamanaIndex, VamanaParams}` but:
- Requires `features = ["vamana"]`
- Not mentioned in docs

**Fix**: Add to `vamana/mod.rs`:
```rust
//! # Feature Flag
//!
//! This module requires the `vamana` feature:
//! ```toml
//! jin = { version = "0.1", features = ["vamana"] }
//! ```
```

### 4. RaBitQ Example: MISSING FEATURE GATE

Requires `features = ["rabitq"]`, not mentioned.

### 5. Streaming Example: MISSING FEATURE GATE

Requires `features = ["hnsw"]` (for `HNSWIndex`).

### 6. Factory Examples: STALE IMPORT PATH

```rust
// WRONG (stale):
use ordino_retrieve::dense::ann::factory::index_factory;

// CORRECT:
use jin::ann::factory::index_factory;
```

Affected files:
- `ann/mod.rs` line 17
- `ann/factory.rs` line 9, 225
- `ann/autotune.rs` line 10

## Feature Gate Matrix

| Module | Feature Required | Default? | Documented? |
|--------|------------------|----------|-------------|
| hnsw | `hnsw` | Yes | No |
| ivf_pq | `ivf_pq` | No | No |
| vamana | `vamana` | No | No |
| diskann | `diskann` | No | No |
| scann | `scann` | No | No |
| sng | `sng` | No | No |
| rabitq | `rabitq` | No | No |
| nsw | `nsw` | No | No |
| persistence | `persistence` | No | No |
| streaming | (uses hnsw) | - | No |

## Recommended Actions

### Priority 1: Fix incorrect examples

1. **IVF-PQ field names** - wrong is wrong, fix immediately

### Priority 2: Add feature documentation

Every feature-gated module needs this in its `mod.rs`:

```rust
//! # Feature Flag
//!
//! Requires `features = ["<feature>"]`:
//! ```toml
//! jin = { version = "0.1", features = ["<feature>"] }
//! ```
```

### Priority 3: Fix stale imports

Replace `ordino_retrieve` with `jin` in 3 files.

### Priority 4: Enable doctests

Change `rust,ignore` to:
- `rust` if example should compile and run
- `rust,no_run` if example compiles but shouldn't run in tests
- `rust,compile_fail` if showing what NOT to do

Keep `rust,ignore` only for pseudocode or incomplete snippets.

## Test Command

After fixes, verify with:

```bash
# Test all features
cargo test --doc --all-features

# Should have 0 ignored, all pass
```
