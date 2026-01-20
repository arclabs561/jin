<p align="center">
  <img src="doc/readme_logo.png" width="180" alt="jin" />
</p>

# jin

Approximate nearest-neighbor (ANN) search in Rust.

This repo is a working surface for:
- graph-based ANN (HNSW + variants)
- coarse-to-fine indexes (IVF-PQ, ScaNN-style ideas)
- quantization / compression experiments

- **Guide**: [GUIDE.md](GUIDE.md)
- **Datasets**: [doc/datasets.md](doc/datasets.md)
- **Testing**: [TESTING.md](TESTING.md)
- **Research notes**: [docs/ANN_RESEARCH_2024_2026.md](docs/ANN_RESEARCH_2024_2026.md)

## Quickstart

Add the crate:

```toml
[dependencies]
jin = "0.1.0"
```

Build an HNSW index and query it:

```rust
use jin::hnsw::HNSWIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // NOTE: `HNSWIndex` currently uses cosine distance.
    // That implies L2-normalized vectors.
    let vectors: Vec<Vec<f32>> = vec![
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
        vec![0.0, 0.0, 0.0, 1.0],
    ];

    let query = vec![1.0, 0.0, 0.0, 0.0];

    let mut index = HNSWIndex::new(4, 16, 32)?; // dim, m, m_max
    for (id, v) in vectors.iter().enumerate() {
        index.add_slice(id as u32, v)?;
    }
    index.build()?;

    let results = index.search(&query, 2, 50)?; // k, ef_search
    println!("{results:?}");
    Ok(())
}
```

## The problem

Given a query vector, find the top-k most similar vectors from a collection.
Brute force computes all N distances (O(N) per query). For 1,000,000 vectors,
that’s 1,000,000 distance computations per query.

ANN systems trade exactness for speed: they aim for **high recall** at much lower latency.

## The key idea (graph search, not magic)

HNSW builds a multi-layer graph where each point has:
- a few long edges (good for jumping across the space)
- and more local edges near the bottom (good for refinement)

A query does a greedy, coarse-to-fine walk:
- **start** from an entry point at the top layer
- **greedily descend** toward the query through progressively denser layers
- **maintain a candidate set** (size `ef_search`) at the bottom to avoid getting stuck

A more accurate mental model than “shortcuts” is:
**HNSW is a cheap way to keep multiple plausible local minima alive until you can locally refine.**

```text
Layer 2 (coarse):      o---------o
                        \       /
                         \  o  /
                          \ | /
Layer 1:          o---o---o-o---o---o
                    \      |      /
                     \     |     /
Layer 0 (dense):  o--o--o--o--o--o--o--o
                         ^
                 keep ~ef_search candidates here,
                 return the best k
```

## Tuning knobs (HNSW)

### `ef_search` (query effort)

In HNSW, `ef_search` controls how many candidates you keep during the bottom-layer search.
Larger values usually increase recall, at the cost of query time.

<p align="center">
  <img src="doc/plots/recall_vs_ef.png" width="720" alt="Recall vs ef_search" style="border-radius: 10px; box-shadow: 0 12px 30px rgba(0,0,0,0.18);" />
</p>

Notes:
- This plot is from `jin`’s bundled “quick” profile (it’s meant to show the *shape* of the curve).
- It does **not** justify a universal claim like “ef_search=50–100 gives >95% recall” for all datasets.

### `M` / graph degree (build-time and memory)

Higher `M` generally improves recall, but increases build time and memory.

<p align="center">
  <img src="doc/plots/build_time_vs_m.png" width="720" alt="Build time vs M" style="border-radius: 10px; box-shadow: 0 12px 30px rgba(0,0,0,0.18);" />
</p>

<p align="center">
  <img src="doc/plots/memory_scaling.png" width="720" alt="Memory scaling" style="border-radius: 10px; box-shadow: 0 12px 30px rgba(0,0,0,0.18);" />
</p>

Notes:
- These plots are for the labeled settings (e.g. 1K vectors for build-time; dim=128, M=16 for memory).
- Treat them as *sanity checks*, not as a stable performance contract.

## Distance semantics (current behavior)

Different components currently assume different distance semantics.
This is intentionally surfaced here because it’s an easy place to make silent mistakes
(e.g. forgetting to normalize vectors).

| Component | Metric | Notes |
|---|---|---|
| `hnsw::HNSWIndex` | cosine distance | Fast path assumes **L2-normalized** vectors |
| `ivf_pq::IVFPQIndex` | cosine distance | Uses dot-based cosine distance for IVF + PQ |
| `scann::SCANNIndex` | inner product / cosine | Uses dot products; reranking uses cosine distance |
| `hnsw::dual_branch::DualBranchHNSW` | L2 distance | Experimental implementation |
| `quantization` | Hamming-like / binary distances | See `quantization::simd_ops::hamming_distance` and ternary helpers |

Planned direction: make distance semantics explicit via a shared metric/normalization contract
so that “same input vectors” means “same meaning” across indexes.

## Algorithms

| Type | Implementations |
|---|---|
| Graph | HNSW, NSW, Vamana (DiskANN), SNG |
| Hash | LSH, MinHash, SimHash |
| Partition | IVF-PQ, ScaNN |
| Quantization | PQ, RaBitQ |

## Features

```toml
[dependencies]
jin = { version = "0.1.0", features = ["hnsw"] }
```

- `hnsw` — HNSW graph index (default)
- `lsh` — locality-sensitive hashing
- `ivf_pq` — inverted file with product quantization
- `persistence` — WAL-based durability

## Running benchmarks / examples

Benchmarks:

```sh
cargo bench
```

Example benchmark driver:

```sh
cargo run --example 03_quick_benchmark --release
JIN_DATASET=hard cargo run --example 03_quick_benchmark --release
```

See [examples/](examples/) for more: semantic search, IVF-PQ, LSH, LID, and real dataset benchmarks.

## References

- Malkov & Yashunin (2018). [Efficient and robust approximate nearest neighbor search using HNSW graphs](https://arxiv.org/abs/1603.09320)
- Subramanya et al. (2019). [DiskANN: Fast Accurate Billion-point Nearest Neighbor Search](https://proceedings.neurips.cc/paper/2019/file/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Paper.pdf)
