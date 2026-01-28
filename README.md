# jin

Approximate nearest neighbor search in Rust.

Algorithms and benchmarks for vector search.

(jin: Chinese 近 "near")

Dual-licensed under MIT or Apache-2.0.

## Distance metrics (what `jin` actually does today)

Different index implementations in `jin` currently assume different distance semantics.
This is not yet uniform across the crate.

| Component | Metric | Notes |
|---|---|---|
| `hnsw::HNSWIndex` | cosine distance | Fast path assumes **L2-normalized** vectors |
| `ivf_pq::IVFPQIndex` | cosine distance | Uses dot-based cosine distance for IVF + PQ |
| `scann::SCANNIndex` | inner product / cosine | Uses dot products; reranking uses cosine distance |
| `hnsw::dual_branch::DualBranchHNSW` | L2 distance | Experimental implementation |
| `quantization` | Hamming-like / binary distances | See `quantization::simd_ops::hamming_distance` and ternary helpers |

```rust
use jin::hnsw::HNSWIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Vectors should be L2-normalized for cosine distance.
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

## The Problem

Given a query vector, find the k most similar vectors from a collection. Brute force computes all N distances — O(N) per query. For 1M vectors, that's 1M distance computations per query.

**ANN algorithms trade exactness for speed.** Instead of guaranteeing the true nearest neighbors, they find neighbors that are *probably* correct, *most* of the time.

## The Key Insight

HNSW (Hierarchical Navigable Small World) builds a graph where:
1. Each vector is a node
2. Edges connect similar vectors
3. Multiple layers provide "highway" shortcuts

Search starts at the top layer (sparse, long-range edges), descends through layers, and greedily follows edges toward the query.

```text
Layer 2:  A -------- B          (sparse, fast traversal)
          |          |
Layer 1:  A -- C -- B -- D      (medium density)
          |    |    |    |
Layer 0:  A-E-C-F-B-G-D-H      (dense, high recall)
```

## Recall vs Speed Tradeoff

The `ef_search` parameter controls how many candidates HNSW explores:

<p align="center">
  <img src="doc/plots/recall_vs_ef.png" width="720" alt="Recall vs ef_search" />
</p>

Higher `ef_search` = better recall, slower queries. The “right” range is workload-dependent; start with `ef_search=50-100` and measure recall@k vs latency for your dataset.

## Dataset Difficulty

Not all datasets are equal. Recall depends on data characteristics:

<p align="center">
  <img src="doc/plots/recall_vs_ef_by_difficulty.png" width="720" alt="Recall by difficulty" />
</p>

Based on He et al. (2012) and Radovanovic et al. (2010):

- **Relative Contrast**: $C_r = \bar{D} / D_{\min}$. Lower = harder.
- **Hubness**: Some points become neighbors to many queries. Higher = harder.
- **Distance Concentration**: In high dims, distances converge. Lower variance = harder.

<p align="center">
  <img src="doc/plots/difficulty_comparison.png" width="720" alt="Difficulty metrics" />
</p>

```sh
cargo run --example 03_quick_benchmark --release                   # bench (medium)
JIN_DATASET=hard cargo run --example 03_quick_benchmark --release  # hard (stress test)
```

## Algorithm Comparison

Different algorithms suit different constraints:

<p align="center">
  <img src="doc/plots/algorithm_comparison.png" width="720" alt="Algorithm comparison" />
</p>

| Algorithm | Best For | Tradeoff |
|-----------|----------|----------|
| **Brute force** | < 10K vectors | Exact, but O(N) |
| **LSH** | Binary/sparse data | Fast, lower recall (see `sketchir`) |
| **IVF-PQ** | Memory-constrained | Compressed, lower recall |
| **HNSW** | General use | Strong recall/latency tradeoff |
| **NSW (flat graph)** | High-dim embeddings; simpler graph | Often comparable recall/latency; less overhead (measure) |

## Build Cost

Graph construction time scales with `M` (edges per node):

<p align="center">
  <img src="doc/plots/build_time_vs_m.png" width="720" alt="Build time vs M" />
</p>

Higher `M` = better recall, but more memory and slower builds.

## Memory Scaling

Memory usage scales linearly with vector count:

<p align="center">
  <img src="doc/plots/memory_scaling.png" width="720" alt="Memory scaling" />
</p>

For dim=128, M=16: approximately 0.5 KB per vector (vector + graph edges).

## Algorithms

| Type | Implementations |
|------|-----------------|
| Graph | HNSW, NSW, Vamana (DiskANN-style), SNG |
| Hash | LSH, MinHash, SimHash (see `sketchir`) |
| Partition | IVF-PQ, ScaNN |
| Quantization | PQ, RaBitQ |

## Features

```toml
[dependencies]
jin = { version = "0.1", features = ["hnsw"] }
```

- `hnsw` — HNSW graph index (default)
- `nsw` — Flat NSW graph index (single-layer)
- `ivf_pq` — Inverted File with Product Quantization
- `persistence` — WAL-based durability

## Flat vs hierarchical graphs (why “H” may not matter)

HNSW’s hierarchy was designed to provide multi-scale “express lanes” for long-range navigation.
However, recent empirical work suggests that on **high-dimensional embedding datasets** a flat
navigable small-world graph can retain the key recall/latency benefits of HNSW, because “hub” nodes
emerge and already provide effective routing.

Concrete reference:
- Munyampirwa et al. (2024). *Down with the Hierarchy: The 'H' in HNSW Stands for "Hubs"* (arXiv:2412.01940).

Practical guidance in `jin`:
- Try `HNSW{m}` first (default; robust).
- If you want to experiment with a simpler flat graph, enable `nsw` and try `NSW{m}` via the factory.
  Benchmark recall@k vs latency on your workload; the “best” choice depends on data and constraints.

## Performance

Build with native CPU optimizations:

```sh
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

Run benchmarks:

```sh
cargo bench
```

See [examples/](examples/) for more: semantic search, IVF-PQ, LID, and real dataset benchmarks.

For benchmarking datasets, see [doc/datasets.md](doc/datasets.md) — covers bundled data, ann-benchmarks.com datasets, and typical embedding dimensions.

For primary sources (papers) backing the algorithms and phenomena mentioned in docs, see [doc/references.md](doc/references.md).

## References

- Malkov & Yashunin (2016/2018). *Efficient and robust approximate nearest neighbor search using HNSW graphs* (HNSW). `https://arxiv.org/abs/1603.09320`
- Malkov et al. (2014). *Approximate nearest neighbor algorithm based on navigable small world graphs* (NSW). `https://doi.org/10.1016/j.is.2013.10.006`
- Munyampirwa et al. (2024). *Down with the Hierarchy: The “H” in HNSW Stands for “Hubs”*. `https://arxiv.org/abs/2412.01940`
- Subramanya et al. (2019). *DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node*. `https://proceedings.neurips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html`
- Jégou, Douze, Schmid (2011). *Product Quantization for Nearest Neighbor Search* (PQ / IVFADC). `https://ieeexplore.ieee.org/document/5432202`
- Ge et al. (2014). *Optimized Product Quantization* (OPQ). `https://arxiv.org/abs/1311.4055`
- Guo et al. (2020). *Accelerating Large-Scale Inference with Anisotropic Vector Quantization* (ScaNN line). `https://arxiv.org/abs/1908.10396`
