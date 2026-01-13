# vicinity

Approximate Nearest Neighbor search in Rust. HNSW, DiskANN, IVF-PQ, ScaNN, and Quantization.

Dual-licensed under MIT or Apache-2.0.

```rust
use vicinity::hnsw::Hnsw;

// dim=128, M=16, ef_construction=200
let mut index = Hnsw::new(128, 16, 200);
index.insert(&vector, id);

// k=10, ef_search=50
let neighbors = index.search(&query, 10, 50);
```

## Features

- **HNSW**: Hierarchical Navigable Small World (state-of-the-art memory-based)
- **DiskANN**: SSD-optimized Vamana graph for larger-than-memory indices
- **IVF-PQ**: Inverted File with Product Quantization (FAISS-style)
- **ScaNN**: Score-aware quantization with anisotropic loss
- **Quantization**:
  - **RaBitQ**: Randomized Binary Quantization
  - **TurboQuant**: Fast SIMD product quantization
  - **SAQ**: Simulated Annealing Quantization
- **Persistence**: WAL-based durability and fast restarts

See [`docs/README_DETAILED.md`](docs/README_DETAILED.md) for benchmarks and architecture.
