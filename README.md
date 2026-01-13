# vicinity

Approximate Nearest Neighbor search in Rust.

Dual-licensed under MIT or Apache-2.0.

```rust
use vicinity::hnsw::Hnsw;

let mut index = Hnsw::new(128, 16, 200); // dim, M, ef_construction
index.insert(&vector, id);

let neighbors = index.search(&query, 10, 50); // k, ef_search
```

## Algorithms

| Type | Algorithms |
|------|------------|
| Graph | HNSW, NSW, Vamana (DiskANN), SNG |
| Hash | LSH, MinHash, SimHash |
| Partition | IVF-PQ, ScaNN |
| Quantization | PQ, RaBitQ, SAQ |

## When to Use What

- **< 10K vectors**: Brute force (no index overhead)
- **Memory-constrained**: IVF-PQ, quantization
- **Disk-based**: Vamana/DiskANN
- **Default choice**: HNSW (best recall/latency tradeoff)

## Features

- `lsh` — LSH, MinHash, SimHash
- `persistence` — WAL-based durability
- `full` — all features
