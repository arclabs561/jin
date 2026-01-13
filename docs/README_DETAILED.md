# vicinity

Approximate Nearest Neighbor search in Rust. HNSW, DiskANN, IVF-PQ, ScaNN, quantization.

Dual-licensed under MIT or Apache-2.0.

```rust
use vicinity::hnsw::Hnsw;

// dim=128, M=16, ef_construction=200
let mut index = Hnsw::new(128, 16, 200);
index.insert(&vector, id);

// k=10, ef_search=50
let neighbors = index.search(&query, 10, 50);
```

## Advanced Algorithms (Research & Experimental)

- **DiskANN**: Disk-based Vamana graph for massive datasets
- **ScaNN**: Anisotropic vector quantization
- **Quantization**: RaBitQ, SAQ, TurboQuant
- **Persistence**: Crash-safe WAL and fast recovery

## Features

- **HNSW**: Hierarchical Navigable Small World
- **IVF-PQ**: Inverted File with Product Quantization
- **LSH**: Locality Sensitive Hashing (SimHash, MinHash)
- **Zero Dependencies**: Pure Rust for core algorithms

## License

MIT OR Apache-2.0
