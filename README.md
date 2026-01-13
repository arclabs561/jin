# vicinity

Approximate Nearest Neighbor search in Rust.

## Algorithms

- **HNSW**: Hierarchical Navigable Small World graphs
- **DiskANN**: Disk-based ANN for datasets larger than memory
- **IVF-PQ**: Inverted file with product quantization
- **LSH**: Locality-sensitive hashing (SimHash, MinHash)
- **Quantization**: Scalar, Product, and RaBitQ quantization

## Features

- Pure Rust, no external dependencies for core algorithms
- SIMD-accelerated distance computations
- Serialization support (optional `serde` feature)
- Comprehensive benchmarks

## Usage

```rust
use vicinity::hnsw::Hnsw;

let mut index = Hnsw::new(128, 16, 200);  // dim, M, ef_construction
index.insert(&vector, id);
let neighbors = index.search(&query, k, ef_search);
```

## License

MIT OR Apache-2.0
