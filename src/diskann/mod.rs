//! DiskANN: Billion-scale ANN on a single machine with SSD.
//!
//! Search 1B+ vectors with ~5ms latency using NVMe storage.
//!
//! # Feature Flag
//!
//! ```toml
//! vicinity = { version = "0.1", features = ["diskann"] }
//! ```
//!
//! # Status: Experimental
//!
//! Current implementation stores data in memory during construction.
//! True disk-based operation is planned.
//!
//! # Quick Start
//!
//! ```ignore
//! use vicinity::diskann::{DiskANNIndex, DiskANNParams};
//!
//! let params = DiskANNParams {
//!     m: 32,
//!     alpha: 1.2,
//!     ef_construction: 100,
//!     ef_search: 100,
//! };
//!
//! let mut index = DiskANNIndex::new(128, params);
//! index.add(0, vec![0.1; 128]);
//! index.build()?;
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # The Problem: 1B Vectors Don't Fit in RAM
//!
//! ```text
//! 1B vectors × 768 dims × 4 bytes = 3 TB
//! ```
//!
//! HNSW/NSW require all data in memory. DiskANN keeps vectors on SSD.
//!
//! # How It Works
//!
//! ```text
//! Memory:  [Cache: ~1% hot nodes] + [Beam search state]
//!              ↓ cache miss
//! NVMe:    [Node 0: vector + edges][Node 1: vector + edges]...
//!              └─ 4KB aligned for efficient I/O
//! ```
//!
//! 1. **Vamana graph**: Single-layer (no hierarchy = sequential I/O)
//! 2. **Co-located storage**: Vector + neighbor list in same disk block
//! 3. **Beam search + prefetch**: Hide SSD latency with async I/O
//!
//! # Performance (from Microsoft paper)
//!
//! | Scale | Recall@10 | Latency | Memory | Storage |
//! |-------|-----------|---------|--------|---------|
//! | 100M | 95% | ~1ms | 8 GB | 50 GB |
//! | 1B | 95% | ~5ms | 64 GB | 500 GB |
//!
//! **Throughput**: ~5000 QPS per node on NVMe
//!
//! # Parameter Recommendations
//!
//! | Dataset | m | alpha | ef_construction |
//! |---------|---|-------|-----------------|
//! | 100M | 32 | 1.2 | 100 |
//! | 1B | 64 | 1.2 | 200 |
//! | > 1B | 96 | 1.4 | 400 |
//!
//! # When to Use
//!
//! - **Dataset exceeds RAM** (the only reason to use this)
//! - Have **NVMe SSD** (HDD is 100x slower)
//! - Can tolerate **1-10ms latency** (vs <1ms in-memory)
//!
//! # When NOT to Use
//!
//! - Dataset fits in RAM → use HNSW (10x faster)
//! - Only have HDD → seek time makes this impractical
//! - Need <1ms latency → use in-memory index with smaller dataset
//!
//! # Why Single-Layer (Vamana) Instead of HNSW?
//!
//! - Hierarchy = random I/O (jump between layers on disk = slow)
//! - Flat graph = sequential reads possible
//! - Alpha-pruning provides long-range connections without layers
//!
//! # References
//!
//! - Jayaram Subramanya et al. (2019). "DiskANN: Fast Accurate Billion-point
//!   Nearest Neighbor Search on a Single Node."
//! - See also: [`vamana`](crate::vamana) for the graph construction algorithm

#![allow(dead_code)]

mod cache;
mod disk_io;
mod graph;

pub use graph::DiskANNIndex;
pub use graph::DiskANNParams;
