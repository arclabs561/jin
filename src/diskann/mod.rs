//! DiskANN: Disk-based approximate nearest neighbor search.
//!
//! # Feature Flag
//!
//! Requires the `diskann` feature:
//! ```toml
//! vicinity = { version = "0.1", features = ["diskann"] }
//! ```
//!
//! # Status: Experimental
//!
//! This module is under active development. The current implementation stores
//! data in memory during construction - true disk-based operation is planned.
//!
//! # The Problem
//!
//! HNSW and other in-memory indices require all data in RAM. For 1 billion
//! 768-dim vectors at float32, that's ~3TB. DiskANN solves this by storing
//! vectors and graph structure on SSD, keeping only a small cache in memory.
//!
//! # Key Insight: Vamana + Disk Layout
//!
//! DiskANN combines:
//! 1. **Vamana graph**: Alpha-pruned proximity graph (like HNSW but single-layer)
//! 2. **Disk-optimized layout**: Co-locate vectors with their adjacency lists
//! 3. **Beam search with prefetching**: Hide SSD latency via async I/O
//!
//! ```text
//! Memory:  [Cache: hot nodes] + [Navigation metadata]
//!              ↓ cache miss
//! SSD:     [Node 0: vector + neighbors][Node 1: vector + neighbors]...
//! ```
//!
//! # Why Single-Layer?
//!
//! Unlike HNSW's hierarchy, DiskANN uses a flat graph because:
//! - Hierarchy adds random I/O (jumping between layers)
//! - Single layer = sequential reads possible
//! - Vamana's alpha-pruning provides long-range connections without layers
//!
//! # Parameters
//!
//! | Parameter | Typical | Effect |
//! |-----------|---------|--------|
//! | `m` | 32-64 | Max edges per node. Higher = better recall, more I/O |
//! | `alpha` | 1.2-1.4 | Pruning aggressiveness. Higher = sparser, faster |
//! | `ef_construction` | 100-200 | Build quality. Higher = slower build, better graph |
//! | `ef_search` | 50-200 | Search quality. Higher = better recall, more I/O |
//!
//! # Performance Expectations
//!
//! On NVMe SSD (from the paper):
//! - 1B vectors, 95% recall@10: ~5ms latency
//! - Throughput: ~5000 QPS per node
//! - Memory: ~100GB for graph metadata (vectors on disk)
//!
//! # When to Use
//!
//! - Dataset > available RAM
//! - Can tolerate 1-10ms latency (vs <1ms for in-memory)
//! - Have fast SSD (NVMe preferred)
//!
//! # When NOT to Use
//!
//! - Dataset fits in RAM (use HNSW instead)
//! - Need sub-millisecond latency
//! - Only have HDD (seek time kills performance)
//!
//! # Usage
//!
//! ```ignore
//! use vicinity::diskann::{DiskANNIndex, DiskANNParams};
//!
//! let params = DiskANNParams {
//!     m: 32,
//!     alpha: 1.2,
//!     ..Default::default()
//! };
//!
//! let mut index = DiskANNIndex::new(128, params);
//! // Note: Current impl stores in memory during build
//! index.add(0, vec![0.1; 128]);
//! index.build()?;
//!
//! let results = index.search(&query, 10)?;
//! ```
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
