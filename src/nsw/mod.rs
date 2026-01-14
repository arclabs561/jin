//! Flat Navigable Small World (NSW) graph.
//!
//! Single-layer variant of HNSW. Same search quality, ~25% less memory.
//!
//! # Feature Flag
//!
//! Requires the `nsw` feature:
//! ```toml
//! plesio = { version = "0.1", features = ["nsw"] }
//! ```
//!
//! # Quick Start
//!
//! ```ignore
//! use plesio::nsw::{NSWIndex, NSWParams};
//!
//! let params = NSWParams { m: 16, ef_construction: 200 };
//! let mut index = NSWIndex::new(768, params);
//!
//! index.add(0, vec![0.1; 768])?;
//! index.build()?;
//!
//! let results = index.search(&query, 10, 50)?;  // k=10, ef=50
//! ```
//!
//! # Why Flat?
//!
//! HNSW's hierarchy (multiple layers) was designed to provide "express lanes"
//! for long-range navigation. But research (2024-2025) shows:
//!
//! - **In high dimensions (d > 32)**, natural "hubs" emerge in the data
//! - These hubs serve the same routing function as explicit hierarchy
//! - The multi-layer overhead becomes redundant
//!
//! ```text
//! HNSW (3 layers):          NSW (1 layer):
//!   Memory: 100%              Memory: ~75%
//!   Build time: 100%          Build time: ~80%
//!   Search QPS: 100%          Search QPS: ~100%  <-- same!
//! ```
//!
//! # When to Use
//!
//! | Situation | Recommendation |
//! |-----------|----------------|
//! | d > 32, memory-constrained | **NSW** |
//! | d < 32, need maximum robustness | HNSW |
//! | Want simpler code to understand | **NSW** |
//! | Production with many edge cases | HNSW (more battle-tested) |
//!
//! # Memory Comparison
//!
//! For 1M vectors at d=768, M=16:
//! - **HNSW**: ~3.5 GB (multi-layer storage)
//! - **NSW**: ~2.7 GB (single layer)
//! - **Savings**: ~800 MB
//!
//! # The Small World Property
//!
//! Even without hierarchy, a well-constructed graph has "six degrees of separation."
//! Greedy routing finds near-optimal paths in O(log n) hops:
//!
//! ```text
//! Query → Hub node → Intermediate → Target (3-4 hops typical)
//! ```
//!
//! Hubs emerge naturally from degree distribution and serve as express lanes.
//!
//! # References
//!
//! - [Why hierarchy doesn't help in high-d](https://arxiv.org/abs/2412.01940) (2024)
//! - See [`hnsw`](crate::hnsw) for the hierarchical variant

pub mod construction;
pub mod graph;
pub mod search;

pub use graph::{NSWIndex, NSWParams};
