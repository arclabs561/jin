//! OPT-SNG: Optimized Sparse Neighborhood Graph.
//!
//! # Feature Flag
//!
//! Requires the `sng` feature:
//! ```toml
//! vicinity = { version = "0.1", features = ["sng"] }
//! ```
//!
//! # Status: Experimental
//!
//! This module implements the 2026 OPT-SNG algorithm. Under active development.
//!
//! # The Problem with Manual Tuning
//!
//! HNSW and similar algorithms require careful parameter tuning:
//! - **M** (max degree): Too low = poor recall, too high = slow search
//! - **ef_construction**: Trade-off between build time and graph quality
//!
//! Getting these wrong can degrade performance 2-5x. OPT-SNG eliminates this.
//!
//! # Key Insight: Martingale Theory
//!
//! OPT-SNG uses martingale theory to model the stochastic evolution of
//! candidate sets during graph construction.
//!
//! **Martingale property**: The expected candidate set size at step n+1,
//! given the current state, equals the current size. This lets us:
//!
//! 1. Predict when to stop expanding candidates
//! 2. Automatically determine optimal truncation radius R
//! 3. Guarantee O(n^{2/3+ε}) maximum out-degree (theoretical)
//!
//! ```text
//! Without martingale:     With martingale:
//!   Keep all candidates     Truncate at optimal R
//!   Dense graph             Sparse graph
//!   Slow search             Fast search
//! ```
//!
//! # Why 5.9x Faster Construction?
//!
//! Traditional methods compute distances to all candidates, then prune.
//! OPT-SNG uses the martingale model to:
//!
//! 1. **Early termination**: Stop when expected improvement < threshold
//! 2. **Adaptive pruning**: Truncation radius R adapts per-node
//! 3. **Variance tracking**: Use variance estimates to avoid over-pruning
//!
//! Peak speedup of 15.4x on sparse data; average 5.9x across benchmarks.
//!
//! # Parameters
//!
//! | Parameter | Default | Effect |
//! |-----------|---------|--------|
//! | `max_degree` | Auto | Maximum out-degree (auto-optimized) |
//! | `num_hash_functions` | 10 | For LSH-based candidate generation |
//!
//! Most parameters are auto-tuned; you rarely need to set them.
//!
//! # Theoretical Guarantees
//!
//! - **Search path**: O(log n) expected
//! - **Max out-degree**: O(n^{2/3+ε})
//! - **Construction**: O(n^{5/3+ε}) expected (vs O(n²) naive)
//!
//! # When to Use
//!
//! - Don't want to tune parameters
//! - Building many indices with varying data characteristics
//! - Research/experimentation where reproducibility matters
//!
//! # When NOT to Use
//!
//! - Production with well-tuned HNSW (HNSW is more battle-tested)
//! - Very small datasets (< 10K, overhead not worth it)
//!
//! # Usage
//!
//! ```ignore
//! use vicinity::sng::{SNGIndex, SNGParams};
//!
//! // Auto-tuning: just provide dimension
//! let mut index = SNGIndex::new(128, SNGParams::default());
//!
//! index.add(0, vec![0.1; 128]);
//! index.add(1, vec![0.2; 128]);
//! index.build()?;  // Parameters auto-optimized
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! # References
//!
//! - Ma et al. (2026). "Graph-Based Approximate Nearest Neighbor Search Revisited:
//!   Theoretical Analysis and Optimization." arXiv:2509.15531

#![allow(dead_code)]

mod graph;
mod martingale;
mod optimization;
mod search;

pub use graph::{SNGIndex, SNGParams};
