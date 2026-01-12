//! OPT-SNG (Optimized Sparse Neighborhood Graph) implementation.
//!
//! Pure Rust implementation of the 2026 OPT-SNG algorithm with:
//! - Martingale-based theoretical model
//! - Automatic parameter optimization (no manual tuning)
//! - 5.9× average construction speedup (15.4× peak)
//! - Theoretical guarantees: O(log n) search path
//!
//! # References
//!
//! - Ma et al. (2026): "Graph-Based Approximate Nearest Neighbor Search Revisited:
//!   Theoretical Analysis and Optimization" - https://arxiv.org/abs/2509.15531
//!
//! # Status: Experimental
//!
//! This module is under active development. Some fields and methods are
//! placeholders for future optimization strategies.

#![allow(dead_code)]

mod graph;
mod martingale;
mod optimization;
mod search;

pub use graph::{SNGIndex, SNGParams};
