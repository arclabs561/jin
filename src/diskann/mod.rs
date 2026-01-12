//! DiskANN implementation.
//!
//! Disk-based approximate nearest neighbor search for very large datasets
//! that don't fit in memory.
//!
//! # References
//!
//! - Jayaram Subramanya et al. (2019): "DiskANN: Fast Accurate Billion-point
//!   Nearest Neighbor Search on a Single Node"
//!
//! # Status: Experimental
//!
//! This module is under active development. Some fields and methods are
//! placeholders for future functionality.

#![allow(dead_code)]

mod cache;
mod disk_io;
mod graph;

pub use graph::DiskANNIndex;
pub use graph::DiskANNParams;
