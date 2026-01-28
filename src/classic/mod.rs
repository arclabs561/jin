//! Classic ANN methods implementation.
//!
//! # Status: Experimental
//!
//! Classic tree-based methods for comparison with modern graph methods.

#![allow(dead_code)]

// Note: hash/LSH methods live in `sketchir`.

#[cfg(any(
    feature = "kdtree",
    feature = "balltree",
    feature = "rptree",
    feature = "kmeans_tree"
))]
pub mod trees;
