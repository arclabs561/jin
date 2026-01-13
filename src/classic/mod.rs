//! Classic ANN methods implementation.
//!
//! # Status: Experimental
//!
//! Classic tree-based methods for comparison with modern graph methods.

#![allow(dead_code)]

// Note: LSH methods moved to top-level `hash` module

#[cfg(any(
    feature = "annoy",
    feature = "kdtree",
    feature = "balltree",
    feature = "rptree",
    feature = "kmeans_tree"
))]
pub mod trees;
