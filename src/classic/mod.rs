//! Classic ANN methods implementation.
//!
//! # Status: Experimental
//!
//! Classic tree-based methods for comparison with modern graph methods.

#![allow(dead_code)]

#[cfg(feature = "lsh")]
pub mod lsh;

#[cfg(any(
    feature = "annoy",
    feature = "kdtree",
    feature = "balltree",
    feature = "rptree",
    feature = "kmeans_tree"
))]
pub mod trees;
