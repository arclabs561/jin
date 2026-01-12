//! Tree-based ANN methods.

#[cfg(feature = "annoy")]
pub mod annoy;

#[cfg(feature = "kdtree")]
pub mod kdtree;

#[cfg(feature = "balltree")]
pub mod balltree;

#[cfg(feature = "rptree")]
pub mod random_projection;

#[cfg(feature = "kmeans_tree")]
pub mod kmeans_tree;
