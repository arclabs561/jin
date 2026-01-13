//! Tree-based ANN methods.

#[cfg(feature = "annoy")]
pub mod rp_forest;  // Random Projection Forest (Annoy-style)

#[cfg(feature = "kdtree")]
pub mod kdtree;

#[cfg(feature = "balltree")]
pub mod balltree;

#[cfg(feature = "rptree")]
pub mod random_projection;

#[cfg(feature = "kmeans_tree")]
pub mod kmeans_tree;
