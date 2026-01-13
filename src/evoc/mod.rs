//! EVōC (Embedding Vector Oriented Clustering) implementation.
//!
//! Pure Rust implementation of fast hierarchical clustering for embeddings.
//!
//! **Technical Name**: Embedding Vector Oriented Clustering
//! **Vendor Name**: EVōC (Tutte Institute)
//!
//! Algorithm:
//! - UMAP-style dimensionality reduction to intermediate space (~15 dimensions)
//! - HDBSCAN-style hierarchical clustering (MST-based)
//! - Multi-granularity cluster layers
//! - Automatic cluster number selection
//! - Near-duplicate detection
//!
//! **Relationships**:
//! - Can replace k-means in partitioning stages (SCANN, IVF-PQ)
//! - Provides hierarchical clustering vs flat k-means
//! - Optimized specifically for embedding vectors
//!
//! # References
//!
//! - Tutte Institute: <https://github.com/TutteInstitute/evoc>
//! - Combines UMAP dimensionality reduction + HDBSCAN hierarchical clustering
//! - Optimized for high-dimensional embedding vectors

pub mod clustering;
pub mod hierarchy;
pub mod reduction;

pub use clustering::{ClusterLayer, EVoC, EVoCParams};
pub use hierarchy::{ClusterHierarchy, ClusterNode};
