//! Unified Approximate Nearest Neighbor (ANN) search algorithms.
//!
//! Pure Rust implementations of state-of-the-art ANN algorithms:
//! - **HNSW**: Hierarchical Navigable Small World (graph-based) - see [`crate::hnsw`]
//! - **NSW**: Flat Navigable Small World (single-layer graph) - see [`crate::nsw`]
//! - **AnisotropicVQ-kmeans**: Anisotropic Vector Quantization with k-means Partitioning
//!   (vendor name: SCANN/ScaNN) - see [`crate::scann`]
//! - **IVF-PQ**: Inverted File Index with Product Quantization - see [`crate::ivf_pq`]
//! - **DiskANN**: Disk-based ANN for very large datasets - see [`crate::diskann`]
//!
//! All algorithms are optimized with SIMD acceleration and minimal dependencies.
//!
//! Note: not every algorithm is enabled by default; many are feature-gated. The factory
//! (`factory::index_factory`) is a convenience layer and does not expose every possible
//! parameter for every index type.
//!
//! # Index Factory
//!
//! Use `factory::index_factory()` to create indexes from string descriptions:
//!
//! ```rust,ignore
//! use jin::ann::{index_factory, ANNIndex};
//!
//! // Create HNSW index (requires "hnsw" feature)
//! let mut index = index_factory(128, "HNSW32")?;
//! let v0 = vec![0.1; 128];
//! index.add_slice(0, &v0)?;
//! index.build()?;
//! ```

// Autotune for automatic parameter optimization
#[cfg(any(feature = "hnsw", feature = "ivf_pq",))]
pub mod autotune;

#[cfg(any(
    feature = "hnsw",
    feature = "nsw",
    feature = "ivf_pq",
    feature = "scann",
))]
pub mod factory;
pub mod traits;

#[cfg(any(
    feature = "hnsw",
    feature = "nsw",
    feature = "ivf_pq",
    feature = "scann",
))]
pub use factory::{index_factory, AnyANNIndex};
pub use traits::{ANNIndex, ANNStats};
