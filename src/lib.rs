//! vicinity: Approximate Nearest Neighbor Search primitives.
//!
//! Provides standalone implementations of state-of-the-art ANN algorithms
//! organized by algorithmic approach:
//!
//! - `graph/`: Graph-based (HNSW, NSW, Vamana, SNG)
//! - `tree/`: Tree-based (K-D tree, Ball tree, RP forest)
//! - `hash/`: Hash-based (LSH, MinHash, SimHash)
//! - `partition/`: Partition-based (IVF, ScaNN)
//! - `quantize/`: Quantization (PQ, RaBitQ, SAQ)

pub mod ann;
pub mod classic;
pub mod diskann;
pub mod evoc;
pub mod hnsw;
pub mod ivf_pq;
pub mod nsw;
pub mod quantization;
pub mod scann;
pub mod sng;
pub mod vamana;

pub mod matryoshka;
pub mod partitioning;

pub mod filtering;
pub mod simd;

// Hash-based methods (LSH, MinHash, SimHash)
#[cfg(feature = "lsh")]
pub mod hash;

// Re-exports
pub use ann::traits::ANNIndex;
pub use error::{Result, RetrieveError};

pub mod persistence;
pub mod compression;
pub mod error;