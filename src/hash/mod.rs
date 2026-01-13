//! Hash-based similarity search methods.
//!
//! This module provides Locality Sensitive Hashing (LSH) algorithms for
//! approximate similarity search:
//!
//! ## Implemented
//!
//! - **Random Projection LSH**: For cosine similarity (vectors)
//! - **MinHash**: For Jaccard similarity (sets/documents)
//! - **SimHash**: For cosine similarity (binary fingerprints)
//!
//! ## Planned
//!
//! - **E2LSH**: For Euclidean distance
//! - **Cross-polytope LSH**: High-dimensional LSH
//!
//! # Use Cases
//!
//! | Algorithm | Input | Similarity | Best For |
//! |-----------|-------|------------|----------|
//! | Random Projection | Vectors | Cosine | Dense embeddings |
//! | MinHash | Sets | Jaccard | Document deduplication |
//! | SimHash | Features | Cosine | Text fingerprinting |
//!
//! # References
//!
//! - Indyk & Motwani (1998): "Approximate nearest neighbors: towards removing
//!   the curse of dimensionality"
//! - Broder (1997): "On the resemblance and containment of documents" (MinHash)
//! - Charikar (2002): "Similarity estimation techniques from rounding algorithms" (SimHash)

mod hash_table;
pub mod minhash;
mod random_projection;
pub mod search;
pub mod simhash;

pub use minhash::{MinHash, MinHashLSH, MinHashSignature};
pub use search::{LSHIndex, LSHParams};
pub use simhash::{SimHash, SimHashFingerprint, SimHashLSH};
