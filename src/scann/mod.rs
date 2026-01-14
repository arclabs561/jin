//! ScaNN: Anisotropic Vector Quantization with k-means Partitioning.
//!
//! # Feature Flag
//!
//! Requires the `scann` feature:
//! ```toml
//! vicinity = { version = "0.1", features = ["scann"] }
//! ```
//!
//! # Status: Experimental
//!
//! Google Research's ScaNN algorithm. Under active development.
//!
//! # The Problem with Standard PQ for MIPS
//!
//! Product Quantization (PQ) minimizes reconstruction error: ||x - Q(x)||².
//! But for Maximum Inner Product Search (MIPS), we care about inner product
//! preservation, not reconstruction. PQ can introduce systematic bias.
//!
//! # Key Insight: Anisotropic Quantization
//!
//! **Anisotropic Vector Quantization (AVQ)** weights dimensions by their
//! importance for inner product computation:
//!
//! ```text
//! Standard PQ:  min ||x - Q(x)||²           (uniform error)
//! Anisotropic:  min ||x - Q(x)||²_W         (weighted by query distribution)
//! ```
//!
//! Where W captures the expected query distribution. If queries tend to
//! have large values in dimension i, errors in dimension i matter more.
//!
//! # Three-Stage Pipeline
//!
//! ```text
//! Query → [1. Partition Search] → [2. Quantized Scan] → [3. Re-ranking]
//!              k-means              AVQ distances         Exact scores
//!              ~256 clusters        Fast approx           Top-k only
//! ```
//!
//! 1. **Partitioning**: k-means clusters database vectors. Query finds
//!    nearest cluster centroids, only scans those partitions.
//!
//! 2. **Quantized scan**: Within partitions, use AVQ codes for fast
//!    approximate inner product estimation.
//!
//! 3. **Re-ranking**: Compute exact inner products for top candidates
//!    from quantized search. Corrects quantization errors.
//!
//! # AVQ vs PQ for MIPS
//!
//! | Aspect | PQ | AVQ |
//! |--------|----|----|
//! | Objective | Minimize reconstruction | Preserve inner products |
//! | Error distribution | Uniform | Weighted by importance |
//! | Best for | L2 distance | Inner product / cosine |
//! | Training | Simpler | Needs query distribution estimate |
//!
//! # Parameters
//!
//! | Parameter | Typical | Effect |
//! |-----------|---------|--------|
//! | `num_partitions` | 256-4096 | More = faster search, slower build |
//! | `num_reorder` | 100-1000 | More = better recall, slower |
//! | `num_codebooks` | 8-32 | More = better accuracy, more memory |
//! | `codebook_size` | 256 | Usually 256 (8-bit codes) |
//!
//! # When to Use
//!
//! - Maximum Inner Product Search (MIPS)
//! - Recommendation systems (user-item inner products)
//! - Large-scale (>10M vectors) where memory matters
//!
//! # When NOT to Use
//!
//! - L2/Euclidean distance (use IVF-PQ instead)
//! - Small datasets (< 100K, overhead not worth it)
//! - Need real-time updates (requires rebuild)
//!
//! # Usage
//!
//! ```ignore
//! use vicinity::scann::{SCANNIndex, SCANNParams};
//!
//! let params = SCANNParams {
//!     num_partitions: 256,
//!     num_reorder: 100,
//!     num_codebooks: 16,
//!     codebook_size: 256,
//! };
//!
//! let mut index = SCANNIndex::new(128, params);
//! index.train(&training_vectors)?;
//! index.add_batch(&database)?;
//! index.build()?;
//!
//! // Returns (index, inner_product) pairs
//! let results = index.search_mips(&query, 10)?;
//! ```
//!
//! # References
//!
//! - Guo et al. (2020). "Accelerating Large-Scale Inference with Anisotropic
//!   Vector Quantization."
//! - Sun et al. (2023). "SOAR: Improved Indexing for Approximate Nearest
//!   Neighbor Search."
//! - See also: [`ivf_pq`](crate::ivf_pq) for L2 distance

pub mod partitioning;
mod quantization;
mod reranking;
pub mod search;

pub use search::{SCANNIndex, SCANNParams};
