//! Anisotropic Vector Quantization with k-means Partitioning implementation.
//!
//! Pure Rust implementation of the three-stage ANN framework combining:
//! 1. **Partitioning**: k-means clustering to divide dataset (coarse search)
//! 2. **Quantization**: Anisotropic vector quantization for compression (fine search)
//! 3. **Re-ranking**: Exact distance computation for top candidates (accuracy refinement)
//!
//! **Technical Name**: Anisotropic Vector Quantization with k-means Partitioning
//! **Vendor Name**: SCANN/ScaNN (Google Research)
//!
//! Optimized for Maximum Inner Product Search (MIPS) and large-scale datasets.
//!
//! **Relationships**:
//! - Combines partitioning (IVF-style) with quantization (AVQ)
//! - AVQ preserves inner products better than standard PQ for MIPS
//! - Complementary to graph-based methods (can be combined)
//! - Similar to IVF-PQ but uses AVQ instead of PQ
//!
//! # References
//!
//! - Guo et al. (2020): "Accelerating Large-Scale Inference with Anisotropic Vector Quantization"
//! - Sun et al. (2023): "SOAR: Improved Indexing for Approximate Nearest Neighbor Search"

pub mod partitioning;
mod quantization;
mod reranking;
pub mod search;

pub use search::{SCANNIndex, SCANNParams};
