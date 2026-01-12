//! Lossless compression for vector IDs in ANN indexes.
//!
//! This module provides compression algorithms that exploit ordering invariance
//! in vector ID collections (IVF clusters, HNSW neighbor lists) to achieve
//! significant compression ratios (5-7x for large sets).
//!
//! Based on "Lossless Compression of Vector IDs for Approximate Nearest Neighbor Search"
//! (Severo et al., 2025).
//!
//! # Compression Methods
//!
//! - **ROC (Random Order Coding)**: Compress sets of IDs using bits-back coding with ANS
//! - **Elias-Fano**: Baseline compression for sorted sequences
//! - **Wavelet Trees**: Full random access compression (future)
//!
//! # Status: Experimental
//!
//! ANS codec fields are placeholders for future integration.
//!
//! # Usage
//!
//! ```rust,ignore
//! use ordino_retrieve::compression::{RocCompressor, IdSetCompressor};
//!
//! let compressor = RocCompressor::new();
//! let ids: Vec<u32> = vec![1, 5, 10, 20, 50];
//! let universe_size = 1000;
//!
//! // Compress
//! let compressed = compressor.compress_set(&ids, universe_size)?;
//!
//! // Decompress
//! let decompressed = compressor.decompress_set(&compressed, universe_size)?;
//! ```

#![allow(dead_code)]

#[cfg(feature = "id-compression")]
mod ans;
mod error;
#[cfg(feature = "id-compression")]
mod roc;
mod traits;

pub use error::CompressionError;
pub use traits::IdSetCompressor;

#[cfg(feature = "id-compression")]
pub use roc::RocCompressor;

/// Compression method selection.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum IdCompressionMethod {
    /// No compression (uncompressed storage).
    #[default]
    None,
    /// Elias-Fano encoding (baseline, sorted sequences).
    EliasFano,
    /// Random Order Coding (optimal for sets, uses bits-back with ANS).
    Roc,
    /// Wavelet tree (full random access, future).
    WaveletTree,
}
