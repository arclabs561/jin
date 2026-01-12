//! Compression trait definitions.

use crate::compression::error::CompressionError;

/// Trait for compressing sets of IDs where order doesn't matter.
///
/// This trait is designed for compressing collections of vector IDs in ANN indexes
/// where the ordering of IDs is irrelevant (e.g., IVF clusters, HNSW neighbor lists).
///
/// # Requirements
///
/// - Input IDs must be sorted and unique
/// - Compression should exploit ordering invariance
/// - Decompression should return sorted IDs
pub trait IdSetCompressor {
    /// Compress a set of IDs (order-invariant).
    ///
    /// # Arguments
    ///
    /// * `ids` - Sorted, unique IDs (must be sorted for correctness)
    /// * `universe_size` - Maximum possible ID value (for entropy calculation)
    ///
    /// # Returns
    ///
    /// Compressed representation as byte vector.
    ///
    /// # Errors
    ///
    /// Returns `CompressionError` if input is invalid or compression fails.
    fn compress_set(&self, ids: &[u32], universe_size: u32) -> Result<Vec<u8>, CompressionError>;

    /// Decompress a set of IDs.
    ///
    /// # Arguments
    ///
    /// * `compressed` - Compressed byte vector
    /// * `universe_size` - Maximum possible ID value (must match compression)
    ///
    /// # Returns
    ///
    /// Sorted vector of IDs.
    ///
    /// # Errors
    ///
    /// Returns `CompressionError` if decompression fails.
    fn decompress_set(
        &self,
        compressed: &[u8],
        universe_size: u32,
    ) -> Result<Vec<u32>, CompressionError>;

    /// Estimate compressed size without full compression.
    ///
    /// Useful for deciding whether to compress.
    ///
    /// # Arguments
    ///
    /// * `num_ids` - Number of IDs in the set
    /// * `universe_size` - Maximum possible ID value
    ///
    /// # Returns
    ///
    /// Estimated compressed size in bytes.
    fn estimate_size(&self, num_ids: usize, universe_size: u32) -> usize;

    /// Get compression ratio (bits per ID).
    ///
    /// # Arguments
    ///
    /// * `num_ids` - Number of IDs in the set
    /// * `universe_size` - Maximum possible ID value
    ///
    /// # Returns
    ///
    /// Average bits per ID (theoretical lower bound).
    fn bits_per_id(&self, num_ids: usize, universe_size: u32) -> f64;
}
