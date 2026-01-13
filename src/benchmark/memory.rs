//! Memory tracking utilities for benchmarking.
//!
//! Provides tools to measure:
//! - Bytes per vector (compression ratio)
//! - Index overhead
//! - Peak memory usage during construction

use std::mem::size_of;

/// Memory statistics for an index.
#[derive(Debug, Clone, Default)]
pub struct IndexMemoryStats {
    /// Number of vectors in the index
    pub n_vectors: usize,
    /// Vector dimensionality
    pub dimension: usize,
    /// Raw vector data size (uncompressed baseline)
    pub raw_vectors_bytes: usize,
    /// Stored vector data size (may be compressed/quantized)
    pub stored_vectors_bytes: usize,
    /// Index structure overhead (graph edges, metadata, etc.)
    pub index_overhead_bytes: usize,
    /// Total index size
    pub total_bytes: usize,
}

impl IndexMemoryStats {
    /// Create stats for an index with the given sizes.
    pub fn new(
        n_vectors: usize,
        dimension: usize,
        stored_vectors_bytes: usize,
        index_overhead_bytes: usize,
    ) -> Self {
        let raw_vectors_bytes = n_vectors * dimension * size_of::<f32>();
        let total_bytes = stored_vectors_bytes + index_overhead_bytes;

        Self {
            n_vectors,
            dimension,
            raw_vectors_bytes,
            stored_vectors_bytes,
            index_overhead_bytes,
            total_bytes,
        }
    }

    /// Bytes per vector (total index size / n_vectors).
    pub fn bytes_per_vector(&self) -> f64 {
        if self.n_vectors == 0 {
            return 0.0;
        }
        self.total_bytes as f64 / self.n_vectors as f64
    }

    /// Compression ratio (raw size / stored size).
    /// 
    /// Values > 1 indicate compression, < 1 indicates expansion.
    pub fn compression_ratio(&self) -> f64 {
        if self.stored_vectors_bytes == 0 {
            return 0.0;
        }
        self.raw_vectors_bytes as f64 / self.stored_vectors_bytes as f64
    }

    /// Index overhead ratio (overhead / raw vectors).
    ///
    /// E.g., 0.5 means the index structure adds 50% overhead.
    pub fn overhead_ratio(&self) -> f64 {
        if self.raw_vectors_bytes == 0 {
            return 0.0;
        }
        self.index_overhead_bytes as f64 / self.raw_vectors_bytes as f64
    }

    /// Bits per dimension for stored vectors.
    pub fn bits_per_dimension(&self) -> f64 {
        if self.n_vectors == 0 || self.dimension == 0 {
            return 0.0;
        }
        (self.stored_vectors_bytes * 8) as f64 / (self.n_vectors * self.dimension) as f64
    }
}

/// Memory tracker for measuring peak usage during operations.
#[derive(Debug, Default)]
pub struct MemoryTracker {
    /// Starting memory (if measurable)
    pub start_bytes: Option<usize>,
    /// Peak memory observed
    pub peak_bytes: Option<usize>,
    /// Final memory
    pub end_bytes: Option<usize>,
}

impl MemoryTracker {
    /// Create a new memory tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record current memory as starting point.
    /// 
    /// Note: Actual implementation would use platform-specific
    /// memory queries (e.g., /proc/self/statm on Linux).
    pub fn start(&mut self) {
        self.start_bytes = Self::current_memory();
        self.peak_bytes = self.start_bytes;
    }

    /// Update peak memory if current is higher.
    pub fn checkpoint(&mut self) {
        if let Some(current) = Self::current_memory() {
            match self.peak_bytes {
                Some(peak) if current > peak => self.peak_bytes = Some(current),
                None => self.peak_bytes = Some(current),
                _ => {}
            }
        }
    }

    /// Record final memory.
    pub fn finish(&mut self) {
        self.end_bytes = Self::current_memory();
        self.checkpoint();
    }

    /// Estimated memory allocated during tracking.
    pub fn allocated(&self) -> Option<usize> {
        match (self.start_bytes, self.end_bytes) {
            (Some(start), Some(end)) if end > start => Some(end - start),
            _ => None,
        }
    }

    /// Peak memory above starting point.
    pub fn peak_allocated(&self) -> Option<usize> {
        match (self.start_bytes, self.peak_bytes) {
            (Some(start), Some(peak)) if peak > start => Some(peak - start),
            _ => None,
        }
    }

    /// Get current process memory usage (platform-specific).
    #[cfg(target_os = "linux")]
    fn current_memory() -> Option<usize> {
        use std::fs;
        // Read /proc/self/statm: values are in pages
        let statm = fs::read_to_string("/proc/self/statm").ok()?;
        let rss_pages: usize = statm.split_whitespace().nth(1)?.parse().ok()?;
        let page_size = 4096; // Usually 4KB
        Some(rss_pages * page_size)
    }

    #[cfg(target_os = "macos")]
    fn current_memory() -> Option<usize> {
        // macOS: use mach APIs or rusage
        // Simplified: return None (not implemented)
        None
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    fn current_memory() -> Option<usize> {
        None
    }
}

/// Calculate theoretical memory for common index types.
pub mod theoretical {
    use super::*;

    /// HNSW memory estimate.
    ///
    /// HNSW stores vectors + graph edges.
    /// Graph edges: ~M * 2 * n_vectors * size_of(u32) for the main layer,
    /// plus additional layers (exponentially fewer nodes).
    pub fn hnsw_memory(n_vectors: usize, dimension: usize, m: usize) -> IndexMemoryStats {
        let stored_vectors_bytes = n_vectors * dimension * size_of::<f32>(); // HNSW stores full vectors

        // Graph overhead estimate:
        // - Layer 0: all nodes, ~2*M edges each
        // - Higher layers: ~1/M^(layer) nodes
        // Simplified: ~2.5 * M edges per node on average
        let avg_edges_per_node = (2.5 * m as f64) as usize;
        let graph_edges_bytes = n_vectors * avg_edges_per_node * size_of::<u32>();

        // Node metadata (entry point, level, etc.)
        let metadata_bytes = n_vectors * (size_of::<u32>() + size_of::<u8>());

        let index_overhead_bytes = graph_edges_bytes + metadata_bytes;

        IndexMemoryStats::new(n_vectors, dimension, stored_vectors_bytes, index_overhead_bytes)
    }

    /// IVF-PQ memory estimate.
    ///
    /// IVF-PQ stores PQ codes (compressed) + cluster assignments.
    pub fn ivf_pq_memory(
        n_vectors: usize,
        dimension: usize,
        n_clusters: usize,
        n_subquantizers: usize,
        bits_per_code: usize,
    ) -> IndexMemoryStats {
        // Note: raw_vectors_bytes computed internally by IndexMemoryStats::new
        let _raw_vectors_bytes = n_vectors * dimension * size_of::<f32>();

        // PQ codes: n_subquantizers bytes per vector (assuming 8-bit codes)
        let code_bytes = (n_subquantizers * bits_per_code).div_ceil(8);
        let stored_vectors_bytes = n_vectors * code_bytes;

        // Overhead: cluster centroids + codebooks
        let centroid_bytes = n_clusters * dimension * size_of::<f32>();
        let codebook_size = 1 << bits_per_code; // e.g., 256 for 8 bits
        let sub_dimension = dimension / n_subquantizers;
        let codebook_bytes = n_subquantizers * codebook_size * sub_dimension * size_of::<f32>();

        // Inverted lists (vector IDs per cluster)
        let invlist_bytes = n_vectors * size_of::<u32>();

        let index_overhead_bytes = centroid_bytes + codebook_bytes + invlist_bytes;

        IndexMemoryStats::new(n_vectors, dimension, stored_vectors_bytes, index_overhead_bytes)
    }

    /// Flat (brute force) index memory.
    pub fn flat_memory(n_vectors: usize, dimension: usize) -> IndexMemoryStats {
        let stored_bytes = n_vectors * dimension * size_of::<f32>();
        IndexMemoryStats::new(n_vectors, dimension, stored_bytes, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_memory_stats() {
        // 1000 vectors, 128 dimensions, uncompressed
        let stats = IndexMemoryStats::new(
            1000,
            128,
            1000 * 128 * 4, // stored = raw (no compression)
            1000 * 32 * 4,  // 32 edges per node
        );

        assert_eq!(stats.n_vectors, 1000);
        assert_eq!(stats.raw_vectors_bytes, 1000 * 128 * 4);
        assert!((stats.compression_ratio() - 1.0).abs() < 0.001);
        assert!(stats.bytes_per_vector() > 128.0 * 4.0); // > raw due to overhead
    }

    #[test]
    fn test_compression_ratio() {
        // 8x compression (32-bit -> 4-bit per dimension)
        let stats = IndexMemoryStats::new(
            1000,
            128,
            1000 * 128 / 2, // 4 bits = 0.5 bytes per dimension
            0,
        );

        assert!((stats.compression_ratio() - 8.0).abs() < 0.001);
    }

    #[test]
    fn test_theoretical_hnsw() {
        let stats = theoretical::hnsw_memory(10000, 128, 16);

        // Should have raw vectors + graph overhead
        assert_eq!(stats.raw_vectors_bytes, 10000 * 128 * 4);
        assert!(stats.index_overhead_bytes > 0);
        assert!(stats.overhead_ratio() > 0.0);

        // Bytes per vector should be > raw (128 * 4 = 512)
        assert!(stats.bytes_per_vector() > 512.0);
    }

    #[test]
    fn test_theoretical_ivf_pq() {
        let stats = theoretical::ivf_pq_memory(
            10000,  // vectors
            128,    // dimension
            256,    // clusters
            16,     // subquantizers
            8,      // bits per code
        );

        // PQ should compress: 128 floats -> 16 bytes
        assert!(stats.compression_ratio() > 1.0);

        // Much smaller than raw
        assert!(stats.stored_vectors_bytes < stats.raw_vectors_bytes);
    }
}
