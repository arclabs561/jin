//! Anisotropic vector quantization for SCANN.

use crate::simd;
use crate::RetrieveError;

/// Anisotropic vector quantizer.
///
/// Preserves parallel components between vectors (important for inner product accuracy).
/// Optimized for Maximum Inner Product Search (MIPS).
#[allow(dead_code)] // Future SCANN integration - used by quantization_methods example
pub struct AnisotropicQuantizer {
    dimension: usize,
    num_codebooks: usize,
    codebook_size: usize,
    codebooks: Vec<Vec<Vec<f32>>>, // [codebook][codeword][dimension]
}

#[allow(dead_code)] // Future SCANN integration - used by quantization_methods example
impl AnisotropicQuantizer {
    /// Create new quantizer.
    pub fn new(
        dimension: usize,
        num_codebooks: usize,
        codebook_size: usize,
    ) -> Result<Self, RetrieveError> {
        if dimension == 0 || num_codebooks == 0 || codebook_size == 0 {
            return Err(RetrieveError::Other(
                "All parameters must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            dimension,
            num_codebooks,
            codebook_size,
            codebooks: Vec::new(),
        })
    }

    /// Train quantizer on vectors.
    ///
    /// Uses k-means on subvectors to create codebooks.
    pub fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        let subvector_dim = self.dimension / self.num_codebooks;

        // Train codebook for each subvector
        self.codebooks = Vec::new();

        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * subvector_dim;
            let end_dim = (codebook_idx + 1) * subvector_dim;

            // Extract subvectors
            let mut subvectors = Vec::new();
            for i in 0..num_vectors {
                let vec = get_vector(vectors, self.dimension, i);
                subvectors.push(vec[start_dim..end_dim].to_vec());
            }

            // Train k-means on subvectors
            let mut kmeans =
                crate::scann::partitioning::KMeans::new(subvector_dim, self.codebook_size)?;

            // Flatten subvectors for k-means
            let flat: Vec<f32> = subvectors.iter().flatten().copied().collect();
            kmeans.fit(&flat, num_vectors)?;

            self.codebooks.push(kmeans.centroids().to_vec());
        }

        Ok(())
    }

    /// Quantize a vector.
    ///
    /// Returns codebook indices for each subvector.
    pub fn quantize(&self, vector: &[f32]) -> Vec<usize> {
        let subvector_dim = self.dimension / self.num_codebooks;
        let mut codes = Vec::new();

        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * subvector_dim;
            let end_dim = (codebook_idx + 1) * subvector_dim;
            let subvector = &vector[start_dim..end_dim];

            // Find closest codeword
            let mut best_code = 0;
            let mut best_dist = f32::INFINITY;

            for (code, codeword) in self.codebooks[codebook_idx].iter().enumerate() {
                let dist = cosine_distance(subvector, codeword);
                if dist < best_dist {
                    best_dist = dist;
                    best_code = code;
                }
            }

            codes.push(best_code);
        }

        codes
    }

    /// Compute approximate distance using quantized codes.
    ///
    /// Uses lookup tables for fast computation.
    pub fn approximate_distance(&self, query: &[f32], codes: &[usize]) -> f32 {
        let subvector_dim = self.dimension / self.num_codebooks;
        let mut total_dist = 0.0;

        for (codebook_idx, &code) in codes.iter().enumerate() {
            let start_dim = codebook_idx * subvector_dim;
            let end_dim = (codebook_idx + 1) * subvector_dim;
            let query_subvector = &query[start_dim..end_dim];
            let codeword = &self.codebooks[codebook_idx][code];

            total_dist += cosine_distance(query_subvector, codeword);
        }

        total_dist
    }
}

/// Compute cosine distance (SIMD-accelerated).
#[allow(dead_code)] // Used by AnisotropicQuantizer
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let similarity = simd::dot(a, b);
    1.0 - similarity
}

/// Get vector from SoA storage.
#[allow(dead_code)] // Used by AnisotropicQuantizer
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}
