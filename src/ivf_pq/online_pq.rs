//! Online Product Quantization (O-PQ) implementation.
//!
//! Adapts to dynamic data sets by updating quantization codebooks and codes online.
//! Handles data streams and incremental data sets without requiring offline retraining.
//!
//! # References
//!
//! - Survey: Section III-B4
//! - Xu et al. (2018): "Online Product Quantization"

#[cfg(not(feature = "scann"))]
compile_error!("Online PQ requires 'scann' feature for k-means clustering");

use crate::simd;
use crate::RetrieveError;

/// Online Product Quantizer.
///
/// Extends Product Quantization with online learning for dynamic datasets.
pub struct OnlineProductQuantizer {
    dimension: usize,
    num_codebooks: usize,
    codebook_size: usize,
    subvector_dim: usize,
    codebooks: Vec<Vec<Vec<f32>>>, // [codebook][codeword][dimension]
    learning_rate: f32,
    forgetting_rate: f32,
    codebook_counts: Vec<Vec<usize>>, // [codebook][codeword] count
}

impl OnlineProductQuantizer {
    /// Create new online product quantizer.
    pub fn new(
        dimension: usize,
        num_codebooks: usize,
        codebook_size: usize,
        learning_rate: f32,
        forgetting_rate: f32,
    ) -> Result<Self, RetrieveError> {
        if dimension == 0 || num_codebooks == 0 || codebook_size == 0 {
            return Err(RetrieveError::Other(
                "All parameters must be greater than 0".to_string(),
            ));
        }

        if dimension % num_codebooks != 0 {
            return Err(RetrieveError::Other(
                "Dimension must be divisible by num_codebooks".to_string(),
            ));
        }

        if learning_rate <= 0.0 || learning_rate > 1.0 {
            return Err(RetrieveError::Other(
                "Learning rate must be in (0, 1]".to_string(),
            ));
        }

        if forgetting_rate < 0.0 || forgetting_rate > 1.0 {
            return Err(RetrieveError::Other(
                "Forgetting rate must be in [0, 1]".to_string(),
            ));
        }

        let subvector_dim = dimension / num_codebooks;
        let codebook_counts = vec![vec![0; codebook_size]; num_codebooks];

        Ok(Self {
            dimension,
            num_codebooks,
            codebook_size,
            subvector_dim,
            codebooks: Vec::new(),
            learning_rate,
            forgetting_rate,
            codebook_counts,
        })
    }

    /// Initialize codebooks using k-means++ on initial vectors.
    pub fn initialize(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        // Train codebook for each subvector using k-means
        self.codebooks = Vec::new();

        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;

            // Extract subvectors
            let mut subvectors = Vec::new();
            for i in 0..num_vectors {
                let vec = get_vector(vectors, self.dimension, i);
                subvectors.push(vec[start_dim..end_dim].to_vec());
            }

            // Train k-means on subvectors
            let mut kmeans = crate::scann::partitioning::KMeans::new(
                self.subvector_dim,
                self.codebook_size,
            )?;

            // Flatten subvectors for k-means (SoA format)
            let mut flat = Vec::with_capacity(num_vectors * self.subvector_dim);
            for subvec in &subvectors {
                flat.extend_from_slice(subvec);
            }
            kmeans.fit(&flat, num_vectors)?;

            self.codebooks.push(kmeans.centroids().to_vec());
        }

        // Initialize counts
        for codebook_idx in 0..self.num_codebooks {
            self.codebook_counts[codebook_idx].fill(0);
        }

        // Count initial assignments
        for i in 0..num_vectors {
            let vec = get_vector(vectors, self.dimension, i);
            let codes = self.quantize_internal(vec);
            for (codebook_idx, &code) in codes.iter().enumerate() {
                self.codebook_counts[codebook_idx][code as usize] += 1;
            }
        }

        Ok(())
    }

    /// Update quantizer with new vector (online learning).
    ///
    /// Updates codebooks and codes incrementally without full retraining.
    pub fn update(&mut self, vector: &[f32]) -> Result<Vec<u8>, RetrieveError> {
        if vector.len() != self.dimension {
            return Err(RetrieveError::Other(format!(
                "Vector dimension {} != {}",
                vector.len(),
                self.dimension
            )));
        }

        // Quantize vector
        let codes = self.quantize_internal(vector);

        // Update codebooks and codes online
        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;
            let subvector = &vector[start_dim..end_dim];
            let code = codes[codebook_idx] as usize;

            // Update centroid using learning rate
            let centroid = &mut self.codebooks[codebook_idx][code];
            for (i, &val) in subvector.iter().enumerate() {
                centroid[i] = (1.0 - self.learning_rate) * centroid[i] + self.learning_rate * val;
            }

            // Update count
            self.codebook_counts[codebook_idx][code] += 1;

            // Apply forgetting rate to other centroids (optional, for adaptation)
            if self.forgetting_rate > 0.0 {
                for (other_code, other_centroid) in
                    self.codebooks[codebook_idx].iter_mut().enumerate()
                {
                    if other_code != code {
                        // Slight decay to allow adaptation
                        for val in other_centroid.iter_mut() {
                            *val *= 1.0 - self.forgetting_rate * 0.01;
                        }
                    }
                }
            }
        }

        Ok(codes)
    }

    /// Batch update with multiple vectors.
    pub fn update_batch(
        &mut self,
        vectors: &[f32],
        num_vectors: usize,
    ) -> Result<Vec<Vec<u8>>, RetrieveError> {
        let mut all_codes = Vec::new();

        for i in 0..num_vectors {
            let vec = get_vector(vectors, self.dimension, i);
            let codes = self.update(vec)?;
            all_codes.push(codes);
        }

        Ok(all_codes)
    }

    /// Quantize a vector (same as standard PQ).
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        self.quantize_internal(vector)
    }

    /// Internal quantization method.
    fn quantize_internal(&self, vector: &[f32]) -> Vec<u8> {
        let mut codes = Vec::with_capacity(self.num_codebooks);

        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;
            let subvector = &vector[start_dim..end_dim];

            // Find closest codeword
            let mut best_code = 0u8;
            let mut best_dist = f32::INFINITY;

            for (code, codeword) in self.codebooks[codebook_idx].iter().enumerate() {
                let dist = cosine_distance(subvector, codeword);
                if dist < best_dist {
                    best_dist = dist;
                    best_code = code.min(255) as u8;
                }
            }

            codes.push(best_code);
        }

        codes
    }

    /// Compute approximate distance using quantized codes.
    pub fn approximate_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        let mut total_dist = 0.0;

        for (codebook_idx, &code) in codes.iter().enumerate() {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;
            let query_subvector = &query[start_dim..end_dim];
            let codeword = &self.codebooks[codebook_idx][code as usize];

            total_dist += cosine_distance(query_subvector, codeword);
        }

        total_dist
    }

    /// Get codebook counts (for monitoring/debugging).
    pub fn codebook_counts(&self) -> &[Vec<usize>] {
        &self.codebook_counts
    }

    /// Get codebooks (for testing/debugging).
    pub fn codebooks(&self) -> &[Vec<Vec<f32>>] {
        &self.codebooks
    }
}

/// Compute cosine distance (SIMD-accelerated).
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let similarity = simd::dot(a, b);
    1.0 - similarity
}

/// Get vector from SoA storage.
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_online_pq_basic() {
        let mut opq = OnlineProductQuantizer::new(128, 8, 256, 0.1, 0.01).unwrap();

        // Initialize with some vectors
        let num_init = 1000;
        let mut init_vectors = Vec::new();
        for i in 0..num_init {
            let mut vec = vec![0.0; 128];
            for j in 0..128 {
                vec[j] = ((i * 128 + j) as f32) * 0.01;
            }
            init_vectors.extend_from_slice(&vec);
        }

        opq.initialize(&init_vectors, num_init).unwrap();

        // Update with new vector
        let new_vec = vec![1.0; 128];
        let codes = opq.update(&new_vec).unwrap();
        assert_eq!(codes.len(), 8);
    }
}
