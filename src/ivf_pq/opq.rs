//! Optimized Product Quantization (OPQ) implementation.
//!
//! OPQ optimizes space decomposition and codebooks to minimize quantization distortions.
//! It uses rotation matrices to optimize the decomposition before applying product quantization.
//!
//! # References
//!
//! - Ge et al. (2013): "Optimized Product Quantization"
//! - Survey: Section III-B4

#[cfg(not(feature = "scann"))]
compile_error!("OPQ requires 'scann' feature for k-means clustering");

use super::pq::ProductQuantizer;
use crate::RetrieveError;

/// Optimized Product Quantizer.
///
/// Extends Product Quantization with optimized space decomposition using rotation matrices.
pub struct OptimizedProductQuantizer {
    dimension: usize,
    num_codebooks: usize,
    codebook_size: usize,
    subvector_dim: usize,
    rotation_matrices: Vec<Vec<Vec<f32>>>, // [codebook][row][col] rotation matrix
    quantizer: ProductQuantizer,
}

impl OptimizedProductQuantizer {
    /// Create new optimized product quantizer.
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

        if dimension % num_codebooks != 0 {
            return Err(RetrieveError::Other(
                "Dimension must be divisible by num_codebooks".to_string(),
            ));
        }

        let subvector_dim = dimension / num_codebooks;

        // Initialize rotation matrices as identity matrices
        let mut rotation_matrices = Vec::new();
        for _ in 0..num_codebooks {
            let mut matrix = vec![vec![0.0; subvector_dim]; subvector_dim];
            for i in 0..subvector_dim {
                matrix[i][i] = 1.0; // Identity matrix
            }
            rotation_matrices.push(matrix);
        }

        Ok(Self {
            dimension,
            num_codebooks,
            codebook_size,
            subvector_dim,
            rotation_matrices,
            quantizer: ProductQuantizer::new(dimension, num_codebooks, codebook_size)?,
        })
    }

    /// Train optimized quantizer on vectors.
    ///
    /// Uses iterative optimization to find optimal rotation matrices and codebooks.
    pub fn fit(
        &mut self,
        vectors: &[f32],
        num_vectors: usize,
        max_iterations: usize,
    ) -> Result<(), RetrieveError> {
        // Initialize rotation matrices (identity)
        // Rotation matrices are already initialized as identity in new()

        // Iterative optimization
        for _iteration in 0..max_iterations {
            // Step 1: Rotate vectors using current rotation matrices
            let rotated_vectors = self.rotate_vectors(vectors, num_vectors)?;

            // Step 2: Train PQ on rotated vectors
            self.quantizer.fit(&rotated_vectors, num_vectors)?;

            // Step 3: Optimize rotation matrices
            self.optimize_rotations(vectors, num_vectors, &rotated_vectors)?;

            // Check convergence (simplified: just iterate max_iterations times)
            // In production, check quantization error reduction
        }

        Ok(())
    }

    /// Rotate vectors using current rotation matrices.
    fn rotate_vectors(
        &self,
        vectors: &[f32],
        num_vectors: usize,
    ) -> Result<Vec<f32>, RetrieveError> {
        let mut rotated = Vec::with_capacity(num_vectors * self.dimension);

        for i in 0..num_vectors {
            let vec = get_vector(vectors, self.dimension, i);
            let mut rotated_vec = vec![0.0; self.dimension];

            for codebook_idx in 0..self.num_codebooks {
                let start_dim = codebook_idx * self.subvector_dim;
                let end_dim = (codebook_idx + 1) * self.subvector_dim;
                let subvector = &vec[start_dim..end_dim];
                let rotation = &self.rotation_matrices[codebook_idx];

                // Apply rotation: rotated = R * subvector
                for row in 0..self.subvector_dim {
                    let mut sum = 0.0;
                    for col in 0..self.subvector_dim {
                        sum += rotation[row][col] * subvector[col];
                    }
                    rotated_vec[start_dim + row] = sum;
                }
            }

            rotated.extend_from_slice(&rotated_vec);
        }

        Ok(rotated)
    }

    /// Optimize rotation matrices to minimize quantization error.
    ///
    /// Uses simplified optimization: update rotation based on quantization residuals.
    fn optimize_rotations(
        &mut self,
        _original_vectors: &[f32],
        num_vectors: usize,
        rotated_vectors: &[f32],
    ) -> Result<(), RetrieveError> {
        // Simplified optimization: update rotation to align with principal components
        // In production, use more sophisticated optimization (e.g., iterative refinement)

        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;

            // Extract subvectors
            let mut subvectors = Vec::new();
            for i in 0..num_vectors {
                let vec = get_vector(rotated_vectors, self.dimension, i);
                subvectors.push(vec[start_dim..end_dim].to_vec());
            }

            // Compute covariance matrix
            let cov = compute_covariance(&subvectors);

            // Update rotation to align with principal components (simplified)
            // In production, use eigenvalue decomposition
            self.update_rotation_from_covariance(codebook_idx, &cov);
        }

        Ok(())
    }

    /// Update rotation matrix from covariance matrix.
    ///
    /// Simplified: uses covariance to guide rotation updates.
    fn update_rotation_from_covariance(&mut self, codebook_idx: usize, cov: &[Vec<f32>]) {
        // Simplified update: gradually adjust rotation towards principal components
        // In production, use proper eigenvalue decomposition

        let learning_rate = 0.1; // Small learning rate for stability
        let rotation = &mut self.rotation_matrices[codebook_idx];

        // Update rotation matrix (simplified gradient step)
        for i in 0..self.subvector_dim {
            for j in 0..self.subvector_dim {
                // Adjust rotation based on covariance
                rotation[i][j] += learning_rate * cov[i][j] * 0.01; // Small adjustment
            }
        }

        // Orthonormalize rotation matrix (simplified Gram-Schmidt)
        self.orthonormalize_rotation(codebook_idx);
    }

    /// Orthonormalize rotation matrix using Gram-Schmidt process.
    fn orthonormalize_rotation(&mut self, codebook_idx: usize) {
        let rotation = &mut self.rotation_matrices[codebook_idx];

        // Gram-Schmidt orthonormalization
        for i in 0..self.subvector_dim {
            // Normalize current row
            let norm: f32 = rotation[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for j in 0..self.subvector_dim {
                    rotation[i][j] /= norm;
                }
            }

            // Orthogonalize against previous rows
            for k in 0..i {
                let dot: f32 = rotation[i]
                    .iter()
                    .zip(rotation[k].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                for j in 0..self.subvector_dim {
                    rotation[i][j] -= dot * rotation[k][j];
                }
            }

            // Renormalize
            let norm: f32 = rotation[i].iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-6 {
                for j in 0..self.subvector_dim {
                    rotation[i][j] /= norm;
                }
            }
        }
    }

    /// Quantize a vector using optimized quantization.
    ///
    /// Returns codebook indices for each subvector.
    pub fn quantize(&self, vector: &[f32]) -> Vec<u8> {
        // Rotate vector
        let mut rotated_vec = vec![0.0; self.dimension];
        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;
            let subvector = &vector[start_dim..end_dim];
            let rotation = &self.rotation_matrices[codebook_idx];

            // Apply rotation
            for row in 0..self.subvector_dim {
                let mut sum = 0.0;
                for col in 0..self.subvector_dim {
                    sum += rotation[row][col] * subvector[col];
                }
                rotated_vec[start_dim + row] = sum;
            }
        }

        // Quantize rotated vector
        self.quantizer.quantize(&rotated_vec)
    }

    /// Compute approximate distance using quantized codes.
    pub fn approximate_distance(&self, query: &[f32], codes: &[u8]) -> f32 {
        // Rotate query
        let mut rotated_query = vec![0.0; self.dimension];
        for codebook_idx in 0..self.num_codebooks {
            let start_dim = codebook_idx * self.subvector_dim;
            let end_dim = (codebook_idx + 1) * self.subvector_dim;
            let subvector = &query[start_dim..end_dim];
            let rotation = &self.rotation_matrices[codebook_idx];

            // Apply rotation
            for row in 0..self.subvector_dim {
                let mut sum = 0.0;
                for col in 0..self.subvector_dim {
                    sum += rotation[row][col] * subvector[col];
                }
                rotated_query[start_dim + row] = sum;
            }
        }

        // Compute distance using quantizer
        self.quantizer.approximate_distance(&rotated_query, codes)
    }

    /// Get rotation matrices (for testing/debugging).
    pub fn rotation_matrices(&self) -> &[Vec<Vec<f32>>] {
        &self.rotation_matrices
    }

    /// Get quantizer (for testing/debugging).
    pub fn quantizer(&self) -> &ProductQuantizer {
        &self.quantizer
    }
}

/// Compute covariance matrix for subvectors.
fn compute_covariance(subvectors: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let dim = subvectors[0].len();
    let n = subvectors.len() as f32;

    // Compute mean
    let mut mean = vec![0.0; dim];
    for subvec in subvectors {
        for (i, &val) in subvec.iter().enumerate() {
            mean[i] += val;
        }
    }
    for i in 0..dim {
        mean[i] /= n;
    }

    // Compute covariance
    let mut cov = vec![vec![0.0; dim]; dim];
    for subvec in subvectors {
        for i in 0..dim {
            for j in 0..dim {
                cov[i][j] += (subvec[i] - mean[i]) * (subvec[j] - mean[j]);
            }
        }
    }

    for i in 0..dim {
        for j in 0..dim {
            cov[i][j] /= n;
        }
    }

    cov
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
    fn test_opq_basic() {
        let mut opq = OptimizedProductQuantizer::new(128, 8, 256).unwrap();

        // Generate test vectors
        let num_vectors = 1000;
        let mut vectors = Vec::new();
        for i in 0..num_vectors {
            let mut vec = vec![0.0; 128];
            for j in 0..128 {
                vec[j] = ((i * 128 + j) as f32) * 0.01;
            }
            vectors.extend_from_slice(&vec);
        }

        // Train
        opq.fit(&vectors, num_vectors, 5).unwrap();

        // Test quantization
        let test_vec = vec![1.0; 128];
        let codes = opq.quantize(&test_vec);
        assert_eq!(codes.len(), 8);
    }
}
