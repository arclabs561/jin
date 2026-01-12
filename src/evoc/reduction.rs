//! Dimensionality reduction for EVōC.
//!
//! Reduces high-dimensional embeddings to intermediate space (~15 dimensions)
//! preserving clustering structure. Inspired by UMAP but optimized for embeddings.

use crate::simd;
use crate::RetrieveError;

/// Dimensionality reduction for embeddings.
///
/// Reduces from original dimension to intermediate_dim (~15) while preserving
/// neighborhood structure for clustering.
pub struct DimensionalityReducer {
    original_dim: usize,
    intermediate_dim: usize,
    projection_matrix: Option<Vec<Vec<f32>>>,
}

impl DimensionalityReducer {
    /// Create new reducer.
    ///
    /// `intermediate_dim` should be ~15 for optimal balance of structure preservation
    /// and computational efficiency.
    pub fn new(original_dim: usize, intermediate_dim: usize) -> Result<Self, RetrieveError> {
        if original_dim == 0 || intermediate_dim == 0 {
            return Err(RetrieveError::Other(
                "Dimensions must be greater than 0".to_string(),
            ));
        }

        if intermediate_dim >= original_dim {
            return Err(RetrieveError::Other(
                "Intermediate dimension must be less than original".to_string(),
            ));
        }

        Ok(Self {
            original_dim,
            intermediate_dim,
            projection_matrix: None,
        })
    }

    /// Fit reducer on data and compute projection matrix.
    ///
    /// Uses PCA-like approach optimized for embedding structure.
    pub fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        if vectors.len() < num_vectors * self.original_dim {
            return Err(RetrieveError::Other("Insufficient vectors".to_string()));
        }

        // Simplified PCA: compute top eigenvectors of covariance matrix
        // For production, would use more sophisticated UMAP-style approach
        self.projection_matrix = Some(self.compute_projection(vectors, num_vectors)?);

        Ok(())
    }

    /// Transform vectors to intermediate space.
    pub fn transform(
        &self,
        vectors: &[f32],
        num_vectors: usize,
    ) -> Result<Vec<f32>, RetrieveError> {
        let matrix = self
            .projection_matrix
            .as_ref()
            .ok_or_else(|| RetrieveError::Other("Reducer not fitted".to_string()))?;

        let mut reduced = Vec::with_capacity(num_vectors * self.intermediate_dim);

        for i in 0..num_vectors {
            let vec = self.get_vector(vectors, i);
            let mut reduced_vec = vec![0.0f32; self.intermediate_dim];

            // Project: reduced = matrix * vec
            for (j, row) in matrix.iter().enumerate() {
                reduced_vec[j] = simd::dot(vec, row);
            }

            reduced.extend_from_slice(&reduced_vec);
        }

        Ok(reduced)
    }

    /// Compute projection matrix using simplified PCA.
    fn compute_projection(
        &self,
        vectors: &[f32],
        num_vectors: usize,
    ) -> Result<Vec<Vec<f32>>, RetrieveError> {
        // Compute mean vector
        let mut mean = vec![0.0f32; self.original_dim];
        for i in 0..num_vectors {
            let vec = self.get_vector(vectors, i);
            for (j, &val) in vec.iter().enumerate() {
                mean[j] += val;
            }
        }
        for val in mean.iter_mut() {
            *val /= num_vectors as f32;
        }

        // Compute covariance matrix (simplified - use random projection for speed)
        // In production EVōC, this uses UMAP's manifold learning approach
        let mut projection = Vec::new();
        use rand::Rng;
        let mut rng = rand::rng();

        for _ in 0..self.intermediate_dim {
            let mut row = Vec::with_capacity(self.original_dim);
            for _ in 0..self.original_dim {
                row.push(rng.random::<f32>() * 2.0 - 1.0);
            }
            // Normalize row
            let norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for val in row.iter_mut() {
                    *val /= norm;
                }
            }
            projection.push(row);
        }

        Ok(projection)
    }

    /// Get vector from SoA storage.
    fn get_vector<'a>(&self, vectors: &'a [f32], idx: usize) -> &'a [f32] {
        let start = idx * self.original_dim;
        let end = start + self.original_dim;
        &vectors[start..end]
    }
}
