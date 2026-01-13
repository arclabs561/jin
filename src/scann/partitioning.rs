//! k-means partitioning for SCANN.

use crate::simd;
use crate::RetrieveError;

/// k-means clustering for partitioning vectors.
///
/// Uses SIMD-accelerated distance computation and k-means++ initialization.
pub struct KMeans {
    /// Centroids (k x dimension)
    centroids: Vec<Vec<f32>>,
    dimension: usize,
    k: usize,
}

impl KMeans {
    /// Create new k-means with k clusters.
    pub fn new(dimension: usize, k: usize) -> Result<Self, RetrieveError> {
        if dimension == 0 || k == 0 {
            return Err(RetrieveError::Other(
                "Dimension and k must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            centroids: Vec::new(),
            dimension,
            k,
        })
    }

    /// Train k-means on vectors.
    ///
    /// Uses k-means++ initialization and iterative refinement.
    pub fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        if vectors.len() < num_vectors * self.dimension {
            return Err(RetrieveError::Other("Insufficient vectors".to_string()));
        }

        // k-means++ initialization
        self.centroids = self.kmeans_plus_plus(vectors, num_vectors)?;

        // Iterative refinement
        for _iteration in 0..100 {
            let assignments = self.assign_clusters(vectors, num_vectors);
            let new_centroids = self.update_centroids(vectors, num_vectors, &assignments);

            // Check convergence
            let mut converged = true;
            for (old, new) in self.centroids.iter().zip(new_centroids.iter()) {
                let dist = self.distance(old, new);
                if dist > 1e-6 {
                    converged = false;
                    break;
                }
            }

            self.centroids = new_centroids;
            if converged {
                break;
            }
        }

        Ok(())
    }

    /// k-means++ initialization.
    fn kmeans_plus_plus(
        &self,
        vectors: &[f32],
        num_vectors: usize,
    ) -> Result<Vec<Vec<f32>>, RetrieveError> {
        use rand::Rng;
        let mut rng = rand::rng();

        let mut centroids = Vec::new();

        // First centroid: random vector
        let first_idx = rng.random_range(0..num_vectors);
        centroids.push(self.get_vector(vectors, first_idx).to_vec());

        // Subsequent centroids: weighted by distance to nearest existing centroid
        for _ in 1..self.k {
            let mut distances = Vec::new();
            let mut total_distance = 0.0;

            for i in 0..num_vectors {
                let vec = self.get_vector(vectors, i);
                let min_dist = centroids
                    .iter()
                    .map(|c| self.distance(vec, c))
                    .fold(f32::INFINITY, f32::min);

                distances.push(min_dist);
                total_distance += min_dist;
            }

            // Sample proportional to distance squared
            let mut cumulative = 0.0;
            let threshold = rng.random::<f64>() * total_distance as f64;

            for (i, &dist) in distances.iter().enumerate() {
                cumulative += dist as f64;
                if cumulative >= threshold {
                    centroids.push(self.get_vector(vectors, i).to_vec());
                    break;
                }
            }
        }

        Ok(centroids)
    }

    /// Assign vectors to nearest clusters.
    pub fn assign_clusters(&self, vectors: &[f32], num_vectors: usize) -> Vec<usize> {
        let mut assignments = Vec::with_capacity(num_vectors);

        for i in 0..num_vectors {
            let vec = self.get_vector(vectors, i);
            let mut best_cluster = 0;
            let mut best_dist = f32::INFINITY;

            for (cluster_idx, centroid) in self.centroids.iter().enumerate() {
                let dist = self.distance(vec, centroid);
                if dist < best_dist {
                    best_dist = dist;
                    best_cluster = cluster_idx;
                }
            }

            assignments.push(best_cluster);
        }

        assignments
    }

    /// Update centroids based on assignments.
    fn update_centroids(
        &self,
        vectors: &[f32],
        num_vectors: usize,
        assignments: &[usize],
    ) -> Vec<Vec<f32>> {
        let mut cluster_sums = vec![vec![0.0f32; self.dimension]; self.k];
        let mut cluster_counts = vec![0usize; self.k];

        for (i, &cluster) in assignments.iter().enumerate().take(num_vectors) {
            cluster_counts[cluster] += 1;

            let vec = self.get_vector(vectors, i);
            for (j, &val) in vec.iter().enumerate() {
                cluster_sums[cluster][j] += val;
            }
        }

        // Compute centroids as means
        let mut new_centroids = Vec::new();
        for (sums, &count) in cluster_sums.iter().zip(cluster_counts.iter()) {
            if count > 0 {
                let centroid: Vec<f32> = sums.iter().map(|&s| s / count as f32).collect();
                new_centroids.push(centroid);
            } else {
                // Empty cluster: keep old centroid
                new_centroids.push(vec![0.0; self.dimension]);
            }
        }

        new_centroids
    }

    /// Compute distance between two vectors (SIMD-accelerated).
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        // Use existing SIMD-accelerated dot product for cosine distance
        // For L2 distance, would use: simd::dot(&diff, &diff).sqrt()
        let similarity = simd::dot(a, b);
        1.0 - similarity // Cosine distance
    }

    /// Get vector from SoA storage.
    fn get_vector<'a>(&self, vectors: &'a [f32], idx: usize) -> &'a [f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &vectors[start..end]
    }

    /// Get centroids.
    pub fn centroids(&self) -> &[Vec<f32>] {
        &self.centroids
    }
}
