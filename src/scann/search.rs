//! SCANN search implementation.

use crate::scann::partitioning::KMeans;
use crate::scann::reranking;
use crate::RetrieveError;

/// Anisotropic Vector Quantization with k-means Partitioning index.
///
/// Three-stage ANN framework:
/// 1. **Partitioning**: k-means clustering (coarse search, reduces search space)
/// 2. **Quantization**: Anisotropic vector quantization (fine search, preserves inner products)
/// 3. **Re-ranking**: Exact distance computation (accuracy refinement)
///
/// **Technical Name**: Anisotropic Vector Quantization with k-means Partitioning
/// **Vendor Name**: SCANN/ScaNN (Google Research)
///
/// **Relationships**:
/// - Similar to IVF-PQ but uses AVQ (anisotropic) instead of PQ (isotropic)
/// - AVQ optimized for MIPS (Maximum Inner Product Search)
/// - Can be combined with graph-based methods for hybrid approaches
#[derive(Debug)]
pub struct SCANNIndex {
    /// Vectors stored in SoA format
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    params: SCANNParams,
    built: bool,

    // Partitioning
    partitions: Vec<Partition>,
    pub(crate) partition_centroids: Vec<Vec<f32>>,
}

/// Anisotropic Vector Quantization with k-means Partitioning parameters.
#[derive(Clone, Debug)]
pub struct SCANNParams {
    /// Number of partitions (clusters)
    pub num_partitions: usize,

    /// Number of candidates to re-rank
    pub num_reorder: usize,

    /// Anisotropic quantization parameters
    pub quantization_bits: usize,
}

impl Default for SCANNParams {
    fn default() -> Self {
        Self {
            num_partitions: 256,
            num_reorder: 100,
            quantization_bits: 8,
        }
    }
}

/// Partition (cluster) containing vector indices.
#[derive(Clone, Debug)]
struct Partition {
    vector_indices: Vec<u32>,
}

impl SCANNIndex {
    /// Create a new SCANN index.
    pub fn new(dimension: usize, params: SCANNParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::EmptyQuery);
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            params,
            built: false,
            partitions: Vec::new(),
            partition_centroids: Vec::new(),
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, _doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::Other(
                "Cannot add vectors after index is built".to_string(),
            ));
        }

        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: self.dimension,
                doc_dim: vector.len(),
            });
        }

        self.vectors.extend_from_slice(&vector);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the index.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Stage 1: k-means partitioning
        let mut kmeans = KMeans::new(self.dimension, self.params.num_partitions)?;
        kmeans.fit(&self.vectors, self.num_vectors)?;
        self.partition_centroids = kmeans.centroids().to_vec();

        // Assign vectors to partitions
        let assignments = kmeans.assign_clusters(&self.vectors, self.num_vectors);
        self.partitions = vec![
            Partition {
                vector_indices: Vec::new()
            };
            self.params.num_partitions
        ];

        for (vector_idx, &partition_idx) in assignments.iter().enumerate() {
            self.partitions[partition_idx]
                .vector_indices
                .push(vector_idx as u32);
        }

        // Stage 2: Anisotropic quantization (placeholder - will implement full PQ)
        // For now, we'll use exact vectors in search

        self.built = true;
        Ok(())
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::Other(
                "Index must be built before search".to_string(),
            ));
        }

        if query.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: self.dimension,
                doc_dim: query.len(),
            });
        }

        // Stage 1: Find closest partitions
        let mut partition_distances: Vec<(usize, f32)> = self
            .partition_centroids
            .iter()
            .enumerate()
            .map(|(idx, centroid)| {
                let dist = 1.0 - crate::simd::dot(query, centroid);
                (idx, dist)
            })
            .collect();

        partition_distances
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)); // Unstable for better performance

        // Search in top partitions
        let num_partitions_to_search = (self.params.num_partitions / 10).clamp(1, 10);
        let mut candidates = Vec::new();

        for (partition_idx, _) in partition_distances.iter().take(num_partitions_to_search) {
            let partition = &self.partitions[*partition_idx];

            for &vector_idx in &partition.vector_indices {
                let vec = self.get_vector(vector_idx as usize);
                let dist = 1.0 - crate::simd::dot(query, vec);
                candidates.push((vector_idx, dist));
            }
        }

        // Stage 2: Re-rank top candidates with exact distances (unstable for better performance)
        candidates
            .sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_candidates: Vec<(u32, f32)> = candidates
            .into_iter()
            .take(self.params.num_reorder.max(k))
            .collect();

        let reranked = reranking::rerank(query, &top_candidates, &self.vectors, self.dimension, k);

        Ok(reranked)
    }

    /// Get vector from SoA storage.
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
}
