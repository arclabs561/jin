//! Partitioning/clustering interface for ANN methods.
//!
//! Provides unified interface for partitioning vectors, supporting both
//! k-means (flat clustering) and EVōC (hierarchical clustering).

pub mod kmeans;

use crate::RetrieveError;

/// Partitioning result: vector index -> partition/cluster ID.
pub type PartitionAssignments = Vec<usize>;

/// Partitioning trait for different clustering methods.
pub trait Partitioner: Send + Sync {
    /// Fit partitioner on vectors.
    fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError>;

    /// Assign vectors to partitions.
    fn assign(
        &self,
        vectors: &[f32],
        num_vectors: usize,
    ) -> Result<PartitionAssignments, RetrieveError>;

    /// Get partition centroids (for routing queries).
    fn centroids(&self) -> &[Vec<f32>];

    /// Number of partitions.
    fn num_partitions(&self) -> usize;
}

#[cfg(feature = "scann")]
use crate::partitioning::kmeans::KMeans;

/// k-means partitioner (flat clustering).
#[cfg(feature = "scann")]
pub struct KMeansPartitioner {
    kmeans: KMeans,
}

#[cfg(feature = "scann")]
impl KMeansPartitioner {
    pub fn new(dimension: usize, k: usize) -> Result<Self, RetrieveError> {
        Ok(Self {
            kmeans: KMeans::new(dimension, k)?,
        })
    }
}

#[cfg(feature = "scann")]
impl Partitioner for KMeansPartitioner {
    fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        self.kmeans.fit(vectors, num_vectors)
    }

    fn assign(
        &self,
        vectors: &[f32],
        num_vectors: usize,
    ) -> Result<PartitionAssignments, RetrieveError> {
        Ok(self.kmeans.assign_clusters(vectors, num_vectors))
    }

    fn centroids(&self) -> &[Vec<f32>] {
        self.kmeans.centroids()
    }

    fn num_partitions(&self) -> usize {
        self.kmeans.centroids().len()
    }
}

/// EVōC partitioner (hierarchical clustering, uses finest layer).
#[cfg(feature = "evoc")]
pub struct EVoCPartitioner {
    evoc: crate::evoc::clustering::EVoC,
    #[allow(dead_code)]
    dimension: usize,
    num_partitions: usize,
    assignments: Option<PartitionAssignments>,
}

#[cfg(feature = "evoc")]
impl EVoCPartitioner {
    pub fn new(dimension: usize, num_partitions: usize) -> Result<Self, RetrieveError> {
        use crate::evoc::clustering::EVoCParams;

        let params = EVoCParams {
            intermediate_dim: 15,
            min_cluster_size: 10,
            noise_level: 0.0,
            min_number_clusters: Some(num_partitions),
        };

        Ok(Self {
            evoc: crate::evoc::clustering::EVoC::new(dimension, params)?,
            dimension,
            num_partitions,
            assignments: None,
        })
    }
}

#[cfg(feature = "evoc")]
impl Partitioner for EVoCPartitioner {
    fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        let assignments_opt = self.evoc.fit_predict(vectors, num_vectors)?;

        // Convert Option<usize> to usize (None -> 0 for noise)
        let assignments: PartitionAssignments = assignments_opt
            .into_iter()
            .map(|opt| opt.unwrap_or(0))
            .collect();

        self.assignments = Some(assignments);
        Ok(())
    }

    fn assign(
        &self,
        _vectors: &[f32],
        _num_vectors: usize,
    ) -> Result<PartitionAssignments, RetrieveError> {
        self.assignments
            .clone()
            .ok_or_else(|| RetrieveError::Other("EVōC not fitted".to_string()))
    }

    fn centroids(&self) -> &[Vec<f32>] {
        // EVōC doesn't have explicit centroids, return empty
        // In practice, would compute centroids from cluster members
        &[]
    }

    fn num_partitions(&self) -> usize {
        self.num_partitions
    }
}
