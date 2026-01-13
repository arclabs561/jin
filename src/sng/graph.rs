//! OPT-SNG graph structure.

use crate::RetrieveError;
use smallvec::SmallVec;

/// OPT-SNG index for approximate nearest neighbor search.
///
/// Optimized version of Sparse Neighborhood Graph with:
/// - Automatic truncation parameter optimization
/// - Martingale-based pruning model
/// - Theoretical guarantees
pub struct SNGIndex {
    /// Vectors stored in SoA format
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    params: SNGParams,
    built: bool,

    /// Graph structure: neighbors[i] = neighbors of vector i
    pub(crate) neighbors: Vec<SmallVec<[u32; 16]>>,

    /// Truncation parameter R (automatically optimized)
    truncation_r: f32,
}

/// OPT-SNG parameters.
#[derive(Clone, Debug)]
pub struct SNGParams {
    /// Maximum out-degree (automatically optimized, but can set initial value)
    pub max_degree: Option<usize>,

    /// Number of hash functions for LSH-based construction (optional)
    pub num_hash_functions: usize,
}

impl Default for SNGParams {
    fn default() -> Self {
        Self {
            max_degree: None, // Will be auto-optimized
            num_hash_functions: 10,
        }
    }
}

impl SNGIndex {
    /// Create a new OPT-SNG index.
    pub fn new(dimension: usize, params: SNGParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::EmptyQuery);
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            params,
            built: false,
            neighbors: Vec::new(),
            truncation_r: 0.0, // Will be optimized during build
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

    /// Build the index with automatic parameter optimization.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Optimize truncation parameter R using closed-form rule
        self.truncation_r =
            crate::sng::optimization::optimize_truncation_r(self.num_vectors, self.dimension)?;

        // Build graph using martingale-based model
        self.construct_graph()?;

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

        crate::sng::search::search_sng(self, query, k)
    }

    /// Get vector from SoA storage.
    pub(crate) fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }

    /// Construct graph using martingale-based model.
    fn construct_graph(&mut self) -> Result<(), RetrieveError> {
        use crate::simd;
        use crate::sng::martingale;
        use smallvec::SmallVec;

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Initialize neighbor lists
        self.neighbors = vec![SmallVec::new(); self.num_vectors];

        // Build graph using martingale-based pruning
        let mut evolution = martingale::CandidateEvolution::new();

        for current_id in 0..self.num_vectors {
            let current_vector = self.get_vector(current_id);

            // Find candidates: all other vectors
            let mut candidates = Vec::new();
            for other_id in 0..self.num_vectors {
                if other_id == current_id {
                    continue;
                }

                let other_vector = self.get_vector(other_id);
                let dist = 1.0 - simd::dot(current_vector, other_vector);
                candidates.push((other_id as u32, dist));
            }

            // Prune using martingale-based model
            let pruned = martingale::prune_candidates_martingale(
                &candidates,
                self.truncation_r,
                &self.vectors,
                self.dimension,
            )?;

            // Update evolution tracker
            evolution.update(pruned.len());

            // Add bidirectional connections
            for &neighbor_id in &pruned {
                // Add connection from current to neighbor
                self.neighbors[current_id].push(neighbor_id);

                // Add reverse connection (if not already present)
                let reverse_neighbors = &mut self.neighbors[neighbor_id as usize];
                if !reverse_neighbors.contains(&(current_id as u32)) {
                    reverse_neighbors.push(current_id as u32);
                }
            }
        }

        Ok(())
    }
}
