//! Flat NSW graph structure.

use crate::RetrieveError;
use smallvec::SmallVec;

/// Flat Navigable Small World index.
///
/// Single-layer graph variant achieving performance parity with HNSW
/// in high-dimensional settings with lower memory overhead.
#[derive(Debug)]
pub struct NSWIndex {
    /// Vectors stored in Structure of Arrays (SoA) format
    pub(crate) vectors: Vec<f32>,

    /// Vector dimension
    pub(crate) dimension: usize,

    /// Number of vectors
    pub(crate) num_vectors: usize,

    /// Single graph layer (no hierarchy)
    pub(crate) neighbors: Vec<SmallVec<[u32; 16]>>,

    /// Parameters
    pub(crate) params: NSWParams,

    /// Whether index has been built
    built: bool,

    /// Entry point for search
    pub(crate) entry_point: Option<u32>,
}

/// NSW parameters.
#[derive(Clone, Debug)]
pub struct NSWParams {
    /// Maximum number of connections per node (typically 16)
    pub m: usize,

    /// Maximum connections for newly inserted nodes (typically 16)
    pub m_max: usize,

    /// Search width during construction (typically 200)
    pub ef_construction: usize,

    /// Default search width during query (typically 50-200)
    pub ef_search: usize,
}

impl Default for NSWParams {
    fn default() -> Self {
        Self {
            m: 16,
            m_max: 16,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

impl NSWIndex {
    /// Create a new NSW index.
    pub fn new(dimension: usize, m: usize, m_max: usize) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::EmptyQuery);
        }
        if m == 0 || m_max == 0 {
            return Err(RetrieveError::Other(
                "m and m_max must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            neighbors: Vec::new(),
            params: NSWParams {
                m,
                m_max,
                ..Default::default()
            },
            built: false,
            entry_point: None,
        })
    }

    /// Create with custom parameters.
    pub fn with_params(dimension: usize, params: NSWParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::EmptyQuery);
        }
        if params.m == 0 || params.m_max == 0 {
            return Err(RetrieveError::Other(
                "m and m_max must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            neighbors: Vec::new(),
            params,
            built: false,
            entry_point: None,
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, _doc_id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        self.add_slice(_doc_id, &vector)
    }

    /// Add a vector to the index from a borrowed slice.
    ///
    /// Notes:
    /// - The index stores vectors internally, so it must copy the slice into its own storage.
    /// - NSW currently ignores `doc_id` and uses insertion order as the internal ID.
    pub fn add_slice(&mut self, _doc_id: u32, vector: &[f32]) -> Result<(), RetrieveError> {
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

        // Store vector in SoA format
        self.vectors.extend_from_slice(vector);
        self.num_vectors += 1;

        Ok(())
    }

    /// Build the index (required before search).
    ///
    /// Constructs the single-layer graph structure.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Construct flat graph
        super::construction::construct_graph(self)?;
        self.built = true;

        Ok(())
    }

    /// Search for k nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
    ) -> Result<Vec<(u32, f32)>, RetrieveError> {
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

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let entry_point = self.entry_point.ok_or(RetrieveError::EmptyIndex)?;

        // Greedy search in single layer
        let results = super::search::greedy_search(
            query,
            entry_point,
            &self.neighbors,
            &self.vectors,
            self.dimension,
            ef.max(k),
        )?;

        // Return top k
        let mut sorted_results: Vec<(u32, f32)> = results.into_iter().take(k).collect();
        sorted_results.sort_by(|a, b| a.1.total_cmp(&b.1));
        Ok(sorted_results)
    }

    /// Get vector by index.
    pub(crate) fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
}
