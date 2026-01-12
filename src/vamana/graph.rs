//! Vamana graph structure and core types.

use crate::RetrieveError;
use smallvec::SmallVec;

#[cfg(feature = "vamana")]
/// Vamana parameters controlling graph structure and search behavior.
#[derive(Clone, Debug)]
pub struct VamanaParams {
    /// Maximum out-degree per node (typically 64-128, higher for SSD serving)
    pub max_degree: usize,

    /// Relaxation factor for RRND (typically 1.3-1.5)
    /// Higher alpha = less pruning, larger graphs
    pub alpha: f32,

    /// Search width during construction (typically 200-400)
    pub ef_construction: usize,

    /// Default search width during query (typically 50-200)
    pub ef_search: usize,
}

#[cfg(feature = "vamana")]
impl Default for VamanaParams {
    fn default() -> Self {
        Self {
            max_degree: 64,
            alpha: 1.3,
            ef_construction: 200,
            ef_search: 50,
        }
    }
}

#[cfg(feature = "vamana")]
/// Vamana index for approximate nearest neighbor search.
///
/// Uses two-pass construction with RRND + RND for high-quality graph structure.
pub struct VamanaIndex {
    /// Vector dimension
    pub(crate) dimension: usize,

    /// Vectors stored in Structure of Arrays (SoA) layout
    pub(crate) vectors: Vec<f32>,

    /// Neighbor lists for each vector
    pub(crate) neighbors: Vec<SmallVec<[u32; 16]>>,

    /// Parameters
    pub(crate) params: VamanaParams,

    /// Number of vectors added
    pub(crate) num_vectors: usize,

    /// Whether index has been built
    built: bool,
}

#[cfg(feature = "vamana")]
impl VamanaIndex {
    /// Create a new Vamana index.
    pub fn new(dimension: usize, params: VamanaParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::Other("Dimension must be > 0".to_string()));
        }

        Ok(Self {
            dimension,
            vectors: Vec::new(),
            neighbors: Vec::new(),
            params,
            num_vectors: 0,
            built: false,
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, _id: u32, vector: Vec<f32>) -> Result<(), RetrieveError> {
        if self.built {
            return Err(RetrieveError::Other(
                "Cannot add vectors after build".to_string(),
            ));
        }

        if vector.len() != self.dimension {
            return Err(RetrieveError::DimensionMismatch {
                query_dim: self.dimension,
                doc_dim: vector.len(),
            });
        }

        // Extend vectors array (SoA layout)
        self.vectors.extend_from_slice(&vector);
        self.neighbors.push(SmallVec::new());
        self.num_vectors += 1;

        Ok(())
    }

    /// Build the index (two-pass construction).
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        if self.built {
            return Err(RetrieveError::Other("Index already built".to_string()));
        }

        // Two-pass construction: RRND + RND
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

        super::search::search(self, query, k, ef)
    }

    /// Get vector by index.
    pub(crate) fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
}

#[cfg(all(test, feature = "vamana"))]
mod tests {
    use super::*;

    #[test]
    fn test_vamana_create() {
        let params = VamanaParams::default();
        let index = VamanaIndex::new(128, params);
        assert!(index.is_ok());
    }

    #[test]
    fn test_vamana_add() {
        let params = VamanaParams::default();
        let mut index = VamanaIndex::new(128, params).unwrap();

        let vector = vec![0.1; 128];
        assert!(index.add(0, vector).is_ok());
        assert_eq!(index.num_vectors, 1);
    }
}
