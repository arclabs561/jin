//! LSH search implementation.

use crate::simd;
use crate::RetrieveError;
use std::collections::HashMap;

/// LSH index for approximate nearest neighbor search.
pub struct LSHIndex {
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    params: LSHParams,
    built: bool,

    /// Hash tables: [table][hash] -> vector indices
    pub(crate) hash_tables: Vec<HashMap<u64, Vec<u32>>>,

    /// Hash functions: random projection vectors
    pub(crate) hash_functions: Vec<Vec<f32>>,
}

/// LSH parameters.
#[derive(Clone, Debug)]
pub struct LSHParams {
    /// Number of hash tables
    pub num_tables: usize,

    /// Number of hash functions per table
    pub num_functions: usize,

    /// Number of candidates to verify
    pub num_candidates: usize,
}

impl Default for LSHParams {
    fn default() -> Self {
        Self {
            num_tables: 10,
            num_functions: 10,
            num_candidates: 100,
        }
    }
}

impl LSHIndex {
    /// Create a new LSH index.
    pub fn new(dimension: usize, params: LSHParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::EmptyQuery);
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            params,
            built: false,
            hash_tables: Vec::new(),
            hash_functions: Vec::new(),
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

    /// Build the index with hash tables.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // Generate hash functions (random projection vectors)
        use rand::Rng;
        let mut rng = rand::rng();

        let total_functions = self.params.num_tables * self.params.num_functions;
        self.hash_functions = (0..total_functions)
            .map(|_| {
                (0..self.dimension)
                    .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                    .collect()
            })
            .collect();

        // Build hash tables
        self.hash_tables = vec![HashMap::new(); self.params.num_tables];

        // Pre-compute all hashes to avoid borrowing conflicts
        let mut hash_values: Vec<Vec<u64>> = Vec::new();
        for vector_idx in 0..self.num_vectors {
            let vec = self.get_vector(vector_idx);
            let mut hashes = Vec::new();
            for table_idx in 0..self.params.num_tables {
                let hash = self.compute_hash(vec, table_idx);
                hashes.push(hash);
            }
            hash_values.push(hashes);
        }

        // Now populate hash tables
        for vector_idx in 0..self.num_vectors {
            for table_idx in 0..self.params.num_tables {
                let hash = hash_values[vector_idx][table_idx];
                self.hash_tables[table_idx]
                    .entry(hash)
                    .or_default()
                    .push(vector_idx as u32);
            }
        }

        self.built = true;
        Ok(())
    }

    /// Compute hash for a vector in a specific table.
    fn compute_hash(&self, vector: &[f32], table_idx: usize) -> u64 {
        let mut hash = 0u64;

        for func_idx in 0..self.params.num_functions {
            let hash_func_idx = table_idx * self.params.num_functions + func_idx;
            let hash_func = &self.hash_functions[hash_func_idx];

            // Random projection: sign of dot product
            let projection = simd::dot(vector, hash_func);
            let bit = if projection >= 0.0 { 1 } else { 0 };

            hash = (hash << 1) | bit;
        }

        hash
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

        // Collect candidates from hash tables
        let mut candidate_set = std::collections::HashSet::new();

        for table_idx in 0..self.params.num_tables {
            let hash = self.compute_hash(query, table_idx);
            if let Some(indices) = self.hash_tables[table_idx].get(&hash) {
                for &idx in indices {
                    candidate_set.insert(idx);
                }
            }
        }

        // Verify candidates with exact distances
        let mut candidates: Vec<(u32, f32)> = candidate_set
            .iter()
            .map(|&idx| {
                let vec = self.get_vector(idx as usize);
                let dist = 1.0 - simd::dot(query, vec);
                (idx, dist)
            })
            .collect();

        // Sort and return top k (unstable for better performance)
        candidates.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
        Ok(candidates.into_iter().take(k).collect())
    }

    /// Get vector from SoA storage.
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }
}
