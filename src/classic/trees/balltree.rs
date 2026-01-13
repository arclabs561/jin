//! Ball Tree implementation.
//!
//! Space-partitioning tree using hyperspheres (balls) instead of hyperplanes.
//! Better than KD-Tree for medium dimensions (20 < d < 100).
//!
//! **Technical Name**: Ball Tree
//!
//! Algorithm:
//! - Recursive space partitioning using hyperspheres
//! - Each node represents a ball (center + radius) containing its vectors
//! - Better for medium dimensions than KD-Tree
//! - More robust to high-dimensional data
//!
//! **Relationships**:
//! - Improvement over KD-Tree for medium dimensions
//! - Uses hyperspheres instead of hyperplanes
//! - Complementary to KD-Tree (KD-Tree better for d < 20, Ball Tree better for 20 < d < 100)
//!
//! # References
//!
//! - Omohundro (1989): "Five balltree construction algorithms"
//! - Liu et al. (2006): "An investigation of practical approximate nearest neighbor algorithms"

use crate::simd;
use crate::RetrieveError;

/// Ball Tree index.
///
/// Space-partitioning tree using hyperspheres for medium-dimensional data.
pub struct BallTreeIndex {
    pub(crate) vectors: Vec<f32>,
    pub(crate) dimension: usize,
    pub(crate) num_vectors: usize,
    params: BallTreeParams,
    built: bool,
    root: Option<BallNode>,
}

/// Ball Tree parameters.
#[derive(Clone, Debug)]
pub struct BallTreeParams {
    /// Maximum leaf size
    pub max_leaf_size: usize,

    /// Maximum depth
    pub max_depth: usize,
}

impl Default for BallTreeParams {
    fn default() -> Self {
        Self {
            max_leaf_size: 10,
            max_depth: 32,
        }
    }
}

/// Ball Tree node.
enum BallNode {
    /// Internal node: has center, radius, and children
    Internal {
        center: Vec<f32>,
        radius: f32,
        left: Box<BallNode>,
        right: Box<BallNode>,
    },
    /// Leaf node: contains vector indices
    Leaf {
        indices: Vec<u32>,
        center: Vec<f32>,
        radius: f32,
    },
}

impl BallTreeIndex {
    /// Create new Ball Tree index.
    pub fn new(dimension: usize, params: BallTreeParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::Other(
                "Dimension must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            vectors: Vec::new(),
            dimension,
            num_vectors: 0,
            params,
            built: false,
            root: None,
        })
    }

    /// Add a vector to the index.
    pub fn add(&mut self, _doc_id: u32, embedding: Vec<f32>) -> Result<(), RetrieveError> {
        if embedding.len() != self.dimension {
            return Err(RetrieveError::Other(format!(
                "Embedding dimension {} != {}",
                embedding.len(),
                self.dimension
            )));
        }

        if self.built {
            return Err(RetrieveError::Other(
                "Cannot add vectors after build".to_string(),
            ));
        }

        self.vectors.extend_from_slice(&embedding);
        self.num_vectors += 1;
        Ok(())
    }

    /// Build the Ball Tree.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        let indices: Vec<u32> = (0..self.num_vectors as u32).collect();
        self.root = Some(self.build_tree(&indices, 0)?);

        self.built = true;
        Ok(())
    }

    /// Build tree recursively.
    fn build_tree(&self, indices: &[u32], depth: usize) -> Result<BallNode, RetrieveError> {
        if indices.is_empty() {
            return Err(RetrieveError::Other("Empty indices".to_string()));
        }

        // Compute center and radius
        let center = self.compute_center(indices);
        let radius = self.compute_radius(indices, &center);

        // Leaf node if small enough or max depth reached
        if indices.len() <= self.params.max_leaf_size || depth >= self.params.max_depth {
            return Ok(BallNode::Leaf {
                indices: indices.to_vec(),
                center,
                radius,
            });
        }

        // Find two farthest points as seeds for splitting
        let (seed1_idx, seed2_idx) = self.find_farthest_pair(indices);

        // Split indices by distance to seeds
        let mut left_indices = Vec::new();
        let mut right_indices = Vec::new();

        for &idx in indices {
            let vec = self.get_vector(idx as usize);
            let dist1 = self.euclidean_distance(vec, &self.get_vector(seed1_idx as usize));
            let dist2 = self.euclidean_distance(vec, &self.get_vector(seed2_idx as usize));

            if dist1 < dist2 {
                left_indices.push(idx);
            } else {
                right_indices.push(idx);
            }
        }

        // Ensure both sides have at least one point
        if left_indices.is_empty() {
            left_indices.push(right_indices.pop().unwrap());
        }
        if right_indices.is_empty() {
            right_indices.push(left_indices.pop().unwrap());
        }

        // Build children
        let left = self.build_tree(&left_indices, depth + 1)?;
        let right = self.build_tree(&right_indices, depth + 1)?;

        Ok(BallNode::Internal {
            center,
            radius,
            left: Box::new(left),
            right: Box::new(right),
        })
    }

    /// Compute center of vectors.
    fn compute_center(&self, indices: &[u32]) -> Vec<f32> {
        let mut center = vec![0.0f32; self.dimension];

        for &idx in indices {
            let vec = self.get_vector(idx as usize);
            for (j, &val) in vec.iter().enumerate() {
                center[j] += val;
            }
        }

        let count = indices.len() as f32;
        for val in center.iter_mut() {
            *val /= count;
        }

        center
    }

    /// Compute radius (max distance from center).
    fn compute_radius(&self, indices: &[u32], center: &[f32]) -> f32 {
        let mut max_radius = 0.0f32;

        for &idx in indices {
            let vec = self.get_vector(idx as usize);
            let dist = self.euclidean_distance(vec, center);
            max_radius = max_radius.max(dist);
        }

        max_radius
    }

    /// Find two farthest points.
    fn find_farthest_pair(&self, indices: &[u32]) -> (u32, u32) {
        let mut max_dist = 0.0f32;
        let mut pair = (indices[0], indices[0]);

        for i in 0..indices.len() {
            for j in (i + 1)..indices.len() {
                let vec1 = self.get_vector(indices[i] as usize);
                let vec2 = self.get_vector(indices[j] as usize);
                let dist = self.euclidean_distance(vec1, vec2);

                if dist > max_dist {
                    max_dist = dist;
                    pair = (indices[i], indices[j]);
                }
            }
        }

        pair
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(u32, f32)>, RetrieveError> {
        if !self.built {
            return Err(RetrieveError::Other("Index not built".to_string()));
        }

        if query.len() != self.dimension {
            return Err(RetrieveError::Other(format!(
                "Query dimension {} != {}",
                query.len(),
                self.dimension
            )));
        }

        let root = self
            .root
            .as_ref()
            .ok_or_else(|| RetrieveError::Other("Tree not built".to_string()))?;

        // Collect candidates from tree traversal
        let mut candidates = Vec::new();
        self.search_recursive(root, query, &mut candidates)?;

        // Compute distances and sort
        let mut results: Vec<(u32, f32)> = candidates
            .iter()
            .map(|&idx| {
                let vec = self.get_vector(idx as usize);
                let dist = self.cosine_distance(query, vec);
                (idx, dist)
            })
            .collect();

        results.sort_by(|a, b| a.1.total_cmp(&b.1));
        results.truncate(k);

        Ok(results)
    }

    /// Search recursively.
    fn search_recursive(
        &self,
        node: &BallNode,
        query: &[f32],
        candidates: &mut Vec<u32>,
    ) -> Result<(), RetrieveError> {
        match node {
            BallNode::Leaf { indices, .. } => {
                candidates.extend_from_slice(indices);
            }
            BallNode::Internal {
                center,
                radius: _, // TODO: Use for pruning optimization
                left,
                right,
            } => {
                // Compute distance from query to ball center
                let _dist_to_center = self.euclidean_distance(query, center); // TODO: Use for pruning

                // Prune if ball is too far (distance > radius + best_dist)
                // For now, traverse both (optimization: add pruning)
                self.search_recursive(left, query, candidates)?;
                self.search_recursive(right, query, candidates)?;
            }
        }

        Ok(())
    }

    /// Get vector from SoA storage.
    fn get_vector(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.vectors[start..end]
    }

    /// Compute Euclidean distance.
    /// Optimized to use SIMD-accelerated operations.
    fn euclidean_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        use crate::simd;
        // Compute squared distance: sum((a[i] - b[i])^2) = sum(a[i]^2) + sum(b[i]^2) - 2*sum(a[i]*b[i])
        let a_squared = simd::dot(a, a);
        let b_squared = simd::dot(b, b);
        let ab_dot = simd::dot(a, b);
        (a_squared + b_squared - 2.0 * ab_dot).sqrt()
    }

    /// Compute cosine distance.
    fn cosine_distance(&self, a: &[f32], b: &[f32]) -> f32 {
        let similarity = simd::dot(a, b);
        1.0 - similarity
    }
}
