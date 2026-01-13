//! Hierarchical clustering for EVōC.
//!
//! Implements HDBSCAN-style hierarchical clustering using MST (Minimum Spanning Tree)
//! on the reduced-dimensional space.

use crate::evoc::hierarchy::ClusterHierarchy;
use crate::evoc::reduction::DimensionalityReducer;
use crate::RetrieveError;
use std::collections::{HashMap, HashSet};

/// EVōC clustering parameters.
#[derive(Clone, Debug)]
pub struct EVoCParams {
    /// Intermediate dimension for reduction (~15 recommended)
    pub intermediate_dim: usize,

    /// Minimum cluster size (base_min_cluster_size)
    pub min_cluster_size: usize,

    /// Noise level tolerance (0.0 = cluster more, higher = more noise)
    pub noise_level: f32,

    /// Minimum number of clusters (optional)
    pub min_number_clusters: Option<usize>,
}

impl Default for EVoCParams {
    fn default() -> Self {
        Self {
            intermediate_dim: 15,
            min_cluster_size: 10,
            noise_level: 0.0,
            min_number_clusters: None,
        }
    }
}

/// Single cluster layer (one granularity level).
#[derive(Clone, Debug)]
pub struct ClusterLayer {
    /// Cluster assignments: vector_idx -> cluster_id (or None for noise)
    pub assignments: Vec<Option<usize>>,

    /// Number of clusters at this layer
    pub num_clusters: usize,

    /// Cluster members: cluster_id -> Vec<vector_idx>
    pub clusters: HashMap<usize, Vec<usize>>,
}

/// EVōC clusterer.
///
/// Fast hierarchical clustering optimized for embedding vectors.
pub struct EVoC {
    params: EVoCParams,
    original_dim: usize,
    reducer: Option<DimensionalityReducer>,
    hierarchy: Option<ClusterHierarchy>,
    cluster_layers: Vec<ClusterLayer>,
    duplicates: Vec<Vec<usize>>, // Groups of near-duplicate vector indices
}

impl EVoC {
    /// Create new EVōC clusterer.
    pub fn new(original_dim: usize, params: EVoCParams) -> Result<Self, RetrieveError> {
        if original_dim == 0 {
            return Err(RetrieveError::Other(
                "Original dimension must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            params,
            original_dim,
            reducer: None,
            hierarchy: None,
            cluster_layers: Vec::new(),
            duplicates: Vec::new(),
        })
    }

    /// Fit clusterer on vectors and extract hierarchical clusters.
    pub fn fit_predict(
        &mut self,
        vectors: &[f32],
        num_vectors: usize,
    ) -> Result<Vec<Option<usize>>, RetrieveError> {
        if vectors.len() < num_vectors * self.original_dim {
            return Err(RetrieveError::Other("Insufficient vectors".to_string()));
        }

        // Step 1: Dimensionality reduction
        let mut reducer =
            DimensionalityReducer::new(self.original_dim, self.params.intermediate_dim)?;
        reducer.fit(vectors, num_vectors)?;
        let reduced_vectors = reducer.transform(vectors, num_vectors)?;
        self.reducer = Some(reducer);

        // Step 2: Build hierarchical clustering (MST-based)
        let hierarchy = self.build_hierarchy(&reduced_vectors, num_vectors)?;
        self.hierarchy = Some(hierarchy);

        // Step 3: Extract multi-granularity cluster layers
        self.cluster_layers = self.extract_cluster_layers(num_vectors)?;

        // Step 4: Detect near-duplicates
        self.duplicates = self.detect_duplicates(&reduced_vectors, num_vectors)?;

        // Return finest-grained layer assignments
        if let Some(layer) = self.cluster_layers.first() {
            Ok(layer.assignments.clone())
        } else {
            Ok(vec![None; num_vectors])
        }
    }

    /// Build hierarchical clustering using MST.
    fn build_hierarchy(
        &self,
        reduced_vectors: &[f32],
        num_vectors: usize,
    ) -> Result<ClusterHierarchy, RetrieveError> {
        // Compute pairwise distances in reduced space
        let mut edges = Vec::new();
        for i in 0..num_vectors {
            let vec_i = self.get_reduced_vector(reduced_vectors, i);
            for j in (i + 1)..num_vectors {
                let vec_j = self.get_reduced_vector(reduced_vectors, j);
                let dist = self.euclidean_distance(vec_i, vec_j);
                edges.push((i, j, dist));
            }
        }

        // Sort by distance (for MST)
        edges.sort_by(|a, b| a.2.total_cmp(&b.2));

        // Build MST using Kruskal's algorithm
        let mut mst_edges = Vec::new();
        let mut union_find = UnionFind::new(num_vectors);

        for (i, j, dist) in edges {
            if union_find.find(i) != union_find.find(j) {
                union_find.union(i, j);
                mst_edges.push((i, j, dist));
            }
        }

        // Build hierarchy from MST
        let hierarchy = ClusterHierarchy::from_mst(mst_edges, num_vectors);

        Ok(hierarchy)
    }

    /// Extract multi-granularity cluster layers.
    fn extract_cluster_layers(
        &self,
        _num_vectors: usize, // Reserved for granularity-based layer sizing
    ) -> Result<Vec<ClusterLayer>, RetrieveError> {
        let hierarchy = self
            .hierarchy
            .as_ref()
            .ok_or_else(|| RetrieveError::Other("Hierarchy not built".to_string()))?;

        // Extract layers at different distance thresholds
        let mut layers = Vec::new();
        let thresholds = self.compute_thresholds(hierarchy)?;

        for threshold in thresholds {
            let layer = hierarchy.extract_layer(threshold, self.params.min_cluster_size)?;
            layers.push(layer);
        }

        // Sort by granularity (finest first)
        layers.sort_by(|a, b| b.num_clusters.cmp(&a.num_clusters));

        Ok(layers)
    }

    /// Compute distance thresholds for layer extraction.
    fn compute_thresholds(&self, hierarchy: &ClusterHierarchy) -> Result<Vec<f32>, RetrieveError> {
        // Extract thresholds from hierarchy nodes
        let mut thresholds = hierarchy.get_all_distances();
        thresholds.sort_by(|a, b| a.total_cmp(b));

        // Select representative thresholds (every Nth)
        let step = (thresholds.len() / 10).max(1);
        let selected: Vec<f32> = thresholds.iter().step_by(step).copied().collect();

        Ok(selected)
    }

    /// Detect near-duplicate vectors.
    fn detect_duplicates(
        &self,
        reduced_vectors: &[f32],
        num_vectors: usize,
    ) -> Result<Vec<Vec<usize>>, RetrieveError> {
        const DUPLICATE_THRESHOLD: f32 = 1e-6;

        let mut duplicates = Vec::new();
        let mut processed = HashSet::new();

        for i in 0..num_vectors {
            if processed.contains(&i) {
                continue;
            }

            let vec_i = self.get_reduced_vector(reduced_vectors, i);
            let mut group = vec![i];

            for j in (i + 1)..num_vectors {
                if processed.contains(&j) {
                    continue;
                }

                let vec_j = self.get_reduced_vector(reduced_vectors, j);
                let dist = self.euclidean_distance(vec_i, vec_j);

                if dist < DUPLICATE_THRESHOLD {
                    group.push(j);
                    processed.insert(j);
                }
            }

            if group.len() > 1 {
                duplicates.push(group);
                processed.insert(i);
            }
        }

        Ok(duplicates)
    }

    /// Get reduced vector from SoA storage.
    fn get_reduced_vector<'a>(&self, vectors: &'a [f32], idx: usize) -> &'a [f32] {
        let start = idx * self.params.intermediate_dim;
        let end = start + self.params.intermediate_dim;
        &vectors[start..end]
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

    /// Get cluster layers (multi-granularity).
    pub fn cluster_layers(&self) -> &[ClusterLayer] {
        &self.cluster_layers
    }

    /// Get cluster hierarchy tree.
    pub fn cluster_tree(&self) -> Option<&ClusterHierarchy> {
        self.hierarchy.as_ref()
    }

    /// Get potential duplicate groups.
    pub fn duplicates(&self) -> &[Vec<usize>] {
        &self.duplicates
    }
}

/// Union-Find data structure for MST construction.
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    fn union(&mut self, x: usize, y: usize) {
        let root_x = self.find(x);
        let root_y = self.find(y);

        if root_x == root_y {
            return;
        }

        if self.rank[root_x] < self.rank[root_y] {
            self.parent[root_x] = root_y;
        } else if self.rank[root_x] > self.rank[root_y] {
            self.parent[root_y] = root_x;
        } else {
            self.parent[root_y] = root_x;
            self.rank[root_x] += 1;
        }
    }
}
