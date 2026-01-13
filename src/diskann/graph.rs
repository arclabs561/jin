//! DiskANN graph structure and Vamana construction.

use crate::RetrieveError;
use rand::seq::SliceRandom;
use rand::Rng;
use smallvec::SmallVec;
use std::collections::HashSet;

/// DiskANN index for disk-based approximate nearest neighbor search.
///
/// Implements the Vamana graph construction algorithm:
/// 1. Random graph initialization
/// 2. Two-pass construction (alpha=1.0, then alpha>1.0)
/// 3. Robust pruning (alpha-pruning) to maintain long-range edges
pub struct DiskANNIndex {
    dimension: usize,
    params: DiskANNParams,
    built: bool,

    // Vectors stored in memory for build (would be on disk in prod)
    vectors: Vec<f32>,
    num_vectors: usize,

    // Graph structure (adjacency list)
    // Using SmallVec to optimize for typical degree M=16-32
    // Stored in memory for construction, serialized to disk later
    adj: Vec<SmallVec<[u32; 32]>>,

    // Entry point for search (medoid)
    start_node: u32,
}

/// DiskANN parameters.
#[derive(Clone, Debug)]
pub struct DiskANNParams {
    /// Maximum connections per node (R in paper)
    pub m: usize,

    /// Beam width for construction search (L in paper)
    pub ef_construction: usize,

    /// Alpha parameter for pruning (typically 1.2 - 1.4)
    pub alpha: f32,

    /// Search width
    pub ef_search: usize,
}

impl Default for DiskANNParams {
    fn default() -> Self {
        Self {
            m: 32,
            ef_construction: 100,
            alpha: 1.2,
            ef_search: 100,
        }
    }
}

/// Candidate for priority queues
#[derive(Clone, Copy, PartialEq)]
struct Candidate {
    id: u32,
    dist: f32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Max-heap: larger distance = higher priority (for results pruning)
        // Use total_cmp for IEEE 754 total ordering (NaN-safe, NaN > all)
        self.dist.total_cmp(&other.dist)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl DiskANNIndex {
    /// Create a new DiskANN index.
    pub fn new(dimension: usize, params: DiskANNParams) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::EmptyQuery);
        }

        Ok(Self {
            dimension,
            params,
            built: false,
            vectors: Vec::new(),
            num_vectors: 0,
            adj: Vec::new(),
            start_node: 0,
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

        self.vectors.extend(vector);
        self.num_vectors += 1;
        self.adj.push(SmallVec::new());
        Ok(())
    }

    /// Build the index using Vamana construction.
    pub fn build(&mut self) -> Result<(), RetrieveError> {
        if self.built {
            return Ok(());
        }

        if self.num_vectors == 0 {
            return Err(RetrieveError::EmptyIndex);
        }

        // 1. Initialize random graph (R-regular)
        self.initialize_random_graph();

        // 2. Compute medoid as start node
        self.start_node = self.compute_medoid();

        // 3. First pass: alpha = 1.0 (approximates RNG)
        // Helps build initial connectivity
        self.vamana_pass(1.0)?;

        // 4. Second pass: alpha = params.alpha (e.g. 1.2)
        // Adds long-range edges for small-world navigation
        self.vamana_pass(self.params.alpha)?;

        self.built = true;
        Ok(())
    }

    /// Initialize random R-regular graph.
    fn initialize_random_graph(&mut self) {
        let mut rng = rand::rng();
        let r = self.params.m;

        for i in 0..self.num_vectors {
            // Pick R random neighbors
            let mut neighbors: HashSet<u32> = HashSet::with_capacity(r);
            while neighbors.len() < r && neighbors.len() < self.num_vectors - 1 {
                let n = rng.random_range(0..self.num_vectors) as u32;
                if n != i as u32 {
                    neighbors.insert(n);
                }
            }
            self.adj[i] = neighbors.into_iter().collect();
        }
    }

    /// Compute geometric medoid of the dataset.
    fn compute_medoid(&self) -> u32 {
        // Approximate medoid by centroid of a sample
        // For simplicity in this implementation, just pick a random node if N is large,
        // or 0. A robust implementation would compute the true centroid.
        // Using 0 is a common valid simplification for prototype.
        0
    }

    /// Single pass of Vamana construction.
    fn vamana_pass(&mut self, alpha: f32) -> Result<(), RetrieveError> {
        // Random permutation of nodes
        let mut nodes: Vec<u32> = (0..self.num_vectors as u32).collect();
        nodes.shuffle(&mut rand::rng());

        for &i in &nodes {
            let query_vec = self.get_vector(i);

            // Greedy search to find candidates
            // We use the graph as it exists so far
            let (visited, _) =
                self.greedy_search(query_vec, self.params.ef_construction, self.start_node);

            // Candidate set V = visited nodes
            // Run RobustPrune on V to find new neighbors for i
            let new_neighbors = self.robust_prune(i, &visited, alpha, self.params.m);

            // Update graph: add directed edges
            self.adj[i as usize] = new_neighbors.into_iter().collect();

            // Note: In full DiskANN, we'd also add reverse edges to keep graph undirected/balanced,
            // but vanilla Vamana works well with directed edges refined this way.
            // For production, we'd enforce max degree on reverse updates.
        }

        Ok(())
    }

    /// RobustPrune (Alpha-Pruning) algorithm.
    ///
    /// Selects neighbors that are close to `node`, but also "orthogonal" to each other
    /// to ensure good coverage of the space.
    fn robust_prune(
        &self,
        node: u32,
        candidates: &[u32],
        alpha: f32,
        max_degree: usize,
    ) -> Vec<u32> {
        let node_vec = self.get_vector(node);

        // 1. Calculate distances to all candidates
        let mut candidates_with_dist: Vec<Candidate> = candidates
            .iter()
            .filter(|&&c| c != node) // distinct
            .map(|&c| Candidate {
                id: c,
                dist: self.dist(node_vec, self.get_vector(c)),
            })
            .collect();

        // Add current neighbors to candidate set (to refine them)
        for &neighbor in &self.adj[node as usize] {
            if !candidates.contains(&neighbor) {
                candidates_with_dist.push(Candidate {
                    id: neighbor,
                    dist: self.dist(node_vec, self.get_vector(neighbor)),
                });
            }
        }

        // 2. Sort by distance (ascending)
        candidates_with_dist.sort_by(|a, b| a.dist.total_cmp(&b.dist));

        // 3. Prune
        let mut new_neighbors: Vec<u32> = Vec::with_capacity(max_degree);

        // Remove duplicates if any
        candidates_with_dist.dedup_by(|a, b| a.id == b.id);

        for cand in candidates_with_dist {
            if new_neighbors.len() >= max_degree {
                break;
            }

            // Check if cand is reachable from any existing neighbor with shorter path
            // alpha parameter controls "shorter": distance(p*, p') <= alpha * distance(p, p')
            let mut prune = false;
            let cand_vec = self.get_vector(cand.id);

            for &existing_neighbor in &new_neighbors {
                let dist_existing_cand = self.dist(self.get_vector(existing_neighbor), cand_vec);

                // If existing neighbor is closer to candidate than node is (scaled by alpha),
                // then candidate is redundant (we can reach it via existing neighbor).
                if alpha * dist_existing_cand <= cand.dist {
                    prune = true;
                    break;
                }
            }

            if !prune {
                new_neighbors.push(cand.id);
            }
        }

        new_neighbors
    }

    /// Greedy search for construction and querying.
    ///
    /// Returns (visited_nodes, nearest_candidates).
    fn greedy_search(
        &self,
        query: &[f32],
        l_size: usize,
        start_node: u32,
    ) -> (Vec<u32>, Vec<Candidate>) {
        let mut visited = HashSet::new();
        // Note: We use retset Vec instead of BinaryHeap for simpler control over L closest

        // Use a max-heap for the working queue to easily pop the worst candidate
        // Wait, standard beam search keeps L closest.
        // Let's implement standard "iterate until convergence" greedy search.

        // Results set (L closest found so far) - sorted vector or binary heap
        // We'll use a vector and sort it, for simplicity in this proto.
        let mut retset: Vec<Candidate> = Vec::with_capacity(l_size + 1);

        let start_dist = self.dist(query, self.get_vector(start_node));
        retset.push(Candidate {
            id: start_node,
            dist: start_dist,
        });
        visited.insert(start_node);

        let mut current_idx = 0;

        while current_idx < retset.len() {
            // Find the closest unvisited node in retset
            // (In optimized impl, we iterate sorted retset)
            retset.sort_by(|a, b| a.dist.total_cmp(&b.dist));

            if current_idx >= retset.len() {
                break;
            }

            let current = retset[current_idx];
            current_idx += 1;

            // If closest unvisited is farther than our worst candidate (and list is full), stop?
            // Vamana doesn't strictly stop, it explores all neighbors.

            for &neighbor in &self.adj[current.id as usize] {
                if visited.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let dist = self.dist(query, self.get_vector(neighbor));

                // Add to retset
                retset.push(Candidate { id: neighbor, dist });
            }

            // Keep only top L
            retset.sort_by(|a, b| a.dist.total_cmp(&b.dist));
            if retset.len() > l_size {
                retset.truncate(l_size);
            }
        }

        let ids: Vec<u32> = retset.iter().map(|c| c.id).collect();
        (ids, retset)
    }

    /// Search for k nearest neighbors.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
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

        let ef = ef_search.max(k);
        let (_, candidates) = self.greedy_search(query, ef, self.start_node);

        // Return top k
        let result = candidates
            .into_iter()
            .take(k)
            .map(|c| (c.id, c.dist))
            .collect();

        Ok(result)
    }

    fn get_vector(&self, idx: u32) -> &[f32] {
        let start = idx as usize * self.dimension;
        &self.vectors[start..start + self.dimension]
    }

    // Euclidean distance (squared)
    fn dist(&self, a: &[f32], b: &[f32]) -> f32 {
        // In full impl, use SIMD from crate::simd
        a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
    }
}
