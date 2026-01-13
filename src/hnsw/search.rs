//! HNSW search algorithm with early termination optimizations.

use std::collections::{BinaryHeap, HashSet};

#[cfg(feature = "hnsw")]
mod distance_impl {
    use crate::hnsw::distance;
    pub use distance::cosine_distance;
}

#[cfg(feature = "hnsw")]
use distance_impl::cosine_distance;

/// Candidate node during search.
#[derive(Clone, PartialEq)]
pub(crate) struct Candidate {
    pub(crate) id: u32,
    pub(crate) distance: f32,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: smaller distance = higher priority
        // Use total_cmp for IEEE 754 total ordering (NaN-safe)
        self.distance.total_cmp(&other.distance).reverse()
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Search state for HNSW search algorithm.
pub(crate) struct SearchState {
    /// Candidate queue (min-heap by distance)
    candidates: BinaryHeap<Candidate>,

    /// Visited nodes (to avoid revisiting)
    visited: HashSet<u32>,

    /// Best distance found so far
    best_distance: f32,

    /// Number of iterations without improvement (for early termination)
    no_improvement_count: usize,
}

impl SearchState {
    #[allow(dead_code)]
    fn new() -> Self {
        Self {
            candidates: BinaryHeap::new(),
            visited: HashSet::new(),
            best_distance: f32::INFINITY,
            no_improvement_count: 0,
        }
    }

    /// Create with pre-allocated capacity for better performance.
    pub(crate) fn with_capacity(ef: usize) -> Self {
        Self {
            candidates: BinaryHeap::with_capacity(ef * 2), // Pre-allocate for ef candidates
            visited: HashSet::with_capacity(ef * 2),       // Pre-allocate visited set
            best_distance: f32::INFINITY,
            no_improvement_count: 0,
        }
    }

    pub(crate) fn add_candidate(&mut self, id: u32, distance: f32) {
        if !self.visited.contains(&id) {
            self.candidates.push(Candidate { id, distance });
        }
    }

    pub(crate) fn pop_candidate(&mut self) -> Option<Candidate> {
        while let Some(candidate) = self.candidates.pop() {
            if !self.visited.contains(&candidate.id) {
                self.visited.insert(candidate.id);
                if candidate.distance < self.best_distance {
                    self.best_distance = candidate.distance;
                    self.no_improvement_count = 0;
                } else {
                    self.no_improvement_count += 1;
                }
                return Some(candidate);
            }
        }
        None
    }
}

/// Greedy search in a single layer.
///
/// Finds the closest node to query in the current layer.
#[cfg(feature = "hnsw")]
pub fn greedy_search_layer(
    query: &[f32],
    entry_point: u32,
    layer: &crate::hnsw::graph::Layer,
    vectors: &[f32],
    dimension: usize,
    ef: usize,
) -> Vec<(u32, f32)> {
    let mut state = SearchState::with_capacity(ef);

    // Start from entry point
    let entry_vector = get_vector(vectors, dimension, entry_point as usize);
    let entry_distance = cosine_distance(query, entry_vector);
    state.add_candidate(entry_point, entry_distance);

    // Greedy search: always explore closest unvisited node
    // Pre-allocate results vector with capacity for ef results
    let mut results = Vec::with_capacity(ef);

    while let Some(candidate) = state.pop_candidate() {
        results.push((candidate.id, candidate.distance));

        if results.len() >= ef {
            break;
        }

        // Explore neighbors
        let neighbors = layer.get_neighbors(candidate.id);
        for &neighbor_id in neighbors.iter() {
            let neighbor_vector = get_vector(vectors, dimension, neighbor_id as usize);
            let neighbor_distance = cosine_distance(query, neighbor_vector);
            state.add_candidate(neighbor_id, neighbor_distance);
        }
    }

    results
}

/// Get vector from SoA storage.
fn get_vector(vectors: &[f32], dimension: usize, idx: usize) -> &[f32] {
    let start = idx * dimension;
    let end = start + dimension;
    &vectors[start..end]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candidate_ordering() {
        let mut heap = BinaryHeap::new();
        heap.push(Candidate {
            id: 0,
            distance: 0.5,
        });
        heap.push(Candidate {
            id: 1,
            distance: 0.1,
        });
        heap.push(Candidate {
            id: 2,
            distance: 0.3,
        });

        // Should pop in order: 0.1, 0.3, 0.5 (min-heap)
        assert_eq!(heap.pop().unwrap().distance, 0.1);
        assert_eq!(heap.pop().unwrap().distance, 0.3);
        assert_eq!(heap.pop().unwrap().distance, 0.5);
    }
}
