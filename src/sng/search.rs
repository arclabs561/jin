//! OPT-SNG search algorithm.

use crate::sng::graph::SNGIndex;
use crate::simd;
use crate::RetrieveError;
use std::collections::{BinaryHeap, HashSet};

/// Candidate during search.
#[derive(Clone, PartialEq)]
struct SearchCandidate {
    id: u32,
    distance: f32,
}

impl Eq for SearchCandidate {}

impl PartialOrd for SearchCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        // Min-heap: smaller distance = higher priority
        other.distance.partial_cmp(&self.distance)
    }
}

impl Ord for SearchCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Use total_cmp for IEEE 754 total ordering (NaN-safe)
        self.distance.total_cmp(&other.distance).reverse()
    }
}

/// Search OPT-SNG graph for k nearest neighbors.
///
/// Uses greedy search with early termination, leveraging the theoretical
/// guarantee of O(log n) search path length.
pub fn search_sng(
    index: &SNGIndex,
    query: &[f32],
    k: usize,
) -> Result<Vec<(u32, f32)>, RetrieveError> {
    if index.num_vectors == 0 {
        return Ok(Vec::new());
    }

    // Start from random entry point (or first vector)
    let mut current = 0u32;
    let mut current_dist = f32::INFINITY;

    // Find good starting point
    for i in 0..index.num_vectors.min(100) {
        let vec = index.get_vector(i);
        let dist = 1.0 - simd::dot(query, vec);
        if dist < current_dist {
            current_dist = dist;
            current = i as u32;
        }
    }

    // Greedy search with early termination
    let mut candidates = BinaryHeap::new();
    let mut visited = HashSet::new();
    let mut results = Vec::new();

    candidates.push(SearchCandidate {
        id: current,
        distance: current_dist,
    });

    // Search with O(log n) guarantee
    let max_iterations = (index.num_vectors as f32).ln().ceil() as usize * 10;
    let mut iterations = 0;

    while let Some(candidate) = candidates.pop() {
        if visited.contains(&candidate.id) {
            continue;
        }

        visited.insert(candidate.id);
        results.push((candidate.id, candidate.distance));

        if results.len() >= k {
            break;
        }

        if iterations >= max_iterations {
            break;
        }
        iterations += 1;

        // Explore neighbors
        if let Some(neighbors) = index.neighbors.get(candidate.id as usize) {
            for &neighbor_id in neighbors.iter() {
                if visited.contains(&neighbor_id) {
                    continue;
                }

                let neighbor_vec = index.get_vector(neighbor_id as usize);
                let dist = 1.0 - simd::dot(query, neighbor_vec);
                candidates.push(SearchCandidate {
                    id: neighbor_id,
                    distance: dist,
                });
            }
        }
    }

    // Sort by distance and return top k
    results.sort_unstable_by(|a, b| a.1.total_cmp(&b.1)); // Unstable for better performance
    Ok(results.into_iter().take(k).collect())
}
