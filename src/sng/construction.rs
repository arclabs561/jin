//! OPT-SNG graph construction.

use crate::RetrieveError;
use crate::sng::graph::SNGIndex;
use crate::sng::martingale;
use crate::simd;
use smallvec::SmallVec;

/// Construct OPT-SNG graph using martingale-based model.
pub fn construct_sng_graph(
    index: &mut SNGIndex,
) -> Result<(), RetrieveError> {
    if index.num_vectors == 0 {
        return Err(RetrieveError::EmptyIndex);
    }
    
    // Initialize neighbor lists
    index.neighbors = vec![SmallVec::new(); index.num_vectors];
    
    // Build graph using martingale-based pruning
    let mut evolution = martingale::CandidateEvolution::new();
    
    for current_id in 0..index.num_vectors {
        let current_vector = index.get_vector(current_id);
        
        // Find candidates: all other vectors
        let mut candidates = Vec::new();
        for other_id in 0..index.num_vectors {
            if other_id == current_id {
                continue;
            }
            
            let other_vector = index.get_vector(other_id);
            let dist = 1.0 - simd::dot(current_vector, other_vector);
            candidates.push((other_id as u32, dist));
        }
        
        // Prune using martingale-based model
        let pruned = martingale::prune_candidates_martingale(
            &candidates,
            index.truncation_r,
            &index.vectors,
            index.dimension,
        )?;
        
        // Update evolution tracker
        evolution.update(pruned.len());
        
        // Add bidirectional connections
        for &neighbor_id in &pruned {
            // Add connection from current to neighbor
            index.neighbors[current_id].push(neighbor_id);
            
            // Add reverse connection (if not already present)
            let reverse_neighbors = &mut index.neighbors[neighbor_id as usize];
            if !reverse_neighbors.contains(&(current_id as u32)) {
                reverse_neighbors.push(current_id as u32);
            }
        }
    }
    
    Ok(())
}
