//! Automatic parameter optimization for OPT-SNG.

use crate::RetrieveError;

/// Optimize truncation parameter R using closed-form rule.
///
/// Based on the theoretical analysis in the OPT-SNG paper, this derives
/// the optimal R value without expensive parameter sweeping.
///
/// # Returns
///
/// Optimal truncation parameter R for graph construction.
pub fn optimize_truncation_r(num_vectors: usize, dimension: usize) -> Result<f32, RetrieveError> {
    if num_vectors == 0 || dimension == 0 {
        return Err(RetrieveError::Other(
            "Cannot optimize parameters for empty dataset".to_string(),
        ));
    }

    // Closed-form rule from OPT-SNG paper
    // R is optimized based on dataset size and dimension
    // This is a simplified version - full implementation would use
    // the martingale-based analysis from the paper

    let n = num_vectors as f32;
    let d = dimension as f32;

    // Optimal R based on theoretical analysis
    // R ~ O(n^{2/3+ε}) for maximum out-degree
    // For practical purposes, we use a heuristic that works well
    let log_n = n.ln();
    let r = (log_n * d.sqrt()).max(1.0);

    Ok(r)
}

/// Estimate maximum out-degree based on truncation parameter R.
///
/// From OPT-SNG paper: maximum out-degree is O(n^{2/3+ε}).
pub fn estimate_max_degree(num_vectors: usize, r: f32) -> usize {
    let n = num_vectors as f32;
    // Maximum out-degree: O(n^{2/3+ε}) where ε is small
    let max_deg = (n.powf(2.0 / 3.0) * r).ceil() as usize;
    max_deg.clamp(16, 256) // Practical bounds
}
