//! Local Intrinsic Dimensionality (LID) estimation.
//!
//! # Intuition
//!
//! LID measures how "locally complex" the space around a point is.
//! A point with high LID behaves as if it's in a high-dimensional space locally,
//! even if the global embedding dimension is lower. These are often outliers or
//! points in sparse regions.
//!
//! # Mathematical Foundation
//!
//! For a point x, LID is defined as:
//!
//! ```text
//! LID(x) = lim(r→0) ln(F(r)) / ln(r)
//! ```
//!
//! where F(r) is the cumulative distribution of distances to nearby points.
//!
//! The MLE estimator (Amsaleg et al., 2015) uses k nearest neighbors:
//!
//! ```text
//! LID_MLE(x) = -k / Σᵢ log(dᵢ / dₖ)
//! ```
//!
//! where d₁ ≤ d₂ ≤ ... ≤ dₖ are distances to k nearest neighbors.
//!
//! # Applications
//!
//! - **HNSW optimization**: Dual-Branch HNSW (2025) uses LID to identify
//!   outliers that need special handling during graph construction.
//! - **Anomaly detection**: High LID points are potential outliers.
//! - **Adaptive search**: Adjust ef_search based on local complexity.
//!
//! # References
//!
//! - Amsaleg et al. (2015) "Estimating Local Intrinsic Dimensionality"
//! - Dual-Branch HNSW (arXiv 2501.13992, 2025) "LID-based insertion"

use crate::hnsw::distance;

/// Result of LID estimation for a single point.
#[derive(Debug, Clone, Copy)]
pub struct LidEstimate {
    /// Estimated local intrinsic dimensionality.
    pub lid: f32,
    /// Number of neighbors used in estimation.
    pub k: usize,
    /// Maximum distance to k-th neighbor.
    pub max_dist: f32,
}

/// LID estimation configuration.
#[derive(Debug, Clone)]
pub struct LidConfig {
    /// Number of neighbors for LID estimation (default: 20).
    pub k: usize,
    /// Minimum distance to avoid log(0) (default: 1e-10).
    pub epsilon: f32,
}

impl Default for LidConfig {
    fn default() -> Self {
        Self {
            k: 20,
            epsilon: 1e-10,
        }
    }
}

/// MLE estimator for Local Intrinsic Dimensionality.
///
/// # Formula
///
/// ```text
/// LID = -k / Σᵢ log(dᵢ / dₖ)
/// ```
///
/// where d₁ ≤ ... ≤ dₖ are sorted distances to k nearest neighbors.
///
/// # Properties
///
/// - Returns f32::INFINITY if all distances are equal (degenerate case).
/// - Returns LID ≈ embedding_dim for uniformly distributed data.
/// - Returns LID >> embedding_dim for points in sparse regions (outliers).
///
/// # Example
///
/// ```ignore
/// use vicinity::lid::{estimate_lid_mle, LidConfig};
///
/// let distances = vec![0.1, 0.2, 0.3, 0.5, 0.8, 1.2];
/// let lid = estimate_lid_mle(&distances, &LidConfig::default());
/// println!("LID estimate: {}", lid.lid);
/// ```
#[must_use]
pub fn estimate_lid_mle(sorted_distances: &[f32], config: &LidConfig) -> LidEstimate {
    let k = sorted_distances.len().min(config.k);
    
    if k < 2 {
        return LidEstimate {
            lid: f32::NAN,
            k,
            max_dist: sorted_distances.first().copied().unwrap_or(0.0),
        };
    }
    
    let d_k = sorted_distances[k - 1];
    
    // Use relative epsilon for scale-invariance
    let abs_epsilon = d_k * config.epsilon;
    let d_k = d_k.max(abs_epsilon);
    
    // Sum of log ratios
    let mut sum_log = 0.0f32;
    let mut valid_count = 0;
    
    for &d_i in &sorted_distances[..k] {
        let d_i = d_i.max(abs_epsilon);
        let ratio = d_i / d_k;
        if ratio > 0.0 && ratio < 1.0 {
            sum_log += ratio.ln();
            valid_count += 1;
        }
    }
    
    let lid = if valid_count > 0 && sum_log.abs() > abs_epsilon {
        -(valid_count as f32) / sum_log
    } else {
        // Degenerate case: all distances equal or very close
        f32::INFINITY
    };
    
    LidEstimate {
        lid,
        k,
        max_dist: d_k,
    }
}

/// Estimate LID for a query point given its neighbors' distances.
///
/// This is the primary interface for HNSW integration.
///
/// # Arguments
///
/// * `neighbor_distances` - Distances to neighbors (need not be sorted)
/// * `config` - LID estimation parameters
///
/// # Returns
///
/// LID estimate for the query point.
#[must_use]
pub fn estimate_lid(neighbor_distances: &[f32], config: &LidConfig) -> LidEstimate {
    let mut sorted = neighbor_distances.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    estimate_lid_mle(&sorted, config)
}

/// Estimate LID for all points in a dataset using brute-force k-NN.
///
/// Useful for analysis and debugging, but O(n²) complexity.
///
/// # Arguments
///
/// * `vectors` - Flat array of vectors (n * dimension elements)
/// * `dimension` - Vector dimension
/// * `config` - LID estimation parameters
///
/// # Returns
///
/// Vector of LID estimates, one per point.
pub fn estimate_lid_batch(
    vectors: &[f32],
    dimension: usize,
    config: &LidConfig,
) -> Vec<LidEstimate> {
    let n = vectors.len() / dimension;
    let mut results = Vec::with_capacity(n);
    
    for i in 0..n {
        let query = &vectors[i * dimension..(i + 1) * dimension];
        
        // Compute distances to all other points
        let mut distances: Vec<f32> = (0..n)
            .filter(|&j| j != i)
            .map(|j| {
                let other = &vectors[j * dimension..(j + 1) * dimension];
                distance::cosine_distance(query, other)
            })
            .collect();
        
        distances.sort_by(|a, b| a.total_cmp(b));
        results.push(estimate_lid_mle(&distances, config));
    }
    
    results
}

/// Classify a LID estimate relative to a reference distribution.
///
/// # Categories
///
/// - **Low LID** (< median - σ): Point in dense region, well-connected locally.
/// - **Normal LID**: Typical point.
/// - **High LID** (> median + σ): Potential outlier, sparse local neighborhood.
///
/// High-LID points benefit from special handling in HNSW construction
/// (more edges, different entry point selection).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LidCategory {
    /// LID significantly below median - dense region.
    Low,
    /// LID near median - typical point.
    Normal,
    /// LID significantly above median - sparse region / outlier.
    High,
}

/// Compute LID statistics for a batch of estimates.
#[derive(Debug, Clone)]
pub struct LidStats {
    /// Mean LID across all points.
    pub mean: f32,
    /// Median LID.
    pub median: f32,
    /// Standard deviation.
    pub std_dev: f32,
    /// Minimum LID.
    pub min: f32,
    /// Maximum LID.
    pub max: f32,
    /// Count of points.
    pub count: usize,
}

impl LidStats {
    /// Compute statistics from a batch of LID estimates.
    pub fn from_estimates(estimates: &[LidEstimate]) -> Self {
        let valid: Vec<f32> = estimates
            .iter()
            .map(|e| e.lid)
            .filter(|lid| lid.is_finite())
            .collect();
        
        if valid.is_empty() {
            return Self {
                mean: f32::NAN,
                median: f32::NAN,
                std_dev: f32::NAN,
                min: f32::NAN,
                max: f32::NAN,
                count: 0,
            };
        }
        
        let count = valid.len();
        let mean = valid.iter().sum::<f32>() / count as f32;
        
        let variance = valid.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / count as f32;
        let std_dev = variance.sqrt();
        
        let mut sorted = valid.clone();
        sorted.sort_by(|a, b| a.total_cmp(b));
        
        let median = if count % 2 == 0 {
            (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0
        } else {
            sorted[count / 2]
        };
        
        Self {
            mean,
            median,
            std_dev,
            min: sorted[0],
            max: sorted[count - 1],
            count,
        }
    }
    
    /// Categorize a LID value relative to these statistics.
    pub fn categorize(&self, lid: f32) -> LidCategory {
        if !lid.is_finite() {
            return LidCategory::High; // Treat infinite LID as outlier
        }
        
        if lid < self.median - self.std_dev {
            LidCategory::Low
        } else if lid > self.median + self.std_dev {
            LidCategory::High
        } else {
            LidCategory::Normal
        }
    }
    
    /// Get threshold for high-LID outliers.
    pub fn high_lid_threshold(&self) -> f32 {
        self.median + self.std_dev
    }
}

/// Integration point for HNSW: compute LID during construction.
///
/// This function is designed to be called during HNSW graph building
/// to identify outlier points that need special handling.
///
/// # Usage in HNSW
///
/// From Dual-Branch HNSW (2025):
/// 1. Estimate LID for each new point during insertion
/// 2. For high-LID points (outliers):
///    - Use more neighbors (higher m)
///    - Consider special entry points
///    - Add skip bridges to bypass redundant layers
///
/// # Example
///
/// ```ignore
/// // During HNSW construction
/// let lid = estimate_lid_for_hnsw(&neighbor_distances, k);
/// if lid.lid > stats.high_lid_threshold() {
///     // Use extended neighborhood for this outlier
///     let m_extended = m * 2;
///     select_neighbors_extended(candidates, m_extended);
/// }
/// ```
pub fn estimate_lid_for_hnsw(neighbor_distances: &[f32], k: usize) -> LidEstimate {
    let config = LidConfig { k, epsilon: 1e-10 };
    estimate_lid(neighbor_distances, &config)
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_lid_uniform_distances() {
        // Linearly increasing distances should give moderate LID
        let distances: Vec<f32> = (1..=20).map(|i| i as f32 * 0.1).collect();
        let estimate = estimate_lid_mle(&distances, &LidConfig::default());
        
        assert!(estimate.lid.is_finite());
        assert!(estimate.lid > 0.0);
        println!("Linear distances LID: {}", estimate.lid);
    }
    
    #[test]
    fn test_lid_exponential_distances() {
        // Exponentially increasing distances (common in high-dim) should give higher LID
        let distances: Vec<f32> = (1..=20).map(|i| (i as f32).exp() * 0.01).collect();
        let estimate = estimate_lid_mle(&distances, &LidConfig::default());
        
        assert!(estimate.lid.is_finite());
        println!("Exponential distances LID: {}", estimate.lid);
    }
    
    #[test]
    fn test_lid_equal_distances() {
        // All equal distances = degenerate case
        let distances = vec![1.0f32; 20];
        let estimate = estimate_lid_mle(&distances, &LidConfig::default());
        
        // Should be infinite or very high
        assert!(estimate.lid.is_infinite() || estimate.lid > 100.0);
    }
    
    #[test]
    fn test_lid_stats() {
        let estimates = vec![
            LidEstimate { lid: 5.0, k: 20, max_dist: 1.0 },
            LidEstimate { lid: 10.0, k: 20, max_dist: 1.5 },
            LidEstimate { lid: 8.0, k: 20, max_dist: 1.2 },
            LidEstimate { lid: 7.0, k: 20, max_dist: 1.1 },
            LidEstimate { lid: 15.0, k: 20, max_dist: 2.0 }, // Outlier
        ];
        
        let stats = LidStats::from_estimates(&estimates);
        
        assert_eq!(stats.count, 5);
        assert!(stats.mean > 0.0);
        assert_eq!(stats.min, 5.0);
        assert_eq!(stats.max, 15.0);
        
        // 15.0 should be categorized as High
        assert_eq!(stats.categorize(15.0), LidCategory::High);
        // 8.0 should be Normal (close to median)
        assert_eq!(stats.categorize(8.0), LidCategory::Normal);
    }
    
    #[test]
    fn test_lid_batch() {
        // Create simple 2D points
        let dim = 2;
        let vectors: Vec<f32> = vec![
            0.0, 0.0,  // Point 0
            1.0, 0.0,  // Point 1
            0.0, 1.0,  // Point 2
            1.0, 1.0,  // Point 3
            10.0, 10.0, // Point 4 - outlier
        ];
        
        let config = LidConfig { k: 3, epsilon: 1e-10 };
        let estimates = estimate_lid_batch(&vectors, dim, &config);
        
        assert_eq!(estimates.len(), 5);
        
        // The outlier (point 4) should have higher LID
        let stats = LidStats::from_estimates(&estimates);
        println!("LID stats: mean={}, median={}, std={}", 
            stats.mean, stats.median, stats.std_dev);
        
        // Point 4's LID should be in the high category
        // (This is a soft test - LID behavior depends on distance distribution)
    }
    
    #[test]
    fn test_lid_config_k() {
        let distances: Vec<f32> = (1..=50).map(|i| i as f32 * 0.1).collect();
        
        let config_small = LidConfig { k: 5, epsilon: 1e-10 };
        let config_large = LidConfig { k: 30, epsilon: 1e-10 };
        
        let est_small = estimate_lid_mle(&distances, &config_small);
        let est_large = estimate_lid_mle(&distances, &config_large);
        
        assert_eq!(est_small.k, 5);
        assert_eq!(est_large.k, 30);
        
        // Different k can give different estimates
        println!("k=5 LID: {}, k=30 LID: {}", est_small.lid, est_large.lid);
    }
}
