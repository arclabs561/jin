//! Distance computation optimized with SIMD.

#![allow(dead_code)]

use crate::simd;

/// Compute cosine distance between two vectors.
///
/// For L2-normalized vectors, cosine distance = 1 - dot product.
/// Uses SIMD-accelerated dot product from existing `simd` module.
///
/// # Arguments
///
/// * `a` - First vector (should be L2-normalized)
/// * `b` - Second vector (should be L2-normalized)
///
/// # Returns
///
/// Cosine distance in [0, 2] range (0 = identical, 2 = opposite)
#[inline]
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    // For normalized vectors: cosine similarity = dot product
    // Distance = 1 - similarity (for normalized vectors)
    let similarity = simd::dot(a, b);
    1.0 - similarity
}

/// Compute L2 (Euclidean) distance between two vectors.
///
/// Uses SIMD-accelerated operations where possible.
/// Optimized to avoid temporary allocations by computing squared distance directly.
#[inline]
pub fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }

    // Compute squared L2 distance: sum((a[i] - b[i])^2)
    // Optimized: compute directly without temporary allocation
    // For SIMD, we can use: sum((a[i] - b[i])^2) = sum(a[i]^2) + sum(b[i]^2) - 2*sum(a[i]*b[i])
    // But simpler: compute difference squared directly with SIMD
    l2_distance_squared(a, b).sqrt()
}

/// Compute squared L2 distance (avoids sqrt for comparisons).
///
/// Uses SIMD-accelerated operations. More efficient than l2_distance when
/// only comparing distances (no need for sqrt).
#[inline]
pub fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }

    // Compute sum((a[i] - b[i])^2) = sum(a[i]^2) + sum(b[i]^2) - 2*sum(a[i]*b[i])
    // This avoids creating a temporary difference vector
    let a_squared = simd::dot(a, a);
    let b_squared = simd::dot(b, b);
    let ab_dot = simd::dot(a, b);
    a_squared + b_squared - 2.0 * ab_dot
}

/// Compute inner product distance (for MIPS - Maximum Inner Product Search).
///
/// For inner product, distance = -inner_product (since we want to minimize distance,
/// which corresponds to maximizing inner product).
#[inline]
pub fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    // Use SIMD-accelerated dot product
    -simd::dot(a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_distance() {
        let a = [1.0, 0.0];
        let b = [1.0, 0.0];
        assert!((cosine_distance(&a, &b) - 0.0).abs() < 1e-5);

        let a = [1.0, 0.0];
        let b = [0.0, 1.0];
        assert!((cosine_distance(&a, &b) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_distance() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((l2_distance(&a, &b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_distance_squared() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        // Squared distance should be 25 (3² + 4² = 9 + 16 = 25)
        assert!((l2_distance_squared(&a, &b) - 25.0).abs() < 1e-5);

        // Verify relationship: distance² = l2_distance_squared
        let a2 = [1.0, 2.0];
        let b2 = [4.0, 6.0];
        let dist = l2_distance(&a2, &b2);
        let dist_sq = l2_distance_squared(&a2, &b2);
        assert!((dist * dist - dist_sq).abs() < 1e-5);
    }
}
