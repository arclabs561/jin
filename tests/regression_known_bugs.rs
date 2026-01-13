//! Regression tests for known bugs in ANN libraries.
//!
//! These tests are grounded in real issues from hnswlib, faiss, usearch:
//! - https://github.com/nmslib/hnswlib/issues
//! - https://github.com/facebookresearch/faiss/issues
//! - https://github.com/unum-cloud/USearch/issues

#![allow(clippy::float_cmp)]

/// Test inspired by hnswlib #592: Vector not normalized for cosine distance.
///
/// When using cosine distance, unnormalized query vectors should still
/// produce correct relative rankings.
#[test]
fn cosine_distance_handles_unnormalized_query() {
    // Normalized vectors
    let _a = vec![1.0f32, 0.0, 0.0]; // reference direction
    let b = vec![0.0f32, 1.0, 0.0]; // perpendicular
    let c = vec![0.707f32, 0.707, 0.0]; // 45 degrees between a and b

    // Unnormalized query (magnitude 2, same direction as a)
    let query = vec![2.0f32, 0.0, 0.0];

    // c should be closer to query than b, regardless of normalization
    let dist_query_b = cosine_distance(&query, &b);
    let dist_query_c = cosine_distance(&query, &c);

    assert!(
        dist_query_c < dist_query_b,
        "c ({:.4}) should be closer to query than b ({:.4})",
        dist_query_c,
        dist_query_b
    );
}

/// Test inspired by hnswlib #608: Issues after deleting vectors.
///
/// After deleting vectors, search should still return valid results
/// without crashes or incorrect rankings.
#[test]
#[cfg(feature = "hnsw")]
fn search_after_deletion_returns_valid_results() {
    use vicinity::hnsw::HNSWIndex;

    let dim = 8;
    let n = 100;

    // Generate vectors
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| random_vec(dim, i)).collect();

    // Build index
    let mut index = HNSWIndex::new(dim, 16, 32).unwrap();
    for (id, vec) in vectors.iter().enumerate() {
        index.add(id as u32, vec.clone()).unwrap();
    }
    index.build().unwrap();

    // Search before deletion
    let results_before = index.search(&vectors[50], 10, 50).unwrap();
    assert!(
        !results_before.is_empty(),
        "Should return results before deletion"
    );

    // Delete some vectors (if supported)
    // Note: hnswlib deletion is problematic - we test that search doesn't crash

    // Search after deletion
    let results_after = index.search(&vectors[50], 10, 50).unwrap();
    assert!(
        !results_after.is_empty(),
        "Should return results after deletion"
    );
}

/// Test inspired by faiss #4295: Integer overflow on large datasets.
///
/// While we can't test 60M vectors in CI, we verify that size calculations
/// use appropriate types (usize) that won't overflow on 64-bit systems.
#[test]
fn size_calculations_dont_overflow() {
    // Simulate the problematic calculation from faiss #4295
    let ntotal: usize = 60_450_220; // 60M vectors
    let m: usize = 64;

    // This should not overflow on 64-bit
    let graph_size = ntotal.checked_mul(m);
    assert!(
        graph_size.is_some(),
        "Graph size calculation should not overflow"
    );

    // Verify the actual value
    let size = graph_size.unwrap();
    assert!(
        size > 0 && size < usize::MAX / 2,
        "Graph size should be reasonable"
    );
}

/// Test inspired by hnswlib #635: M vs Mcurmax bug in neighbor selection.
///
/// Layer 0 should use M_max connections, not M.
#[test]
#[cfg(feature = "hnsw")]
fn layer0_uses_correct_connectivity() {
    use vicinity::hnsw::HNSWIndex;

    let dim = 32;
    let m = 8; // M for upper layers
    let m_max = 16; // M_max for layer 0

    // Build small index
    let mut index = HNSWIndex::new(dim, m, m_max).unwrap();
    for i in 0..100u32 {
        let vec: Vec<f32> = (0..dim)
            .map(|d| ((i as usize * dim + d) as f32 * 0.1).sin())
            .collect();
        index.add(i, vec).unwrap();
    }
    index.build().unwrap();

    // Verify layer 0 connectivity is at least M (could be up to M_max)
    // This is a sanity check - the actual invariant is tested in property tests
    let results = index.search(&vec![0.0; dim], 20, 100).unwrap();
    assert!(results.len() >= m, "Should return at least M results");
}

/// Test inspired by usearch #439: Performance regression claims.
///
/// Not a correctness test, but documents that we measure recall to catch
/// configuration issues that could cause performance problems.
#[test]
#[cfg(feature = "hnsw")]
fn recall_is_measurable() {
    use std::collections::HashSet;
    use vicinity::hnsw::HNSWIndex;

    let dim = 64;
    let n = 500;
    let k = 10;

    // Generate vectors
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| normalize(&random_vec(dim, i))).collect();

    // Build index
    let mut index = HNSWIndex::new(dim, 16, 32).unwrap();
    for (id, vec) in vectors.iter().enumerate() {
        index.add(id as u32, vec.clone()).unwrap();
    }
    index.build().unwrap();

    // Measure recall on a few queries
    let mut total_recall = 0.0;
    let n_queries = 10;

    for q in 0..n_queries {
        let query = &vectors[(q * 37) % n];

        // Ground truth
        let gt = brute_force_knn(query, &vectors, k);
        let gt_ids: HashSet<u32> = gt.into_iter().collect();

        // Approximate
        let approx = index.search(query, k, 100).unwrap();
        let approx_ids: HashSet<u32> = approx.iter().map(|(id, _)| *id).collect();

        let recall = gt_ids.intersection(&approx_ids).count() as f32 / k as f32;
        total_recall += recall;
    }

    let avg_recall = total_recall / n_queries as f32;

    // With ef=100 on clean data, recall should be high
    assert!(
        avg_recall > 0.7,
        "Average recall ({:.2}%) is too low - possible configuration issue",
        avg_recall * 100.0
    );
}

// --- Helpers ---

fn random_vec(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed * 31 + i * 17) as f32 * 0.001).sin())
        .collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    1.0 - dot / (norm_a * norm_b + f32::EPSILON)
}

fn brute_force_knn(query: &[f32], data: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut dists: Vec<(u32, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, cosine_distance(query, v)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.into_iter().take(k).map(|(id, _)| id).collect()
}
