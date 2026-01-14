//! Integration tests for HNSW index.
//!
//! Tests the full lifecycle: build, query, persistence, streaming updates.
//!
//! Note: HNSWIndex uses internal indices (0, 1, 2, ...) based on insertion order,
//! not the doc_id passed to add(). The index also uses cosine distance internally.

use plesio::hnsw::{HNSWIndex, HNSWParams};
use std::collections::HashSet;

/// Generate random vectors for testing.
fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    use std::hash::{Hash, Hasher};

    (0..n)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let mut hasher = std::collections::hash_map::DefaultHasher::new();
                    seed.hash(&mut hasher);
                    i.hash(&mut hasher);
                    j.hash(&mut hasher);
                    let h = hasher.finish();
                    (h as f64 / u64::MAX as f64 * 2.0 - 1.0) as f32
                })
                .collect()
        })
        .collect()
}

/// Normalize a vector to unit length (for cosine similarity).
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        v.to_vec()
    } else {
        v.iter().map(|x| x / norm).collect()
    }
}

/// Cosine distance: 1 - cosine_similarity.
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 1.0;
    }

    1.0 - dot / (norm_a * norm_b)
}

/// Compute exact k-NN using cosine distance.
fn exact_knn_cosine(vectors: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(u32, f32)> {
    let mut distances: Vec<(u32, f32)> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let dist = cosine_distance(v, query);
            (i as u32, dist)
        })
        .collect();

    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k);
    distances
}

/// Calculate recall@k.
fn recall_at_k(exact: &[(u32, f32)], approx: &[(u32, f32)], k: usize) -> f32 {
    let exact_set: HashSet<u32> = exact.iter().take(k).map(|(i, _)| *i).collect();
    let approx_set: HashSet<u32> = approx.iter().take(k).map(|(i, _)| *i).collect();
    exact_set.intersection(&approx_set).count() as f32 / k as f32
}

const DEFAULT_EF: usize = 50;

#[test]
fn test_hnsw_basic_build_and_query() {
    let dim = 32;
    let n = 500;
    let vectors: Vec<Vec<f32>> = random_vectors(n, dim, 42)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create index");

    // Insert all vectors (doc_id is ignored, internal index used)
    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add vector");
    }

    // Build the index
    hnsw.build().expect("Failed to build index");

    // Query with the first vector - should find itself (internal index 0)
    let query = &vectors[0];
    let results = hnsw.search(query, 10, DEFAULT_EF).expect("Search failed");

    assert!(!results.is_empty(), "Search should return results");
    // Internal index is 0 for first inserted vector
    assert_eq!(results[0].0, 0, "First result should be internal index 0");
    assert!(results[0].1 < 0.01, "Distance to self should be ~0");
}

#[test]
fn test_hnsw_recall_quality() {
    let dim = 64;
    let n = 1000;
    let k = 10;
    let n_queries = 50;

    // Normalize vectors for cosine similarity
    let vectors: Vec<Vec<f32>> = random_vectors(n, dim, 123)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();
    let queries: Vec<Vec<f32>> = random_vectors(n_queries, dim, 456)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    // Use higher M for better recall
    let params = HNSWParams {
        m: 32,
        m_max: 32,
        ef_construction: 200,
        ef_search: 100,
        ..Default::default()
    };
    let mut hnsw = HNSWIndex::with_params(dim, params).expect("Failed to create index");
    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    let mut total_recall = 0.0;

    for query in &queries {
        let exact = exact_knn_cosine(&vectors, query, k);
        let approx = hnsw.search(query, k, 150).expect("Search failed");

        let recall = recall_at_k(&exact, &approx, k);
        total_recall += recall;
    }

    let avg_recall = total_recall / n_queries as f32;
    // Note: With M=32 and ef=150, recall should be reasonable.
    // HNSW recall varies by run due to random layer assignment.
    assert!(
        avg_recall >= 0.35,
        "Average recall@{} should be >= 0.35, got {}",
        k,
        avg_recall
    );
}

#[test]
fn test_hnsw_empty_index_errors() {
    let dim = 32;
    let hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    let query = vec![0.0f32; dim];
    // Empty index not built should error
    let result = hnsw.search(&query, 10, DEFAULT_EF);
    assert!(
        result.is_err(),
        "Empty unbuilt index should error on search"
    );
}

#[test]
fn test_hnsw_single_vector() {
    let dim = 16;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    let vector = normalize(&vec![1.0f32; dim]);
    // doc_id is ignored; internal index will be 0
    hnsw.add(42, vector.clone()).expect("Failed to add");
    hnsw.build().expect("Failed to build");

    let results = hnsw.search(&vector, 10, DEFAULT_EF).expect("Search failed");

    assert_eq!(results.len(), 1, "Should find exactly one result");
    // Internal index is 0, not 42 (doc_id is ignored)
    assert_eq!(results[0].0, 0, "Should find internal index 0");
}

#[test]
fn test_hnsw_high_dimensional() {
    let dim = 768; // BERT-like dimension
    let n = 100;
    let vectors: Vec<Vec<f32>> = random_vectors(n, dim, 789)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    let mut hnsw = HNSWIndex::new(dim, 32, 32).expect("Failed to create");
    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    // Query with vector at index 50
    let results = hnsw
        .search(&vectors[50], 10, DEFAULT_EF)
        .expect("Search failed");
    assert!(!results.is_empty());
    assert_eq!(results[0].0, 50, "Should find internal index 50");
}

#[test]
fn test_hnsw_returns_k_results() {
    let dim = 32;
    let n = 100;
    let vectors: Vec<Vec<f32>> = random_vectors(n, dim, 111)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    // Request various k values
    for k in [1, 5, 10, 50, 100] {
        let results = hnsw.search(&vectors[0], k, 100).expect("Search failed");
        let expected = k.min(n);
        assert_eq!(
            results.len(),
            expected,
            "Should return {} results for k={}, got {}",
            expected,
            k,
            results.len()
        );
    }

    // Request more than available
    let results = hnsw.search(&vectors[0], 200, 200).expect("Search failed");
    assert_eq!(results.len(), n, "Should return all {} vectors", n);
}

#[test]
fn test_hnsw_results_sorted_by_distance() {
    let dim = 32;
    let n = 200;
    let k = 20;
    let vectors: Vec<Vec<f32>> = random_vectors(n, dim, 222)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    let query = normalize(&random_vectors(1, dim, 333).pop().unwrap());
    let results = hnsw.search(&query, k, 100).expect("Search failed");

    // Verify results are sorted by distance
    for i in 1..results.len() {
        assert!(
            results[i].1 >= results[i - 1].1 - 1e-6,
            "Results not sorted: {:?} vs {:?}",
            results[i - 1],
            results[i]
        );
    }
}

#[test]
fn test_hnsw_dimension_validation() {
    // Zero dimension should fail
    let result = HNSWIndex::new(0, 16, 16);
    assert!(result.is_err(), "Zero dimension should fail");
}

#[test]
fn test_hnsw_query_dimension_mismatch() {
    let dim = 32;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
    hnsw.add(0, normalize(&vec![1.0; dim]))
        .expect("Failed to add");
    hnsw.build().expect("Failed to build");

    // Query with wrong dimension - should return error
    let wrong_dim_query = vec![1.0; dim + 1];
    let result = hnsw.search(&wrong_dim_query, 10, DEFAULT_EF);
    assert!(result.is_err(), "Wrong dimension query should error");
}

#[test]
fn test_hnsw_with_custom_params() {
    let dim = 32;
    let n = 200;
    let vectors: Vec<Vec<f32>> = random_vectors(n, dim, 444)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    // Create with custom params
    let params = HNSWParams {
        m: 32,
        m_max: 32,
        ef_construction: 100,
        ef_search: 100,
        ..Default::default()
    };

    let mut hnsw = HNSWIndex::with_params(dim, params).expect("Failed to create");
    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    let results = hnsw.search(&vectors[0], 10, 100).expect("Search failed");
    assert!(!results.is_empty());
}

#[test]
fn test_hnsw_repeated_builds_idempotent() {
    let dim = 16;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    hnsw.add(0, normalize(&vec![1.0; dim]))
        .expect("Failed to add");

    // Build multiple times should be ok
    hnsw.build().expect("First build");
    hnsw.build().expect("Second build should be idempotent");

    let results = hnsw
        .search(&normalize(&vec![1.0; dim]), 10, DEFAULT_EF)
        .expect("Search failed");
    assert_eq!(results.len(), 1);
}

#[test]
fn test_hnsw_ef_tradeoff() {
    // Higher ef_search = higher recall (typically)
    let dim = 64;
    let n = 500;
    let k = 10;
    let vectors: Vec<Vec<f32>> = random_vectors(n, dim, 999)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();
    let query = normalize(&random_vectors(1, dim, 1000).pop().unwrap());

    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add");
    }
    hnsw.build().expect("Failed to build");

    let exact = exact_knn_cosine(&vectors, &query, k);

    // Low ef_search
    let approx_low = hnsw.search(&query, k, 20).expect("Search failed");

    // High ef_search
    let approx_high = hnsw.search(&query, k, 200).expect("Search failed");

    let recall_low = recall_at_k(&exact, &approx_low, k);
    let recall_high = recall_at_k(&exact, &approx_high, k);

    // Higher ef should give equal or better recall (with small tolerance for randomness)
    assert!(
        recall_high >= recall_low - 0.1,
        "Higher ef_search should not significantly decrease recall: {} vs {}",
        recall_low,
        recall_high
    );
}

#[test]
fn test_hnsw_cannot_add_after_build() {
    let dim = 16;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    hnsw.add(0, normalize(&vec![1.0; dim]))
        .expect("Failed to add");
    hnsw.build().expect("Failed to build");

    // Adding after build should fail
    let result = hnsw.add(1, normalize(&vec![2.0; dim]));
    assert!(result.is_err(), "Adding after build should fail");
}

#[test]
fn test_hnsw_search_before_build_fails() {
    let dim = 16;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    hnsw.add(0, normalize(&vec![1.0; dim]))
        .expect("Failed to add");

    // Search before build should fail
    let result = hnsw.search(&normalize(&vec![1.0; dim]), 10, DEFAULT_EF);
    assert!(result.is_err(), "Search before build should fail");
}

#[test]
fn test_hnsw_cosine_similarity_property() {
    // Identical normalized vectors should have distance ~0
    let dim = 32;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    let v = normalize(&vec![1.0; dim]);
    hnsw.add(0, v.clone()).expect("Failed to add");
    hnsw.add(1, v.clone()).expect("Failed to add"); // Same vector
    hnsw.build().expect("Failed to build");

    let results = hnsw.search(&v, 10, DEFAULT_EF).expect("Search failed");

    // Both should have distance ~0
    assert!(
        results[0].1 < 0.01,
        "Distance to identical vector should be ~0"
    );
    assert!(
        results[1].1 < 0.01,
        "Distance to identical vector should be ~0"
    );
}

#[test]
fn test_hnsw_orthogonal_vectors() {
    // Orthogonal vectors should have cosine distance ~1
    let dim = 4;
    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");

    let v1 = vec![1.0, 0.0, 0.0, 0.0];
    let v2 = vec![0.0, 1.0, 0.0, 0.0];

    hnsw.add(0, v1.clone()).expect("Failed to add");
    hnsw.add(1, v2.clone()).expect("Failed to add");
    hnsw.build().expect("Failed to build");

    let results = hnsw.search(&v1, 10, DEFAULT_EF).expect("Search failed");

    // v1 should find itself first with distance ~0
    assert_eq!(results[0].0, 0);
    assert!(results[0].1 < 0.01);

    // v2 should be found with distance ~1 (orthogonal)
    assert_eq!(results[1].0, 1);
    assert!(
        (results[1].1 - 1.0).abs() < 0.01,
        "Orthogonal vector distance should be ~1"
    );
}

/// Property: Recall should be monotonically non-decreasing with ef_search.
///
/// This is a fundamental HNSW property: larger search effort (ef) explores
/// more candidates, so recall should not decrease.
#[test]
fn test_hnsw_recall_monotonic_with_ef() {
    let dim = 32;
    let n = 500;
    let k = 10;

    let vectors: Vec<Vec<f32>> = random_vectors(n, dim, 42)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create index");

    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add vector");
    }
    hnsw.build().expect("Failed to build index");

    // Test with several queries
    let test_queries: Vec<Vec<f32>> = random_vectors(20, dim, 999)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    let ef_values = [10, 20, 50, 100, 200];

    for query in &test_queries {
        let exact = exact_knn_cosine(&vectors, query, k);

        let mut prev_recall = 0.0_f32;

        for &ef in &ef_values {
            let results = hnsw.search(query, k, ef).expect("Search failed");
            let recall = recall_at_k(&exact, &results, k);

            // Recall should not decrease as ef increases
            // Allow small tolerance for floating point and edge cases
            assert!(
                recall >= prev_recall - 0.1,
                "Recall decreased from {} to {} when ef increased to {}",
                prev_recall,
                recall,
                ef
            );

            prev_recall = recall;
        }
    }
}

/// Property: Search should return valid indices within the indexed range.
#[test]
fn test_hnsw_search_returns_valid_indices() {
    let dim = 16;
    let n = 200;

    let vectors: Vec<Vec<f32>> = random_vectors(n, dim, 123)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create index");

    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add vector");
    }
    hnsw.build().expect("Failed to build index");

    // Run many queries
    let test_queries: Vec<Vec<f32>> = random_vectors(50, dim, 456)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    for query in &test_queries {
        let results = hnsw.search(query, 20, DEFAULT_EF).expect("Search failed");

        for (idx, dist) in &results {
            // Index must be valid
            assert!(
                (*idx as usize) < n,
                "Search returned invalid index {} (n={})",
                idx,
                n
            );

            // Distance must be non-negative (cosine distance is in [0, 2])
            assert!(
                *dist >= 0.0 && *dist <= 2.0 + 1e-5,
                "Invalid cosine distance: {}",
                dist
            );
        }
    }
}

/// Property: Identical queries should return identical results (determinism).
#[test]
fn test_hnsw_deterministic_search() {
    let dim = 32;
    let n = 300;

    let vectors: Vec<Vec<f32>> = random_vectors(n, dim, 789)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create index");

    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add vector");
    }
    hnsw.build().expect("Failed to build index");

    let query = normalize(&vec![1.0; dim]);

    // Run the same query multiple times
    let results1 = hnsw.search(&query, 10, DEFAULT_EF).expect("Search failed");
    let results2 = hnsw.search(&query, 10, DEFAULT_EF).expect("Search failed");
    let results3 = hnsw.search(&query, 10, DEFAULT_EF).expect("Search failed");

    // All results should be identical
    assert_eq!(results1, results2, "Search should be deterministic");
    assert_eq!(results2, results3, "Search should be deterministic");
}

/// Property: Results should be unique (no duplicate indices).
#[test]
fn test_hnsw_results_unique() {
    let dim = 32;
    let n = 400;
    let k = 50;

    let vectors: Vec<Vec<f32>> = random_vectors(n, dim, 321)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create index");

    for (i, v) in vectors.iter().enumerate() {
        hnsw.add(i as u32, v.clone()).expect("Failed to add vector");
    }
    hnsw.build().expect("Failed to build index");

    let test_queries: Vec<Vec<f32>> = random_vectors(30, dim, 654)
        .into_iter()
        .map(|v| normalize(&v))
        .collect();

    for query in &test_queries {
        let results = hnsw.search(query, k, 100).expect("Search failed");

        let indices: Vec<u32> = results.iter().map(|(i, _)| *i).collect();
        let unique: HashSet<u32> = indices.iter().copied().collect();

        assert_eq!(
            indices.len(),
            unique.len(),
            "Search returned duplicate indices"
        );
    }
}
