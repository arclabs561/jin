//! Property-based tests for vicinity ANN components.
//!
//! These tests verify invariants that should hold regardless of input:
//! - Distance metrics satisfy metric space properties
//! - Recall is always in [0, 1]
//! - Memory estimates are consistent
//! - Ground truth computation is correct

use proptest::prelude::*;

mod distance_props {
    use super::*;

    fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            return 1.0;
        }

        1.0 - dot / (norm_a * norm_b)
    }

    prop_compose! {
        fn arb_vector(dim: usize)(vec in prop::collection::vec(-10.0f32..10.0, dim)) -> Vec<f32> {
            vec
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn l2_distance_non_negative(
            a in arb_vector(64),
            b in arb_vector(64),
        ) {
            let dist = l2_distance_squared(&a, &b);
            prop_assert!(dist >= 0.0, "L2 distance must be non-negative, got {}", dist);
        }

        #[test]
        fn l2_distance_symmetric(
            a in arb_vector(32),
            b in arb_vector(32),
        ) {
            let d_ab = l2_distance_squared(&a, &b);
            let d_ba = l2_distance_squared(&b, &a);
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-6,
                "L2 distance not symmetric: {} vs {}",
                d_ab, d_ba
            );
        }

        #[test]
        fn l2_distance_self_is_zero(
            a in arb_vector(32),
        ) {
            let dist = l2_distance_squared(&a, &a);
            prop_assert!(
                dist.abs() < 1e-10,
                "Distance to self should be 0, got {}",
                dist
            );
        }

        #[test]
        fn l2_triangle_inequality(
            a in arb_vector(16),
            b in arb_vector(16),
            c in arb_vector(16),
        ) {
            // For squared L2, triangle inequality is:
            // sqrt(d(a,c)) <= sqrt(d(a,b)) + sqrt(d(b,c))
            let d_ac = l2_distance_squared(&a, &c).sqrt();
            let d_ab = l2_distance_squared(&a, &b).sqrt();
            let d_bc = l2_distance_squared(&b, &c).sqrt();

            prop_assert!(
                d_ac <= d_ab + d_bc + 1e-5,
                "Triangle inequality violated: {} > {} + {}",
                d_ac, d_ab, d_bc
            );
        }

        #[test]
        fn cosine_distance_in_range(
            a in arb_vector(32),
            b in arb_vector(32),
        ) {
            let dist = cosine_distance(&a, &b);
            // Cosine distance should be in [0, 2] (or 1 for zero vectors)
            // Due to floating point, allow small violations
            prop_assert!(
                dist >= -0.001 && dist <= 2.001,
                "Cosine distance out of range: {}",
                dist
            );
        }

        #[test]
        fn cosine_distance_symmetric(
            a in arb_vector(32),
            b in arb_vector(32),
        ) {
            let d_ab = cosine_distance(&a, &b);
            let d_ba = cosine_distance(&b, &a);
            prop_assert!(
                (d_ab - d_ba).abs() < 1e-5,
                "Cosine distance not symmetric: {} vs {}",
                d_ab, d_ba
            );
        }
    }
}

mod recall_props {
    use super::*;
    use std::collections::HashSet;

    fn recall_at_k(ground_truth: &[u32], retrieved: &[u32], k: usize) -> f32 {
        if k == 0 || ground_truth.is_empty() {
            return 0.0;
        }

        let gt_set: HashSet<u32> = ground_truth.iter().take(k).copied().collect();
        let ret_set: HashSet<u32> = retrieved.iter().take(k).copied().collect();

        gt_set.intersection(&ret_set).count() as f32 / k as f32
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        #[test]
        fn recall_in_unit_interval(
            gt in prop::collection::vec(0u32..1000, 1..50),
            ret in prop::collection::vec(0u32..1000, 1..50),
            k in 1usize..20,
        ) {
            let recall = recall_at_k(&gt, &ret, k);
            prop_assert!(
                recall >= 0.0 && recall <= 1.0,
                "Recall must be in [0,1], got {}",
                recall
            );
        }

        #[test]
        fn perfect_recall_when_identical(
            // Use HashSet to guarantee unique values
            gt_set in prop::collection::hash_set(0u32..1000, 1..20),
        ) {
            let gt: Vec<u32> = gt_set.into_iter().collect();
            let k = gt.len();
            let recall = recall_at_k(&gt, &gt, k);
            prop_assert!(
                (recall - 1.0).abs() < 1e-6,
                "Recall should be 1.0 for identical sets, got {}",
                recall
            );
        }

        #[test]
        fn zero_recall_disjoint_sets(
            offset in 0u32..1000,
            size in 1usize..20,
        ) {
            let gt: Vec<u32> = (0..size as u32).collect();
            let ret: Vec<u32> = (offset + 1000..(offset + 1000 + size as u32)).collect();
            let recall = recall_at_k(&gt, &ret, size);
            prop_assert!(
                recall.abs() < 1e-6,
                "Recall should be 0 for disjoint sets, got {}",
                recall
            );
        }

        #[test]
        fn recall_monotonic_with_overlap(
            base in prop::collection::vec(0u32..50, 10..20),
        ) {
            // As we include more of the ground truth in retrieved,
            // recall should increase (or stay same)
            let k = base.len();
            let mut retrieved: Vec<u32> = (100..100 + k as u32).collect();

            let mut prev_recall = recall_at_k(&base, &retrieved, k);

            for (i, &gt_item) in base.iter().enumerate().take(k / 2) {
                retrieved[i] = gt_item;
                let new_recall = recall_at_k(&base, &retrieved, k);
                prop_assert!(
                    new_recall >= prev_recall - 1e-6,
                    "Recall decreased from {} to {} when adding correct items",
                    prev_recall, new_recall
                );
                prev_recall = new_recall;
            }
        }
    }
}

mod memory_props {
    use super::*;

    fn theoretical_hnsw_memory(n_vectors: usize, dimension: usize, m: usize) -> usize {
        let raw_bytes = n_vectors * dimension * 4;
        let avg_edges = (2.5 * m as f64) as usize;
        let graph_bytes = n_vectors * avg_edges * 4;
        let metadata_bytes = n_vectors * 5;
        raw_bytes + graph_bytes + metadata_bytes
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn memory_increases_with_vectors(
            n1 in 100usize..1000,
            n2 in 1000usize..10000,
            dim in 32usize..256,
            m in 8usize..32,
        ) {
            let mem1 = theoretical_hnsw_memory(n1, dim, m);
            let mem2 = theoretical_hnsw_memory(n2, dim, m);
            prop_assert!(
                mem2 > mem1,
                "Memory should increase with vectors: {} vs {}",
                mem1, mem2
            );
        }

        #[test]
        fn memory_increases_with_dimension(
            n in 1000usize..5000,
            dim1 in 32usize..128,
            dim2 in 256usize..512,
            m in 8usize..24,
        ) {
            let mem1 = theoretical_hnsw_memory(n, dim1, m);
            let mem2 = theoretical_hnsw_memory(n, dim2, m);
            prop_assert!(
                mem2 > mem1,
                "Memory should increase with dimension: {} vs {}",
                mem1, mem2
            );
        }

        #[test]
        fn memory_increases_with_m(
            n in 1000usize..5000,
            dim in 64usize..256,
            m1 in 4usize..16,
            m2 in 24usize..48,
        ) {
            let mem1 = theoretical_hnsw_memory(n, dim, m1);
            let mem2 = theoretical_hnsw_memory(n, dim, m2);
            prop_assert!(
                mem2 > mem1,
                "Memory should increase with M: {} (M={}) vs {} (M={})",
                mem1, m1, mem2, m2
            );
        }

        #[test]
        fn memory_positive(
            n in 1usize..10000,
            dim in 1usize..512,
            m in 1usize..64,
        ) {
            let mem = theoretical_hnsw_memory(n, dim, m);
            prop_assert!(mem > 0, "Memory must be positive");
        }
    }
}

mod ground_truth_props {
    use super::*;

    fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    fn compute_ground_truth(query: &[f32], database: &[Vec<f32>], k: usize) -> Vec<(u32, f32)> {
        let mut distances: Vec<(u32, f32)> = database
            .iter()
            .enumerate()
            .map(|(i, vec)| (i as u32, l2_distance_squared(query, vec)))
            .collect();
        distances.sort_by(|a, b| a.1.total_cmp(&b.1));
        distances.truncate(k);
        distances
    }

    prop_compose! {
        fn arb_database(n: usize, dim: usize)(
            db in prop::collection::vec(
                prop::collection::vec(-5.0f32..5.0, dim),
                n
            )
        ) -> Vec<Vec<f32>> {
            db
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        #[test]
        fn ground_truth_returns_k_results(
            db in arb_database(50, 16),
            k in 1usize..20,
        ) {
            let query: Vec<f32> = (0..16).map(|_| 0.0f32).collect();
            let gt = compute_ground_truth(&query, &db, k);
            let expected_k = k.min(db.len());
            prop_assert_eq!(
                gt.len(),
                expected_k,
                "Ground truth should return {} results, got {}",
                expected_k,
                gt.len()
            );
        }

        #[test]
        fn ground_truth_sorted_by_distance(
            db in arb_database(30, 16),
        ) {
            let query: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
            let gt = compute_ground_truth(&query, &db, db.len());

            for i in 1..gt.len() {
                prop_assert!(
                    gt[i].1 >= gt[i - 1].1,
                    "Ground truth not sorted: {} > {} at position {}",
                    gt[i - 1].1, gt[i].1, i
                );
            }
        }

        #[test]
        fn ground_truth_unique_ids(
            db in arb_database(30, 16),
        ) {
            let query: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
            let gt = compute_ground_truth(&query, &db, db.len());

            let ids: Vec<u32> = gt.iter().map(|(id, _)| *id).collect();
            let mut sorted_ids = ids.clone();
            sorted_ids.sort_unstable();
            sorted_ids.dedup();

            prop_assert_eq!(
                ids.len(),
                sorted_ids.len(),
                "Ground truth contains duplicate IDs"
            );
        }

        #[test]
        fn ground_truth_ids_valid(
            db in arb_database(30, 16),
        ) {
            let query: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
            let gt = compute_ground_truth(&query, &db, db.len());

            for (id, _) in &gt {
                prop_assert!(
                    (*id as usize) < db.len(),
                    "Ground truth ID {} out of range (db size {})",
                    id, db.len()
                );
            }
        }

        #[test]
        fn ground_truth_distances_match(
            db in arb_database(20, 8),
        ) {
            let query: Vec<f32> = (0..8).map(|i| i as f32 * 0.2).collect();
            let gt = compute_ground_truth(&query, &db, db.len());

            for (id, dist) in &gt {
                let actual_dist = l2_distance_squared(&query, &db[*id as usize]);
                prop_assert!(
                    (dist - actual_dist).abs() < 1e-5,
                    "Distance mismatch for ID {}: {} vs {}",
                    id, dist, actual_dist
                );
            }
        }
    }
}

// =============================================================================
// LID (Local Intrinsic Dimensionality) Properties
// =============================================================================

mod lid_props {
    use super::*;
    use vicinity::lid::{estimate_lid_mle, LidConfig, LidEstimate, LidStats};

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        /// LID should be positive for valid distance sequences.
        #[test]
        fn lid_positive_for_valid_distances(
            // Generate increasing distances (realistic scenario)
            base in 0.01f32..0.1,
            increments in prop::collection::vec(0.01f32..0.5, 19),
        ) {
            let mut distances = vec![base];
            let mut cumsum = base;
            for inc in increments {
                cumsum += inc;
                distances.push(cumsum);
            }

            let config = LidConfig::default();
            let estimate = estimate_lid_mle(&distances, &config);

            prop_assert!(estimate.lid > 0.0 || estimate.lid.is_infinite(),
                "LID should be positive, got {}", estimate.lid);
        }

        /// LID should be invariant to uniform scaling of distances.
        #[test]
        fn lid_scale_invariant(
            base in 0.01f32..0.1,
            increments in prop::collection::vec(0.01f32..0.5, 19),
            scale in 0.1f32..10.0,
        ) {
            let mut distances = vec![base];
            let mut cumsum = base;
            for inc in &increments {
                cumsum += inc;
                distances.push(cumsum);
            }

            let scaled: Vec<f32> = distances.iter().map(|d| d * scale).collect();

            let config = LidConfig::default();
            let est1 = estimate_lid_mle(&distances, &config);
            let est2 = estimate_lid_mle(&scaled, &config);

            // LID should be approximately scale-invariant
            // (exact invariance holds for continuous distributions)
            if est1.lid.is_finite() && est2.lid.is_finite() {
                let relative_diff = (est1.lid - est2.lid).abs() / est1.lid.max(1.0);
                prop_assert!(relative_diff < 0.3,
                    "LID not scale invariant: {} vs {} (scale={})",
                    est1.lid, est2.lid, scale);
            }
        }

        /// k parameter should be respected.
        #[test]
        fn lid_respects_k(
            distances in prop::collection::vec(0.1f32..10.0, 30..50),
            k in 5usize..25,
        ) {
            let mut sorted = distances.clone();
            sorted.sort_by(|a, b| a.total_cmp(b));

            let config = LidConfig { k, epsilon: 1e-10 };
            let estimate = estimate_lid_mle(&sorted, &config);

            prop_assert_eq!(estimate.k, k.min(sorted.len()),
                "k should be min of config.k and distances.len()");
        }

        /// LidStats should produce valid statistics.
        #[test]
        fn lid_stats_valid(
            lids in prop::collection::vec(1.0f32..50.0, 10..30),
        ) {
            let estimates: Vec<LidEstimate> = lids.iter()
                .map(|&lid| LidEstimate { lid, k: 20, max_dist: 1.0 })
                .collect();

            let stats = LidStats::from_estimates(&estimates);

            prop_assert_eq!(stats.count, estimates.len());
            prop_assert!(stats.min <= stats.mean);
            prop_assert!(stats.mean <= stats.max);
            prop_assert!(stats.std_dev >= 0.0);
        }

        /// High LID threshold should be above median.
        #[test]
        fn high_lid_threshold_above_median(
            lids in prop::collection::vec(1.0f32..50.0, 5..20),
        ) {
            let estimates: Vec<LidEstimate> = lids.iter()
                .map(|&lid| LidEstimate { lid, k: 20, max_dist: 1.0 })
                .collect();

            let stats = LidStats::from_estimates(&estimates);

            if stats.std_dev > 0.0 {
                prop_assert!(stats.high_lid_threshold() > stats.median,
                    "threshold {} should be > median {}",
                    stats.high_lid_threshold(), stats.median);
            }
        }

        /// LID categorization is consistent with thresholds.
        #[test]
        fn lid_categorization_consistent(
            lids in prop::collection::vec(1.0f32..100.0, 20..50),
        ) {
            use vicinity::lid::LidCategory;

            let estimates: Vec<LidEstimate> = lids.iter()
                .map(|&lid| LidEstimate { lid, k: 20, max_dist: 1.0 })
                .collect();

            let stats = LidStats::from_estimates(&estimates);

            for &lid in &lids {
                let category = stats.categorize(lid);
                match category {
                    LidCategory::Low => {
                        prop_assert!(lid < stats.median - stats.std_dev + 1e-5,
                            "Low category but lid {} >= median - std = {}",
                            lid, stats.median - stats.std_dev);
                    }
                    LidCategory::High => {
                        prop_assert!(lid > stats.median + stats.std_dev - 1e-5,
                            "High category but lid {} <= median + std = {}",
                            lid, stats.median + stats.std_dev);
                    }
                    LidCategory::Normal => {
                        // Normal is in between
                    }
                }
            }
        }

        /// LID for uniform distances gives finite positive result.
        #[test]
        fn lid_uniform_distances_finite(
            n in 10usize..50,
            step in 0.01f32..0.5,
        ) {
            let distances: Vec<f32> = (0..n).map(|i| (i + 1) as f32 * step).collect();
            let config = LidConfig::default();
            let estimate = estimate_lid_mle(&distances, &config);

            prop_assert!(estimate.lid.is_finite(),
                "LID should be finite for uniform distances, got {}", estimate.lid);
            prop_assert!(estimate.lid > 0.0,
                "LID should be positive, got {}", estimate.lid);
        }
    }
}

// =============================================================================
// Additional Distance Metric Properties
// =============================================================================

mod metric_space_props {
    use super::*;

    fn arb_vec(dim: usize) -> impl Strategy<Value = Vec<f32>> {
        prop::collection::vec(-10.0f32..10.0, dim)
    }

    fn normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }

    fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(200))]

        /// Cosine distance is bounded [0, 2].
        #[test]
        fn cosine_distance_bounded(
            a in arb_vec(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6)),
            b in arb_vec(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6)),
        ) {
            let a_norm = normalize(&a);
            let b_norm = normalize(&b);

            let dot: f32 = a_norm.iter().zip(b_norm.iter()).map(|(x, y)| x * y).sum();
            let cosine_dist = 1.0 - dot;

            prop_assert!(
                (-0.01..=2.01).contains(&cosine_dist),
                "Cosine distance out of bounds: {}",
                cosine_dist
            );
        }

        /// Cosine distance to self is zero.
        #[test]
        fn cosine_distance_self_zero(
            v in arb_vec(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6)),
        ) {
            let v_norm = normalize(&v);
            let dot: f32 = v_norm.iter().map(|x| x * x).sum();
            let cosine_dist = 1.0 - dot;

            prop_assert!(
                cosine_dist.abs() < 1e-5,
                "Cosine distance to self should be 0, got {}",
                cosine_dist
            );
        }

        /// L2 distance satisfies identity of indiscernibles.
        #[test]
        fn l2_identity_of_indiscernibles(v in arb_vec(32)) {
            let dist = l2_squared(&v, &v);
            prop_assert!(dist.abs() < 1e-6, "L2(v, v) should be 0, got {}", dist);
        }

        /// L2 distance is symmetric.
        #[test]
        fn l2_symmetric_test(
            a in arb_vec(32),
            b in arb_vec(32),
        ) {
            let ab = l2_squared(&a, &b);
            let ba = l2_squared(&b, &a);

            prop_assert!(
                (ab - ba).abs() < 1e-6,
                "L2 not symmetric: {} != {}",
                ab, ba
            );
        }

        /// L2 distance satisfies triangle inequality.
        #[test]
        fn l2_triangle_inequality_test(
            a in arb_vec(16),
            b in arb_vec(16),
            c in arb_vec(16),
        ) {
            let ab = l2_squared(&a, &b).sqrt();
            let bc = l2_squared(&b, &c).sqrt();
            let ac = l2_squared(&a, &c).sqrt();

            prop_assert!(
                ac <= ab + bc + 1e-4,
                "Triangle inequality violated: {} > {} + {} = {}",
                ac, ab, bc, ab + bc
            );
        }

        /// Dot product is bilinear: dot(a + b, c) = dot(a, c) + dot(b, c).
        #[test]
        fn dot_bilinear(
            a in arb_vec(32),
            b in arb_vec(32),
            c in arb_vec(32),
        ) {
            let ab: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

            let dot_ab_c: f32 = ab.iter().zip(c.iter()).map(|(x, y)| x * y).sum();
            let dot_a_c: f32 = a.iter().zip(c.iter()).map(|(x, y)| x * y).sum();
            let dot_b_c: f32 = b.iter().zip(c.iter()).map(|(x, y)| x * y).sum();

            let expected = dot_a_c + dot_b_c;
            let tolerance = expected.abs() * 1e-4 + 1e-4;

            prop_assert!(
                (dot_ab_c - expected).abs() < tolerance,
                "Bilinearity violated: {} != {} + {} = {}",
                dot_ab_c, dot_a_c, dot_b_c, expected
            );
        }

        /// Scaling preserves direction.
        #[test]
        fn scaling_preserves_direction(
            v in arb_vec(32).prop_filter("non-zero", |v| v.iter().any(|x| x.abs() > 1e-6)),
            scale in 0.1f32..10.0,
        ) {
            let v_norm = normalize(&v);
            let scaled: Vec<f32> = v.iter().map(|x| x * scale).collect();
            let scaled_norm = normalize(&scaled);

            for (a, b) in v_norm.iter().zip(scaled_norm.iter()) {
                prop_assert!(
                    (a - b).abs() < 1e-5,
                    "Scaling changed direction"
                );
            }
        }
    }
}

// =============================================================================
// HNSW Invariant Tests
// =============================================================================

mod hnsw_props {
    use super::*;
    use vicinity::hnsw::HNSWIndex;

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

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(50))]

        /// HNSW search returns k results for k <= n.
        #[test]
        fn hnsw_returns_k_results(
            n in 20usize..100,
            k in 1usize..20,
            seed in any::<u64>(),
        ) {
            let dim = 16;
            let k = k.min(n);
            let vectors = random_vectors(n, dim, seed);

            let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
            for (i, v) in vectors.iter().enumerate() {
                hnsw.add(i as u32, v.clone()).expect("Failed to add");
            }
            hnsw.build().expect("Failed to build");

            let results = hnsw.search(&vectors[0], k, 100).expect("Search failed");

            prop_assert_eq!(
                results.len(),
                k,
                "Should return {} results, got {}",
                k,
                results.len()
            );
        }

        /// HNSW results are sorted by distance.
        #[test]
        fn hnsw_results_sorted(
            n in 30usize..80,
            seed in any::<u64>(),
        ) {
            let dim = 16;
            let vectors = random_vectors(n, dim, seed);

            let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
            for (i, v) in vectors.iter().enumerate() {
                hnsw.add(i as u32, v.clone()).expect("Failed to add");
            }
            hnsw.build().expect("Failed to build");

            let results = hnsw.search(&vectors[n / 2], 10, 100).expect("Search failed");

            for i in 1..results.len() {
                prop_assert!(
                    results[i].1 >= results[i - 1].1 - 1e-6,
                    "Results not sorted: {} > {} at position {}",
                    results[i - 1].1,
                    results[i].1,
                    i
                );
            }
        }

        /// HNSW result IDs are unique.
        #[test]
        fn hnsw_unique_results(
            n in 30usize..80,
            seed in any::<u64>(),
        ) {
            let dim = 16;
            let vectors = random_vectors(n, dim, seed);

            let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
            for (i, v) in vectors.iter().enumerate() {
                hnsw.add(i as u32, v.clone()).expect("Failed to add");
            }
            hnsw.build().expect("Failed to build");

            let results = hnsw.search(&vectors[0], 15, 100).expect("Search failed");

            let ids: std::collections::HashSet<u32> = results.iter().map(|(id, _)| *id).collect();
            prop_assert_eq!(
                ids.len(),
                results.len(),
                "Result IDs are not unique"
            );
        }

        /// Same query gives same results (determinism).
        #[test]
        fn hnsw_deterministic(
            n in 30usize..60,
            seed in any::<u64>(),
        ) {
            let dim = 16;
            let vectors = random_vectors(n, dim, seed);

            let mut hnsw = HNSWIndex::new(dim, 16, 16).expect("Failed to create");
            for (i, v) in vectors.iter().enumerate() {
                hnsw.add(i as u32, v.clone()).expect("Failed to add");
            }
            hnsw.build().expect("Failed to build");

            let query = &vectors[0];
            let results1 = hnsw.search(query, 10, 100).expect("Search failed");
            let results2 = hnsw.search(query, 10, 100).expect("Search failed");
            let results3 = hnsw.search(query, 10, 100).expect("Search failed");

            // Same IDs should be returned
            let ids1: std::collections::HashSet<u32> = results1.iter().map(|(id, _)| *id).collect();
            let ids2: std::collections::HashSet<u32> = results2.iter().map(|(id, _)| *id).collect();
            let ids3: std::collections::HashSet<u32> = results3.iter().map(|(id, _)| *id).collect();

            prop_assert_eq!(&ids1, &ids2, "Results not deterministic (run 1 vs 2)");
            prop_assert_eq!(&ids2, &ids3, "Results not deterministic (run 2 vs 3)");
        }
    }
}
