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
