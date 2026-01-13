//! Dual-Branch HNSW Demo
//!
//! Demonstrates LID-aware HNSW construction with skip bridges for better
//! handling of outliers and non-uniform data distributions.
//!
//! ```bash
//! cargo run --example dual_branch_hnsw_demo --release
//! ```

use std::collections::HashSet;
use std::time::Instant;
use vicinity::hnsw::dual_branch::{DualBranchConfig, DualBranchHNSW};
use vicinity::hnsw::HNSWIndex;
use vicinity::lid::{estimate_lid_batch, LidConfig, LidStats};

fn main() -> vicinity::Result<()> {
    println!("Dual-Branch HNSW with LID-Aware Construction");
    println!("=============================================\n");

    println!("Key ideas:");
    println!("  1. Local Intrinsic Dimensionality (LID) identifies 'outlier' points");
    println!("  2. High-LID points get more neighbors (better connectivity)");
    println!("  3. Skip bridges connect high-LID nodes for faster navigation\n");

    demo_lid_detection()?;
    demo_clustered_with_outliers()?;
    demo_stats()?;

    Ok(())
}

fn demo_lid_detection() -> vicinity::Result<()> {
    println!("1. LID Detection on Clustered Data");
    println!("   --------------------------------\n");

    let dim = 16;
    let n_clusters = 5;
    let points_per_cluster = 50;
    let n_outliers = 10;

    // Generate clusters
    let mut data: Vec<f32> = Vec::new();

    for c in 0..n_clusters {
        let center: Vec<f32> = (0..dim)
            .map(|j| ((c * dim + j) as f32 * 0.618).sin() * 10.0)
            .collect();

        for i in 0..points_per_cluster {
            let point: Vec<f32> = center
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    let noise = ((c * points_per_cluster + i) * dim + j) as f32 * 0.01;
                    v + noise.sin() * 0.5
                })
                .collect();
            data.extend(point);
        }
    }

    let n_normal = n_clusters * points_per_cluster;

    // Add outliers (random high-dimensional noise)
    for i in 0..n_outliers {
        let outlier: Vec<f32> = (0..dim)
            .map(|j| ((i * dim + j) as f32 * 0.7).sin() * 50.0) // Far from clusters
            .collect();
        data.extend(outlier);
    }

    let n_total = n_normal + n_outliers;

    // Estimate LID for all points
    let config = LidConfig {
        k: 15,
        epsilon: 1e-10,
    };
    let estimates = estimate_lid_batch(&data, dim, &config);
    let stats = LidStats::from_estimates(&estimates);

    println!(
        "   Generated {} cluster points + {} outliers",
        n_normal, n_outliers
    );
    println!("   LID statistics:");
    println!("     Mean:   {:.2}", stats.mean);
    println!("     Median: {:.2}", stats.median);
    println!("     Std:    {:.2}", stats.std_dev);
    println!(
        "     High threshold: {:.2} (median + 1 sigma)",
        stats.high_lid_threshold()
    );

    // Count detected outliers
    let high_lid_threshold = stats.high_lid_threshold();
    let mut detected_as_outlier = 0;
    let mut cluster_detected_as_high = 0;

    for (i, estimate) in estimates.iter().enumerate() {
        let is_actual_outlier = i >= n_normal;
        let is_high_lid = estimate.lid > high_lid_threshold;

        if is_actual_outlier && is_high_lid {
            detected_as_outlier += 1;
        } else if !is_actual_outlier && is_high_lid {
            cluster_detected_as_high += 1;
        }
    }

    println!("\n   Outlier detection:");
    println!(
        "     True outliers detected: {}/{}",
        detected_as_outlier, n_outliers
    );
    println!(
        "     Cluster points flagged: {}/{}",
        cluster_detected_as_high, n_normal
    );
    println!();

    Ok(())
}

fn demo_clustered_with_outliers() -> vicinity::Result<()> {
    println!("2. Dual-Branch HNSW on Non-Uniform Data");
    println!("   ------------------------------------\n");

    let dim = 32;
    let n_clusters = 10;
    let points_per_cluster = 100;
    let n_outliers = 50;

    // Generate data
    let mut data: Vec<f32> = Vec::new();

    for c in 0..n_clusters {
        let center: Vec<f32> = (0..dim)
            .map(|j| ((c * dim + j) as f32 * 0.618).sin() * 20.0)
            .collect();

        for i in 0..points_per_cluster {
            let point: Vec<f32> = center
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    let noise = ((c * points_per_cluster + i) * dim + j) as f32 * 0.01;
                    v + noise.sin() * 1.0
                })
                .collect();
            data.extend(point);
        }
    }

    let n_normal = n_clusters * points_per_cluster;

    // Add scattered outliers
    for i in 0..n_outliers {
        let outlier: Vec<f32> = (0..dim)
            .map(|j| ((i * dim + j) as f32 * 1.414).sin() * 100.0)
            .collect();
        data.extend(outlier);
    }

    let n_total = n_normal + n_outliers;

    // Build vectors for search
    let vectors: Vec<Vec<f32>> = (0..n_total)
        .map(|i| data[i * dim..(i + 1) * dim].to_vec())
        .collect();

    // Generate queries (some from clusters, some outliers)
    let n_queries = 50;
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| {
            if i < n_queries / 2 {
                // Query near a cluster
                let cluster_idx = (i * 7) % n_normal;
                let base = &vectors[cluster_idx];
                base.iter()
                    .enumerate()
                    .map(|(j, &v)| v + ((i * dim + j) as f32 * 0.001).sin() * 0.1)
                    .collect()
            } else {
                // Query in sparse region
                (0..dim)
                    .map(|j| ((i * dim + j) as f32 * 0.999).cos() * 80.0)
                    .collect()
            }
        })
        .collect();

    // Build standard HNSW
    let standard_build_start = Instant::now();
    let mut standard_index = HNSWIndex::new(dim, 16, 32)?;
    for (i, vec) in vectors.iter().enumerate() {
        standard_index.add(i as u32, vec.clone())?;
    }
    standard_index.build()?;
    let standard_build_time = standard_build_start.elapsed();

    // Build Dual-Branch HNSW
    let dual_config = DualBranchConfig {
        m: 16,
        m_high_lid: 24, // More edges for outliers
        ef_construction: 100,
        ef_search: 50,
        lid_k: 15,
        lid_threshold_sigma: 1.0,
        skip_bridge_probability: 0.3,
        max_skip_length: 3,
        seed: Some(42),
    };

    let dual_build_start = Instant::now();
    let mut dual_index = DualBranchHNSW::new(dim, dual_config);
    dual_index.add_vectors(&data)?;
    dual_index.build()?;
    let dual_build_time = dual_build_start.elapsed();

    println!(
        "   Dataset: {} points ({}+{} outliers), dim={}",
        n_total, n_normal, n_outliers, dim
    );
    println!("   Build time:");
    println!("     Standard HNSW: {:?}", standard_build_time);
    println!("     Dual-Branch:   {:?}", dual_build_time);

    // Search and compare
    let k = 10;
    let ef = 50;

    // Brute force ground truth
    let ground_truth: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(u32, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| (i as u32, l2_distance(q, v)))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            dists.into_iter().take(k).map(|(id, _)| id).collect()
        })
        .collect();

    // Standard HNSW search
    let standard_start = Instant::now();
    let mut standard_results = Vec::with_capacity(n_queries);
    for query in &queries {
        let results = standard_index.search(query, k, ef)?;
        standard_results.push(results.iter().map(|r| r.0).collect::<Vec<_>>());
    }
    let standard_search_time = standard_start.elapsed();

    // Dual-Branch search
    let dual_start = Instant::now();
    let mut dual_results = Vec::with_capacity(n_queries);
    for query in &queries {
        let results = dual_index.search(query, k)?;
        dual_results.push(results.iter().map(|r| r.0).collect::<Vec<_>>());
    }
    let dual_search_time = dual_start.elapsed();

    // Calculate recall
    let standard_recall = calculate_recall(&ground_truth, &standard_results, k);
    let dual_recall = calculate_recall(&ground_truth, &dual_results, k);

    println!("\n   Search ({} queries, k={}):", n_queries, k);
    println!(
        "     Standard: {:?}, recall@{}: {:.1}%",
        standard_search_time,
        k,
        standard_recall * 100.0
    );
    println!(
        "     Dual-Branch: {:?}, recall@{}: {:.1}%",
        dual_search_time,
        k,
        dual_recall * 100.0
    );

    // Get dual-branch stats
    let stats = dual_index.stats();
    println!("\n   Dual-Branch structure:");
    println!("     High-LID nodes: {}", stats.high_lid_nodes);
    println!("     Skip bridges: {}", stats.num_skip_bridges);
    println!("     Total edges: {}", stats.num_edges);
    let avg_degree = if stats.num_vectors > 0 {
        stats.num_edges as f64 / stats.num_vectors as f64
    } else {
        0.0
    };
    println!("     Avg neighbors: {:.1}", avg_degree);

    println!();
    Ok(())
}

fn demo_stats() -> vicinity::Result<()> {
    println!("3. Understanding Skip Bridges");
    println!("   --------------------------\n");

    println!("   Skip bridges are long-range connections from high-LID nodes");
    println!("   to distant reachable nodes. They help search escape local minima.");
    println!("\n   When are they useful?");
    println!("     - Data with isolated clusters");
    println!("     - Non-uniform density distributions");
    println!("     - Queries in sparse regions");
    println!("\n   Configuration parameters:");
    println!("     - m_high_lid: extra neighbors for high-LID nodes (default: 1.5x m)");
    println!("     - lid_threshold_sigma: how many std devs above median = 'high' LID");
    println!("     - max_skip_length: random walk steps to find bridge targets");
    println!("     - num_skip_bridges: bridges per high-LID node");

    Ok(())
}

fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

fn calculate_recall(ground_truth: &[Vec<u32>], results: &[Vec<u32>], k: usize) -> f64 {
    let mut total = 0.0;
    for (gt, res) in ground_truth.iter().zip(results.iter()) {
        let gt_set: HashSet<u32> = gt.iter().take(k).copied().collect();
        let res_set: HashSet<u32> = res.iter().take(k).copied().collect();
        total += gt_set.intersection(&res_set).count() as f64 / k as f64;
    }
    total / ground_truth.len() as f64
}
