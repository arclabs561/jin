//! LID-based Outlier Detection
//!
//! Demonstrates using Local Intrinsic Dimensionality to identify outliers.
//! Points with high LID are in complex regions; points with low LID are in
//! dense clusters. Extreme values in either direction can indicate anomalies.
//!
//! ```bash
//! cargo run --example lid_outlier_detection --release
//! ```

use jin::lid::{
    aggregate_lid, estimate_lid, estimate_twonn_with_ci, LidAggregation, LidCategory, LidConfig,
    LidStats,
};

fn main() {
    println!("LID-based Outlier Detection");
    println!("============================\n");

    // Create a dataset with:
    // - A dense cluster (low LID)
    // - A sparse region (high LID)
    // - Some outliers
    let dim = 32;
    let mut points: Vec<Vec<f32>> = Vec::new();

    // Dense cluster around origin (80 points)
    for i in 0..80 {
        let point: Vec<f32> = (0..dim)
            .map(|j| {
                let seed = (i * dim + j) as f32;
                0.1 * (seed * 0.01).sin() // small spread
            })
            .collect();
        points.push(point);
    }

    // Sparse region (15 points spread out)
    for i in 0..15 {
        let point: Vec<f32> = (0..dim)
            .map(|j| {
                let seed = (i * dim + j + 1000) as f32;
                5.0 + 2.0 * (seed * 0.01).cos() // larger spread, offset
            })
            .collect();
        points.push(point);
    }

    // Outliers (5 points far from everything)
    for i in 0..5 {
        let point: Vec<f32> = (0..dim)
            .map(|j| {
                let seed = (i * dim + j + 2000) as f32;
                20.0 + 10.0 * (seed * 0.01).sin() // very far out
            })
            .collect();
        points.push(point);
    }

    println!("Dataset: {} points in {} dimensions", points.len(), dim);
    println!("  - Dense cluster: 80 points (indices 0-79)");
    println!("  - Sparse region: 15 points (indices 80-94)");
    println!("  - Outliers: 5 points (indices 95-99)");
    println!();

    // Compute pairwise distances for LID estimation
    let distances = compute_distance_matrix(&points);

    // Estimate LID for each point using k=20 neighbors
    let k = 20;
    let config = LidConfig {
        k,
        ..Default::default()
    };

    println!("Estimating LID with k={} neighbors...\n", k);

    let mut estimates = Vec::new();
    for i in 0..points.len() {
        // Get sorted distances to other points
        let mut dists: Vec<f32> = distances[i]
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &d)| d)
            .collect();
        dists.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let estimate = estimate_lid(&dists, &config);
        estimates.push((i, estimate));
    }

    // Aggregate statistics
    let all_estimates: Vec<_> = estimates.iter().map(|(_, e)| e.clone()).collect();
    let stats = LidStats::from_estimates(&all_estimates);

    println!("Global LID Statistics:");
    println!("  Mean:   {:.2}", stats.mean);
    println!("  Median: {:.2}", stats.median);
    println!("  Std:    {:.2}", stats.std_dev);
    println!("  Min:    {:.2}", stats.min);
    println!("  Max:    {:.2}", stats.max);
    println!();

    // Categorize points
    let mut low_lid = Vec::new();
    let mut normal_lid = Vec::new();
    let mut high_lid = Vec::new();

    for (i, estimate) in &estimates {
        match stats.categorize(estimate.lid) {
            LidCategory::Low => low_lid.push(*i),
            LidCategory::Normal => normal_lid.push(*i),
            LidCategory::High => high_lid.push(*i),
        }
    }

    println!("Point Categories:");
    println!(
        "  Low LID (dense):   {} points (indices: {:?}...)",
        low_lid.len(),
        &low_lid[..low_lid.len().min(5)]
    );
    println!("  Normal LID:        {} points", normal_lid.len());
    println!(
        "  High LID (sparse): {} points (indices: {:?})",
        high_lid.len(),
        high_lid
    );
    println!();

    // Show some specific examples
    println!("Example LID values:");
    for &idx in &[0, 40, 80, 90, 95, 99] {
        if idx < estimates.len() {
            let (_, ref est) = estimates[idx];
            let category = stats.categorize(est.lid);
            println!("  Point {}: LID={:.2} ({:?})", idx, est.lid, category);
        }
    }
    println!();

    // Demonstrate TwoNN estimator with confidence intervals
    println!("TwoNN Estimation (global):");
    let mu_ratios: Vec<f32> = estimates
        .iter()
        .filter_map(|(i, _)| {
            let mut dists: Vec<f32> = distances[*i]
                .iter()
                .enumerate()
                .filter(|(j, _)| *j != *i)
                .map(|(_, &d)| d)
                .collect();
            dists.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if dists.len() >= 2 && dists[0] > 0.0 {
                Some(dists[1] / dists[0])
            } else {
                None
            }
        })
        .collect();

    let twonn = estimate_twonn_with_ci(&mu_ratios, 0.1);
    println!(
        "  Dimension: {:.2} (95% CI: [{:.2}, {:.2}])",
        twonn.dimension, twonn.ci_lower, twonn.ci_upper
    );
    println!("  Samples used: {}", twonn.n_samples);
    println!();

    // Demonstrate aggregation methods
    println!("Aggregation Comparison:");
    let mean = aggregate_lid(&all_estimates, LidAggregation::Mean);
    let median = aggregate_lid(&all_estimates, LidAggregation::Median);
    let harmonic = aggregate_lid(&all_estimates, LidAggregation::HarmonicMean);
    println!("  Arithmetic Mean:  {:.2}", mean);
    println!("  Median:           {:.2}", median);
    println!("  Harmonic Mean:    {:.2}", harmonic);
}

fn compute_distance_matrix(points: &[Vec<f32>]) -> Vec<Vec<f32>> {
    let n = points.len();
    let mut distances = vec![vec![0.0f32; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let d = euclidean_distance(&points[i], &points[j]);
            distances[i][j] = d;
            distances[j][i] = d;
        }
    }

    distances
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
