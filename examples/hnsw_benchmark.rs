//! HNSW vs Brute Force Benchmark
//!
//! Demonstrates that HNSW is faster than brute force for larger datasets
//! while maintaining high recall.
//!
//! ```bash
//! cargo run --example hnsw_benchmark --release
//! ```

use std::time::Instant;
use vicinity::hnsw::HNSWIndex;

fn main() -> vicinity::Result<()> {
    println!("HNSW vs Brute Force Benchmark");
    println!("==============================\n");

    let dim = 128;
    let sizes = [1_000, 5_000, 10_000, 25_000];

    for &n in &sizes {
        benchmark_size(n, dim)?;
    }

    Ok(())
}

fn benchmark_size(n: usize, dim: usize) -> vicinity::Result<()> {
    println!("Dataset: {} vectors, {} dimensions", n, dim);
    println!("{}", "-".repeat(50));

    // Generate random vectors (deterministic for reproducibility)
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..dim)
                .map(|j| {
                    let seed = (i * dim + j) as f32;
                    (seed * 0.0001).sin()
                })
                .collect()
        })
        .collect();

    // Generate query vectors (slight perturbations of random database vectors)
    let n_queries = 100;
    let queries: Vec<Vec<f32>> = (0..n_queries)
        .map(|i| {
            // Pick a random base vector and perturb it slightly
            let base_idx = (i * 7) % n;
            vectors[base_idx]
                .iter()
                .enumerate()
                .map(|(j, &v)| {
                    let noise = ((i * dim + j) as f32 * 0.0001).sin() * 0.01;
                    v + noise
                })
                .collect()
        })
        .collect();

    let k = 10;

    // Build HNSW index
    let build_start = Instant::now();
    let mut index = HNSWIndex::new(dim, 16, 32)?;
    for (i, vec) in vectors.iter().enumerate() {
        index.add(i as u32, vec.clone())?;
    }
    index.build()?;
    let build_time = build_start.elapsed();
    println!("HNSW build time: {:?}", build_time);

    // HNSW search
    let ef = 50; // search beam width
    let hnsw_start = Instant::now();
    let mut hnsw_results = Vec::with_capacity(n_queries);
    for query in &queries {
        let results = index.search(query, k, ef)?;
        hnsw_results.push(results);
    }
    let hnsw_time = hnsw_start.elapsed();

    // Brute force search
    let brute_start = Instant::now();
    let mut brute_results = Vec::with_capacity(n_queries);
    for query in &queries {
        let mut distances: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, euclidean_distance(query, v)))
            .collect();
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        brute_results.push(distances.into_iter().take(k).collect::<Vec<_>>());
    }
    let brute_time = brute_start.elapsed();

    // Calculate recall
    let mut total_recall = 0.0;
    for (hnsw, brute) in hnsw_results.iter().zip(brute_results.iter()) {
        let hnsw_ids: std::collections::HashSet<u32> = hnsw.iter().map(|r| r.0).collect();
        let brute_ids: std::collections::HashSet<u32> = brute.iter().map(|r| r.0 as u32).collect();
        let intersection = hnsw_ids.intersection(&brute_ids).count();
        total_recall += intersection as f64 / k as f64;
    }
    let avg_recall = total_recall / n_queries as f64;

    // Report results
    let hnsw_qps = n_queries as f64 / hnsw_time.as_secs_f64();
    let brute_qps = n_queries as f64 / brute_time.as_secs_f64();
    let speedup = brute_time.as_secs_f64() / hnsw_time.as_secs_f64();

    println!("HNSW:        {:>8.1} QPS ({:?} total)", hnsw_qps, hnsw_time);
    println!(
        "Brute force: {:>8.1} QPS ({:?} total)",
        brute_qps, brute_time
    );
    println!("Speedup:     {:>8.1}x", speedup);
    println!("Recall@{}:   {:>8.1}%", k, avg_recall * 100.0);
    println!();

    Ok(())
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
