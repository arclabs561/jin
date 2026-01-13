//! GloVe-25 Real ANN Benchmark
//!
//! Benchmarks HNSW against the GloVe-25-angular dataset from ann-benchmarks.
//! This is a standard benchmark with 1.18M word embeddings and ground truth.
//!
//! Dataset: http://ann-benchmarks.com/glove-25-angular.hdf5
//! Vectors: 1,183,514 train + 10,000 test queries
//! Dimensions: 25
//! Distance: Angular (cosine)
//!
//! Run:
//! ```bash
//! cargo run --example glove_benchmark --release
//! ```

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

use vicinity::hnsw::HNSWIndex;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("GloVe-25 ANN Benchmark (Real Data)");
    println!("==================================\n");

    // Try multiple possible locations for data
    let data_paths = [
        "data",                     // relative to example
        "vicinity/data",            // from workspace root
        "../vicinity/data",         // from other crate
        env!("CARGO_MANIFEST_DIR"), // manifest dir
    ];

    let data_dir = data_paths
        .iter()
        .map(|p| {
            if *p == env!("CARGO_MANIFEST_DIR") {
                format!("{}/data", p)
            } else {
                p.to_string()
            }
        })
        .find(|p| Path::new(p).join("glove25_train.bin").exists())
        .unwrap_or_else(|| {
            eprintln!("GloVe-25 data not found. To generate:");
            eprintln!();
            eprintln!(
                "  1. Download: curl -o glove.hdf5 http://ann-benchmarks.com/glove-25-angular.hdf5"
            );
            eprintln!("  2. Convert: python scripts/convert_hdf5.py glove.hdf5 data/");
            eprintln!();
            eprintln!("Or run with synthetic data: cargo run --example hnsw_benchmark --release");
            std::process::exit(1);
        });

    println!("Data directory: {}\n", data_dir);

    // Load train vectors
    let train_path = format!("{}/glove25_train.bin", data_dir);
    let (train, dim) = load_vectors(&train_path)?;
    println!("Train: {} vectors x {} dims", train.len(), dim);

    // Load test queries
    let test_path = format!("{}/glove25_test.bin", data_dir);
    let (test, _) = load_vectors(&test_path)?;
    println!("Test:  {} queries", test.len());

    // Load ground truth neighbors
    let neighbors_path = format!("{}/glove25_neighbors.bin", data_dir);
    let (neighbors, k_gt) = load_neighbors(&neighbors_path)?;
    println!("Ground truth: {} neighbors per query\n", k_gt);

    // Build HNSW index
    let m = 16; // connections per layer
    let ef_build = 200; // construction beam width

    println!(
        "Building HNSW index (M={}, ef_construction={})...",
        m, ef_build
    );
    let build_start = Instant::now();

    let mut index = HNSWIndex::new(dim, m, ef_build)?;
    for (i, vec) in train.iter().enumerate() {
        index.add(i as u32, vec.clone())?;
    }
    index.build()?;

    let build_time = build_start.elapsed();
    println!(
        "Build time: {:.2}s ({:.0} vectors/sec)\n",
        build_time.as_secs_f64(),
        train.len() as f64 / build_time.as_secs_f64()
    );

    // Benchmark at different ef values
    let k = 10;
    println!(
        "{:>8} {:>12} {:>12} {:>10}",
        "ef", "QPS", "Latency", "Recall@10"
    );
    println!("{}", "-".repeat(50));

    for ef in [50, 100, 200, 400, 800] {
        let search_start = Instant::now();
        let mut total_recall = 0.0;

        for (i, query) in test.iter().enumerate() {
            let results = index.search(query, k, ef)?;

            // Calculate recall against ground truth
            let gt: HashSet<i32> = neighbors[i].iter().take(k).copied().collect();
            let found: HashSet<i32> = results.iter().map(|r| r.0 as i32).collect();
            let intersection = gt.intersection(&found).count();
            total_recall += intersection as f64 / k as f64;
        }

        let search_time = search_start.elapsed();
        let qps = test.len() as f64 / search_time.as_secs_f64();
        let latency_us = search_time.as_micros() as f64 / test.len() as f64;
        let recall = total_recall / test.len() as f64 * 100.0;

        println!(
            "{:>8} {:>12.0} {:>10.0}us {:>9.1}%",
            ef, qps, latency_us, recall
        );
    }

    Ok(())
}

/// Load vectors from binary format: VEC1 + n(u32) + d(u32) + data(f32)
fn load_vectors(path: &str) -> Result<(Vec<Vec<f32>>, usize), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"VEC1" {
        return Err("Invalid vector file format".into());
    }

    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;
    let n = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let d = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    // Read data
    let mut data = vec![0u8; n * d * 4];
    reader.read_exact(&mut data)?;

    // Convert to Vec<Vec<f32>>
    let vectors: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..d)
                .map(|j| {
                    let offset = (i * d + j) * 4;
                    f32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ])
                })
                .collect()
        })
        .collect();

    Ok((vectors, d))
}

/// Load neighbors from binary format: NBR1 + n(u32) + k(u32) + data(i32)
fn load_neighbors(path: &str) -> Result<(Vec<Vec<i32>>, usize), Box<dyn std::error::Error>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);

    // Read header
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"NBR1" {
        return Err("Invalid neighbors file format".into());
    }

    let mut header = [0u8; 8];
    reader.read_exact(&mut header)?;
    let n = u32::from_le_bytes([header[0], header[1], header[2], header[3]]) as usize;
    let k = u32::from_le_bytes([header[4], header[5], header[6], header[7]]) as usize;

    // Read data
    let mut data = vec![0u8; n * k * 4];
    reader.read_exact(&mut data)?;

    // Convert to Vec<Vec<i32>>
    let neighbors: Vec<Vec<i32>> = (0..n)
        .map(|i| {
            (0..k)
                .map(|j| {
                    let offset = (i * k + j) * 4;
                    i32::from_le_bytes([
                        data[offset],
                        data[offset + 1],
                        data[offset + 2],
                        data[offset + 3],
                    ])
                })
                .collect()
        })
        .collect();

    Ok((neighbors, k))
}
