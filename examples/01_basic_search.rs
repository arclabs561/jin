//! Basic HNSW Search
//!
//! The minimal example: build an index, search it.
//!
//! ```bash
//! cargo run --example 01_basic_search --release
//! ```

use plesio::hnsw::HNSWIndex;

fn main() -> plesio::Result<()> {
    // Sample data: 100 vectors of dimension 64
    let dim = 64;
    let n = 100;

    // Generate normalized vectors (important for cosine similarity!)
    let vectors: Vec<Vec<f32>> = (0..n).map(|i| normalize(&random_vec(dim, i))).collect();

    // 1. Create index
    //    - dim: vector dimension
    //    - M=16: edges per node (affects recall/memory)
    //    - M_max=32: max edges at layer 0
    let mut index = HNSWIndex::new(dim, 16, 32)?;

    // 2. Add vectors
    for (id, vec) in vectors.iter().enumerate() {
        index.add(id as u32, vec.clone())?;
    }

    // 3. Build the graph
    index.build()?;

    // 4. Search
    //    - query: the vector to find neighbors for
    //    - k=5: return 5 nearest neighbors
    //    - ef=50: search width (higher = better recall, slower)
    let query = &vectors[42]; // Use vector 42 as query
    let results = index.search(query, 5, 50)?;

    println!("Query: vector 42");
    println!("Top 5 nearest neighbors:");
    for (id, distance) in &results {
        let similarity = 1.0 - distance; // cosine distance -> similarity
        println!("  id={:3}, similarity={:.4}", id, similarity);
    }

    // The query itself should be the closest (similarity ~1.0)
    println!("\nClosest match: id={} (expected: 42)", results[0].0);

    Ok(())
}

fn random_vec(dim: usize, seed: usize) -> Vec<f32> {
    (0..dim)
        .map(|i| ((seed * 31 + i * 17) as f32 * 0.001).sin())
        .collect()
}

fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}
