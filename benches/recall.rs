//! Recall vs Latency benchmarks.
//!
//! Measures the fundamental ANN tradeoff: how much accuracy do you sacrifice for speed?
//! Generates recall@k curves at various ef_search/nprobe settings.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

// Inline distance and recall to avoid depending on lib structure
fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

fn compute_ground_truth(query: &[f32], database: &[Vec<f32>], k: usize) -> Vec<u32> {
    let mut distances: Vec<(u32, f32)> = database
        .iter()
        .enumerate()
        .map(|(i, vec)| (i as u32, l2_distance_squared(query, vec)))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.into_iter().take(k).map(|(id, _)| id).collect()
}

fn recall_at_k(ground_truth: &[u32], retrieved: &[u32], k: usize) -> f32 {
    let gt_set: HashSet<u32> = ground_truth.iter().take(k).copied().collect();
    let ret_set: HashSet<u32> = retrieved.iter().take(k).copied().collect();
    gt_set.intersection(&ret_set).count() as f32 / k as f32
}

/// Simple HNSW for benchmarking (same as in hnsw.rs bench)
struct SimpleHnsw {
    vectors: Vec<Vec<f32>>,
    neighbors: Vec<Vec<u32>>,
    m: usize,
}

impl SimpleHnsw {
    fn new(m: usize) -> Self {
        Self {
            vectors: Vec::new(),
            neighbors: Vec::new(),
            m,
        }
    }

    fn insert(&mut self, vec: Vec<f32>) {
        let id = self.vectors.len() as u32;
        self.vectors.push(vec);

        if self.vectors.len() == 1 {
            self.neighbors.push(Vec::new());
            return;
        }

        // Find nearest neighbors among existing vectors
        let query = &self.vectors[id as usize];
        let mut candidates: Vec<(u32, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .take(id as usize)
            .map(|(i, v)| (i as u32, l2_distance_squared(query, v)))
            .collect();
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let selected: Vec<u32> = candidates
            .iter()
            .take(self.m)
            .map(|(idx, _)| *idx)
            .collect();

        // Add bidirectional edges
        for &neighbor_id in &selected {
            self.neighbors[neighbor_id as usize].push(id);
            // Prune if too many
            if self.neighbors[neighbor_id as usize].len() > self.m * 2 {
                let neighbor_vec = &self.vectors[neighbor_id as usize];
                let mut with_dist: Vec<(u32, f32)> = self.neighbors[neighbor_id as usize]
                    .iter()
                    .map(|&n| {
                        (
                            n,
                            l2_distance_squared(neighbor_vec, &self.vectors[n as usize]),
                        )
                    })
                    .collect();
                with_dist.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
                self.neighbors[neighbor_id as usize] =
                    with_dist.iter().take(self.m).map(|(n, _)| *n).collect();
            }
        }

        self.neighbors.push(selected);
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        if self.vectors.is_empty() {
            return Vec::new();
        }

        let mut visited = vec![false; self.vectors.len()];
        let mut candidates: Vec<(u32, f32)> = Vec::with_capacity(ef);

        // Start from random entry point
        let entry = 0u32;
        let dist = l2_distance_squared(query, &self.vectors[0]);
        candidates.push((entry, dist));
        visited[0] = true;

        let mut i = 0;
        while i < candidates.len() && candidates.len() < ef {
            let (current, _) = candidates[i];
            for &neighbor in &self.neighbors[current as usize] {
                if !visited[neighbor as usize] {
                    visited[neighbor as usize] = true;
                    let d = l2_distance_squared(query, &self.vectors[neighbor as usize]);
                    candidates.push((neighbor, d));
                }
            }
            i += 1;
        }

        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        candidates.truncate(k);
        candidates
    }
}

fn create_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
        .collect()
}

/// Benchmark recall at various ef_search values.
fn bench_recall_vs_ef(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_vs_ef");
    group.sample_size(20); // Fewer samples for expensive benchmarks

    let n_vectors = 5000;
    let n_queries = 50;
    let dimension = 128;
    let k = 10;
    let m = 16;

    // Build index
    let database = create_dataset(n_vectors, dimension, 42);
    let queries = create_dataset(n_queries, dimension, 123);

    let mut index = SimpleHnsw::new(m);
    for vec in &database {
        index.insert(vec.clone());
    }

    // Precompute ground truth
    let ground_truths: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| compute_ground_truth(q, &database, k))
        .collect();

    for ef in [16, 32, 64, 128, 256] {
        group.bench_with_input(BenchmarkId::new("ef", ef), &ef, |b, &ef| {
            b.iter(|| {
                let mut total_recall = 0.0;
                for (i, query) in queries.iter().enumerate() {
                    let results = index.search(black_box(query), k, ef);
                    let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
                    total_recall += recall_at_k(&ground_truths[i], &retrieved, k);
                }
                total_recall / queries.len() as f32
            })
        });
    }

    group.finish();
}

/// Benchmark recall at various k values.
fn bench_recall_vs_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_vs_k");
    group.sample_size(20);

    let n_vectors = 5000;
    let n_queries = 50;
    let dimension = 128;
    let m = 16;
    let ef = 64;

    let database = create_dataset(n_vectors, dimension, 42);
    let queries = create_dataset(n_queries, dimension, 123);

    let mut index = SimpleHnsw::new(m);
    for vec in &database {
        index.insert(vec.clone());
    }

    for k in [1, 5, 10, 20, 50, 100] {
        let ground_truths: Vec<Vec<u32>> = queries
            .iter()
            .map(|q| compute_ground_truth(q, &database, k))
            .collect();

        group.bench_with_input(BenchmarkId::new("k", k), &k, |b, &k| {
            b.iter(|| {
                let mut total_recall = 0.0;
                for (i, query) in queries.iter().enumerate() {
                    let results = index.search(black_box(query), k, ef.max(k));
                    let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
                    total_recall += recall_at_k(&ground_truths[i], &retrieved, k);
                }
                total_recall / queries.len() as f32
            })
        });
    }

    group.finish();
}

/// Measure actual recall values (not just timing).
/// This prints recall statistics for documentation.
fn bench_recall_measurement(c: &mut Criterion) {
    let mut group = c.benchmark_group("recall_measurement");
    group.sample_size(10);

    let n_vectors = 5000;
    let n_queries = 100;
    let dimension = 128;
    let k = 10;
    let m = 16;

    let database = create_dataset(n_vectors, dimension, 42);
    let queries = create_dataset(n_queries, dimension, 123);

    let mut index = SimpleHnsw::new(m);
    for vec in &database {
        index.insert(vec.clone());
    }

    let ground_truths: Vec<Vec<u32>> = queries
        .iter()
        .map(|q| compute_ground_truth(q, &database, k))
        .collect();

    // Compute and print recall at various ef values
    for ef in [16, 32, 64, 128, 256] {
        let recalls: Vec<f32> = queries
            .iter()
            .enumerate()
            .map(|(i, query)| {
                let results = index.search(query, k, ef);
                let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
                recall_at_k(&ground_truths[i], &retrieved, k)
            })
            .collect();

        let mean_recall = recalls.iter().sum::<f32>() / recalls.len() as f32;
        eprintln!("ef={}: recall@{}={:.3}", ef, k, mean_recall);
    }

    // Benchmark a single representative case
    group.bench_function("measure_ef64", |b| {
        b.iter(|| {
            queries
                .iter()
                .enumerate()
                .map(|(i, query)| {
                    let results = index.search(black_box(query), k, 64);
                    let retrieved: Vec<u32> = results.iter().map(|(id, _)| *id).collect();
                    recall_at_k(&ground_truths[i], &retrieved, k)
                })
                .sum::<f32>()
                / queries.len() as f32
        })
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_recall_vs_ef,
    bench_recall_vs_k,
    bench_recall_measurement,
);
criterion_main!(benches);
