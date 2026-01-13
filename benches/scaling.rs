//! Scaling benchmarks: how performance changes with dataset size and dimensions.
//!
//! Key questions:
//! - How does construction time scale with n?
//! - How does query time scale with n?
//! - How does query time scale with dimension?
//! - At what n does brute force become slower than ANN?

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn l2_distance_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Simple HNSW for benchmarking
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

        for &neighbor_id in &selected {
            self.neighbors[neighbor_id as usize].push(id);
            if self.neighbors[neighbor_id as usize].len() > self.m * 2 {
                let neighbor_vec = &self.vectors[neighbor_id as usize];
                let mut with_dist: Vec<(u32, f32)> = self.neighbors[neighbor_id as usize]
                    .iter()
                    .map(|&n| (n, l2_distance_squared(neighbor_vec, &self.vectors[n as usize])))
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

/// Brute force k-NN
fn brute_force_knn(query: &[f32], database: &[Vec<f32>], k: usize) -> Vec<(u32, f32)> {
    let mut distances: Vec<(u32, f32)> = database
        .iter()
        .enumerate()
        .map(|(i, v)| (i as u32, l2_distance_squared(query, v)))
        .collect();
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances.truncate(k);
    distances
}

fn create_dataset(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.random::<f32>()).collect())
        .collect()
}

/// Benchmark query time scaling with dataset size.
fn bench_query_scaling_n(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_scaling_n");
    group.sample_size(20);

    let dimension = 128;
    let k = 10;
    let ef = 64;
    let m = 16;

    for n in [1_000usize, 2_000, 5_000, 10_000, 20_000] {
        let database = create_dataset(n, dimension, 42);
        let queries = create_dataset(10, dimension, 123);

        // Build index
        let mut index = SimpleHnsw::new(m);
        for vec in &database {
            index.insert(vec.clone());
        }

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("hnsw", n), &n, |b, _| {
            b.iter(|| {
                for query in &queries {
                    black_box(index.search(query, k, ef));
                }
            })
        });

        // Only benchmark brute force for smaller sizes
        if n <= 10_000 {
            group.bench_with_input(BenchmarkId::new("brute", n), &n, |b, _| {
                b.iter(|| {
                    for query in &queries {
                        black_box(brute_force_knn(query, &database, k));
                    }
                })
            });
        }
    }

    group.finish();
}

/// Benchmark query time scaling with dimension.
fn bench_query_scaling_dim(c: &mut Criterion) {
    let mut group = c.benchmark_group("query_scaling_dim");
    group.sample_size(20);

    let n = 10_000;
    let k = 10;
    let ef = 64;
    let m = 16;

    for dim in [32, 64, 128, 256, 512] {
        let database = create_dataset(n, dim, 42);
        let queries = create_dataset(10, dim, 123);

        let mut index = SimpleHnsw::new(m);
        for vec in &database {
            index.insert(vec.clone());
        }

        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("hnsw", dim), &dim, |b, _| {
            b.iter(|| {
                for query in &queries {
                    black_box(index.search(query, k, ef));
                }
            })
        });
    }

    group.finish();
}

/// Benchmark construction time scaling.
fn bench_construction_scaling(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction_scaling");
    group.sample_size(10); // Construction is slow

    let dimension = 128;
    let m = 16;

    for n in [500usize, 1_000, 2_000, 5_000] {
        let database = create_dataset(n, dimension, 42);

        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("hnsw", n), &n, |b, _| {
            b.iter(|| {
                let mut index = SimpleHnsw::new(m);
                for vec in &database {
                    index.insert(vec.clone());
                }
                index
            })
        });
    }

    group.finish();
}

/// Find crossover point where HNSW beats brute force.
fn bench_crossover_point(c: &mut Criterion) {
    let mut group = c.benchmark_group("crossover_point");
    group.sample_size(20);

    let dimension = 128;
    let k = 10;
    let ef = 64;
    let m = 16;

    // Test small sizes to find where HNSW becomes faster
    for n in [100usize, 200, 500, 1_000, 2_000, 5_000] {
        let database = create_dataset(n, dimension, 42);
        let queries = create_dataset(10, dimension, 123);

        let mut index = SimpleHnsw::new(m);
        for vec in &database {
            index.insert(vec.clone());
        }

        group.bench_with_input(BenchmarkId::new("hnsw", n), &n, |b, _| {
            b.iter(|| {
                for query in &queries {
                    black_box(index.search(query, k, ef));
                }
            })
        });

        group.bench_with_input(BenchmarkId::new("brute", n), &n, |b, _| {
            b.iter(|| {
                for query in &queries {
                    black_box(brute_force_knn(query, &database, k));
                }
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_query_scaling_n,
    bench_query_scaling_dim,
    bench_construction_scaling,
    bench_crossover_point,
);
criterion_main!(benches);
