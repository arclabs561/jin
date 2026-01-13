//! Benchmarks for HNSW index construction and search.
//!
//! These benchmarks measure end-to-end performance on synthetic data.
//! For reproducible comparisons with ann-benchmarks, use standardized
//! datasets (SIFT-1M, GloVe, etc.) via the data loading utilities.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;
use std::collections::BinaryHeap;

// === Synthetic Data Generation ===

fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>()).collect())
        .collect()
}

fn normalized_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    random_vectors(n, dim, seed)
        .into_iter()
        .map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / (norm + 1e-10)).collect()
        })
        .collect()
}

// === Distance Functions ===

fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

// === Simplified HNSW for Benchmarking ===
//
// This is a minimal HNSW implementation for benchmarking purposes.
// The real implementation is in src/hnsw/ with more features.

#[derive(Clone)]
struct HnswNode {
    vector: Vec<f32>,
    neighbors: Vec<Vec<u32>>, // neighbors[level]
}

struct SimpleHnsw {
    nodes: Vec<HnswNode>,
    entry_point: u32,
    max_level: usize,
    m: usize, // max neighbors per layer
    ef_construction: usize,
    ml: f64, // level multiplier
}

impl SimpleHnsw {
    fn new(m: usize, ef_construction: usize) -> Self {
        Self {
            nodes: Vec::new(),
            entry_point: 0,
            max_level: 0,
            m,
            ef_construction,
            ml: 1.0 / (m as f64).ln(),
        }
    }

    fn random_level(&self, rng: &mut impl Rng) -> usize {
        let r: f64 = rng.gen();
        ((-r.ln() * self.ml).floor() as usize).min(16)
    }

    fn insert(&mut self, vector: Vec<f32>, rng: &mut impl Rng) {
        let id = self.nodes.len() as u32;
        let level = self.random_level(rng);

        // Ensure levels exist
        while level >= self.max_level {
            self.max_level += 1;
        }

        let mut neighbors = vec![Vec::new(); level + 1];

        if self.nodes.is_empty() {
            self.nodes.push(HnswNode { vector, neighbors });
            return;
        }

        // Find entry point through upper layers (greedy search)
        let mut current = self.entry_point;
        for l in (level + 1..=self.max_level).rev() {
            current = self.search_layer_entry(&vector, current, l);
        }

        // Insert at each level
        for l in (0..=level).rev() {
            let candidates = self.search_layer(&vector, current, self.ef_construction, l);
            let selected = self.select_neighbors(&vector, &candidates, self.m);

            neighbors[l] = selected.clone();

            // Add bidirectional links
            for &neighbor_id in &selected {
                let neighbor = &mut self.nodes[neighbor_id as usize];
                if neighbor.neighbors.len() > l {
                    if neighbor.neighbors[l].len() < self.m * 2 {
                        neighbor.neighbors[l].push(id);
                    }
                }
            }

            if !candidates.is_empty() {
                current = candidates[0].0;
            }
        }

        self.nodes.push(HnswNode { vector, neighbors });

        // Update entry point if new node has higher level
        if level > self.max_level - 1 {
            self.entry_point = id;
        }
    }

    fn search_layer_entry(&self, query: &[f32], entry: u32, level: usize) -> u32 {
        let mut current = entry;
        let mut current_dist = l2_squared(query, &self.nodes[current as usize].vector);

        loop {
            let mut improved = false;
            let node = &self.nodes[current as usize];

            if node.neighbors.len() > level {
                for &neighbor in &node.neighbors[level] {
                    let dist = l2_squared(query, &self.nodes[neighbor as usize].vector);
                    if dist < current_dist {
                        current = neighbor;
                        current_dist = dist;
                        improved = true;
                    }
                }
            }

            if !improved {
                break;
            }
        }

        current
    }

    fn search_layer(&self, query: &[f32], entry: u32, ef: usize, level: usize) -> Vec<(u32, f32)> {
        let mut visited = std::collections::HashSet::new();
        let mut candidates: BinaryHeap<std::cmp::Reverse<(ordered_float::OrderedFloat<f32>, u32)>> =
            BinaryHeap::new();
        let mut results: BinaryHeap<(ordered_float::OrderedFloat<f32>, u32)> = BinaryHeap::new();

        let dist = l2_squared(query, &self.nodes[entry as usize].vector);
        visited.insert(entry);
        candidates.push(std::cmp::Reverse((
            ordered_float::OrderedFloat(dist),
            entry,
        )));
        results.push((ordered_float::OrderedFloat(dist), entry));

        while let Some(std::cmp::Reverse((_, current))) = candidates.pop() {
            let node = &self.nodes[current as usize];

            // Get worst result distance for pruning
            let worst_dist = if results.len() >= ef {
                results.peek().map(|(d, _)| d.0).unwrap_or(f32::INFINITY)
            } else {
                f32::INFINITY
            };

            if node.neighbors.len() > level {
                for &neighbor in &node.neighbors[level] {
                    if visited.insert(neighbor) {
                        let dist = l2_squared(query, &self.nodes[neighbor as usize].vector);

                        if dist < worst_dist || results.len() < ef {
                            candidates.push(std::cmp::Reverse((
                                ordered_float::OrderedFloat(dist),
                                neighbor,
                            )));
                            results.push((ordered_float::OrderedFloat(dist), neighbor));

                            while results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        results.into_iter().map(|(d, id)| (id, d.0)).collect()
    }

    fn select_neighbors(&self, _query: &[f32], candidates: &[(u32, f32)], m: usize) -> Vec<u32> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        sorted.into_iter().take(m).map(|(id, _)| id).collect()
    }

    fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(u32, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        // Find entry through upper layers
        let mut current = self.entry_point;
        for l in (1..=self.max_level).rev() {
            current = self.search_layer_entry(query, current, l);
        }

        // Search at layer 0
        let mut results = self.search_layer(query, current, ef, 0);
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        results.truncate(k);
        results
    }
}

// === Benchmarks ===

fn bench_hnsw_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_construction");

    let dim = 128;

    for n in [1000, 5000, 10000].iter() {
        group.throughput(Throughput::Elements(*n as u64));

        let vectors = random_vectors(*n, dim, 42);

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |bench, _| {
            bench.iter(|| {
                let mut hnsw = SimpleHnsw::new(16, 100);
                let mut rng = StdRng::seed_from_u64(42);
                for v in &vectors {
                    hnsw.insert(black_box(v.clone()), &mut rng);
                }
                hnsw
            });
        });
    }

    group.finish();
}

fn bench_hnsw_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search");

    let dim = 128;
    let n_vectors = 10000;
    let n_queries = 100;

    // Build index once
    let vectors = random_vectors(n_vectors, dim, 42);
    let queries = random_vectors(n_queries, dim, 123);

    let mut hnsw = SimpleHnsw::new(16, 100);
    let mut rng = StdRng::seed_from_u64(42);
    for v in &vectors {
        hnsw.insert(v.clone(), &mut rng);
    }

    for ef in [10, 50, 100, 200].iter() {
        group.throughput(Throughput::Elements(n_queries as u64));

        group.bench_with_input(BenchmarkId::new("ef", ef), ef, |bench, &ef| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|q| hnsw.search(black_box(q), 10, ef))
                    .collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

fn bench_hnsw_search_k(c: &mut Criterion) {
    let mut group = c.benchmark_group("hnsw_search_k");

    let dim = 128;
    let n_vectors = 10000;
    let n_queries = 100;

    // Build index once
    let vectors = random_vectors(n_vectors, dim, 42);
    let queries = random_vectors(n_queries, dim, 123);

    let mut hnsw = SimpleHnsw::new(16, 100);
    let mut rng = StdRng::seed_from_u64(42);
    for v in &vectors {
        hnsw.insert(v.clone(), &mut rng);
    }

    for k in [1, 10, 50, 100].iter() {
        group.throughput(Throughput::Elements(n_queries as u64));

        group.bench_with_input(BenchmarkId::new("k", k), k, |bench, &k| {
            bench.iter(|| {
                queries
                    .iter()
                    .map(|q| hnsw.search(black_box(q), k, 100))
                    .collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_hnsw_construction,
    bench_hnsw_search,
    bench_hnsw_search_k,
);
criterion_main!(benches);
