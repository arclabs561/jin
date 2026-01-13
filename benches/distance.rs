//! Benchmarks for distance computations.
//!
//! These benchmarks measure the core distance functions that dominate
//! ANN search performance.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::prelude::*;

// === Distance Functions ===

/// Euclidean (L2) distance squared.
fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Dot product.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Cosine distance (1 - cosine_similarity).
fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot = dot_product(a, b);
    let norm_a = dot_product(a, a).sqrt();
    let norm_b = dot_product(b, b).sqrt();
    1.0 - dot / (norm_a * norm_b + 1e-10)
}

/// Inner product distance (negative dot product for max similarity).
fn inner_product_distance(a: &[f32], b: &[f32]) -> f32 {
    -dot_product(a, b)
}

// === Generators ===

fn random_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(42);
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect())
        .collect()
}

fn normalized_vectors(n: usize, dim: usize) -> Vec<Vec<f32>> {
    random_vectors(n, dim)
        .into_iter()
        .map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            v.into_iter().map(|x| x / (norm + 1e-10)).collect()
        })
        .collect()
}

// === Benchmarks ===

fn bench_l2_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("l2_squared");

    for dim in [64, 128, 256, 384, 768, 1536].iter() {
        group.throughput(Throughput::Elements(*dim as u64));

        let vectors = random_vectors(2, *dim);
        let a = &vectors[0];
        let b = &vectors[1];

        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |bench, _| {
            bench.iter(|| l2_squared(black_box(a), black_box(b)));
        });
    }

    group.finish();
}

fn bench_dot_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("dot_product");

    for dim in [64, 128, 256, 384, 768, 1536].iter() {
        group.throughput(Throughput::Elements(*dim as u64));

        let vectors = random_vectors(2, *dim);
        let a = &vectors[0];
        let b = &vectors[1];

        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |bench, _| {
            bench.iter(|| dot_product(black_box(a), black_box(b)));
        });
    }

    group.finish();
}

fn bench_cosine_dimensions(c: &mut Criterion) {
    let mut group = c.benchmark_group("cosine_distance");

    for dim in [64, 128, 256, 384, 768, 1536].iter() {
        group.throughput(Throughput::Elements(*dim as u64));

        let vectors = normalized_vectors(2, *dim);
        let a = &vectors[0];
        let b = &vectors[1];

        group.bench_with_input(BenchmarkId::from_parameter(dim), dim, |bench, _| {
            bench.iter(|| cosine_distance(black_box(a), black_box(b)));
        });
    }

    group.finish();
}

fn bench_batch_distances(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_l2");

    let dim = 384; // Common embedding dimension

    for n in [10, 100, 1000].iter() {
        group.throughput(Throughput::Elements(*n as u64));

        let vectors = random_vectors(*n + 1, dim);
        let query = &vectors[0];
        let candidates: Vec<&[f32]> = vectors[1..].iter().map(|v| v.as_slice()).collect();

        group.bench_with_input(BenchmarkId::from_parameter(n), n, |bench, _| {
            bench.iter(|| {
                candidates
                    .iter()
                    .map(|c| l2_squared(black_box(query), black_box(c)))
                    .collect::<Vec<_>>()
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_l2_dimensions,
    bench_dot_dimensions,
    bench_cosine_dimensions,
    bench_batch_distances,
);
criterion_main!(benches);
