//! Memory footprint benchmarks.
//!
//! Measures bytes per vector for different index configurations.
//! This is a "data" benchmark - it computes sizes rather than timing.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use std::mem::size_of;

/// Compute theoretical HNSW memory.
fn hnsw_memory(n_vectors: usize, dimension: usize, m: usize) -> (usize, usize, f64) {
    let raw_bytes = n_vectors * dimension * size_of::<f32>();

    // Graph: ~2.5 * M edges per node on average
    let avg_edges = (2.5 * m as f64) as usize;
    let graph_bytes = n_vectors * avg_edges * size_of::<u32>();
    let metadata_bytes = n_vectors * (size_of::<u32>() + size_of::<u8>());

    let total_bytes = raw_bytes + graph_bytes + metadata_bytes;
    let bytes_per_vec = total_bytes as f64 / n_vectors as f64;

    (raw_bytes, total_bytes, bytes_per_vec)
}

/// Compute theoretical IVF-PQ memory.
fn ivf_pq_memory(
    n_vectors: usize,
    dimension: usize,
    n_clusters: usize,
    n_subquantizers: usize,
) -> (usize, usize, f64, f64) {
    let raw_bytes = n_vectors * dimension * size_of::<f32>();

    // PQ codes: 1 byte per subquantizer
    let pq_bytes = n_vectors * n_subquantizers;

    // Codebooks: 256 centroids per subquantizer
    let sub_dim = dimension / n_subquantizers;
    let codebook_bytes = n_subquantizers * 256 * sub_dim * size_of::<f32>();

    // Cluster centroids
    let centroid_bytes = n_clusters * dimension * size_of::<f32>();

    // Inverted lists (just IDs)
    let invlist_bytes = n_vectors * size_of::<u32>();

    let total_bytes = pq_bytes + codebook_bytes + centroid_bytes + invlist_bytes;
    let bytes_per_vec = total_bytes as f64 / n_vectors as f64;
    let compression = raw_bytes as f64 / pq_bytes as f64;

    (raw_bytes, total_bytes, bytes_per_vec, compression)
}

/// Benchmark memory scaling with dataset size.
fn bench_memory_vs_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_vs_size");

    let dimension = 128;
    let m = 16;

    for n in [1_000usize, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(n as u64));

        group.bench_with_input(BenchmarkId::new("hnsw", n), &n, |b, &n| {
            b.iter(|| hnsw_memory(n, dimension, m))
        });
    }

    // Print summary
    eprintln!("\n=== HNSW Memory (dim={}, m={}) ===", dimension, m);
    eprintln!(
        "{:>12} {:>12} {:>12} {:>12}",
        "n_vectors", "raw_MB", "total_MB", "bytes/vec"
    );
    for n in [1_000usize, 10_000, 100_000, 1_000_000] {
        let (raw, total, bpv) = hnsw_memory(n, dimension, m);
        eprintln!(
            "{:>12} {:>12.2} {:>12.2} {:>12.1}",
            n,
            raw as f64 / 1e6,
            total as f64 / 1e6,
            bpv
        );
    }

    group.finish();
}

/// Benchmark memory scaling with dimension.
fn bench_memory_vs_dimension(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_vs_dimension");

    let n_vectors = 100_000;
    let m = 16;

    for dim in [32, 64, 128, 256, 512, 768, 1536] {
        group.throughput(Throughput::Elements(dim as u64));

        group.bench_with_input(BenchmarkId::new("hnsw", dim), &dim, |b, &dim| {
            b.iter(|| hnsw_memory(n_vectors, dim, m))
        });
    }

    // Print summary
    eprintln!("\n=== HNSW Memory (n={}, m={}) ===", n_vectors, m);
    eprintln!(
        "{:>8} {:>12} {:>12} {:>12}",
        "dim", "raw_MB", "total_MB", "bytes/vec"
    );
    for dim in [32, 64, 128, 256, 512, 768, 1536] {
        let (raw, total, bpv) = hnsw_memory(n_vectors, dim, m);
        eprintln!(
            "{:>8} {:>12.2} {:>12.2} {:>12.1}",
            dim,
            raw as f64 / 1e6,
            total as f64 / 1e6,
            bpv
        );
    }

    group.finish();
}

/// Benchmark IVF-PQ compression ratios.
fn bench_ivfpq_compression(c: &mut Criterion) {
    let mut group = c.benchmark_group("ivfpq_compression");

    let n_vectors = 100_000;
    let dimension = 128;
    let n_clusters = 256;

    for n_sq in [8, 16, 32, 64] {
        if dimension % n_sq != 0 {
            continue;
        }

        group.bench_with_input(
            BenchmarkId::new("subquantizers", n_sq),
            &n_sq,
            |b, &n_sq| b.iter(|| ivf_pq_memory(n_vectors, dimension, n_clusters, n_sq)),
        );
    }

    // Print summary
    eprintln!(
        "\n=== IVF-PQ Memory (n={}, dim={}, clusters={}) ===",
        n_vectors, dimension, n_clusters
    );
    eprintln!(
        "{:>8} {:>12} {:>12} {:>12} {:>12}",
        "n_sq", "raw_MB", "total_MB", "bytes/vec", "compress"
    );
    for n_sq in [8, 16, 32, 64] {
        if dimension % n_sq != 0 {
            continue;
        }
        let (raw, total, bpv, compress) = ivf_pq_memory(n_vectors, dimension, n_clusters, n_sq);
        eprintln!(
            "{:>8} {:>12.2} {:>12.2} {:>12.1} {:>12.1}x",
            n_sq,
            raw as f64 / 1e6,
            total as f64 / 1e6,
            bpv,
            compress
        );
    }

    group.finish();
}

/// Compare HNSW vs IVF-PQ memory.
fn bench_memory_comparison(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_comparison");

    let n_vectors = 1_000_000;
    let dimension = 128;

    group.bench_function("hnsw_m16", |b| {
        b.iter(|| hnsw_memory(n_vectors, dimension, 16))
    });

    group.bench_function("ivfpq_sq16", |b| {
        b.iter(|| ivf_pq_memory(n_vectors, dimension, 1024, 16))
    });

    // Print comparison
    let (hnsw_raw, hnsw_total, hnsw_bpv) = hnsw_memory(n_vectors, dimension, 16);
    let (ivfpq_raw, ivfpq_total, ivfpq_bpv, ivfpq_compress) =
        ivf_pq_memory(n_vectors, dimension, 1024, 16);

    eprintln!(
        "\n=== Memory Comparison (n={}, dim={}) ===",
        n_vectors, dimension
    );
    eprintln!("{:>12} {:>12} {:>12}", "Index", "total_MB", "bytes/vec");
    eprintln!(
        "{:>12} {:>12.2} {:>12.1}",
        "HNSW",
        hnsw_total as f64 / 1e6,
        hnsw_bpv
    );
    eprintln!(
        "{:>12} {:>12.2} {:>12.1}",
        "IVF-PQ",
        ivfpq_total as f64 / 1e6,
        ivfpq_bpv
    );
    eprintln!(
        "IVF-PQ achieves {:.1}x memory reduction vs raw vectors",
        ivfpq_compress
    );

    group.finish();
}

criterion_group!(
    benches,
    bench_memory_vs_size,
    bench_memory_vs_dimension,
    bench_ivfpq_compression,
    bench_memory_comparison,
);
criterion_main!(benches);
