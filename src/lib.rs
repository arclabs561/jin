// Crate-level lint configuration
// Dead code is allowed since this is research code with partial implementations
#![allow(dead_code)]
// Allow unsafe operations in unsafe fn without explicit unsafe blocks
// (Rust 2024 edition strictness - this is a SIMD crate where unsafe is pervasive)
#![allow(unsafe_op_in_unsafe_fn)]

//! vicinity: Approximate Nearest Neighbor Search primitives.
//!
//! Provides standalone implementations of state-of-the-art ANN algorithms:
//!
//! - **Graph-based**: [`hnsw`], [`nsw`], [`sng`], [`vamana`]
//! - **Hash-based**: `hash` (LSH, MinHash, SimHash) — requires `lsh` feature
//! - **Partition-based**: [`ivf_pq`], [`scann`]
//! - **Quantization**: [`quantization`] (PQ, RaBitQ)
//!
//! # Which Index Should I Use?
//!
//! | Situation | Recommendation | Feature |
//! |-----------|----------------|---------|
//! | **General Purpose** (Best Recall/Speed) | [`hnsw::HNSWIndex`] | `hnsw` (default) |
//! | **Billion-Scale** (Memory Constrained) | [`ivf_pq::IVFPQIndex`] | `ivf_pq` |
//! | **Flat Graph** (Simpler, d > 32) | [`nsw::NSWIndex`] | `nsw` |
//! | **Attribute Filtering** | [`hnsw::filtered`] | `hnsw` |
//! | **Out-of-Core** (SSD-based) | [`diskann`] | `diskann` (experimental) |
//!
//! **Default features**: `hnsw`, `innr` (SIMD).
//!
//! ## Recommendation Logic
//!
//! 1. **Start with HNSW**. It's the industry standard for a reason. It offers the best
//!    trade-off between search speed and recall for datasets that fit in RAM.
//!
//! 2. **Use IVF-PQ** if your dataset is too large for RAM (e.g., > 10M vectors on a laptop).
//!    It compresses vectors (32x-64x) but has lower recall than HNSW.
//!
//! 3. **Use NSW (Flat)** if you specifically want to avoid the hierarchy overhead of HNSW,
//!    or if your dimensionality is high enough (d > 32) that the "small world" navigation
//!    benefit of hierarchy is diminished. *Note: HNSW is generally more robust.*
//!
//! 4. **Use DiskANN** (experimental) if you have an NVMe SSD and 1B+ vectors.
//!
//! ```toml
//! # Minimal (HNSW + SIMD)
//! vicinity = "0.1"
//!
//! # With quantization support
//! vicinity = { version = "0.1", features = ["ivf_pq"] }
//! ```
//!
//! # Critical Nuances & The HNSW Critique (2025)
//!
//! ## 1. The HNSW Dominance & Its Cracks
//! HNSW is the default because it's "good enough" for most. But research (2024-2025)
//! highlights structural weaknesses:
//!
//! - **Local Minima**: HNSW's greedy search is prone to getting stuck in local optima,
//!   especially in clustered datasets. Newer graphs like **NSG** (Navigable Small World)
//!   and **Vamana** (DiskANN) use better edge selection (e.g., RNG) to ensure
//!   angular diversity, allowing "glances" around obstacles.
//! - **Memory Bloat**: The hierarchical layers add 30-40% overhead. For d > 32,
//!   the hierarchy often provides negligible speedup over a well-constructed flat graph.
//! - **Construction Sensitivity**: HNSW quality depends heavily on insertion order.
//!   Vamana's two-pass build (random graph -> refined) is more robust.
//!
//! **Verdict**: Use HNSW for RAM-based, low-latency search. Look at Vamana/DiskANN
//! for higher recall or SSD-resident data.
//!
//! ## 2. The Hubness Problem
//!
//! In high-dimensional spaces, some vectors become **hubs**—appearing as
//! nearest neighbors to many other points, while **antihubs** rarely appear
//! in any neighbor lists. This creates asymmetric NN relationships.
//!
//! **Mitigation**: Local scaling, cosine over Euclidean, or dimensionality reduction.
//!
//! ## 3. Curse of Dimensionality
//!
//! For k-NN with neighborhood radius ℓ=0.1, you need n ≈ k × 10^d samples.
//! For d > 100, this exceeds atoms in the observable universe.
//!
//! **Why ANN works anyway**: Real data lies on low-dimensional manifolds.
//! Intrinsic dimensionality << embedding dimensionality.
//!
//! ## 4. When Exact Search Beats Approximate
//!
//! - Small datasets (< 10K vectors): Brute force is faster (SIMD is powerful).
//! - Very high recall requirements (> 99.9%): ANN overhead not worth it.
//! - Low intrinsic dimensionality: KD-trees can be exact and fast.

pub mod ann;
pub mod classic;
pub mod diskann;
pub mod evoc;
pub mod hnsw;
pub mod ivf_pq;
pub mod nsw;
pub mod quantization;
pub mod scann;
pub mod sng;
pub mod vamana;

pub mod adaptive;
pub mod matryoshka;
pub mod partitioning;

pub mod filtering;
pub mod lid;
pub mod pq_simd;
pub mod simd;

// Hash-based methods (LSH, MinHash, SimHash)
#[cfg(feature = "lsh")]
pub mod hash;

// Re-exports
pub use ann::traits::ANNIndex;
pub use error::{Result, RetrieveError};

pub mod benchmark;
pub mod compression;
pub mod error;
pub mod persistence;
pub mod streaming;
