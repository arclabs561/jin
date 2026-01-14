// Crate-level lint configuration
// Dead code is allowed since this is research code with partial implementations
#![allow(dead_code)]
// Allow unsafe operations in unsafe fn without explicit unsafe blocks
// (Rust 2024 edition strictness - this is a SIMD crate where unsafe is pervasive)
#![allow(unsafe_op_in_unsafe_fn)]

//! plesio: Approximate Nearest Neighbor Search primitives.
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
//! plesio = "0.1"
//!
//! # With quantization support
//! plesio = { version = "0.1", features = ["ivf_pq"] }
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
//! - **Hub Highway Hypothesis (2025)**: Research demonstrated that flat NSW graphs
//!   retain *all* benefits of HNSW on high-d data. Hub nodes form a "hub highway"
//!   that serves the same routing function as explicit hierarchy layers.
//! - **Construction Sensitivity**: HNSW quality depends heavily on insertion order.
//!   LID-based ordering (descending) can improve recall by 12%. Vamana's two-pass
//!   build (random graph -> refined) is more robust.
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
//!
//! ## 5. Quantization Trade-offs (2024-2025 Research)
//!
//! | Method | Compression | Recall | Query Speedup | Use Case |
//! |--------|-------------|--------|---------------|----------|
//! | **Binary** | 32x | ~96% | 24x | Extreme speed, lower accuracy OK |
//! | **Scalar (INT8)** | 4x | ~99.3% | 4x | Balance of speed and accuracy |
//! | **Product (PQ)** | 8-64x | ~95-98% | 8-32x | Memory-constrained billion-scale |
//! | **OPQ** | 8-64x | ~97-99% | 8-32x | Better than PQ via rotation |
//!
//! Binary quantization uses Hamming distance (2 CPU cycles per comparison).
//! Matryoshka embeddings enable dimension reduction before quantization for
//! further compression (e.g., 256x total with MRL + binary).
//!
//! ## 6. Future Directions (2025+ Research)
//!
//! Several promising research directions are not yet implemented:
//!
//! | Technique | Benefit | Reference |
//! |-----------|---------|-----------|
//! | **GPU Indexing** | 9.3x speedup, 3.75x cost reduction | OpenSearch cuVS 2025 |
//! | **Filtered ANNS** | Correctness guarantees for metadata filters | ACORN (SIGMOD 2024) |
//! | **Probabilistic Routing** | 1.6-2.5x throughput via PEOs | Theoretical ANN 2024 |
//! | **FreshDiskANN** | Real-time updates with 2-hop reconnection | Microsoft Research |
//! | **Learned Indexes** | RL/DL for routing decisions | CRINN, learned ANN |
//!
//! ### GPU Acceleration
//!
//! NVIDIA cuVS (via OpenSearch) showed 9.3x query speedup and 3.75x cost reduction
//! on billion-scale workloads. GPU acceleration is most beneficial for:
//! - Large batch queries (amortize kernel launch)
//! - High-dimensional data (d > 256)
//! - Index construction (embarrassingly parallel)
//!
//! ### Filtered ANNS (ACORN)
//!
//! Traditional approach: post-filter ANN results. Problem: may return < k results.
//! ACORN builds predicate-aware graphs with formal correctness guarantees.
//! Useful when: attribute filtering is common (e.g., "similar products in category X").
//!
//! ### Probabilistic Routing
//!
//! Instead of deterministic greedy search, use learned probabilistic routing.
//! Position-Encoded Outputs (PEOs) achieve 1.6-2.5x throughput improvement
//! with theoretical guarantees on recall.

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
