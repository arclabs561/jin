//! Hierarchical Navigable Small World (HNSW) approximate nearest neighbor search.
//!
//! Pure Rust implementation optimized for performance with SIMD acceleration and
//! cache-friendly memory layouts.
//!
//! # Algorithm
//!
//! HNSW constructs a multi-layer graph where:
//! - **Upper layers**: Sparse, long-range connections for fast navigation
//! - **Lower layers**: Dense, local connections for precise search
//! - **Search**: Start at top layer, navigate down to base layer, greedy search
//!
//! # Critical Note: Hierarchy Benefits
//!
//! Recent research (2024-2025) suggests that **the hierarchical structure provides
//! minimal or no practical benefit in high-dimensional settings** (d > 32). Flat
//! Navigable Small World (NSW) graphs achieve performance parity with hierarchical
//! HNSW in both median and tail latency, while using less memory.
//!
//! The explanation: **hubness** in high-dimensional spaces creates natural "highway"
//! nodes that serve the same functional role as explicit hierarchy. When metric hubs
//! already provide efficient routing, explicit hierarchical layers become redundant.
//!
//! **Implications**:
//! - For high-dimensional data (d > 32), consider flat NSW variants for memory savings
//! - Hierarchy may still help for low-dimensional data (d < 32) or angular distance metrics
//! - See `docs/CRITICAL_PERSPECTIVES_AND_LIMITATIONS.md` for detailed analysis
//!
//! # Performance
//!
//! - **SIMD-accelerated**: Uses existing `simd` module for distance computation (8-16x speedup)
//! - **Cache-optimized**: Structure of Arrays (SoA) layout for better cache locality
//! - **Early termination**: Multiple strategies to reduce unnecessary distance computations
//! - **O(log n) complexity**: Logarithmic search time vs O(n) brute-force
//!
//! # Usage
//!
//! ```rust
//! use ordino_retrieve::dense::hnsw::HNSWIndex;
//!
//! # fn main() -> Result<(), ordino_retrieve::RetrieveError> {
//! let mut index = HNSWIndex::new(128, 16, 16)?;
//!
//! // Add vectors
//! index.add(0, vec![0.1; 128])?;
//! index.add(1, vec![0.2; 128])?;
//!
//! // Build index (required before search)
//! index.build()?;
//!
//! // Search
//! let results = index.search(&vec![0.15; 128], 10, 50)?;
//! # Ok(())
//! # }
//! ```
//!
//! # References
//!
//! - Malkov & Yashunin (2016): "Efficient and robust approximate nearest neighbor search
//!   using Hierarchical Navigable Small World graphs"
//! - See `docs/HNSW_IMPLEMENTATION_PLAN.md` for detailed implementation notes
//!
//! # Status
//!
//! Some compression-related fields are placeholders for future ID compression support.

#![allow(dead_code)] // Compression fields are placeholders

#[cfg(feature = "hnsw")]
pub(crate) mod construction;
#[cfg(feature = "hnsw")]
pub(crate) mod distance;
#[cfg(feature = "hnsw")]
pub(crate) mod graph;
#[cfg(feature = "hnsw")]
mod memory;
#[cfg(feature = "hnsw")]
mod search;

#[cfg(feature = "hnsw")]
pub use graph::{HNSWIndex, HNSWParams, NeighborhoodDiversification, SeedSelectionStrategy};

// Filtered search (ACORN-style)
#[cfg(feature = "hnsw")]
pub mod filtered;
#[cfg(feature = "hnsw")]
pub use filtered::{
    acorn_search, AcornConfig, FilterPredicate, FilterStrategy, FnFilter, NoFilter,
};

// Graph repair (MN-RU algorithm for deletions)
#[cfg(feature = "hnsw")]
pub mod repair;
#[cfg(feature = "hnsw")]
pub use repair::{
    compute_repair_operations, validate_connectivity, GraphRepairer, RepairConfig, RepairStats,
};

// Vamana graph construction (DiskANN-style alpha-pruning)
#[cfg(feature = "hnsw")]
pub mod vamana;
#[cfg(feature = "hnsw")]
pub use vamana::{build_vamana_graph, search_vamana, VamanaConfig, VamanaGraph};

// FusedANN: Attribute-vector fusion for filtered search
#[cfg(feature = "hnsw")]
pub mod fused;
#[cfg(feature = "hnsw")]
pub use fused::{
    recommend_alpha, AttributeDefinition, AttributeEmbedder, AttributeSchema, AttributeType,
    AttributeValue, FusedConfig, FusedIndex, FusedVector,
};

// Random Walk-based graph repair (alternative to MN-RU)
#[cfg(feature = "hnsw")]
pub mod random_walk_repair;
#[cfg(feature = "hnsw")]
pub use random_walk_repair::{
    estimate_hitting_time_change, random_walk_repair, ImportanceScores, RandomWalkConfig,
    RandomWalkRepairer, RandomWalkStats,
};

// Dynamic Edge Navigation Graph (DEG) for bimodal data
#[cfg(feature = "hnsw")]
pub mod deg;
#[cfg(feature = "hnsw")]
pub use deg::{DEGConfig, DEGIndex, DensityInfo};

// In-place updates (IP-DiskANN style)
#[cfg(feature = "hnsw")]
pub mod inplace;
#[cfg(feature = "hnsw")]
pub use inplace::{InPlaceConfig, InPlaceIndex, InPlaceStats};

// Probabilistic edge routing (PEOs) for QPS improvement
#[cfg(feature = "hnsw")]
pub mod probabilistic_routing;
#[cfg(feature = "hnsw")]
pub use probabilistic_routing::{
    EdgeProbabilityEstimator, ProbabilisticEdgeSelector, ProbabilisticRouter,
    ProbabilisticRoutingConfig, ProbabilisticStats,
};

// HNSW index merging algorithms (NGM, IGTM, CGTM)
#[cfg(feature = "hnsw")]
pub mod merge;
#[cfg(feature = "hnsw")]
pub use merge::{
    cross_graph_traversal_merge, intra_graph_traversal_merge, naive_graph_merge, MergeConfig,
    MergeGraph, MergeNode, MergeStats,
};
