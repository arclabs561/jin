//! Hierarchical Navigable Small World (HNSW) approximate nearest neighbor search.
//!
//! The industry-standard graph-based ANN algorithm. Pure Rust with SIMD acceleration.
//!
//! # Quick Start
//!
//! ```rust,no_run
//! use plesio::hnsw::HNSWIndex;
//!
//! fn main() -> Result<(), plesio::RetrieveError> {
//!     // dimension=128, M=16, ef_construction=200
//!     let mut index = HNSWIndex::new(128, 16, 200)?;
//!
//!     index.add(0, vec![0.1; 128])?;
//!     index.add(1, vec![0.2; 128])?;
//!     index.build()?;
//!
//!     // k=10, ef_search=50
//!     let results = index.search(&vec![0.15; 128], 10, 50)?;
//!     Ok(())
//! }
//! ```
//!
//! # Parameter Recommendations
//!
//! | Dataset Size | M | ef_construction | ef_search | Memory/vector |
//! |--------------|---|-----------------|-----------|---------------|
//! | < 100K | 16 | 100 | 50 | ~1.2 KB |
//! | 100K - 1M | 16 | 200 | 100 | ~1.2 KB |
//! | 1M - 10M | 32 | 200 | 100-200 | ~2.4 KB |
//! | > 10M | 48 | 400 | 200+ | ~3.6 KB |
//!
//! **Memory formula**: `n × (d × 4 + M × 8 + overhead)` bytes
//! - For 1M vectors at d=768, M=16: ~3.5 GB
//!
//! # The Small-World Insight
//!
//! HNSW exploits the **small-world network property**: in well-constructed graphs,
//! any two nodes can be reached in O(log n) hops via greedy routing.
//!
//! ```text
//! Layer 3: [sparse]     o───────────────o  (long jumps, few nodes)
//! Layer 2:          o───o───o───o───o───o  (medium connections)
//! Layer 1:       o─o─o─o─o─o─o─o─o─o─o─o─o  (local connections)
//! Layer 0:     ooooooooooooooooooooooooooo  (all nodes, dense)
//! ```
//!
//! Search starts at sparse top layer, descends using each layer as a better starting point.
//!
//! # Parameter Effects
//!
//! | Parameter | Effect when increased |
//! |-----------|----------------------|
//! | **M** | Better recall, more memory, slower build |
//! | **ef_construction** | Better graph quality, slower build |
//! | **ef_search** | Better recall, slower search |
//!
//! # When NOT to Use HNSW
//!
//! - **< 10K vectors**: Brute force is faster (no graph overhead)
//! - **> 99.9% recall required**: Graph methods have a recall ceiling
//! - **Memory constrained + > 10M vectors**: Use IVF-PQ instead
//!
//! # Hierarchy in High Dimensions
//!
//! Research (2024-2025) shows hierarchy provides **minimal benefit for d > 32**.
//! In high dimensions, "hubs" emerge naturally and serve the same routing function.
//!
//! **Practical advice**: HNSW is still the safe default. The hierarchy overhead
//! is small, and the algorithm is battle-tested. Consider flat NSW only if you
//! specifically need to minimize memory.
//!
//! # Advanced Features
//!
//! - [`filtered`]: ACORN-style attribute filtering
//! - [`dual_branch`]: LID-based insertion with skip bridges (2025)
//! - [`fused`]: Attribute-vector fusion for filtered search
//! - [`merge`]: Index merging algorithms (NGM, IGTM, CGTM)
//!
//! # References
//!
//! - Malkov & Yashunin (2016). "Efficient and robust approximate nearest neighbor
//!   search using Hierarchical Navigable Small World graphs."
//! - [NSW hierarchy analysis](https://arxiv.org/abs/2412.01940) (2024)

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
pub use inplace::{InPlaceConfig, InPlaceIndex, InPlaceStats, MappedInPlaceIndex};

// Incremental learning patterns (edge refinement, temporal locality)
#[cfg(feature = "hnsw")]
pub mod incremental;
#[cfg(feature = "hnsw")]
pub use incremental::{
    DriftTracker, EdgeStats, IncrementalConfig, RecencyWeighting, RefinementAnalyzer,
    RefinementSuggestions,
};

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

// Dual-Branch HNSW with LID-based insertion and skip bridges (arXiv 2501.13992)
#[cfg(feature = "hnsw")]
pub mod dual_branch;
#[cfg(feature = "hnsw")]
pub use dual_branch::{DualBranchConfig, DualBranchHNSW, DualBranchStats, SkipBridge};
