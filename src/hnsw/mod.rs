//! Hierarchical Navigable Small World (HNSW) approximate nearest neighbor search.
//!
//! Pure Rust implementation optimized for performance with SIMD acceleration and
//! cache-friendly memory layouts.
//!
//! # The Small-World Insight
//!
//! HNSW exploits the **small-world network property**: in well-constructed graphs,
//! any two nodes can be reached in O(log n) hops via greedy routing. This is the
//! same phenomenon that gives "six degrees of separation" in social networks.
//!
//! **Greedy routing**: From any node, examine neighbors and move to the one closest
//! to your target. Repeat until no closer neighbor exists (local minimum). In
//! small-world graphs, this converges to near-optimal paths.
//!
//! # Why Hierarchy?
//!
//! Flat navigable small-world (NSW) graphs work, but require many distance
//! computations in the "zoom-out" phase (finding long-range connections). HNSW
//! adds hierarchy to reduce this:
//!
//! ```text
//! Layer 3: [sparse]     ●───────────────●  (long jumps, few nodes)
//! Layer 2:          ●───●───●───●───●───●  (medium connections)
//! Layer 1:       ●─●─●─●─●─●─●─●─●─●─●─●─●  (local connections)
//! Layer 0:     ●●●●●●●●●●●●●●●●●●●●●●●●●●●  (all nodes, dense)
//! ```
//!
//! Search starts at sparse top layer, then descends. Each descent provides a
//! better starting point for greedy search at the next level.
//!
//! # Exponential Layer Assignment
//!
//! Nodes are assigned to layers probabilistically:
//!
//! ```text
//! max_layer = floor(-ln(rand(0,1)) × mL)
//! ```
//!
//! where mL ≈ 1/ln(M). This creates exponentially decreasing node density:
//! - Layer 0: All n nodes
//! - Layer 1: ~n/M nodes
//! - Layer 2: ~n/M² nodes
//!
//! Upper layers are sparse → few distance computations for coarse routing.
//!
//! # Key Parameters
//!
//! - **M**: Max connections per node. Higher = better recall, more memory.
//! - **ef_construction**: Search width during build. Higher = better graph quality.
//! - **ef**: Search width during query. Higher = better recall, slower search.
//!
//! # Critical Note: Hierarchy in High Dimensions
//!
//! Recent research (2024-2025) suggests that **the hierarchical structure provides
//! minimal or no practical benefit in high-dimensional settings** (d > 32). Flat
//! NSW graphs achieve performance parity with HNSW in both median and tail latency.
//!
//! The explanation: **hubness**. In high dimensions, some nodes naturally become
//! "hubs" with high connectivity, serving the same routing function as explicit
//! hierarchy. When metric hubs already provide efficient routing, explicit layers
//! become redundant.
//!
//! **Implications**:
//! - For d > 32: Consider flat NSW variants for memory savings
//! - For d < 32 or angular metrics: Hierarchy may still help
//! - See `docs/CRITICAL_PERSPECTIVES_AND_LIMITATIONS.md` for analysis
//!
//! # References
//!
//! - Malkov & Yashunin (2016). "Efficient and robust approximate nearest neighbor
//!   search using Hierarchical Navigable Small World graphs."
//! - [Recent NSW analysis](https://arxiv.org/abs/2412.01940) on hierarchy benefits
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
//! ```rust,no_run
//! use vicinity::hnsw::HNSWIndex;
//!
//! fn main() -> Result<(), vicinity::RetrieveError> {
//!     let mut index = HNSWIndex::new(128, 16, 16)?;
//!
//!     // Add vectors
//!     index.add(0, vec![0.1; 128])?;
//!     index.add(1, vec![0.2; 128])?;
//!
//!     // Build index (required before search)
//!     index.build()?;
//!
//!     // Search
//!     let results = index.search(&vec![0.15; 128], 10, 50)?;
//!     println!("Found {} results", results.len());
//!     Ok(())
//! }
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
