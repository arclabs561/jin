//! Flat Navigable Small World (NSW) implementation.
//!
//! Single-layer graph variant of HNSW that achieves performance parity with
//! hierarchical HNSW in high-dimensional settings while using less memory.
//!
//! # Critical Finding
//!
//! Recent research (2024-2025) demonstrates that **hierarchical structure provides
//! minimal or no practical benefit in high-dimensional settings** (d > 32). Flat NSW
//! achieves equivalent performance with 20-30% lower memory overhead.
//!
//! # Algorithm
//!
//! NSW constructs a single-layer graph where:
//! - **Graph structure**: Navigable Small World with RNG-based neighbor selection
//! - **Search**: Greedy search from entry point
//! - **No hierarchy**: All vectors in single layer (simpler, less memory)
//!
//! # When to Use NSW vs HNSW
//!
//! **Use NSW (flat)** when:
//! - High-dimensional data (d > 32)
//! - Memory-constrained environments
//! - Want simpler implementation with fewer parameters
//!
//! **Use HNSW (hierarchical)** when:
//! - Low-dimensional data (d < 32)
//! - Angular distance metrics (weaker hubness)
//! - Non-Euclidean metrics where hierarchy may help
//!
//! # Performance
//!
//! - **Equivalent search performance** to HNSW in high dimensions
//! - **20-30% lower memory** (no multi-layer overhead)
//! - **Simpler construction** (no layer assignment)
//! - **Same search complexity**: O(log n) with proper graph structure
//!
//! # References
//!
//! - Recent research (2024-2025) on HNSW hierarchy effectiveness
//! - Hubness and curse of dimensionality literature
//! - See `docs/CRITICAL_PERSPECTIVES_AND_LIMITATIONS.md` for detailed analysis

pub mod construction;
pub mod graph;
pub mod search;

pub use graph::{NSWIndex, NSWParams};
