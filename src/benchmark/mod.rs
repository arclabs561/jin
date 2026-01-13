//! Benchmark utilities for ANN evaluation.
//!
//! Provides metrics, dataset generation, and evaluation utilities for
//! measuring ANN index quality across multiple dimensions:
//!
//! - **Accuracy**: recall@k, precision@k
//! - **Speed**: QPS, latency percentiles
//! - **Memory**: bytes per vector, index overhead
//! - **Scaling**: performance vs dataset size, dimensionality

pub mod datasets;
pub mod metrics;
pub mod memory;

pub use datasets::{compute_ground_truth, create_benchmark_dataset, Dataset};
pub use metrics::{precision_at_k, recall_at_k};
pub use memory::{IndexMemoryStats, MemoryTracker};
