//! Streaming updates for vector indices.
//!
//! # The Problem
//!
//! Real-world vector databases face continuous updates:
//! - Documents added as they're created
//! - Documents deleted for GDPR/legal compliance
//! - Embeddings updated when models change
//!
//! Naive approaches rebuild the entire index, which is expensive.
//! This module provides efficient streaming update primitives.
//!
//! # Architecture
//!
//! ```text
//! Stream of Updates
//!     │
//!     ▼
//! ┌──────────────┐
//! │ StreamBuffer │ ◄── Batches small updates
//! └──────┬───────┘
//!        │
//!        ▼ (periodic merge)
//! ┌──────────────┐
//! │  Main Index  │ ◄── HNSW, DiskANN, etc.
//! └──────────────┘
//! ```
//!
//! # Update Types
//!
//! | Type | Complexity | Notes |
//! |------|------------|-------|
//! | Insert | O(log n) amortized | Buffered, merged periodically |
//! | Delete | O(degree) | Mark-and-repair (IP-DiskANN pattern) |
//! | Update | Delete + Insert | Atomic replacement |
//!
//! # Research Context
//!
//! - **FreshDiskANN** (2024): StreamingMerge for efficient insert/delete streams
//! - **IP-DiskANN** (2025): In-neighbor tracking for O(1) delete preparation
//! - **Delta-EMG** (2025): Monotonic update guarantees (no recall degradation)
//!
//! # Example
//!
//! ```rust,ignore
//! use vicinity::streaming::{StreamingIndex, UpdateOp};
//! use vicinity::hnsw::HNSWIndex;
//!
//! let mut index = StreamingIndex::new(HNSWIndex::new(128)?);
//!
//! // Stream of updates
//! index.apply(UpdateOp::Insert { id: 0, vector: vec![0.1; 128] })?;
//! index.apply(UpdateOp::Insert { id: 1, vector: vec![0.2; 128] })?;
//! index.apply(UpdateOp::Delete { id: 0 })?;
//!
//! // Search works across buffered and merged data
//! let results = index.search(&query, 10)?;
//!
//! // Periodic compaction
//! index.compact()?;
//! ```

mod buffer;
mod ops;
mod coordinator;

pub use buffer::{StreamBuffer, StreamBufferConfig};
pub use ops::{UpdateOp, UpdateBatch, UpdateStats};
pub use coordinator::{StreamingCoordinator, StreamingConfig};

use crate::error::Result;

/// Trait for indices that support streaming updates.
pub trait StreamingIndex {
    /// Apply a single update operation.
    fn apply(&mut self, op: UpdateOp) -> Result<()>;
    
    /// Apply a batch of updates atomically.
    fn apply_batch(&mut self, batch: UpdateBatch) -> Result<UpdateStats>;
    
    /// Compact/merge buffered updates into main index.
    fn compact(&mut self) -> Result<UpdateStats>;
    
    /// Get streaming statistics.
    fn stats(&self) -> StreamingStats;
}

/// Statistics about streaming index state.
#[derive(Debug, Clone, Default)]
pub struct StreamingStats {
    /// Number of vectors in main index.
    pub main_index_size: usize,
    /// Number of vectors in buffer.
    pub buffer_size: usize,
    /// Number of pending deletes.
    pub pending_deletes: usize,
    /// Total inserts since creation.
    pub total_inserts: u64,
    /// Total deletes since creation.
    pub total_deletes: u64,
    /// Total compactions performed.
    pub total_compactions: u64,
}
