//! IVF-PQ: Inverted File with Product Quantization.
//!
//! The workhorse of billion-scale similarity search. Combines two ideas:
//!
//! 1. **IVF (Inverted File)**: Partition space into Voronoi cells, only search
//!    cells near the query
//! 2. **PQ (Product Quantization)**: Compress vectors to 8-32 bytes while
//!    preserving distance estimation
//!
//! ## Why IVF?
//!
//! Brute-force search is O(n). With IVF:
//! - Partition n vectors into k clusters (k ≈ √n)
//! - At query time, only search `nprobe` nearest clusters
//! - Effective search cost: O(nprobe × n/k)
//!
//! ```text
//!           Query
//!             |
//!     +-------+-------+
//!     |               |
//!   Cell A          Cell B      (probe 2 cells)
//!   |__|__|         |__|__|
//!   v  v  v         v  v  v
//!  [vectors]       [vectors]    (compare within cells)
//! ```
//!
//! ## Why Product Quantization?
//!
//! A 128-dim float32 vector is 512 bytes. At 1 billion vectors, that's 500GB.
//! PQ compresses to ~16 bytes (32x compression) while keeping 90%+ recall.
//!
//! **The Key Insight** ([Jégou et al. 2011](https://lear.inrialpes.fr/pubs/2011/JDS11/jegou_searching_with_quantization.pdf)):
//!
//! Split the vector into M subvectors. Quantize each independently using a
//! small codebook (256 entries). Store only the codebook indices.
//!
//! ```text
//! Original:  [v₁ v₂ v₃ v₄ ... v₁₂₈]  (128 floats = 512 bytes)
//!            └──┴──┘ └──┴──┘ ... └──┘
//!              ↓       ↓         ↓
//!            [c₁]    [c₂]  ... [cₘ]   (M=8 codebook indices = 8 bytes)
//! ```
//!
//! **Distance estimation**: Precompute distances from query subvectors to all
//! codebook entries. Then distance to any compressed vector is just M table lookups.
//!
//! ## Asymmetric Distance Computation (ADC)
//!
//! Don't compress the query—only compress database vectors.
//!
//! ```text
//! query (exact) → codebook distances → lookup for each DB vector
//!
//! d(query, db) ≈ Σᵢ lookup[i][db_code[i]]
//! ```
//!
//! This gives better accuracy than symmetric (both compressed) at same memory.
//!
//! ## OPQ: Optimized Product Quantization
//!
//! PQ assumes subspaces are independent. Real data has correlations.
//! OPQ learns a rotation matrix R that decorrelates dimensions before
//! quantization, improving accuracy by 10-30%.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use vicinity::ivf_pq::{IVFPQIndex, IVFPQParams};
//!
//! let params = IVFPQParams {
//!     num_centroids: 1024,  // sqrt(n) is a good default
//!     num_codebooks: 8,     // 8 bytes per vector
//!     codebook_bits: 8,     // 256 codewords per codebook
//!     nprobe: 10,           // search 10 nearest cells
//! };
//!
//! let mut index = IVFPQIndex::new(128, params)?;
//! index.train(&training_vectors)?;
//! index.add_batch(&vectors)?;
//!
//! let results = index.search(&query, 10)?;
//! ```
//!
//! ## Trade-offs
//!
//! | Parameter | ↑ Effect |
//! |-----------|----------|
//! | nprobe | Better recall, slower search |
//! | num_centroids | Better partitioning, slower training |
//! | num_codebooks | More memory, better accuracy |
//!
//! ## References
//!
//! - Jégou, Douze, Schmid (2011). "Product Quantization for Nearest Neighbor Search."
//! - Ge et al. (2014). "Optimized Product Quantization."
//! - See `opq.rs` for the rotation learning algorithm.

// IVF-PQ core implementation (always available when ivf_pq feature is enabled)
pub mod search;
pub use search::{IVFPQIndex, IVFPQParams};

// NOTE: Advanced IVF-PQ components (OPQ, pure IVF) are stubs awaiting implementation.
// The core IVFPQIndex in search.rs provides the main functionality.
// TODO: Implement full OPQ (rotation matrix learning) and separate IVF index
