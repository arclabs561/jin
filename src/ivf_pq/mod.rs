//! IVF-PQ (Inverted File Index with Product Quantization) implementation.
//!
//! Memory-efficient ANN algorithm combining:
//! - **IVF**: Inverted file index (clustering-based partitioning)
//! - **PQ**: Product quantization (vector compression)
//!
//! Best for billion-scale datasets with memory constraints.
//!
//! # References
//!
//! - Jégou et al. (2011): "Product Quantization for Nearest Neighbor Search"
//!
//! # Status: Experimental
//!
//! Some fields (e.g., `pq_codebooks`) are placeholders for Product Quantization
//! integration that is under active development.

#![allow(dead_code)]

mod ivf;
#[cfg(feature = "scann")]
mod online_pq;
#[cfg(feature = "scann")]
mod opq;
pub mod pq;
pub mod search;

#[cfg(feature = "scann")]
pub use online_pq::OnlineProductQuantizer;
#[cfg(feature = "scann")]
pub use opq::OptimizedProductQuantizer;
pub use search::{IVFPQIndex, IVFPQParams};
