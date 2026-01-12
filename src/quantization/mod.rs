//! Vector quantization methods.
//!
//! # Status: Experimental
//!
//! These quantization methods are under development.
//!
//! # Available Methods
//!
//! - **RaBitQ**: Randomized binary quantization with corrective factors (1-8 bit per dim)
//! - **SAQ**: Segmented adaptive quantization with optimal bit allocation
//! - **Ternary**: Ultra-quantization with 1.58-bit encodings {-1, 0, +1}
//! - **TurboQuant**: High-performance quantization (when available)
//! - **simd_ops**: SIMD-optimized bit packing and distance operations

#![allow(dead_code)]

#[cfg(feature = "rabitq")]
pub mod rabitq;

#[cfg(feature = "rabitq")]
pub mod simd_ops;

#[cfg(feature = "saq")]
pub mod saq;

#[cfg(feature = "saq")]
pub mod ternary;

#[cfg(feature = "turboquant")]
pub mod turboquant;

// Re-exports for convenience
#[cfg(feature = "saq")]
pub use ternary::{
    asymmetric_cosine_distance, asymmetric_inner_product, ternary_cosine_similarity,
    ternary_hamming, ternary_inner_product, TernaryConfig, TernaryQuantizer, TernaryVector,
};
