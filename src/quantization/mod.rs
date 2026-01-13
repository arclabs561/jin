//! Vector quantization: compress vectors while preserving distance.
//!
//! Quantization is the art of representing continuous vectors with discrete
//! codes. The goal: **use fewer bits per dimension while keeping distance
//! computations accurate**.
//!
//! ## Why Quantize?
//!
//! A 768-dim float32 embedding is 3KB. At 1 billion vectors, that's 3TB.
//! Quantization can achieve:
//!
//! | Method | Bits/dim | Compression | Recall |
//! |--------|----------|-------------|--------|
//! | float32 | 32 | 1x | 100% |
//! | RaBitQ 4-bit | 4 | 8x | 95%+ |
//! | Ternary | 1.58 | 20x | 85%+ |
//! | Binary | 1 | 32x | 75%+ |
//!
//! ## Scalar vs Vector Quantization
//!
//! **Scalar quantization** (this module): Quantize each dimension independently.
//! Simple but loses correlations between dimensions.
//!
//! **Vector quantization** (see `ivf_pq`): Learn codebooks that capture
//! multi-dimensional structure. Better accuracy, more complex.
//!
//! ## RaBitQ: The Modern Approach
//!
//! [Gao et al. 2024](https://arxiv.org/abs/2409.09913) introduces randomized
//! binary quantization with corrective factors.
//!
//! **Key insight**: Random rotation before quantization spreads information
//! evenly across dimensions. Then:
//!
//! 1. **Sign bit**: Direction of each rotated dimension
//! 2. **Extended bits**: Magnitude refinement (optional)
//! 3. **Corrective factors**: Learned f_add, f_scale for distance estimation
//!
//! ```text
//! Original: [0.2, -0.7, 0.1, 0.9]
//!    ↓ random rotation
//! Rotated: [0.5, 0.3, -0.6, 0.4]
//!    ↓ quantize
//! Codes:   [+, +, -, +]  (1-bit: signs only)
//!    or    [+2, +1, -2, +1]  (4-bit: with magnitude)
//! ```
//!
//! **Why random rotation?** It makes the quantization error independent
//! across dimensions, which allows accurate distance estimation via
//! expectation formulas.
//!
//! ## Ternary Quantization
//!
//! Ultra-aggressive: map each dimension to {-1, 0, +1}.
//!
//! - **1.58 bits/dim** (log₂(3))
//! - **Hamming-like distance** with popcount operations
//! - Best for high-dimensional embeddings where redundancy is high
//!
//! ## Distance Computation
//!
//! The magic of good quantization: distance in quantized space approximates
//! distance in original space.
//!
//! **Asymmetric**: Query is exact, database is quantized
//! ```text
//! d(q, x) ≈ d(q, Q(x)) + correction
//! ```
//!
//! **Symmetric**: Both quantized (faster but less accurate)
//! ```text
//! d(q, x) ≈ d(Q(q), Q(x))
//! ```
//!
//! ## Usage
//!
//! ```rust,ignore
//! use vicinity::quantization::rabitq::{RaBitQ, RaBitQConfig};
//!
//! let config = RaBitQConfig::bits4();  // 4-bit quantization
//! let mut quantizer = RaBitQ::new(768, config);
//!
//! // Train on sample vectors
//! quantizer.fit(&sample_vectors)?;
//!
//! // Quantize database
//! let codes: Vec<_> = vectors.iter()
//!     .map(|v| quantizer.encode(v))
//!     .collect();
//!
//! // Distance estimation
//! let dist = quantizer.asymmetric_distance(&query, &codes[0]);
//! ```
//!
//! ## References
//!
//! - Gao et al. (2024). "RaBitQ: Quantizing High-Dimensional Vectors with
//!   Randomized Binary Quantization."
//! - See also: Product Quantization (`ivf_pq`), ScaNN (`scann`).

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
