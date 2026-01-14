//! Ultra-Quantization: 1.58-bit ternary encodings.
//!
//! Implements extreme quantization where each dimension is represented by
//! one of three values: {-1, 0, +1}. This yields log2(3) ≈ 1.58 bits per
//! dimension, providing massive compression while maintaining surprising
//! accuracy for similarity search.
//!
//! # Algorithm
//!
//! 1. Normalize input vectors to unit length
//! 2. For each dimension, quantize to:
//!    - +1 if value > threshold_high
//!    - -1 if value < threshold_low
//!    - 0 otherwise (values near zero)
//! 3. Store as packed 2-bit values (00=0, 01=+1, 10=-1, 11=unused)
//!
//! # Distance Computation
//!
//! For ternary vectors a, b:
//! - Inner product: sum of element-wise products
//! - Efficiently computed via popcount operations
//!
//! # References
//!
//! - Connor et al. (2025): "Ultra-Quantisation: Efficient Embedding Search
//!   via 1.58-bit Encodings" - <https://arxiv.org/abs/2506.00528>

use crate::RetrieveError;

/// Ternary quantized vector.
///
/// Each dimension is stored as 2 bits:
/// - 00 = 0
/// - 01 = +1
/// - 10 = -1
/// - 11 = reserved
#[derive(Clone, Debug)]
pub struct TernaryVector {
    /// Packed 2-bit values (4 values per byte)
    data: Vec<u8>,
    /// Original dimension
    dimension: usize,
    /// Number of +1 values
    positive_count: usize,
    /// Number of -1 values  
    negative_count: usize,
    /// Norm of original vector (for asymmetric distance)
    original_norm: f32,
}

impl TernaryVector {
    /// Get dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get value at index.
    pub fn get(&self, idx: usize) -> i8 {
        if idx >= self.dimension {
            return 0;
        }
        let byte_idx = idx / 4;
        let bit_offset = (idx % 4) * 2;
        let bits = (self.data[byte_idx] >> bit_offset) & 0b11;
        match bits {
            0b00 => 0,
            0b01 => 1,
            0b10 => -1,
            _ => 0, // Reserved
        }
    }

    /// Get sparsity (fraction of zeros).
    pub fn sparsity(&self) -> f32 {
        let nonzero = self.positive_count + self.negative_count;
        1.0 - (nonzero as f32 / self.dimension as f32)
    }

    /// Memory size in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len()
    }
}

/// Ternary quantizer configuration.
#[derive(Clone, Debug)]
pub struct TernaryConfig {
    /// Upper threshold for +1 (as fraction of max magnitude)
    pub threshold_high: f32,
    /// Lower threshold for -1 (as fraction of max magnitude)
    pub threshold_low: f32,
    /// Whether to normalize vectors before quantization
    pub normalize: bool,
    /// Target sparsity (fraction of zeros), adjusts thresholds adaptively
    pub target_sparsity: Option<f32>,
}

impl Default for TernaryConfig {
    fn default() -> Self {
        Self {
            threshold_high: 0.3,
            threshold_low: -0.3,
            normalize: true,
            target_sparsity: None,
        }
    }
}

/// Ternary quantizer.
pub struct TernaryQuantizer {
    config: TernaryConfig,
    dimension: usize,
    /// Learned thresholds per dimension (optional)
    adaptive_thresholds: Option<Vec<(f32, f32)>>,
    /// Mean vector (for centering)
    mean: Option<Vec<f32>>,
}

impl TernaryQuantizer {
    /// Create new ternary quantizer.
    pub fn new(dimension: usize, config: TernaryConfig) -> Self {
        Self {
            config,
            dimension,
            adaptive_thresholds: None,
            mean: None,
        }
    }

    /// Create with default config.
    pub fn with_dimension(dimension: usize) -> Self {
        Self::new(dimension, TernaryConfig::default())
    }

    /// Fit quantizer on training data for adaptive thresholds.
    pub fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        if vectors.len() != num_vectors * self.dimension {
            return Err(RetrieveError::Other("Vector count mismatch".to_string()));
        }

        // Compute mean for centering
        let mut mean = vec![0.0f32; self.dimension];
        for i in 0..num_vectors {
            let vec = &vectors[i * self.dimension..(i + 1) * self.dimension];
            for (j, &v) in vec.iter().enumerate() {
                mean[j] += v;
            }
        }
        for m in &mut mean {
            *m /= num_vectors as f32;
        }
        self.mean = Some(mean);

        // If target sparsity is set, compute adaptive thresholds
        if let Some(target_sparsity) = self.config.target_sparsity {
            let mut thresholds = Vec::with_capacity(self.dimension);

            for d in 0..self.dimension {
                // Collect centered values for this dimension
                let mut values: Vec<f32> = (0..num_vectors)
                    .map(|i| {
                        let v = vectors[i * self.dimension + d];
                        if let Some(ref m) = self.mean {
                            v - m[d]
                        } else {
                            v
                        }
                    })
                    .collect();

                values.sort_by(|a, b| a.total_cmp(b));

                // Find thresholds for target sparsity
                let zero_fraction = target_sparsity;
                let nonzero_fraction = (1.0 - zero_fraction) / 2.0;

                let low_idx = (nonzero_fraction * num_vectors as f32) as usize;
                let high_idx = ((1.0 - nonzero_fraction) * num_vectors as f32) as usize;

                let low_idx = low_idx.min(num_vectors - 1);
                let high_idx = high_idx.min(num_vectors - 1);

                let threshold_low = values[low_idx];
                let threshold_high = values[high_idx];

                thresholds.push((threshold_low, threshold_high));
            }

            self.adaptive_thresholds = Some(thresholds);
        }

        Ok(())
    }

    /// Quantize a single vector.
    pub fn quantize(&self, vector: &[f32]) -> Result<TernaryVector, RetrieveError> {
        if vector.len() != self.dimension {
            return Err(RetrieveError::Other(format!(
                "Expected {} dimensions, got {}",
                self.dimension,
                vector.len()
            )));
        }

        // Center if mean is available
        let centered: Vec<f32> = if let Some(ref mean) = self.mean {
            vector
                .iter()
                .zip(mean.iter())
                .map(|(&v, &m)| v - m)
                .collect()
        } else {
            vector.to_vec()
        };

        // Normalize if configured
        let processed: Vec<f32> = if self.config.normalize {
            let norm: f32 = centered.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                centered.iter().map(|&x| x / norm).collect()
            } else {
                centered
            }
        } else {
            centered
        };

        // Compute original norm for asymmetric distance
        let original_norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

        // Allocate packed data
        let num_bytes = self.dimension.div_ceil(4);
        let mut data = vec![0u8; num_bytes];
        let mut positive_count = 0;
        let mut negative_count = 0;

        for (i, &v) in processed.iter().enumerate() {
            let (thresh_low, thresh_high) = if let Some(ref thresholds) = self.adaptive_thresholds {
                thresholds[i]
            } else {
                (self.config.threshold_low, self.config.threshold_high)
            };

            let bits: u8 = if v > thresh_high {
                positive_count += 1;
                0b01 // +1
            } else if v < thresh_low {
                negative_count += 1;
                0b10 // -1
            } else {
                0b00 // 0
            };

            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            data[byte_idx] |= bits << bit_offset;
        }

        Ok(TernaryVector {
            data,
            dimension: self.dimension,
            positive_count,
            negative_count,
            original_norm,
        })
    }

    /// Quantize multiple vectors.
    pub fn quantize_batch(
        &self,
        vectors: &[f32],
        num_vectors: usize,
    ) -> Result<Vec<TernaryVector>, RetrieveError> {
        if vectors.len() != num_vectors * self.dimension {
            return Err(RetrieveError::Other("Vector count mismatch".to_string()));
        }

        (0..num_vectors)
            .map(|i| {
                let vec = &vectors[i * self.dimension..(i + 1) * self.dimension];
                self.quantize(vec)
            })
            .collect()
    }
}

/// Compute inner product between two ternary vectors.
///
/// Efficient implementation using bit manipulation.
pub fn ternary_inner_product(a: &TernaryVector, b: &TernaryVector) -> i32 {
    if a.dimension != b.dimension {
        return 0;
    }

    let mut sum: i32 = 0;

    // Process 4 values at a time (one byte)
    for (byte_a, byte_b) in a.data.iter().zip(b.data.iter()) {
        for i in 0..4 {
            let bits_a = (*byte_a >> (i * 2)) & 0b11;
            let bits_b = (*byte_b >> (i * 2)) & 0b11;

            let val_a = match bits_a {
                0b01 => 1i32,
                0b10 => -1,
                _ => 0,
            };
            let val_b = match bits_b {
                0b01 => 1i32,
                0b10 => -1,
                _ => 0,
            };

            sum += val_a * val_b;
        }
    }

    sum
}

/// Compute approximate cosine similarity using ternary vectors.
pub fn ternary_cosine_similarity(a: &TernaryVector, b: &TernaryVector) -> f32 {
    let ip = ternary_inner_product(a, b) as f32;

    // Norm of ternary vector is sqrt(positive_count + negative_count)
    let norm_a = ((a.positive_count + a.negative_count) as f32).sqrt();
    let norm_b = ((b.positive_count + b.negative_count) as f32).sqrt();

    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }

    ip / (norm_a * norm_b)
}

/// Compute asymmetric inner product: f32 query against ternary vector.
///
/// This is more accurate than symmetric ternary comparison.
pub fn asymmetric_inner_product(query: &[f32], quantized: &TernaryVector) -> f32 {
    if query.len() != quantized.dimension {
        return 0.0;
    }

    let mut sum = 0.0f32;

    for (i, &q) in query.iter().enumerate() {
        let val = quantized.get(i);
        sum += q * (val as f32);
    }

    sum
}

/// Compute asymmetric cosine distance.
pub fn asymmetric_cosine_distance(query: &[f32], quantized: &TernaryVector) -> f32 {
    let ip = asymmetric_inner_product(query, quantized);

    // Query norm
    let query_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();

    // Ternary norm
    let ternary_norm = ((quantized.positive_count + quantized.negative_count) as f32).sqrt();

    if query_norm < 1e-10 || ternary_norm < 1e-10 {
        return 1.0;
    }

    1.0 - (ip / (query_norm * ternary_norm))
}

/// Hamming-like distance for ternary vectors.
///
/// Counts positions where values differ.
pub fn ternary_hamming(a: &TernaryVector, b: &TernaryVector) -> usize {
    if a.dimension != b.dimension {
        return a.dimension.max(b.dimension);
    }

    let mut diff = 0;

    for i in 0..a.dimension {
        if a.get(i) != b.get(i) {
            diff += 1;
        }
    }

    diff
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_quantization() {
        let quantizer = TernaryQuantizer::with_dimension(8);
        let vector = vec![0.5, -0.5, 0.1, -0.1, 0.8, -0.8, 0.0, 0.2];

        let quantized = quantizer.quantize(&vector).unwrap();

        assert_eq!(quantized.dimension(), 8);
        assert!(quantized.memory_bytes() <= 2); // 8 dims * 2 bits / 8 = 2 bytes
    }

    #[test]
    fn test_ternary_values() {
        let config = TernaryConfig {
            threshold_high: 0.3,
            threshold_low: -0.3,
            normalize: false,
            target_sparsity: None,
        };
        let quantizer = TernaryQuantizer::new(4, config);

        // Clear +1, -1, 0, 0 pattern
        let vector = vec![0.5, -0.5, 0.1, -0.1];
        let quantized = quantizer.quantize(&vector).unwrap();

        assert_eq!(quantized.get(0), 1); // 0.5 > 0.3
        assert_eq!(quantized.get(1), -1); // -0.5 < -0.3
        assert_eq!(quantized.get(2), 0); // 0.1 in [-0.3, 0.3]
        assert_eq!(quantized.get(3), 0); // -0.1 in [-0.3, 0.3]
    }

    #[test]
    fn test_inner_product() {
        let config = TernaryConfig {
            threshold_high: 0.3,
            threshold_low: -0.3,
            normalize: false,
            target_sparsity: None,
        };
        let quantizer = TernaryQuantizer::new(4, config);

        let v1 = vec![0.5, -0.5, 0.1, 0.0]; // +1, -1, 0, 0
        let v2 = vec![0.5, 0.5, -0.5, 0.0]; // +1, +1, -1, 0

        let q1 = quantizer.quantize(&v1).unwrap();
        let q2 = quantizer.quantize(&v2).unwrap();

        let ip = ternary_inner_product(&q1, &q2);
        // (+1)(+1) + (-1)(+1) + (0)(-1) + (0)(0) = 1 - 1 + 0 + 0 = 0
        assert_eq!(ip, 0);
    }

    #[test]
    fn test_asymmetric_distance() {
        let config = TernaryConfig {
            threshold_high: 0.3,
            threshold_low: -0.3,
            normalize: false,
            target_sparsity: None,
        };
        let quantizer = TernaryQuantizer::new(4, config);

        let vec = vec![0.5, -0.5, 0.1, 0.0]; // quantizes to +1, -1, 0, 0
        let quantized = quantizer.quantize(&vec).unwrap();

        let query = vec![1.0, 1.0, 1.0, 1.0];
        let ip = asymmetric_inner_product(&query, &quantized);
        // 1*1 + 1*(-1) + 1*0 + 1*0 = 0
        assert_eq!(ip, 0.0);

        let query2 = vec![1.0, -1.0, 0.0, 0.0];
        let ip2 = asymmetric_inner_product(&query2, &quantized);
        // 1*1 + (-1)*(-1) + 0*0 + 0*0 = 2
        assert_eq!(ip2, 2.0);
    }

    #[test]
    fn test_sparsity() {
        let config = TernaryConfig {
            threshold_high: 0.5,
            threshold_low: -0.5,
            normalize: false,
            target_sparsity: None,
        };
        let quantizer = TernaryQuantizer::new(4, config);

        // Only 2 out of 4 are non-zero
        let vec = vec![0.6, -0.6, 0.1, 0.2];
        let quantized = quantizer.quantize(&vec).unwrap();

        assert!((quantized.sparsity() - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_fit_adaptive() {
        let mut quantizer = TernaryQuantizer::new(
            4,
            TernaryConfig {
                target_sparsity: Some(0.5),
                normalize: false,
                ..Default::default()
            },
        );

        // Training data
        let vectors: Vec<f32> = vec![
            0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4, 0.5, 0.6, 0.7, 0.8, -0.5, -0.6, -0.7, -0.8,
        ];

        quantizer.fit(&vectors, 4).unwrap();

        assert!(quantizer.adaptive_thresholds.is_some());
    }

    #[test]
    fn test_batch_quantize() {
        let quantizer = TernaryQuantizer::with_dimension(4);

        let vectors = vec![0.5, -0.5, 0.0, 0.0, -0.5, 0.5, 0.0, 0.0];

        let quantized = quantizer.quantize_batch(&vectors, 2).unwrap();

        assert_eq!(quantized.len(), 2);
        assert_eq!(quantized[0].dimension(), 4);
        assert_eq!(quantized[1].dimension(), 4);
    }

    #[test]
    fn test_hamming_distance() {
        let config = TernaryConfig {
            threshold_high: 0.3,
            threshold_low: -0.3,
            normalize: false,
            target_sparsity: None,
        };
        let quantizer = TernaryQuantizer::new(4, config);

        let v1 = vec![0.5, -0.5, 0.0, 0.0]; // +1, -1, 0, 0
        let v2 = vec![0.5, 0.5, 0.0, -0.5]; // +1, +1, 0, -1

        let q1 = quantizer.quantize(&v1).unwrap();
        let q2 = quantizer.quantize(&v2).unwrap();

        let hamming = ternary_hamming(&q1, &q2);
        // Positions 1 and 3 differ: 2 differences
        assert_eq!(hamming, 2);
    }

    #[test]
    fn test_memory_efficiency() {
        let quantizer = TernaryQuantizer::with_dimension(1024);
        let vector: Vec<f32> = (0..1024).map(|i| (i as f32 / 1024.0) - 0.5).collect();

        let quantized = quantizer.quantize(&vector).unwrap();

        // 1024 dimensions * 2 bits / 8 bits per byte = 256 bytes
        assert_eq!(quantized.memory_bytes(), 256);

        // Original f32 vector: 1024 * 4 = 4096 bytes
        // Compression ratio: 4096 / 256 = 16x
    }
}
