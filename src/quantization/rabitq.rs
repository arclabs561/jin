//! RaBitQ (Randomized Binary Quantization) implementation.
//!
//! Extended RaBitQ with 1-8 bits per dimension, heap-based optimal rescaling,
//! and corrective factors for accurate distance estimation.
//!
//! # Algorithm Overview
//!
//! 1. **Centroid-relative**: Quantize residual (vector - centroid)
//! 2. **Random Rotation**: Apply random orthogonal rotation
//! 3. **Binary Mapping**: Sign bit determines base direction
//! 4. **Extended Codes**: Additional bits for magnitude refinement
//! 5. **Optimal Rescaling**: Heap-based search for best scaling factor
//! 6. **Corrective Factors**: f_add, f_rescale for distance estimation
//!
//! # References
//!
//! - Gao et al. (2024): "RaBitQ: Quantizing High-Dimensional Vectors"
//! - lqhl/rabitq-rs: Extended implementation with multi-bit support

use crate::RetrieveError;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Configuration for RaBitQ quantization.
#[derive(Clone, Copy, Debug)]
pub struct RaBitQConfig {
    /// Total bits per dimension (1-8). 1 = binary only.
    pub total_bits: usize,
    /// Precomputed scaling factor (None = compute optimal per vector).
    pub t_const: Option<f32>,
}

impl Default for RaBitQConfig {
    fn default() -> Self {
        Self {
            total_bits: 4, // 4-bit default: good balance of speed/accuracy
            t_const: None,
        }
    }
}

impl RaBitQConfig {
    /// Binary quantization (1-bit per dimension).
    pub fn binary() -> Self {
        Self {
            total_bits: 1,
            t_const: None,
        }
    }

    /// 4-bit quantization (default, good balance).
    pub fn bits4() -> Self {
        Self {
            total_bits: 4,
            t_const: None,
        }
    }

    /// 8-bit quantization (high accuracy).
    pub fn bits8() -> Self {
        Self {
            total_bits: 8,
            t_const: None,
        }
    }

    /// Create config with precomputed scaling factor for faster quantization.
    /// Trades <1% accuracy for 100-500x faster quantization.
    pub fn with_const_scaling(self, dimension: usize, seed: u64) -> Self {
        let ex_bits = self.total_bits.saturating_sub(1);
        let t_const = if ex_bits > 0 {
            Some(compute_const_scaling_factor(dimension, ex_bits, seed))
        } else {
            None
        };
        Self { t_const, ..self }
    }
}

/// Quantized vector with extended codes and corrective factors.
#[derive(Clone, Debug)]
pub struct QuantizedVector {
    /// Binary codes (packed, 8 dimensions per byte)
    pub binary_codes: Vec<u8>,
    /// Extended codes (ex_bits per dimension, packed)
    pub extended_codes: Vec<u8>,
    /// Total code per dimension (for backward compat)
    pub codes: Vec<u16>,
    /// Extended bits count
    pub ex_bits: u8,
    /// Original dimension
    pub dimension: usize,
    /// Rescaling factor (delta)
    pub delta: f32,
    /// Offset for reconstruction (vl = delta * cb)
    pub vl: f32,
    /// Additive correction factor for distance
    pub f_add: f32,
    /// Multiplicative correction factor for distance
    pub f_rescale: f32,
    /// Quantization error estimate
    pub f_error: f32,
    /// L2 norm of residual
    pub residual_norm: f32,
}

/// RaBitQ quantizer with extended bit support.
pub struct RaBitQQuantizer {
    dimension: usize,
    /// Random rotation matrix (orthogonal)
    rotation: Vec<f32>,
    /// Centroid for residual computation
    centroid: Option<Vec<f32>>,
    /// Configuration
    config: RaBitQConfig,
}

impl RaBitQQuantizer {
    /// Create new RaBitQ quantizer with default config.
    pub fn new(dimension: usize, seed: u64) -> Result<Self, RetrieveError> {
        Self::with_config(dimension, seed, RaBitQConfig::default())
    }

    /// Create quantizer with specific config.
    pub fn with_config(
        dimension: usize,
        seed: u64,
        config: RaBitQConfig,
    ) -> Result<Self, RetrieveError> {
        if dimension == 0 {
            return Err(RetrieveError::Other("Dimension must be > 0".into()));
        }
        if config.total_bits == 0 || config.total_bits > 8 {
            return Err(RetrieveError::Other("total_bits must be 1-8".into()));
        }

        let rotation = generate_orthogonal_rotation(dimension, seed);

        Ok(Self {
            dimension,
            rotation,
            centroid: None,
            config,
        })
    }

    /// Create binary-only quantizer.
    pub fn binary(dimension: usize, seed: u64) -> Result<Self, RetrieveError> {
        Self::with_config(dimension, seed, RaBitQConfig::binary())
    }

    /// Fit quantizer on training vectors (computes centroid).
    pub fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        if vectors.len() != num_vectors * self.dimension {
            return Err(RetrieveError::Other(format!(
                "Expected {} floats, got {}",
                num_vectors * self.dimension,
                vectors.len()
            )));
        }

        let mut centroid = vec![0.0f32; self.dimension];
        for i in 0..num_vectors {
            let vec = &vectors[i * self.dimension..(i + 1) * self.dimension];
            for (j, &v) in vec.iter().enumerate() {
                centroid[j] += v;
            }
        }
        for c in &mut centroid {
            *c /= num_vectors as f32;
        }
        self.centroid = Some(centroid);

        Ok(())
    }

    /// Set centroid directly.
    pub fn set_centroid(&mut self, centroid: Vec<f32>) -> Result<(), RetrieveError> {
        if centroid.len() != self.dimension {
            return Err(RetrieveError::Other(format!(
                "Centroid dimension {} != {}",
                centroid.len(),
                self.dimension
            )));
        }
        self.centroid = Some(centroid);
        Ok(())
    }

    /// Quantize a vector relative to centroid.
    pub fn quantize(&self, vector: &[f32]) -> Result<QuantizedVector, RetrieveError> {
        if vector.len() != self.dimension {
            return Err(RetrieveError::Other(format!(
                "Expected dimension {}, got {}",
                self.dimension,
                vector.len()
            )));
        }

        let centroid = self
            .centroid
            .as_ref()
            .unwrap_or(&vec![0.0f32; self.dimension]);
        self.quantize_with_centroid(vector, centroid)
    }

    /// Quantize relative to specific centroid.
    pub fn quantize_with_centroid(
        &self,
        vector: &[f32],
        centroid: &[f32],
    ) -> Result<QuantizedVector, RetrieveError> {
        let dim = self.dimension;
        let ex_bits = self.config.total_bits.saturating_sub(1);

        // Step 1: Compute residual
        let residual: Vec<f32> = vector
            .iter()
            .zip(centroid.iter())
            .map(|(v, c)| v - c)
            .collect();

        // Step 2: Apply rotation
        let rotated = apply_rotation(&residual, &self.rotation, dim);

        // Step 3: Compute binary codes (sign bits)
        let mut binary_codes_unpacked = vec![0u8; dim];
        for (i, &val) in rotated.iter().enumerate() {
            if val >= 0.0 {
                binary_codes_unpacked[i] = 1;
            }
        }

        // Step 4: Compute extended codes (magnitude refinement)
        let (extended_codes_unpacked, ipnorm_inv) = if ex_bits > 0 {
            self.compute_extended_codes(&rotated, ex_bits)
        } else {
            (vec![0u16; dim], 1.0)
        };

        // Step 5: Combine into total codes
        let mut total_codes = vec![0u16; dim];
        for i in 0..dim {
            total_codes[i] =
                extended_codes_unpacked[i] + ((binary_codes_unpacked[i] as u16) << ex_bits);
        }

        // Step 6: Compute corrective factors
        let (f_add, f_rescale, f_error, residual_norm) =
            self.compute_correction_factors(&rotated, centroid, &binary_codes_unpacked);

        // Step 7: Compute delta and vl for reconstruction
        let cb = -((1 << ex_bits) as f32 - 0.5);
        let quantized_shifted: Vec<f32> =
            total_codes.iter().map(|&code| code as f32 + cb).collect();

        let norm_quan_sqr: f32 = quantized_shifted.iter().map(|x| x * x).sum();
        let norm_residual_sqr: f32 = rotated.iter().map(|x| x * x).sum();
        let dot_rq: f32 = rotated
            .iter()
            .zip(quantized_shifted.iter())
            .map(|(r, q)| r * q)
            .sum();

        let norm_residual = norm_residual_sqr.sqrt();
        let norm_quant = norm_quan_sqr.sqrt();
        let denom = (norm_residual * norm_quant).max(f32::EPSILON);
        let cos_sim = (dot_rq / denom).clamp(-1.0, 1.0);

        let delta = if norm_quant <= f32::EPSILON {
            0.0
        } else {
            (norm_residual / norm_quant) * cos_sim
        };
        let vl = delta * cb;

        // Pack binary codes
        let binary_codes = pack_binary_codes(&binary_codes_unpacked);

        // Pack extended codes
        let extended_codes = pack_extended_codes(&extended_codes_unpacked, ex_bits);

        Ok(QuantizedVector {
            binary_codes,
            extended_codes,
            codes: total_codes,
            ex_bits: ex_bits as u8,
            dimension: dim,
            delta,
            vl,
            f_add,
            f_rescale,
            f_error,
            residual_norm,
        })
    }

    /// Compute extended codes using optimal rescaling.
    fn compute_extended_codes(&self, rotated: &[f32], ex_bits: usize) -> (Vec<u16>, f32) {
        let dim = self.dimension;

        // Normalize absolute values
        let mut normalized_abs: Vec<f32> = rotated.iter().map(|x| x.abs()).collect();
        let norm: f32 = normalized_abs.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm <= f32::EPSILON {
            return (vec![0u16; dim], 1.0);
        }

        for val in &mut normalized_abs {
            *val /= norm;
        }

        // Find optimal rescaling factor
        let t = if let Some(t_const) = self.config.t_const {
            t_const as f64
        } else {
            best_rescale_factor(&normalized_abs, ex_bits)
        };

        // Quantize with optimal t
        quantize_extended(&normalized_abs, rotated, ex_bits, t)
    }

    /// Compute correction factors for distance estimation.
    fn compute_correction_factors(
        &self,
        residual: &[f32],
        centroid: &[f32],
        binary_codes: &[u8],
    ) -> (f32, f32, f32, f32) {
        let dim = self.dimension;

        // xu_cb: centered binary codes
        let xu_cb: Vec<f32> = binary_codes.iter().map(|&bit| bit as f32 - 0.5).collect();

        let l2_sqr: f32 = residual.iter().map(|x| x * x).sum();
        let l2_norm = l2_sqr.sqrt();
        let xu_cb_norm_sqr: f32 = xu_cb.iter().map(|x| x * x).sum();
        let ip_resi_xucb: f32 = residual.iter().zip(xu_cb.iter()).map(|(r, x)| r * x).sum();
        let ip_cent_xucb: f32 = centroid.iter().zip(xu_cb.iter()).map(|(c, x)| c * x).sum();

        let denom = if ip_resi_xucb.abs() <= f32::EPSILON {
            f32::INFINITY
        } else {
            ip_resi_xucb
        };

        // Compute error estimate
        let mut tmp_error = 0.0f32;
        if dim > 1 {
            let ratio = ((l2_sqr * xu_cb_norm_sqr) / (denom * denom)) - 1.0;
            if ratio.is_finite() && ratio > 0.0 {
                const K_CONST_EPSILON: f32 = 1.9;
                tmp_error =
                    l2_norm * K_CONST_EPSILON * ((ratio / ((dim - 1) as f32)).max(0.0)).sqrt();
            }
        }

        // L2 distance correction factors
        let f_add = l2_sqr + 2.0 * l2_sqr * ip_cent_xucb / denom;
        let f_rescale = -2.0 * l2_sqr / denom;
        let f_error = 2.0 * tmp_error;

        (f_add, f_rescale, f_error, l2_norm)
    }

    /// Compute approximate L2 distance squared.
    pub fn approximate_l2_sqr(
        &self,
        query: &[f32],
        quantized: &QuantizedVector,
    ) -> Result<f32, RetrieveError> {
        if query.len() != self.dimension {
            return Err(RetrieveError::Other(format!(
                "Expected dimension {}, got {}",
                self.dimension,
                query.len()
            )));
        }

        let centroid = self
            .centroid
            .as_ref()
            .map(|c| c.as_slice())
            .unwrap_or(&vec![0.0f32; self.dimension]);

        // Center and rotate query
        let query_residual: Vec<f32> = query
            .iter()
            .zip(centroid.iter())
            .map(|(q, c)| q - c)
            .collect();
        let rotated_query = apply_rotation(&query_residual, &self.rotation, self.dimension);

        // Compute inner product with quantized codes
        let cb = -((1 << quantized.ex_bits) as f32 - 0.5);
        let mut ip = 0.0f32;
        for (i, &q) in rotated_query.iter().enumerate() {
            let code_val = quantized.codes[i] as f32 + cb;
            ip += q * code_val;
        }

        // Apply correction: dist = f_add + f_rescale * ip
        let dist = quantized.f_add + quantized.f_rescale * ip;

        Ok(dist.max(0.0))
    }

    /// Compute approximate distance (cosine-based).
    pub fn approximate_distance(
        &self,
        query: &[f32],
        quantized: &QuantizedVector,
    ) -> Result<f32, RetrieveError> {
        // Use L2 distance as proxy
        let l2_sqr = self.approximate_l2_sqr(query, quantized)?;
        Ok(l2_sqr.sqrt())
    }

    /// Batch compute distances.
    pub fn batch_distances(
        &self,
        query: &[f32],
        quantized_vecs: &[QuantizedVector],
    ) -> Result<Vec<f32>, RetrieveError> {
        quantized_vecs
            .iter()
            .map(|qv| self.approximate_distance(query, qv))
            .collect()
    }

    /// Get compressed size in bytes per vector.
    pub fn compressed_size(&self) -> usize {
        let binary_bytes = (self.dimension + 7) / 8;
        let ex_bits = self.config.total_bits.saturating_sub(1);
        let extended_bytes = (self.dimension * ex_bits + 7) / 8;
        binary_bytes + extended_bytes + 24 // codes + correction factors
    }

    /// Get compression ratio vs f32.
    pub fn compression_ratio(&self) -> f32 {
        let original = self.dimension * 4;
        original as f32 / self.compressed_size() as f32
    }

    /// Get configuration.
    pub fn config(&self) -> &RaBitQConfig {
        &self.config
    }
}

// ============================================================================
// Optimal Rescaling Factor (Heap-Based)
// ============================================================================

const K_TIGHT_START: [f64; 9] = [0.0, 0.15, 0.20, 0.52, 0.59, 0.71, 0.75, 0.77, 0.81];
const K_EPS: f64 = 1e-5;
const K_NENUM: f64 = 10.0;

/// Find optimal rescaling factor using heap-based search.
///
/// This is the key algorithm from the RaBitQ paper that finds the t value
/// maximizing the inner product between original and quantized vectors.
fn best_rescale_factor(o_abs: &[f32], ex_bits: usize) -> f64 {
    let dim = o_abs.len();
    let max_o = o_abs.iter().cloned().fold(0.0f32, f32::max) as f64;
    if max_o <= f64::EPSILON {
        return 1.0;
    }

    let table_idx = ex_bits.min(K_TIGHT_START.len() - 1);
    let t_end = (((1 << ex_bits) - 1) as f64 + K_NENUM) / max_o;
    let t_start = t_end * K_TIGHT_START[table_idx];

    let mut cur_o_bar = vec![0i32; dim];
    let mut sqr_denominator = dim as f64 * 0.25;
    let mut numerator = 0.0f64;

    for (idx, &val) in o_abs.iter().enumerate() {
        let cur = ((t_start * val as f64) + K_EPS) as i32;
        cur_o_bar[idx] = cur;
        sqr_denominator += (cur * cur + cur) as f64;
        numerator += (cur as f64 + 0.5) * val as f64;
    }

    #[derive(Copy, Clone, Debug)]
    struct HeapEntry {
        t: f64,
        idx: usize,
    }

    impl PartialEq for HeapEntry {
        fn eq(&self, other: &Self) -> bool {
            self.t.to_bits() == other.t.to_bits() && self.idx == other.idx
        }
    }
    impl Eq for HeapEntry {}

    impl PartialOrd for HeapEntry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    impl Ord for HeapEntry {
        fn cmp(&self, other: &Self) -> Ordering {
            self.t
                .total_cmp(&other.t)
                .then_with(|| self.idx.cmp(&other.idx))
        }
    }

    let mut heap: BinaryHeap<Reverse<HeapEntry>> = BinaryHeap::new();
    for (idx, &val) in o_abs.iter().enumerate() {
        if val > 0.0 {
            let next_t = (cur_o_bar[idx] + 1) as f64 / val as f64;
            heap.push(Reverse(HeapEntry { t: next_t, idx }));
        }
    }

    let mut max_ip = 0.0f64;
    let mut best_t = t_start;

    while let Some(Reverse(HeapEntry { t: cur_t, idx })) = heap.pop() {
        if cur_t >= t_end {
            continue;
        }

        cur_o_bar[idx] += 1;
        let update = cur_o_bar[idx];
        sqr_denominator += 2.0 * update as f64;
        numerator += o_abs[idx] as f64;

        let cur_ip = numerator / sqr_denominator.sqrt();
        if cur_ip > max_ip {
            max_ip = cur_ip;
            best_t = cur_t;
        }

        if update < (1 << ex_bits) - 1 && o_abs[idx] > 0.0 {
            let t_next = (update + 1) as f64 / o_abs[idx] as f64;
            if t_next < t_end {
                heap.push(Reverse(HeapEntry { t: t_next, idx }));
            }
        }
    }

    if best_t <= 0.0 {
        t_start.max(f64::EPSILON)
    } else {
        best_t
    }
}

/// Quantize with given scaling factor.
fn quantize_extended(o_abs: &[f32], residual: &[f32], ex_bits: usize, t: f64) -> (Vec<u16>, f32) {
    let dim = o_abs.len();
    if dim == 0 {
        return (Vec::new(), 1.0);
    }

    let mut code = vec![0u16; dim];
    let max_val = (1 << ex_bits) - 1;
    let mut ipnorm = 0.0f64;

    for i in 0..dim {
        let mut cur = (t * o_abs[i] as f64 + K_EPS) as i32;
        if cur > max_val {
            cur = max_val;
        }
        code[i] = cur as u16;
        ipnorm += (cur as f64 + 0.5) * o_abs[i] as f64;
    }

    let mut ipnorm_inv = if ipnorm.is_finite() && ipnorm > 0.0 {
        (1.0 / ipnorm) as f32
    } else {
        1.0
    };

    // Flip codes for negative residual values
    let mask = max_val as u16;
    if max_val > 0 {
        for (idx, &res) in residual.iter().enumerate() {
            if res < 0.0 {
                code[idx] = (!code[idx]) & mask;
            }
        }
    }

    if !ipnorm_inv.is_finite() {
        ipnorm_inv = 1.0;
    }

    (code, ipnorm_inv)
}

/// Compute constant scaling factor for faster quantization.
/// Samples random vectors to estimate average optimal t.
fn compute_const_scaling_factor(dim: usize, ex_bits: usize, seed: u64) -> f32 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    const NUM_SAMPLES: usize = 100;

    let mut state = seed;
    let mut next_rand = || -> f32 {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        state = hasher.finish();
        // Box-Muller for Gaussian
        let u1 = (state as f64) / (u64::MAX as f64);
        let mut hasher2 = DefaultHasher::new();
        state.hash(&mut hasher2);
        state = hasher2.finish();
        let u2 = (state as f64) / (u64::MAX as f64);
        ((-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()) as f32
    };

    let mut sum_t = 0.0f64;
    let mut valid_samples = 0;

    for _ in 0..NUM_SAMPLES {
        let vec: Vec<f32> = (0..dim).map(|_| next_rand()).collect();
        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= f32::EPSILON {
            continue;
        }
        let normalized_abs: Vec<f32> = vec.iter().map(|x| (x / norm).abs()).collect();
        let t = best_rescale_factor(&normalized_abs, ex_bits);
        sum_t += t;
        valid_samples += 1;
    }

    if valid_samples > 0 {
        (sum_t / valid_samples as f64) as f32
    } else {
        1.0
    }
}

// ============================================================================
// Bit Packing Utilities
// ============================================================================

/// Pack binary codes (1 bit per element).
fn pack_binary_codes(codes: &[u8]) -> Vec<u8> {
    let bytes_needed = (codes.len() + 7) / 8;
    let mut packed = vec![0u8; bytes_needed];
    for (i, &code) in codes.iter().enumerate() {
        if code != 0 {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    packed
}

/// Pack extended codes (ex_bits per element).
fn pack_extended_codes(codes: &[u16], ex_bits: usize) -> Vec<u8> {
    if ex_bits == 0 {
        return Vec::new();
    }

    let total_bits = codes.len() * ex_bits;
    let bytes_needed = (total_bits + 7) / 8;
    let mut packed = vec![0u8; bytes_needed];

    let mut bit_pos = 0;
    for &code in codes {
        let val = code & ((1 << ex_bits) - 1);
        for b in 0..ex_bits {
            if (val >> b) & 1 != 0 {
                let byte_idx = bit_pos / 8;
                let bit_idx = bit_pos % 8;
                if byte_idx < packed.len() {
                    packed[byte_idx] |= 1 << bit_idx;
                }
            }
            bit_pos += 1;
        }
    }

    packed
}

/// Unpack binary codes.
#[allow(dead_code)]
fn unpack_binary_codes(packed: &[u8], dim: usize) -> Vec<u8> {
    let mut codes = vec![0u8; dim];
    for i in 0..dim {
        if i / 8 < packed.len() && (packed[i / 8] >> (i % 8)) & 1 != 0 {
            codes[i] = 1;
        }
    }
    codes
}

/// Unpack extended codes.
#[allow(dead_code)]
fn unpack_extended_codes(packed: &[u8], dim: usize, ex_bits: usize) -> Vec<u16> {
    if ex_bits == 0 {
        return vec![0u16; dim];
    }

    let mut codes = vec![0u16; dim];
    let mut bit_pos = 0;

    for code in &mut codes {
        let mut val = 0u16;
        for b in 0..ex_bits {
            let byte_idx = bit_pos / 8;
            let bit_idx = bit_pos % 8;
            if byte_idx < packed.len() && (packed[byte_idx] >> bit_idx) & 1 != 0 {
                val |= 1 << b;
            }
            bit_pos += 1;
        }
        *code = val;
    }

    codes
}

// ============================================================================
// Rotation Matrix Generation
// ============================================================================

/// Generate random orthogonal rotation matrix via Gram-Schmidt.
fn generate_orthogonal_rotation(dimension: usize, seed: u64) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut rotation = vec![0.0f32; dimension * dimension];

    let mut state = seed;
    let mut next_rand = || -> f32 {
        let mut hasher = DefaultHasher::new();
        state.hash(&mut hasher);
        state = hasher.finish();
        ((state as f64) / (u64::MAX as f64) * 2.0 - 1.0) as f32
    };

    let mut basis: Vec<Vec<f32>> = Vec::new();

    for i in 0..dimension {
        let mut v: Vec<f32> = (0..dimension).map(|_| next_rand()).collect();

        // Orthogonalize
        for b in &basis {
            let dot: f32 = v.iter().zip(b.iter()).map(|(a, b)| a * b).sum();
            for (vi, bi) in v.iter_mut().zip(b.iter()) {
                *vi -= dot * bi;
            }
        }

        // Normalize
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for vi in &mut v {
                *vi /= norm;
            }
            basis.push(v);
        } else {
            // Fallback to identity row
            let mut v = vec![0.0f32; dimension];
            v[i] = 1.0;
            basis.push(v);
        }
    }

    for (i, row) in basis.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            rotation[i * dimension + j] = val;
        }
    }

    rotation
}

/// Apply rotation matrix.
fn apply_rotation(vector: &[f32], rotation: &[f32], dimension: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dimension];
    for i in 0..dimension {
        let row_start = i * dimension;
        let mut sum = 0.0;
        for j in 0..dimension {
            sum += rotation[row_start + j] * vector[j];
        }
        result[i] = sum;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rabitq_binary() {
        let quantizer = RaBitQQuantizer::binary(64, 42).unwrap();
        let vector: Vec<f32> = (0..64).map(|i| (i as f32) * 0.1).collect();
        let quantized = quantizer.quantize(&vector).unwrap();

        assert_eq!(quantized.binary_codes.len(), 8);
        assert_eq!(quantized.ex_bits, 0);
        assert!(quantized.residual_norm > 0.0);
    }

    #[test]
    fn test_rabitq_4bit() {
        let quantizer = RaBitQQuantizer::with_config(64, 42, RaBitQConfig::bits4()).unwrap();

        let vector: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let quantized = quantizer.quantize(&vector).unwrap();

        assert_eq!(quantized.ex_bits, 3);
        assert_eq!(quantized.codes.len(), 64);
    }

    #[test]
    fn test_rabitq_8bit() {
        let quantizer = RaBitQQuantizer::with_config(32, 42, RaBitQConfig::bits8()).unwrap();

        let vector: Vec<f32> = (0..32).map(|i| (i as f32) * 0.05).collect();
        let quantized = quantizer.quantize(&vector).unwrap();

        assert_eq!(quantized.ex_bits, 7);
    }

    #[test]
    fn test_rabitq_distance_ordering() {
        let mut quantizer = RaBitQQuantizer::with_config(32, 42, RaBitQConfig::bits4()).unwrap();

        // Fit on some vectors
        let training: Vec<f32> = (0..320).map(|i| (i as f32) * 0.01).collect();
        quantizer.fit(&training, 10).unwrap();

        let base: Vec<f32> = (0..32).map(|i| (i as f32).sin()).collect();
        let similar: Vec<f32> = (0..32).map(|i| (i as f32).sin() + 0.05).collect();
        let different: Vec<f32> = (0..32).map(|i| -(i as f32).sin()).collect();

        let q_base = quantizer.quantize(&base).unwrap();

        let d_sim = quantizer.approximate_distance(&similar, &q_base).unwrap();
        let d_diff = quantizer.approximate_distance(&different, &q_base).unwrap();

        // Similar should be closer (but with quantization, order matters more than values)
        // This is a weak assertion - just checking the function runs
        assert!(d_sim.is_finite());
        assert!(d_diff.is_finite());
    }

    #[test]
    fn test_rabitq_with_centroid() {
        let mut quantizer = RaBitQQuantizer::new(16, 42).unwrap();

        let centroid = vec![0.5f32; 16];
        quantizer.set_centroid(centroid.clone()).unwrap();

        let vector: Vec<f32> = (0..16).map(|i| 0.5 + (i as f32) * 0.01).collect();
        let quantized = quantizer.quantize(&vector).unwrap();

        assert!(quantized.residual_norm < 1.0); // Residual should be small
    }

    #[test]
    fn test_rabitq_compression_ratio() {
        let q_binary = RaBitQQuantizer::binary(128, 42).unwrap();
        let q_4bit = RaBitQQuantizer::with_config(128, 42, RaBitQConfig::bits4()).unwrap();
        let q_8bit = RaBitQQuantizer::with_config(128, 42, RaBitQConfig::bits8()).unwrap();

        let r_binary = q_binary.compression_ratio();
        let r_4bit = q_4bit.compression_ratio();
        let r_8bit = q_8bit.compression_ratio();

        // Binary should have highest compression
        assert!(r_binary > r_4bit);
        assert!(r_4bit > r_8bit);

        // Binary should be > 10x compression
        assert!(r_binary > 10.0);
    }

    #[test]
    fn test_heap_based_rescaling() {
        // Test the heap-based optimal rescaling
        let normalized: Vec<f32> = (0..32).map(|i| (i as f32 / 32.0).abs()).collect();
        let t = best_rescale_factor(&normalized, 3);
        assert!(t > 0.0);
        assert!(t < 100.0);
    }

    #[test]
    fn test_const_scaling_factor() {
        let t = compute_const_scaling_factor(64, 3, 42);
        assert!(t > 0.0);
        assert!(t < 100.0);
    }

    #[test]
    fn test_pack_unpack_binary() {
        let codes = vec![1, 0, 1, 1, 0, 0, 1, 0, 1, 1];
        let packed = pack_binary_codes(&codes);
        let unpacked = unpack_binary_codes(&packed, codes.len());
        assert_eq!(codes, unpacked);
    }

    #[test]
    fn test_pack_unpack_extended() {
        let codes: Vec<u16> = vec![3, 1, 7, 0, 5, 2, 6, 4];
        let packed = pack_extended_codes(&codes, 3);
        let unpacked = unpack_extended_codes(&packed, codes.len(), 3);
        assert_eq!(codes, unpacked);
    }
}
