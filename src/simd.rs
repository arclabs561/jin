//! Vector operations with SIMD acceleration.
//!
//! Copied from `ordino-retrieve` into `prox` because many ANN algorithms
//! need fast dot/cosine during graph construction/search.
//!
//! Notes:
//! - This is CPU-only and uses runtime feature detection on x86_64.
//! - For normalized embeddings, prefer `dot()` over `cosine()`.

const MIN_DIM_SIMD: usize = 16;
const NORM_EPSILON: f32 = 1e-9;

#[inline]
#[must_use]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());

    #[cfg(target_arch = "x86_64")]
    {
        if n >= MIN_DIM_SIMD && is_x86_feature_detected!("avx512f") {
            return unsafe { dot_avx512(a, b) };
        }
        if n >= MIN_DIM_SIMD && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")
        {
            return unsafe { dot_avx2(a, b) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if n >= MIN_DIM_SIMD {
            return unsafe { dot_neon(a, b) };
        }
    }
    #[allow(unreachable_code)]
    dot_portable(a, b)
}

#[inline]
#[must_use]
pub fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

#[inline]
#[must_use]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let d = dot(a, b);
    let na = norm(a);
    let nb = norm(b);
    if na > NORM_EPSILON && nb > NORM_EPSILON {
        d / (na * nb)
    } else {
        0.0
    }
}

#[inline]
#[must_use]
pub fn dot_portable(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[inline]
#[must_use]
pub fn sparse_dot(a_indices: &[u32], a_values: &[f32], b_indices: &[u32], b_values: &[f32]) -> f32 {
    if a_indices.len() < 8 || b_indices.len() < 8 {
        return sparse_dot_portable(a_indices, a_values, b_indices, b_values);
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return unsafe { sparse_dot_avx512(a_indices, a_values, b_indices, b_values) };
        }
        if is_x86_feature_detected!("avx2") {
            return unsafe { sparse_dot_avx2(a_indices, a_values, b_indices, b_values) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { sparse_dot_neon(a_indices, a_values, b_indices, b_values) };
    }
    #[allow(unreachable_code)]
    sparse_dot_portable(a_indices, a_values, b_indices, b_values)
}

#[inline]
#[must_use]
pub fn sparse_dot_portable(
    a_indices: &[u32],
    a_values: &[f32],
    b_indices: &[u32],
    b_values: &[f32],
) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut result = 0.0;

    while i < a_indices.len() && j < b_indices.len() {
        if a_indices[i] < b_indices[j] {
            i += 1;
        } else if a_indices[i] > b_indices[j] {
            j += 1;
        } else {
            result += a_values[i] * b_values[j];
            i += 1;
            j += 1;
        }
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX-512 (x86_64)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn dot_avx512(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        __m256, __m512, _mm256_add_ps, _mm512_castps512_ps256, _mm512_extractf32x8_ps,
        _mm512_fmadd_ps, _mm512_loadu_ps, _mm512_setzero_ps,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 16;
    let remainder = n % 16;

    let mut sum: __m512 = _mm512_setzero_ps();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 16;
        let va = _mm512_loadu_ps(a_ptr.add(offset));
        let vb = _mm512_loadu_ps(b_ptr.add(offset));
        sum = _mm512_fmadd_ps(va, vb, sum);
    }

    let sum256_lo: __m256 = _mm512_castps512_ps256(sum);
    let sum256_hi: __m256 = _mm512_extractf32x8_ps::<1>(sum);
    let sum256: __m256 = _mm256_add_ps(sum256_lo, sum256_hi);

    use std::arch::x86_64::{
        _mm256_castps256_ps128, _mm256_extractf128_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32,
        _mm_movehl_ps, _mm_shuffle_ps,
    };
    let hi = _mm256_extractf128_ps(sum256, 1);
    let lo = _mm256_castps256_ps128(sum256);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    let tail_start = chunks * 16;
    for i in 0..remainder {
        result += *a.get_unchecked(tail_start + i) * *b.get_unchecked(tail_start + i);
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// AVX2 + FMA (x86_64)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::x86_64::{
        __m256, _mm256_castps256_ps128, _mm256_extractf128_ps, _mm256_fmadd_ps, _mm256_loadu_ps,
        _mm256_setzero_ps, _mm_add_ps, _mm_add_ss, _mm_cvtss_f32, _mm_movehl_ps, _mm_shuffle_ps,
    };

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 8;
    let remainder = n % 8;

    let mut sum: __m256 = _mm256_setzero_ps();
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 8;
        let va = _mm256_loadu_ps(a_ptr.add(offset));
        let vb = _mm256_loadu_ps(b_ptr.add(offset));
        sum = _mm256_fmadd_ps(va, vb, sum);
    }

    let hi = _mm256_extractf128_ps(sum, 1);
    let lo = _mm256_castps256_ps128(sum);
    let sum128 = _mm_add_ps(lo, hi);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    let mut result = _mm_cvtss_f32(sum32);

    let tail_start = chunks * 8;
    for i in 0..remainder {
        result += *a.get_unchecked(tail_start + i) * *b.get_unchecked(tail_start + i);
    }

    result
}

// ─────────────────────────────────────────────────────────────────────────────
// NEON (aarch64)
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_neon(a: &[f32], b: &[f32]) -> f32 {
    use std::arch::aarch64::{float32x4_t, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32};

    let n = a.len().min(b.len());
    if n == 0 {
        return 0.0;
    }

    let chunks = n / 4;
    let remainder = n % 4;

    let mut sum: float32x4_t = vdupq_n_f32(0.0);
    let a_ptr = a.as_ptr();
    let b_ptr = b.as_ptr();

    for i in 0..chunks {
        let offset = i * 4;
        let va = vld1q_f32(a_ptr.add(offset));
        let vb = vld1q_f32(b_ptr.add(offset));
        sum = vfmaq_f32(sum, va, vb);
    }

    let mut result = vaddvq_f32(sum);
    let tail_start = chunks * 4;
    for i in 0..remainder {
        result += *a.get_unchecked(tail_start + i) * *b.get_unchecked(tail_start + i);
    }

    result
}

// Sparse SIMD implementations: keep portable-only on unsupported targets.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn sparse_dot_avx512(
    a_indices: &[u32],
    a_values: &[f32],
    b_indices: &[u32],
    b_values: &[f32],
) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut result = 0.0;

    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi32, _mm256_cmpgt_epi32, _mm256_loadu_si256};

    while i + 8 <= a_indices.len() && j + 8 <= b_indices.len() {
        let a_idx = _mm256_loadu_si256(a_indices.as_ptr().add(i) as *const __m256i);
        let b_idx = _mm256_loadu_si256(b_indices.as_ptr().add(j) as *const __m256i);
        let _eq_mask = _mm256_cmpeq_epi32(a_idx, b_idx);
        let _gt_mask = _mm256_cmpgt_epi32(a_idx, b_idx);

        let a_min = a_indices[i];
        let a_max = a_indices[i + 7];
        let b_min = b_indices[j];
        let b_max = b_indices[j + 7];

        let mut ai = i;
        let mut bj = j;
        while ai < i + 8 && bj < j + 8 {
            if a_indices[ai] < b_indices[bj] {
                ai += 1;
            } else if a_indices[ai] > b_indices[bj] {
                bj += 1;
            } else {
                result += a_values[ai] * b_values[bj];
                ai += 1;
                bj += 1;
            }
        }

        if a_max < b_min {
            i += 8;
        } else if b_max < a_min {
            j += 8;
        } else if a_max <= b_max {
            i += 8;
        } else {
            j += 8;
        }
    }

    sparse_dot_portable(&a_indices[i..], &a_values[i..], &b_indices[j..], &b_values[j..]) + result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sparse_dot_avx2(
    a_indices: &[u32],
    a_values: &[f32],
    b_indices: &[u32],
    b_values: &[f32],
) -> f32 {
    let mut i = 0;
    let mut j = 0;
    let mut result = 0.0;

    use std::arch::x86_64::{__m256i, _mm256_cmpeq_epi32, _mm256_cmpgt_epi32, _mm256_loadu_si256};

    while i + 8 <= a_indices.len() && j + 8 <= b_indices.len() {
        let a_idx = _mm256_loadu_si256(a_indices.as_ptr().add(i) as *const __m256i);
        let b_idx = _mm256_loadu_si256(b_indices.as_ptr().add(j) as *const __m256i);
        let _eq_mask = _mm256_cmpeq_epi32(a_idx, b_idx);
        let _gt_mask = _mm256_cmpgt_epi32(a_idx, b_idx);

        let mut ai = i;
        let mut bj = j;
        while ai < i + 8 && bj < j + 8 {
            if a_indices[ai] < b_indices[bj] {
                ai += 1;
            } else if a_indices[ai] > b_indices[bj] {
                bj += 1;
            } else {
                result += a_values[ai] * b_values[bj];
                ai += 1;
                bj += 1;
            }
        }

        let a_max = a_indices[i + 7];
        let b_max = b_indices[j + 7];
        if a_max < b_indices[j] {
            i += 8;
        } else if b_max < a_indices[i] {
            j += 8;
        } else if a_max <= b_max {
            i += 8;
        } else {
            j += 8;
        }
    }

    sparse_dot_portable(&a_indices[i..], &a_values[i..], &b_indices[j..], &b_values[j..]) + result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sparse_dot_neon(
    a_indices: &[u32],
    a_values: &[f32],
    b_indices: &[u32],
    b_values: &[f32],
) -> f32 {
    // Keep simple: scalar merge for now.
    sparse_dot_portable(a_indices, a_values, b_indices, b_values)
}

