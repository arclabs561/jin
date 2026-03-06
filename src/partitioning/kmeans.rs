//! k-means clustering -- thin wrapper around `clump::Kmeans<CosineDistance>`.
//!
//! All computation is delegated to `clump`. This module provides the same public API
//! that vicinity's partitioning and IVF-PQ code expects (SoA flat-buffer input, mutable
//! `fit`, `assign_clusters`, `centroids`).

use crate::clump_compat::{map_clump_error, soa_to_aos};
use crate::RetrieveError;

/// k-means clustering for partitioning vectors.
///
/// Uses cosine distance (via `clump::CosineDistance`) with k-means++ initialization.
pub struct KMeans {
    dimension: usize,
    k: usize,
    seed: Option<u64>,
    /// Stored fit result from clump; `None` before `fit()` is called.
    fit: Option<clump::KmeansFit<clump::CosineDistance>>,
}

impl KMeans {
    /// Create new k-means with k clusters.
    pub fn new(dimension: usize, k: usize) -> Result<Self, RetrieveError> {
        if dimension == 0 || k == 0 {
            return Err(RetrieveError::InvalidParameter(
                "Dimension and k must be greater than 0".to_string(),
            ));
        }

        Ok(Self {
            dimension,
            k,
            seed: None,
            fit: None,
        })
    }

    /// Configure a deterministic seed for k-means++ initialization.
    ///
    /// When set, repeated `fit(...)` calls on the same inputs produce identical results.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Train k-means on vectors.
    ///
    /// `vectors` is a flat SoA buffer of length `num_vectors * dimension`.
    pub fn fit(&mut self, vectors: &[f32], num_vectors: usize) -> Result<(), RetrieveError> {
        if vectors.len() < num_vectors * self.dimension {
            return Err(RetrieveError::InvalidParameter(
                "Insufficient vectors".to_string(),
            ));
        }

        let data = soa_to_aos(vectors, num_vectors, self.dimension);

        // Clamp k to the number of available vectors. The old implementation
        // silently produced fewer centroids when k > n; clump validates k <= n
        // and returns an error. Clamping preserves backward compatibility.
        let effective_k = self.k.min(num_vectors);

        let mut builder =
            clump::Kmeans::with_metric(effective_k, clump::CosineDistance).with_tol(1e-6);

        if let Some(s) = self.seed {
            builder = builder.with_seed(s);
        }

        let result = builder.fit(&data).map_err(map_clump_error)?;
        self.fit = Some(result);

        Ok(())
    }

    /// Assign vectors to nearest clusters.
    ///
    /// Uses the centroids learned during `fit()`. Falls back to a simple
    /// nearest-centroid scan when `fit()` has not been called (returns empty vec
    /// if no centroids exist, preserving the previous behavior).
    pub fn assign_clusters(&self, vectors: &[f32], num_vectors: usize) -> Vec<usize> {
        let Some(ref fit) = self.fit else {
            return Vec::new();
        };

        let data = soa_to_aos(vectors, num_vectors, self.dimension);

        // `predict` can only fail on empty input or dimension mismatch, which
        // callers should never hit after a successful `fit`.
        fit.predict(&data).unwrap_or_default()
    }

    /// Get centroids.
    pub fn centroids(&self) -> &[Vec<f32>] {
        match self.fit {
            Some(ref f) => &f.centroids,
            None => &[],
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, clippy::expect_used)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn l2_normalize_in_place(vecs: &mut [f32], num_vectors: usize, dimension: usize) {
        for i in 0..num_vectors {
            let start = i * dimension;
            let end = start + dimension;
            let v = &mut vecs[start..end];
            let norm2: f32 = v.iter().map(|&x| x * x).sum();
            let norm = norm2.sqrt();
            if norm > 0.0 {
                for x in v {
                    *x /= norm;
                }
            } else if !v.is_empty() {
                v[0] = 1.0;
            }
        }
    }

    proptest! {
        #[test]
        fn prop_kmeans_fit_is_deterministic_given_seed(
            seed in any::<u64>(),
            dimension in 1usize..16,
            num_vectors in 2usize..64,
            k in 1usize..16,
            raw in proptest::collection::vec(-1.0f32..1.0f32, 2usize..(64*16)),
        ) {
            prop_assume!(k <= num_vectors);
            let needed = num_vectors * dimension;
            prop_assume!(raw.len() >= needed);

            let mut vectors = raw[..needed].to_vec();
            l2_normalize_in_place(&mut vectors, num_vectors, dimension);

            let mut km1 = KMeans::new(dimension, k).unwrap().with_seed(seed);
            let mut km2 = KMeans::new(dimension, k).unwrap().with_seed(seed);

            km1.fit(&vectors, num_vectors).unwrap();
            km2.fit(&vectors, num_vectors).unwrap();

            let a1 = km1.assign_clusters(&vectors, num_vectors);
            let a2 = km2.assign_clusters(&vectors, num_vectors);
            prop_assert_eq!(a1, a2);
        }
    }
}
