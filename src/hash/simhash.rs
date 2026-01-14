//! SimHash for cosine similarity via binary fingerprints.
//!
//! SimHash produces binary fingerprints where Hamming distance approximates
//! angular distance (and thus cosine similarity).
//!
//! ## Algorithm
//!
//! For a document represented as weighted features:
//! 1. Initialize a V-dimensional vector to 0
//! 2. For each feature f with weight w:
//!    - Hash f to get a V-bit hash h
//!    - For each bit: `V[i] += w` if `h[i]=1`, else `V[i] -= w`
//! 3. `Fingerprint[i] = 1` if `V[i] > 0`, else 0
//!
//! ## Properties
//!
//! - Hamming distance between fingerprints relates to cosine distance
//! - Very fast comparison (XOR + popcount)
//! - Fixed-size fingerprints regardless of document size
//!
//! ## Use Cases
//!
//! - Web page deduplication
//! - Near-duplicate text detection
//! - Image similarity (with feature vectors)
//!
//! ## References
//!
//! - Charikar (2002). "Similarity estimation techniques from rounding algorithms"
//! - Manku et al. (2007). "Detecting near-duplicates for web crawling"

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// SimHash fingerprint generator.
#[derive(Debug, Clone)]
pub struct SimHash {
    /// Number of bits in the fingerprint.
    bits: usize,
}

impl SimHash {
    /// Create a new SimHash with specified fingerprint size.
    ///
    /// Common values: 64 bits for fast comparison, 128+ for higher accuracy.
    pub fn new(bits: usize) -> Self {
        assert!(bits > 0 && bits <= 128, "bits must be 1-128");
        Self { bits }
    }

    /// Create a 64-bit SimHash (most common).
    pub fn new_64() -> Self {
        Self { bits: 64 }
    }

    /// Compute SimHash fingerprint from weighted features.
    ///
    /// Features are (item, weight) pairs. Higher weight = more influence.
    pub fn fingerprint<T: Hash>(&self, features: &[(T, f64)]) -> SimHashFingerprint {
        let mut v = vec![0.0f64; self.bits];

        for (feature, weight) in features {
            let hash = self.hash_feature(feature);

            for i in 0..self.bits {
                if (hash >> i) & 1 == 1 {
                    v[i] += weight;
                } else {
                    v[i] -= weight;
                }
            }
        }

        // Convert to binary fingerprint
        let mut fingerprint = 0u128;
        for i in 0..self.bits {
            if v[i] > 0.0 {
                fingerprint |= 1u128 << i;
            }
        }

        SimHashFingerprint {
            value: fingerprint,
            bits: self.bits,
        }
    }

    /// Compute SimHash from unweighted features (all weight = 1).
    pub fn fingerprint_unweighted<T: Hash, I: IntoIterator<Item = T>>(
        &self,
        features: I,
    ) -> SimHashFingerprint {
        let weighted: Vec<(T, f64)> = features.into_iter().map(|f| (f, 1.0)).collect();
        self.fingerprint(&weighted)
    }

    /// Compute SimHash from a string (using character n-grams as features).
    pub fn fingerprint_text(&self, text: &str, ngram_size: usize) -> SimHashFingerprint {
        let chars: Vec<char> = text.chars().collect();
        let mut features: Vec<(String, f64)> = Vec::new();

        // Generate n-grams
        for window in chars.windows(ngram_size) {
            let ngram: String = window.iter().collect();
            features.push((ngram, 1.0));
        }

        // Also add individual words
        for word in text.split_whitespace() {
            features.push((word.to_lowercase(), 2.0)); // Higher weight for words
        }

        self.fingerprint(&features)
    }

    /// Hash a feature to get bits.
    fn hash_feature<T: Hash>(&self, feature: &T) -> u128 {
        let mut hasher = DefaultHasher::new();
        feature.hash(&mut hasher);
        let h1 = hasher.finish();

        // Use a second hash for bits beyond 64
        let mut hasher2 = DefaultHasher::new();
        h1.hash(&mut hasher2);
        let h2 = hasher2.finish();

        (h1 as u128) | ((h2 as u128) << 64)
    }

    /// Number of bits in fingerprints.
    pub fn bits(&self) -> usize {
        self.bits
    }
}

impl Default for SimHash {
    fn default() -> Self {
        Self::new_64()
    }
}

/// A SimHash fingerprint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SimHashFingerprint {
    /// The fingerprint value.
    value: u128,
    /// Number of bits used.
    bits: usize,
}

impl SimHashFingerprint {
    /// Hamming distance to another fingerprint.
    ///
    /// Number of bit positions that differ.
    pub fn hamming_distance(&self, other: &SimHashFingerprint) -> usize {
        let xor = self.value ^ other.value;
        xor.count_ones() as usize
    }

    /// Estimated cosine similarity from Hamming distance.
    ///
    /// Based on cos(π * d / bits) approximation.
    pub fn estimated_cosine(&self, other: &SimHashFingerprint) -> f64 {
        let d = self.hamming_distance(other);
        let theta = std::f64::consts::PI * (d as f64) / (self.bits as f64);
        theta.cos()
    }

    /// Check if similar within threshold (Hamming distance).
    pub fn is_similar(&self, other: &SimHashFingerprint, max_distance: usize) -> bool {
        self.hamming_distance(other) <= max_distance
    }

    /// Get the raw fingerprint value.
    pub fn value(&self) -> u128 {
        self.value
    }

    /// Get the 64-bit value (for 64-bit fingerprints).
    pub fn value_64(&self) -> u64 {
        self.value as u64
    }

    /// Number of bits in the fingerprint.
    pub fn bits(&self) -> usize {
        self.bits
    }
}

/// SimHash LSH index using bit sampling.
#[derive(Debug)]
pub struct SimHashLSH {
    /// Number of tables (for recall).
    num_tables: usize,
    /// Bits per table (for precision).
    bits_per_table: usize,
    /// Hash tables: table_idx -> (key -> doc_ids)
    tables: Vec<std::collections::HashMap<u64, Vec<usize>>>,
    /// Bit masks for each table.
    masks: Vec<(Vec<usize>, u128)>,
    /// Stored fingerprints.
    fingerprints: Vec<SimHashFingerprint>,
}

impl SimHashLSH {
    /// Create a new SimHash LSH index.
    ///
    /// - `num_tables`: More tables = better recall, slower query
    /// - `bits_per_table`: Fewer bits = more candidates, better recall
    pub fn new(num_tables: usize, bits_per_table: usize, total_bits: usize) -> Self {
        let mut masks = Vec::with_capacity(num_tables);
        let mut rng_state = 12345u64;

        for _ in 0..num_tables {
            let mut bit_indices = Vec::with_capacity(bits_per_table);
            let mut mask = 0u128;

            // Select random bits for this table
            while bit_indices.len() < bits_per_table {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let bit = (rng_state as usize) % total_bits;
                if !bit_indices.contains(&bit) {
                    bit_indices.push(bit);
                    mask |= 1u128 << bit;
                }
            }

            bit_indices.sort_unstable();
            masks.push((bit_indices, mask));
        }

        Self {
            num_tables,
            bits_per_table,
            tables: (0..num_tables)
                .map(|_| std::collections::HashMap::new())
                .collect(),
            masks,
            fingerprints: Vec::new(),
        }
    }

    /// Insert a fingerprint into the index.
    pub fn insert(&mut self, fingerprint: SimHashFingerprint) -> usize {
        let doc_id = self.fingerprints.len();

        for (table_idx, (bit_indices, _)) in self.masks.iter().enumerate() {
            let key = self.extract_bits(&fingerprint, bit_indices);
            self.tables[table_idx].entry(key).or_default().push(doc_id);
        }

        self.fingerprints.push(fingerprint);
        doc_id
    }

    /// Query for similar fingerprints.
    pub fn query(&self, fingerprint: &SimHashFingerprint) -> Vec<usize> {
        let mut candidates = std::collections::HashSet::new();

        for (table_idx, (bit_indices, _)) in self.masks.iter().enumerate() {
            let key = self.extract_bits(fingerprint, bit_indices);
            if let Some(docs) = self.tables[table_idx].get(&key) {
                candidates.extend(docs.iter().copied());
            }
        }

        candidates.into_iter().collect()
    }

    /// Query and return results with Hamming distance.
    pub fn query_with_distance(&self, fingerprint: &SimHashFingerprint) -> Vec<(usize, usize)> {
        let candidates = self.query(fingerprint);
        let mut results: Vec<(usize, usize)> = candidates
            .into_iter()
            .map(|id| {
                let dist = fingerprint.hamming_distance(&self.fingerprints[id]);
                (id, dist)
            })
            .collect();

        results.sort_by_key(|(_, d)| *d);
        results
    }

    /// Extract selected bits as a key.
    fn extract_bits(&self, fingerprint: &SimHashFingerprint, bit_indices: &[usize]) -> u64 {
        let mut key = 0u64;
        for (i, &bit_idx) in bit_indices.iter().enumerate() {
            if (fingerprint.value >> bit_idx) & 1 == 1 {
                key |= 1u64 << i;
            }
        }
        key
    }

    /// Number of documents in the index.
    pub fn len(&self) -> usize {
        self.fingerprints.len()
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.fingerprints.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simhash_identical() {
        let sh = SimHash::new_64();
        let fp1 = sh.fingerprint_text("hello world", 3);
        let fp2 = sh.fingerprint_text("hello world", 3);

        assert_eq!(fp1.hamming_distance(&fp2), 0);
    }

    #[test]
    fn test_simhash_similar_text() {
        let sh = SimHash::new_64();
        let fp1 = sh.fingerprint_text("the quick brown fox jumps", 3);
        let fp2 = sh.fingerprint_text("the quick brown dog jumps", 3);

        let distance = fp1.hamming_distance(&fp2);
        // Similar texts should have small Hamming distance
        assert!(distance < 20);
    }

    #[test]
    fn test_simhash_different_text() {
        let sh = SimHash::new_64();
        let fp1 = sh.fingerprint_text("the quick brown fox", 3);
        let fp2 = sh.fingerprint_text("completely different text here", 3);

        let distance = fp1.hamming_distance(&fp2);
        // Different texts should have larger Hamming distance
        // (close to 32 for random 64-bit fingerprints)
        assert!(distance > 15);
    }

    #[test]
    fn test_simhash_lsh() {
        let sh = SimHash::new_64();
        let mut lsh = SimHashLSH::new(10, 8, 64);

        let fp1 = sh.fingerprint_text("document about machine learning", 3);
        let fp2 = sh.fingerprint_text("document about deep learning", 3);
        let fp3 = sh.fingerprint_text("recipe for chocolate cake", 3);

        lsh.insert(fp1);
        lsh.insert(fp2);
        lsh.insert(fp3);

        // Query with similar document
        let query = sh.fingerprint_text("document about neural networks", 3);
        let results = lsh.query_with_distance(&query);

        // Should find candidates
        assert!(!results.is_empty());
    }

    #[test]
    fn test_estimated_cosine() {
        let sh = SimHash::new_64();
        let fp1 = sh.fingerprint_text("hello world", 3);
        let fp2 = sh.fingerprint_text("hello world", 3);

        assert!((fp1.estimated_cosine(&fp2) - 1.0).abs() < 0.001);
    }
}
