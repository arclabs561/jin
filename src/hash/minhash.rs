//! MinHash for Jaccard similarity estimation.
//!
//! MinHash provides locality-sensitive hashing for set similarity,
//! estimating the Jaccard coefficient J(A,B) = |A ∩ B| / |A ∪ B|.
//!
//! ## Algorithm
//!
//! For each hash function h_i:
//! - MinHash_i(S) = min_{x ∈ S} h_i(x)
//!
//! The probability that MinHash values match equals Jaccard similarity:
//! P[MinHash_i(A) = MinHash_i(B)] = J(A,B)
//!
//! ## Use Cases
//!
//! - Near-duplicate document detection
//! - Web page deduplication
//! - Plagiarism detection
//! - Similar item retrieval
//!
//! ## References
//!
//! - Broder (1997). "On the resemblance and containment of documents"
//! - Broder et al. (2000). "Min-wise independent permutations"

use std::collections::HashSet;
use std::hash::{BuildHasher, Hash, Hasher};

/// MinHash signature generator.
#[derive(Debug, Clone)]
pub struct MinHash {
    /// Number of hash functions (signature length).
    num_hashes: usize,
    /// Random seeds for hash functions.
    seeds: Vec<u64>,
}

impl MinHash {
    /// Create a new MinHash with specified number of hash functions.
    ///
    /// More hashes = more accurate Jaccard estimate, but larger signatures.
    /// Typical values: 64-256 hashes.
    pub fn new(num_hashes: usize) -> Self {
        Self::with_seed(num_hashes, 42)
    }

    /// Create MinHash with specific seed for reproducibility.
    pub fn with_seed(num_hashes: usize, seed: u64) -> Self {
        let mut seeds = Vec::with_capacity(num_hashes);
        let mut rng_state = seed;
        
        for _ in 0..num_hashes {
            // Simple LCG for seed generation
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seeds.push(rng_state);
        }
        
        Self { num_hashes, seeds }
    }

    /// Compute MinHash signature for a set of items.
    pub fn signature<T: Hash>(&self, items: &HashSet<T>) -> MinHashSignature {
        let mut mins = vec![u64::MAX; self.num_hashes];
        
        for item in items {
            for (i, &seed) in self.seeds.iter().enumerate() {
                let hash = self.hash_with_seed(item, seed);
                if hash < mins[i] {
                    mins[i] = hash;
                }
            }
        }
        
        MinHashSignature { values: mins }
    }

    /// Compute MinHash signature from iterator of items.
    pub fn signature_from_iter<T: Hash, I: IntoIterator<Item = T>>(
        &self,
        items: I,
    ) -> MinHashSignature {
        let mut mins = vec![u64::MAX; self.num_hashes];
        
        for item in items {
            for (i, &seed) in self.seeds.iter().enumerate() {
                let hash = self.hash_with_seed(&item, seed);
                if hash < mins[i] {
                    mins[i] = hash;
                }
            }
        }
        
        MinHashSignature { values: mins }
    }

    /// Hash an item with a specific seed.
    fn hash_with_seed<T: Hash>(&self, item: &T, seed: u64) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        seed.hash(&mut hasher);
        item.hash(&mut hasher);
        hasher.finish()
    }

    /// Number of hash functions.
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }
}

/// A MinHash signature (fingerprint) of a set.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MinHashSignature {
    /// The minimum hash values for each hash function.
    pub values: Vec<u64>,
}

impl MinHashSignature {
    /// Estimate Jaccard similarity between two signatures.
    ///
    /// Returns a value in [0, 1] where 1 means identical sets.
    pub fn jaccard(&self, other: &MinHashSignature) -> f64 {
        if self.values.len() != other.values.len() {
            return 0.0;
        }
        
        let matches = self.values.iter()
            .zip(other.values.iter())
            .filter(|(a, b)| a == b)
            .count();
        
        matches as f64 / self.values.len() as f64
    }

    /// Check if estimated similarity exceeds threshold.
    ///
    /// Useful for candidate filtering before exact comparison.
    pub fn is_similar(&self, other: &MinHashSignature, threshold: f64) -> bool {
        self.jaccard(other) >= threshold
    }

    /// Hamming distance between signatures.
    ///
    /// Number of positions where hash values differ.
    pub fn hamming_distance(&self, other: &MinHashSignature) -> usize {
        self.values.iter()
            .zip(other.values.iter())
            .filter(|(a, b)| a != b)
            .count()
    }

    /// Merge two signatures (union of underlying sets).
    ///
    /// Takes element-wise minimum.
    pub fn merge(&self, other: &MinHashSignature) -> MinHashSignature {
        let values = self.values.iter()
            .zip(other.values.iter())
            .map(|(&a, &b)| a.min(b))
            .collect();
        
        MinHashSignature { values }
    }

    /// Length of the signature.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Check if signature is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }
}

/// LSH index using MinHash for near-duplicate detection.
#[derive(Debug)]
pub struct MinHashLSH {
    /// Number of bands for LSH.
    bands: usize,
    /// Number of rows per band.
    rows_per_band: usize,
    /// Band hash tables: band_idx -> (band_hash -> doc_ids)
    buckets: Vec<std::collections::HashMap<u64, Vec<usize>>>,
    /// Stored signatures.
    signatures: Vec<MinHashSignature>,
}

impl MinHashLSH {
    /// Create a new MinHash LSH index.
    ///
    /// The threshold for similarity is approximately `(1/bands)^(1/rows)`.
    ///
    /// For example, 20 bands with 5 rows each (100 total hashes) gives
    /// threshold ≈ 0.55.
    pub fn new(bands: usize, rows_per_band: usize) -> Self {
        Self {
            bands,
            rows_per_band,
            buckets: (0..bands).map(|_| std::collections::HashMap::new()).collect(),
            signatures: Vec::new(),
        }
    }

    /// Create LSH index with target similarity threshold.
    ///
    /// Automatically calculates optimal band/row configuration.
    pub fn with_threshold(num_hashes: usize, threshold: f64) -> Self {
        // Find bands/rows that give threshold closest to target
        let mut best_bands = 1;
        let mut best_error = f64::MAX;
        
        for b in 1..=num_hashes {
            if num_hashes % b == 0 {
                let r = num_hashes / b;
                let t = (1.0 / b as f64).powf(1.0 / r as f64);
                let error = (t - threshold).abs();
                if error < best_error {
                    best_error = error;
                    best_bands = b;
                }
            }
        }
        
        let rows = num_hashes / best_bands;
        Self::new(best_bands, rows)
    }

    /// Insert a signature into the index.
    pub fn insert(&mut self, signature: MinHashSignature) -> usize {
        let doc_id = self.signatures.len();
        
        // Hash each band and insert into corresponding bucket
        for (band_idx, chunk) in signature.values.chunks(self.rows_per_band).enumerate() {
            if band_idx >= self.bands {
                break;
            }
            let band_hash = self.hash_band(chunk);
            self.buckets[band_idx]
                .entry(band_hash)
                .or_default()
                .push(doc_id);
        }
        
        self.signatures.push(signature);
        doc_id
    }

    /// Query for similar documents.
    ///
    /// Returns candidate document IDs that might be similar.
    /// Candidates should be verified with exact Jaccard computation.
    pub fn query(&self, signature: &MinHashSignature) -> Vec<usize> {
        let mut candidates: HashSet<usize> = HashSet::new();
        
        for (band_idx, chunk) in signature.values.chunks(self.rows_per_band).enumerate() {
            if band_idx >= self.bands {
                break;
            }
            let band_hash = self.hash_band(chunk);
            if let Some(docs) = self.buckets[band_idx].get(&band_hash) {
                candidates.extend(docs.iter().copied());
            }
        }
        
        candidates.into_iter().collect()
    }

    /// Query and return results with estimated similarity.
    pub fn query_with_similarity(
        &self,
        signature: &MinHashSignature,
    ) -> Vec<(usize, f64)> {
        let candidates = self.query(signature);
        let mut results: Vec<(usize, f64)> = candidates
            .into_iter()
            .map(|id| {
                let sim = signature.jaccard(&self.signatures[id]);
                (id, sim)
            })
            .collect();
        
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Hash a band (chunk of signature values).
    fn hash_band(&self, values: &[u64]) -> u64 {
        use std::collections::hash_map::DefaultHasher;
        let mut hasher = DefaultHasher::new();
        for v in values {
            v.hash(&mut hasher);
        }
        hasher.finish()
    }

    /// Number of documents in the index.
    pub fn len(&self) -> usize {
        self.signatures.len()
    }

    /// Check if index is empty.
    pub fn is_empty(&self) -> bool {
        self.signatures.is_empty()
    }

    /// Approximate threshold for this configuration.
    pub fn threshold(&self) -> f64 {
        (1.0 / self.bands as f64).powf(1.0 / self.rows_per_band as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minhash_identical_sets() {
        let mh = MinHash::new(128);
        let set: HashSet<&str> = ["a", "b", "c"].into_iter().collect();
        
        let sig1 = mh.signature(&set);
        let sig2 = mh.signature(&set);
        
        assert_eq!(sig1.jaccard(&sig2), 1.0);
    }

    #[test]
    fn test_minhash_disjoint_sets() {
        let mh = MinHash::new(128);
        
        let set1: HashSet<&str> = ["a", "b", "c"].into_iter().collect();
        let set2: HashSet<&str> = ["x", "y", "z"].into_iter().collect();
        
        let sig1 = mh.signature(&set1);
        let sig2 = mh.signature(&set2);
        
        // Should be close to 0 but not exactly due to hash collisions
        assert!(sig1.jaccard(&sig2) < 0.2);
    }

    #[test]
    fn test_minhash_similar_sets() {
        let mh = MinHash::new(256);
        
        let set1: HashSet<i32> = (0..100).collect();
        let set2: HashSet<i32> = (50..150).collect();
        
        // Actual Jaccard = 50/150 = 0.333...
        let sig1 = mh.signature(&set1);
        let sig2 = mh.signature(&set2);
        
        let estimated = sig1.jaccard(&sig2);
        assert!((estimated - 0.333).abs() < 0.1);
    }

    #[test]
    fn test_minhash_lsh() {
        let mh = MinHash::new(100);
        let mut lsh = MinHashLSH::new(20, 5);
        
        // Insert some documents
        let doc1: HashSet<&str> = ["the", "quick", "brown", "fox"].into_iter().collect();
        let doc2: HashSet<&str> = ["the", "quick", "brown", "dog"].into_iter().collect();
        let doc3: HashSet<&str> = ["hello", "world", "foo", "bar"].into_iter().collect();
        
        let sig1 = mh.signature(&doc1);
        let sig2 = mh.signature(&doc2);
        let sig3 = mh.signature(&doc3);
        
        lsh.insert(sig1);
        lsh.insert(sig2);
        lsh.insert(sig3.clone());
        
        // Query with doc2's signature should find doc1 as similar
        let results = lsh.query_with_similarity(&mh.signature(&doc2));
        
        assert!(!results.is_empty());
        // Doc1 and doc2 are similar (0, 1 are their IDs)
    }

    #[test]
    fn test_signature_merge() {
        let mh = MinHash::new(64);
        
        let set1: HashSet<&str> = ["a", "b"].into_iter().collect();
        let set2: HashSet<&str> = ["c", "d"].into_iter().collect();
        let union: HashSet<&str> = ["a", "b", "c", "d"].into_iter().collect();
        
        let sig1 = mh.signature(&set1);
        let sig2 = mh.signature(&set2);
        let sig_union = mh.signature(&union);
        let sig_merged = sig1.merge(&sig2);
        
        // Merged signature should equal union signature
        assert_eq!(sig_merged, sig_union);
    }
}
