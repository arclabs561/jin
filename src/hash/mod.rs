//! Hash-based similarity search methods.
//!
//! This module provides Locality Sensitive Hashing (LSH) algorithms for
//! approximate similarity search. The core idea: **design hash functions where
//! similar items collide more often than dissimilar ones**.
//!
//! ## The LSH Intuition
//!
//! Traditional hash functions try to *minimize* collisions. LSH does the opposite
//! for similar items—it's designed so that:
//!
//! - P[h(a) = h(b)] is high when a and b are similar
//! - P[h(a) = h(b)] is low when a and b are dissimilar
//!
//! This enables sublinear search: instead of comparing against all items,
//! you only compare against items in the same hash bucket.
//!
//! ## Algorithms by Similarity Metric
//!
//! | Algorithm | Similarity | Input Type | Best For |
//! |-----------|------------|------------|----------|
//! | [Random Projection][search] | Cosine | Dense vectors | Embeddings |
//! | [MinHash][minhash] | Jaccard | Sets | Document deduplication |
//! | [SimHash][simhash] | Cosine | Weighted features | Text fingerprinting |
//!
//! ## MinHash: Jaccard Similarity for Sets
//!
//! **Problem**: Given millions of documents, find near-duplicates.
//! Exact Jaccard comparison is O(n²)—too slow.
//!
//! **Key insight** (Broder 1997): For a random permutation π of the universe,
//!
//! ```text
//! P[min(π(A)) = min(π(B))] = |A ∩ B| / |A ∪ B| = Jaccard(A, B)
//! ```
//!
//! Why? The minimum element of A ∪ B is equally likely to be any element.
//! It's in A ∩ B with probability |A ∩ B| / |A ∪ B|.
//!
//! **Algorithm**:
//! 1. Represent documents as sets of shingles (k-grams)
//! 2. Apply k random hash functions (simulating permutations)
//! 3. Signature = [min h₁(S), min h₂(S), ..., min hₖ(S)]
//! 4. Jaccard ≈ (# matching positions) / k
//!
//! **Amplification with bands**: Divide signature into b bands of r rows.
//! Hash each band. Similar items collide in *any* band with high probability.
//!
//! ```rust
//! use jin::hash::{MinHash, MinHashLSH};
//! use std::collections::HashSet;
//!
//! let mh = MinHash::new(128);  // 128 hash functions
//! let mut lsh = MinHashLSH::new(16, 8);  // 16 bands × 8 rows
//!
//! // Insert documents (as sets of words/shingles)
//! let doc1: HashSet<&str> = ["the", "quick", "brown", "fox"].into();
//! let doc2: HashSet<&str> = ["the", "quick", "brown", "dog"].into();
//!
//! lsh.insert(mh.signature(&doc1));
//! lsh.insert(mh.signature(&doc2));
//!
//! // Query returns candidates; verify with exact Jaccard
//! let query = mh.signature(&["the", "fast", "brown", "fox"].into());
//! let candidates = lsh.query(&query);
//! ```
//!
//! ## SimHash: Binary Fingerprints for Text
//!
//! **Problem**: Detect near-duplicate web pages at Google scale.
//!
//! **Key insight** (Charikar 2002): Project high-dimensional feature vectors
//! onto random hyperplanes. Similar vectors land on the same side more often.
//!
//! ```text
//! P[sign(r·a) = sign(r·b)] = 1 - θ(a,b)/π
//! ```
//!
//! where θ is the angle between a and b. Cosine similarity = cos(θ).
//!
//! **Algorithm**:
//! 1. Extract weighted features from document (words, n-grams)
//! 2. Hash each feature to a 64-bit value
//! 3. For each bit position, sum weights where bit is 1, subtract where 0
//! 4. Final fingerprint: `bit[i] = 1` if `sum[i] > 0`, else 0
//!
//! **Comparison**: Hamming distance (XOR + popcount) in O(1) time.
//! Small Hamming distance → similar documents.
//!
//! ```rust
//! use jin::hash::SimHash;
//!
//! let sh = SimHash::new_64();
//! let fp1 = sh.fingerprint_text("the quick brown fox jumps over", 3);
//! let fp2 = sh.fingerprint_text("the quick brown dog jumps over", 3);
//!
//! // Similar texts have small Hamming distance (< 32 for 64-bit fingerprints)
//! assert!(fp1.hamming_distance(&fp2) < 20);
//! ```
//!
//! ## Choosing an Algorithm
//!
//! - **Document deduplication**: MinHash (Jaccard captures overlap well)
//! - **Semantic similarity**: Random Projection LSH on embeddings
//! - **Web crawling/plagiarism**: SimHash (fast, small fingerprints)
//! - **Recommendation systems**: Depends on your similarity metric
//!
//! ## References
//!
//! - Broder (1997). "On the resemblance and containment of documents." (MinHash)
//! - Charikar (2002). "Similarity estimation techniques from rounding algorithms." (SimHash)
//! - Indyk & Motwani (1998). "Approximate nearest neighbors: towards removing
//!   the curse of dimensionality." (LSH theory)
//! - Manku et al. (2007). "Detecting near-duplicates for web crawling." (SimHash at Google)

mod hash_table;
pub mod minhash;
mod random_projection;
pub mod search;
pub mod simhash;

pub use minhash::{MinHash, MinHashLSH, MinHashSignature};
pub use search::{LSHIndex, LSHParams};
pub use simhash::{SimHash, SimHashFingerprint, SimHashLSH};
