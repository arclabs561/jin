//! Locality Sensitive Hashing Demo
//!
//! LSH is the opposite of traditional hashing: similar items *should* collide.
//! This enables sublinear search by only comparing items in the same bucket.
//!
//! ```bash
//! cargo run --example lsh_demo --features lsh
//! ```

use std::collections::HashSet;
use vicinity::hash::{MinHash, MinHashLSH, SimHash};

fn main() {
    println!("Locality Sensitive Hashing");
    println!("==========================\n");

    demo_minhash();
    demo_simhash();
    demo_lsh_tradeoffs();
}

/// MinHash: Jaccard similarity for sets (document deduplication)
fn demo_minhash() {
    println!("1. MinHash (Jaccard Similarity)");
    println!("   ----------------------------");
    println!("   Problem: Find near-duplicate documents among millions.");
    println!("   Key insight: P[min(h(A)) = min(h(B))] = Jaccard(A, B)\n");

    // Create documents as sets of words (shingles in real use)
    let doc1: HashSet<&str> = ["the", "quick", "brown", "fox", "jumps"]
        .into_iter()
        .collect();
    let doc2: HashSet<&str> = ["the", "quick", "brown", "dog", "jumps"]
        .into_iter()
        .collect();
    let doc3: HashSet<&str> = ["lorem", "ipsum", "dolor", "sit", "amet"]
        .into_iter()
        .collect();

    // Exact Jaccard for reference
    let jaccard_12 = jaccard(&doc1, &doc2);
    let jaccard_13 = jaccard(&doc1, &doc3);

    println!("   Documents:");
    println!("     doc1: {:?}", doc1);
    println!("     doc2: {:?}", doc2);
    println!("     doc3: {:?}", doc3);
    println!("\n   Exact Jaccard:");
    println!("     J(doc1, doc2) = {:.3} (4/6 overlap)", jaccard_12);
    println!("     J(doc1, doc3) = {:.3} (disjoint)", jaccard_13);

    // MinHash estimation
    let mh = MinHash::new(128); // 128 hash functions
    let sig1 = mh.signature(&doc1);
    let sig2 = mh.signature(&doc2);
    let sig3 = mh.signature(&doc3);

    println!("\n   MinHash estimates (128 hashes):");
    println!("     J(doc1, doc2) ~ {:.3}", sig1.jaccard(&sig2));
    println!("     J(doc1, doc3) ~ {:.3}", sig1.jaccard(&sig3));

    // LSH for candidate retrieval
    // Use more bands (32) with fewer rows (4) to catch J=0.67 reliably
    // P(collision) = 1 - (1 - 0.67^4)^32 ≈ 99.9%
    let mut lsh = MinHashLSH::new(32, 4); // 32 bands x 4 rows = 128 hashes
    let id1 = lsh.insert(sig1.clone());
    let _id2 = lsh.insert(sig2);
    let _id3 = lsh.insert(sig3);

    let candidates = lsh.query(&sig1);
    println!(
        "\n   LSH candidates for doc1 (id={}): {:?}",
        id1, candidates
    );
    println!("   Expected: [0, 1] (doc1 and doc2). doc3 (id=2) filtered out.");
    println!();
}

/// SimHash: Cosine similarity via binary fingerprints
fn demo_simhash() {
    println!("2. SimHash (Cosine via Hamming)");
    println!("   ----------------------------");
    println!("   Problem: Detect near-duplicate web pages at Google scale.");
    println!("   Key insight: sign(r.a) = sign(r.b) with prob 1 - theta/pi\n");

    let sh = SimHash::new_64();

    // Similar texts
    let fp1 = sh.fingerprint_text("the quick brown fox jumps over the lazy dog", 3);
    let fp2 = sh.fingerprint_text("the quick brown fox leaps over the lazy dog", 3);
    let fp3 = sh.fingerprint_text("lorem ipsum dolor sit amet consectetur", 3);

    println!("   Text fingerprints (64-bit):");
    println!("     text1: \"the quick brown fox jumps...\"");
    println!("     text2: \"the quick brown fox leaps...\" (1 word diff)");
    println!("     text3: \"lorem ipsum dolor...\" (unrelated)");

    println!("\n   Hamming distances (lower = more similar):");
    println!("     d(text1, text2) = {} bits", fp1.hamming_distance(&fp2));
    println!("     d(text1, text3) = {} bits", fp1.hamming_distance(&fp3));

    // Threshold for "near-duplicate"
    let threshold = 10;
    println!("\n   With threshold = {} bits:", threshold);
    println!(
        "     text1 ~ text2: {}",
        if fp1.hamming_distance(&fp2) <= threshold {
            "DUPLICATE"
        } else {
            "different"
        }
    );
    println!(
        "     text1 ~ text3: {}",
        if fp1.hamming_distance(&fp3) <= threshold {
            "DUPLICATE"
        } else {
            "different"
        }
    );
    println!();
}

/// Trade-offs in LSH parameters
fn demo_lsh_tradeoffs() {
    println!("3. LSH Parameter Trade-offs");
    println!("   ------------------------");
    println!("   bands x rows = total hashes (signature length)");
    println!("   More bands: higher recall (find more similar pairs)");
    println!("   More rows per band: higher precision (fewer false positives)\n");

    // The "S-curve": probability of becoming a candidate
    println!("   Probability of collision vs Jaccard similarity:");
    println!("   (bands=16, rows=8 -> 128 hashes)");
    println!();
    println!("   Jaccard | P(collision)");
    println!("   --------|-------------");

    let bands: i32 = 16;
    let rows: i32 = 8;
    for j in [0.2_f64, 0.4, 0.5, 0.6, 0.8, 0.9] {
        // P(collision in any band) = 1 - (1 - s^r)^b
        let p = 1.0 - (1.0 - j.powi(rows)).powi(bands);
        println!("   {:.1}     | {:.1}%", j, p * 100.0);
    }

    println!("\n   The S-curve effect: pairs with J > ~0.5 are found with high");
    println!("   probability; pairs with J < ~0.3 are filtered out.");
    println!("\n   To shift the threshold:");
    println!("   - More bands -> catches lower similarity pairs (but more candidates)");
    println!("   - More rows -> only catches high similarity pairs (but may miss some)");
}

fn jaccard<T: Eq + std::hash::Hash>(a: &HashSet<T>, b: &HashSet<T>) -> f64 {
    let intersection = a.intersection(b).count();
    let union = a.union(b).count();
    if union == 0 {
        0.0
    } else {
        intersection as f64 / union as f64
    }
}
