# vicinity User Guide

A systematic guide to approximate nearest neighbor search and intrinsic dimensionality.

## Contents

1. [Distance Metrics and Similarity](#1-distance-metrics-and-similarity)
   - 1.1 [Cosine Similarity](#11-cosine-similarity)
   - 1.2 [Euclidean (L2) Distance](#12-euclidean-l2-distance)
   - 1.3 [Choosing the Right Metric](#13-choosing-the-right-metric)

2. [Index Structures](#2-index-structures)
   - 2.1 [HNSW: Hierarchical Navigable Small Worlds](#21-hnsw-hierarchical-navigable-small-worlds)
   - 2.2 [Dual-Branch HNSW](#22-dual-branch-hnsw)
   - 2.3 [Parameter Tuning](#23-parameter-tuning)

3. [Local Intrinsic Dimensionality](#3-local-intrinsic-dimensionality)
   - 3.1 [The MLE Estimator](#31-the-mle-estimator)
   - 3.2 [TwoNN Estimator](#32-twonn-estimator)
   - 3.3 [Applications: Outlier Detection](#33-applications-outlier-detection)

4. [Practical Workflows](#4-practical-workflows)
   - 4.1 [Semantic Search Pipeline](#41-semantic-search-pipeline)
   - 4.2 [Recall vs Latency Tradeoffs](#42-recall-vs-latency-tradeoffs)

5. [Tips on Practical Use](#5-tips-on-practical-use)

---

## 1. Distance Metrics and Similarity

### 1.1 Cosine Similarity

Cosine similarity measures the angle between two vectors, ignoring magnitude:

```
cosine(a, b) = dot(a, b) / (||a|| * ||b||)
```

**When to use**: Text embeddings, normalized vectors, when direction matters more than magnitude.

```rust
use vicinity::distance::cosine_similarity;

let a = vec![1.0, 0.0, 0.0];
let b = vec![0.707, 0.707, 0.0];
let sim = cosine_similarity(&a, &b);  // ~0.707
```

### 1.2 Euclidean (L2) Distance

Standard geometric distance in n-dimensional space:

```
l2(a, b) = sqrt(sum((a[i] - b[i])^2))
```

**When to use**: Image features, when absolute positions matter, scientific data.

```rust
use vicinity::distance::l2_distance;

let dist = l2_distance(&[0.0, 0.0], &[3.0, 4.0]);  // 5.0
```

### 1.3 Choosing the Right Metric

| Use Case | Recommended Metric | Reason |
|----------|-------------------|--------|
| Sentence embeddings | Cosine | Embeddings are often pre-normalized |
| Image features (raw) | L2 | Pixel values have meaningful magnitude |
| TF-IDF vectors | Cosine | Sparse, high-dimensional |
| Geographic coordinates | L2 (or Haversine) | Actual distances matter |

**Key insight**: For L2-normalized vectors, cosine similarity and L2 distance are monotonically related:
```
l2(a, b)^2 = 2 * (1 - cosine(a, b))
```

---

## 2. Index Structures

### 2.1 HNSW: Hierarchical Navigable Small Worlds

HNSW builds a multi-layer graph where:
- Higher layers have fewer nodes but longer edges (for fast navigation)
- Lower layers have more nodes with shorter edges (for precision)
- Search starts at the top and descends, greedily approaching the query

**Mathematical foundation**: Based on "small-world" networks where any two nodes can be reached in O(log N) hops.

```rust
use vicinity::hnsw::HNSWIndex;

// Create index
let mut index = HNSWIndex::new(
    768,    // dimension
    16,     // M: connections per node
    32,     // M_max: max connections at layer 0
)?;

// Add vectors
for (id, vec) in documents.iter().enumerate() {
    index.add(id as u32, vec.clone())?;
}
index.build()?;

// Search
let results = index.search(&query, k, ef_search)?;
```

**Example**: `cargo run --example hnsw_benchmark --release`

### 2.2 Dual-Branch HNSW

Standard HNSW struggles with outliers (high-LID points). Dual-Branch HNSW addresses this:

1. **LID-based layer assignment**: High-LID points get more connections
2. **Skip bridges**: Long-range edges that bypass sparse regions
3. **Dual search paths**: Explores both local and skip-bridge routes

```rust
use vicinity::hnsw::dual_branch::{DualBranchConfig, DualBranchHNSW};

let config = DualBranchConfig {
    m: 16,
    m_high_lid: 24,  // Extra connections for outliers
    lid_threshold_sigma: 1.5,
    ..Default::default()
};

let mut index = DualBranchHNSW::new(768, config);
index.add_vectors(&data)?;
index.build()?;
```

**Example**: `cargo run --example dual_branch_demo --release`

### 2.3 Parameter Tuning

| Parameter | Effect | Typical Values |
|-----------|--------|----------------|
| `M` | Graph connectivity | 8-64 (higher = better recall, more memory) |
| `ef_construction` | Build-time search width | 100-500 |
| `ef_search` | Query-time search width | 50-500 (tune for recall/latency) |

**Rule of thumb**:
- Start with M=16, ef_construction=200
- Tune ef_search based on your recall requirements
- Higher M for high-dimensional data (>256 dims)

---

## 3. Local Intrinsic Dimensionality

LID measures how "spread out" points are around a query. Low LID = dense, clustered. High LID = sparse, outlier-like.

### 3.1 The MLE Estimator

The Maximum Likelihood Estimator for LID:

```
LID_MLE(x) = -k / sum(log(d_i / d_k))
```

where d_1 <= d_2 <= ... <= d_k are distances to k nearest neighbors.

**Intuition**: In a d-dimensional space, the volume of a ball grows as r^d. LID estimates this exponent locally.

```rust
use vicinity::lid::{estimate_lid, LidConfig};

let config = LidConfig { k: 20, ..Default::default() };
let distances = compute_knn_distances(&query, &data);
let estimate = estimate_lid(&distances, &config);
println!("LID: {:.2}", estimate.lid);
```

### 3.2 TwoNN Estimator

Uses only the two nearest neighbors for a robust, scale-free estimate:

```
TwoNN: d = slope of linear fit: -log(1 - F_emp(mu)) vs log(mu)
```

where mu = d_1 / d_2 is the ratio of nearest to second-nearest distance.

**Example**: `cargo run --example lid_demo --release`

### 3.3 Applications: Outlier Detection

Points with unusually high LID are outliers:

```rust
use vicinity::lid::{estimate_lid_batch, LidCategory};

let estimates = estimate_lid_batch(&data, k)?;
let outliers: Vec<_> = estimates.iter()
    .filter(|e| e.category == LidCategory::Sparse)
    .collect();
```

**Example**: `cargo run --example lid_outlier_detection --release`

---

## 4. Practical Workflows

### 4.1 Semantic Search Pipeline

```rust
// 1. Build index from document embeddings
let mut index = HNSWIndex::new(768, 16, 32)?;
for (id, embedding) in documents.iter().enumerate() {
    index.add(id as u32, normalize(embedding))?;
}
index.build()?;

// 2. Search with query embedding
let query_embedding = embed_query("What is machine learning?");
let results = index.search(&normalize(&query_embedding), 10, 100)?;

// 3. Return ranked documents
for (doc_id, distance) in results {
    let similarity = 1.0 - distance;
    println!("Doc {}: {:.3}", doc_id, similarity);
}
```

**Example**: `cargo run --example semantic_search_demo --release`

### 4.2 Recall vs Latency Tradeoffs

```
ef_search    Recall@10    Latency    QPS
------------------------------------------------
10           48.8%        59us       17,034
50           85.2%        172us      5,812
100          92.1%        301us      3,323
200          96.5%        643us      1,554
```

**Guidance**:
- Production search: ef=50-100 for 85-95% recall
- High-precision: ef=200-500 for 95%+ recall
- Real-time: ef=10-20 for sub-100us latency

---

## 5. Tips on Practical Use

### Normalize your vectors

For cosine similarity, always L2-normalize:
```rust
fn normalize(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    v.iter().map(|x| x / norm).collect()
}
```

### Use appropriate M for your dimension

| Dimension | Suggested M |
|-----------|-------------|
| 64-128    | 8-12        |
| 256-512   | 12-16       |
| 768-1536  | 16-24       |
| 2048+     | 24-32       |

### Monitor recall on a holdout set

Always measure actual recall, don't assume:
```rust
let gt = brute_force_knn(&query, &data, k);
let approx = index.search(&query, k, ef)?;
let recall = compute_recall(&gt, &approx);
```

### Consider LID for difficult datasets

If recall varies significantly across queries, high-LID outliers may be the cause. Use Dual-Branch HNSW or increase M for those regions.

---

## Examples Index

| Example | Topic | Run Command |
|---------|-------|-------------|
| `hnsw_benchmark` | Basic HNSW usage and performance | `cargo run --example hnsw_benchmark --release` |
| `dual_branch_demo` | LID-aware HNSW for outliers | `cargo run --example dual_branch_demo --release` |
| `lid_demo` | LID estimation methods | `cargo run --example lid_demo --release` |
| `lid_outlier_detection` | Outlier detection via LID | `cargo run --example lid_outlier_detection --release` |
| `semantic_search_demo` | Full search pipeline | `cargo run --example semantic_search_demo --release` |
