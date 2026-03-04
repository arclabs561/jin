# Audit Report -- vicinity
Date: 2026-03-04 | Scope: arch, qa, release

## Summary

52 findings: 14 structural, 11 coherence, 12 surface, 15 hygiene

The three most critical issues:
1. **`hiqlite` is a non-optional heavy dependency** (Raft consensus stack) used only in persistence code -- every user pays the compile cost.
2. **`cdylib` crate-type is unconditional** -- forces C linkage for all builds, not just PyO3.
3. **`durability` path dep blocks `cargo publish`** -- even though optional, Cargo validates all dep specs at publish time.

---

## Architecture

### STRUCTURAL

- **[Cargo.toml:15]** `crate-type = ["rlib", "cdylib"]` is unconditional. Every `cargo build` produces a cdylib, preventing LTO and causing linker issues on non-Python targets. Should be gated behind the `python` feature or moved to maturin config.

- **[Cargo.toml:59]** `hiqlite = { version = "0.12", features = ["full"] }` is non-optional but only used in `src/persistence/locking.rs:38-57` and `src/persistence/error.rs:79-83`. Pulls in tokio, tonic, raft-rs, rocksdb transitively. Must be `optional = true` behind `persistence`.

- **[Cargo.toml:70]** `durability = { path = "../durability" }` is a path dep that blocks `cargo publish`, even though optional. Needs version-only entry for registry, with `[patch]` at workspace root for local dev.

- **[Cargo.toml:53]** `kuji = "0.1.0"` is non-optional but **never used** -- zero `use kuji::` references in `src/`. Dead mandatory dependency.

- **[src/lib.rs:83-93]** All algorithm modules (`diskann`, `nsw`, `sng`, `scann`, `ivf_pq`, `vamana`, `classic`, `evoc`) are `pub mod` without `#[cfg(feature = ...)]` gates. Feature flags only gate trait impls and factory wiring, not the modules themselves. Consumers see empty modules in docs when features are off.

- **[src/vamana/construction.rs:3-5]** Vamana imports `hnsw::construction::select_neighbors`, `hnsw::distance`, `hnsw::graph::NeighborhoodDiversification` directly. Shared logic should live in a common module, not reach into hnsw internals.

- **[src/ann/autotune.rs:18-19]** `autotune` unconditionally imports `crate::benchmark::datasets` and `crate::benchmark::recall_at_k`. Benchmark utilities compile for all library users. Should be feature-gated or dev-only.

- **[src/diskann/disk_io.rs:5]** DiskANN depends on `crate::persistence::error::{PersistenceError, PersistenceResult}` which compiles unconditionally. Works by accident; proper feature gating of persistence would break diskann.

### COHERENCE

- **[src/lib.rs:3]** `#![allow(dead_code)]` at crate root + 9 module-level `#![allow(dead_code)]` suppress all dead-code warnings across ~108 files. Makes it impossible to detect orphaned code during refactoring.

- **[src/lib.rs:6]** `#![allow(unsafe_op_in_unsafe_fn)]` overrides `Cargo.toml:181` lint `unsafe_op_in_unsafe_fn = "warn"`. The manifest lint is dead.

- **[src/error.rs vs src/persistence/error.rs vs src/compression/error.rs]** Three separate error hierarchies with no `From` conversions between `RetrieveError` and `CompressionError`. `ivf_pq/search.rs:181` returns `CompressionError` while the rest of IVF-PQ uses `RetrieveError`.

- **[src/persistence/error.rs:79-82]** `From<hiqlite::Error>` not feature-gated, making `hiqlite` a hard compile-time dep regardless of features.

- **[src/persistence/locking.rs:35-57]** `DistributedLock` uses `hiqlite::Lock`/`hiqlite::Client` unconditionally. Should be behind `#[cfg(feature = "persistence")]`.

- **[src/hnsw/mod.rs]** Re-exports 40+ public types from 12 sub-modules. Over-coupled hub mixing stable API (core index, search) with research prototypes (DEG, probabilistic routing, dual-branch).

- **[src/scann/partitioning.rs:2]** `pub use crate::partitioning::kmeans::KMeans` leaks internal type into public API surface.

### SURFACE

- **[src/ann/traits.rs:49-92]** `ANNIndex` impl for `HNSWIndex` accesses structural fields (`vectors`, `layers`, `num_vectors`, `dimension`, `params.ef_search`) directly. Internal representation exposed.

- **[src/lib.rs:110]** `ANNIndex::search` returns `Vec<(u32, f32)>`, coupling public API to `u32` document IDs. Prevents future migration to `u64` or generic IDs.

- **[src/lib.rs:112]** Only `RetrieveError` is re-exported; `PersistenceError` and `CompressionError` are also `pub` but require sub-module imports. Fragmented error surface.

- **[src/ivf_pq/search.rs:10-15]** `Quantizer` enum with `Product`/`Optimized` variants is `pub`, exposing implementation-detail types.

### HYGIENE

- **[src/hnsw/graph.rs:304,332,344]** `std::sync::Mutex::lock().unwrap()` in library code. Panics on poisoned mutex. Should use `parking_lot::Mutex` (already a dependency; no poisoning).

- **[src/hnsw/graph.rs:195]** Uses `std::sync::Mutex` while rest of crate uses `parking_lot::Mutex`. Inconsistent.

- **[src/hnsw/dual_branch.rs:210]** `.expect("entry_point should be set")` in library code. Should return `Err(RetrieveError::EmptyIndex)`.

- **[src/sng/alpha_pruning.rs:31]** `partial_cmp(&...).unwrap()` panics on NaN distances.

- **[src/quantization/saq.rs:128]** `partial_cmp(...).unwrap()` inside `sort_by` panics on NaN variance.

- **[Cargo.toml:62-66]** `byteorder`, `crc32fast`, `libc`, `hex`, `fst` are non-optional. `fst` and `hex` are persistence-specific; inflate dep tree for all users.

---

## Quality

### STRUCTURAL

- **[src/sng/construction.rs]** File exists on disk but is not declared as a module in `sng/mod.rs`. Contains `pub fn construct_sng_graph` -- dead orphan code duplicating `sng/graph.rs:131-186`.

- **[src/sng/alpha_pruning.rs]** Exists on disk, not declared in `sng/mod.rs`. Contains `pub trait ProxGraph` and `pub struct AlphaCng` -- unreachable.

- **[src/ivf_pq/ivf.rs]** Empty stub file (5 lines, just a comment). Exists on disk, not compiled.

- **[src/ivf_pq/online_pq.rs]** Has real code with `#[cfg(test)]` tests, but not declared in `ivf_pq/mod.rs`. Tests never compile.

### COHERENCE

- **[src/nsw/graph.rs:65, src/sng/graph.rs:50]** `NSWIndex::new` and `SNGIndex::new` return `Err(RetrieveError::EmptyQuery)` for `dimension == 0`. Semantically wrong -- should be `InvalidParameter`. HNSW uses `RetrieveError::Other` (also wrong, but closer).

- **[src/nsw/graph.rs:111-120, src/sng/graph.rs:65, src/diskann/graph.rs:371-380]** NSW, SNG, and DiskANN accept `doc_id: u32` in `add()`/`add_slice()` but prefix with `_` and silently ignore it. Search results return internal indices instead of user-provided doc_ids. HNSW correctly maps these.

- **[src/ann/traits.rs:62-63]** `ANNIndex::search` bakes in a fixed `ef_search` from `self.params`, with no way for callers to override per-query. Loses the main tuning knob.

- **[src/streaming/buffer.rs:153-167]** `StreamBuffer::search` uses private `euclidean_distance` (L2) while main indexes use cosine. Merged search compares incompatible distance values.

- **[src/vamana/search.rs:60-63]** Vamana picks a **random** entry point every query (`rng.random_range(0..n)`). Non-deterministic and degrades quality vs. medoid entry.

### SURFACE

- **[src/sng/mod.rs:22]** Doc example `SNGIndex::new(128, SNGParams::default())` omits `Result` handling. Would not compile (marked `ignore` so cosmetic only).

### HYGIENE

- **[nsw, scann, diskann, sng, evoc]** Five major modules have **zero test coverage**. Of 206 `#[test]` functions, coverage concentrates in `hnsw` sub-modules and utilities.

- **[src/vamana/graph.rs:142-162]** Only tests are `test_vamana_create` and `test_vamana_add`. No test for `build()` or `search()`.

- **[src/streaming/buffer.rs:176-182]** Private `euclidean_distance` reimplements `distance::l2_distance` without SIMD. Duplicated logic, potential semantic mismatch (sqrt vs. no sqrt).

- **[examples/]** 15 examples, but only 2 (`ivf_pq_demo`, `rabitq_demo`) declare `required-features`. The other 13 lack `[[example]]` entries, so they fail to compile if needed features aren't active.

---

## Release Readiness

### STRUCTURAL

- **[Cargo.toml:70]** Path dep `durability` blocks `cargo publish`. Even `optional = true` does not help -- Cargo validates all deps.

- **[Cargo.toml:75-116]** 35 feature flags for a 0.1.0 crate. Many are "organizational toggles" (per comment at line 90) rather than real compile gates. Maintenance burden and confusing for downstream users.

- **[src/lib.rs:86]** `pub mod evoc;` has no `#[cfg(feature = "evoc")]` gate. Module compiles unconditionally despite `evoc` being a feature flag.

- **[src/persistence/{error.rs, locking.rs}]** Compile unconditionally, dragging in `hiqlite` for all builds.

### COHERENCE

- **[README.md:144]** Documents that `persistence` requires sibling `durability/` checkout. Users installing from crates.io get a build failure when enabling `persistence`.

- **[ci.yml:85]** CI excludes `persistence` from feature matrix (`--exclude-features persistence,memmap,persistence-bincode`). Regressions in persistence code go undetected.

- **[Cargo.toml:101]** `id-compression` feature depends on `cnk/sbits`. If `cnk` changes its feature set, this breaks silently.

### SURFACE

- **[Cargo.toml:92]** `dense = []` feature gates nothing standalone. Only used in `persistence/mod.rs:48` as `cfg(all(feature = "persistence", feature = "dense"))`.

- **[Cargo.toml:94-97]** `kdtree`, `balltree`, `kmeans_tree`, `rptree` features: actual tree modules in `src/classic/trees/` compile unconditionally. Features only gate `ann/traits.rs` impls. Users cannot turn off the dead code.

- **[Cargo.toml:109]** `turboquant` feature gates one line (`quantization/mod.rs:132`). The `turboquant.rs` module compiles unconditionally.

- **[Cargo.toml:115]** `vquant = ["qntz"]` -- no `cfg(feature = "vquant")` in source. Pure alias adding surface area.

- **[Cargo.toml:104]** `memmap = ["persistence"]` -- no `cfg(feature = "memmap")` in source. Users enabling `memmap` get the full (broken) persistence stack.

- **[Cargo.toml:99]** `evoc = []` -- module always compiles; feature only gates `partitioning.rs` usage.

- **[Cargo.toml:80-81]** `ivf_pq`, `diskann`, `scann`, `sng` modules compile unconditionally despite having feature flags. Flags only gate trait impls and factory.

### HYGIENE

- **[Cargo.toml:8]** Repository URL uses `arclabs561` org. Verify this matches actual GitHub org to avoid dead crates.io link.

- **[Cargo.toml:165-167]** `check-cfg` allowlist includes phantom values (`"simd"`, `"rand"`, `"hdf5"`) that are not declared features. Code gated behind them is dead. Also omits many real features, potentially causing spurious `unexpected_cfgs` warnings.

- **No CHANGELOG**: Acceptable at 0.1.0, but needed before 0.2.0.

- **`cargo publish` would fail**: `durability` path dep + `cdylib` crate-type are blockers.

- **[.github/workflows/ci.yml]** No `cargo test --all-features` job. Feature interactions never tested together. `cargo-hack --each-feature` only checks compilation, not tests.

---

## Priority Triage (recommended fix order)

### P0 -- Publish blockers
1. Make `hiqlite` optional behind `persistence` feature
2. Remove `kuji` (unused) or make optional
3. Gate `cdylib` crate-type behind `python` feature (or move to maturin config)
4. Resolve `durability` path dep for publishability
5. Feature-gate `persistence/error.rs` `From<hiqlite::Error>` and `persistence/locking.rs` `DistributedLock`

### P1 -- Correctness
6. Fix doc_id silently ignored in NSW/SNG/DiskANN `add()` (returns wrong IDs in search results)
7. Fix `partial_cmp().unwrap()` in `sng/alpha_pruning.rs:31` and `quantization/saq.rs:128` (NaN panics)
8. Replace `std::sync::Mutex` with `parking_lot::Mutex` in `hnsw/graph.rs` (poison panics)
9. Fix streaming buffer distance mismatch (L2 vs cosine)

### P2 -- Hygiene / quality
10. Remove crate-level `#![allow(dead_code)]` and clean up per-module
11. Delete orphan files: `sng/construction.rs`, `sng/alpha_pruning.rs`, `ivf_pq/ivf.rs`, `ivf_pq/online_pq.rs`
12. Add tests for nsw, scann, diskann, sng, evoc, vamana (search)
13. Clean up feature flags (remove/merge unused ones)
14. Make `benchmark` module dev-only or feature-gated
15. Add `[[example]]` entries with `required-features` for all examples
