# jin

<p align="center">
  <img
    src="doc/readme_logo.png"
    width="160"
    alt="jin logo"
    style="box-shadow: 0 10px 28px rgba(0,0,0,0.18); border-radius: 14px;"
  />
</p>

Approximate nearest neighbor search in Rust.

Dual-licensed under MIT or Apache-2.0.

```rust
use jin::hnsw::HNSWIndex;

let mut index = HNSWIndex::new(128, 16, 32)?; // dim, M, ef_construction
for (id, v) in vectors.iter().enumerate() {
    index.add_slice(id as u32, v)?;
}
index.build()?;

let hits = index.search(&query, 10, 50)?; // k, ef_search
println!("{hits:?}");
```

More:
- `GUIDE.md`
- `examples/`
- `doc/`