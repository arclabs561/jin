use smallvec::SmallVec;

/// A proximity graph where nodes are vectors and edges connect "nearby" nodes.
pub trait ProxGraph {
    fn neighbors(&self, node: u32) -> &[u32];
    fn add_edge(&mut self, u: u32, v: u32);
}

/// α-Convergent Neighborhood Graph (α-CNG) pruner.
/// 
/// Uses shifted-scaled triangle inequality to aggressively prune edges
/// while maintaining connectivity and search guarantees.
pub struct AlphaCng {
    pub alpha: f32,
    pub beta: f32,
}

impl AlphaCng {
    pub fn new(alpha: f32, beta: f32) -> Self {
        Self { alpha, beta }
    }

    /// Prune neighbors of `u` using α-CNG rule.
    /// 
    /// Keep `v` only if: d(u, v) <= alpha * d(u, w) + beta * d(w, v)
    /// for all previously accepted neighbors `w`.
    pub fn prune(&self, u: u32, candidates: &[u32], dist_fn: impl Fn(u32, u32) -> f32) -> Vec<u32> {
        let mut kept = Vec::new();
        // Sort by distance to u (heuristic)
        let mut sorted_candidates = candidates.to_vec();
        sorted_candidates.sort_by(|&a, &b| dist_fn(u, a).partial_cmp(&dist_fn(u, b)).unwrap());

        for &v in &sorted_candidates {
            let d_uv = dist_fn(u, v);
            let mut keep = true;

            for &w in &kept {
                let d_uw = dist_fn(u, w);
                let d_wv = dist_fn(w, v);

                // α-CNG pruning rule
                if d_uv > self.alpha * d_uw + self.beta * d_wv {
                    keep = false;
                    break;
                }
            }

            if keep {
                kept.push(v);
            }
        }
        kept
    }
}
