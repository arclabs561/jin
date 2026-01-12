//! Working set cache for DiskANN.

use std::collections::HashMap;

/// LRU cache for frequently accessed vectors.
pub struct WorkingSetCache {
    capacity: usize,
    cache: HashMap<u32, Vec<f32>>,
}

impl WorkingSetCache {
    /// Create new cache with specified capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cache: HashMap::new(),
        }
    }

    /// Get vector from cache.
    pub fn get(&self, id: u32) -> Option<&Vec<f32>> {
        self.cache.get(&id)
    }

    /// Insert vector into cache.
    pub fn insert(&mut self, id: u32, vector: Vec<f32>) {
        // Simple implementation: if cache is full, remove oldest (FIFO)
        // In production, would use proper LRU
        if self.cache.len() >= self.capacity {
            // Remove first entry (simple FIFO)
            if let Some(&first_key) = self.cache.keys().next() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(id, vector);
    }

    /// Clear cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }
}
