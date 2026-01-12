//! Memory layout optimizations for cache efficiency.

#![allow(dead_code)]

/// Structure of Arrays (SoA) storage for vectors.
///
/// Stores all vectors contiguously: [v0[0..d], v1[0..d], ..., vn[0..d]]
/// This layout enables:
/// - Better cache locality when accessing multiple vectors
/// - SIMD-friendly batch operations
/// - Reduced memory fragmentation
pub struct VectorStorage {
    /// Flattened vector storage
    data: Vec<f32>,

    /// Vector dimension
    dimension: usize,

    /// Number of vectors
    count: usize,
}

impl VectorStorage {
    /// Create new vector storage.
    pub fn new(dimension: usize) -> Self {
        Self {
            data: Vec::new(),
            dimension,
            count: 0,
        }
    }

    /// Add a vector.
    pub fn add(&mut self, vector: &[f32]) {
        assert_eq!(vector.len(), self.dimension);
        self.data.extend_from_slice(vector);
        self.count += 1;
    }

    /// Get vector by index.
    pub fn get(&self, idx: usize) -> &[f32] {
        let start = idx * self.dimension;
        let end = start + self.dimension;
        &self.data[start..end]
    }

    /// Get number of vectors.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Reserve capacity for n vectors.
    pub fn reserve(&mut self, n: usize) {
        self.data.reserve(n * self.dimension);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vector_storage() {
        let mut storage = VectorStorage::new(3);

        storage.add(&[1.0, 2.0, 3.0]);
        storage.add(&[4.0, 5.0, 6.0]);

        assert_eq!(storage.len(), 2);
        assert_eq!(storage.get(0), &[1.0, 2.0, 3.0]);
        assert_eq!(storage.get(1), &[4.0, 5.0, 6.0]);
    }
}
