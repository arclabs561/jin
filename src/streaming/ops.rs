//! Update operations for streaming indices.

/// A single update operation.
#[derive(Debug, Clone)]
pub enum UpdateOp {
    /// Insert a new vector.
    Insert { id: u32, vector: Vec<f32> },
    /// Delete an existing vector.
    Delete { id: u32 },
    /// Update (atomic delete + insert).
    Update { id: u32, vector: Vec<f32> },
}

impl UpdateOp {
    /// Get the ID affected by this operation.
    pub fn id(&self) -> u32 {
        match self {
            UpdateOp::Insert { id, .. } => *id,
            UpdateOp::Delete { id } => *id,
            UpdateOp::Update { id, .. } => *id,
        }
    }

    /// Check if this is a delete operation.
    pub fn is_delete(&self) -> bool {
        matches!(self, UpdateOp::Delete { .. })
    }

    /// Check if this operation adds data (insert or update).
    pub fn adds_data(&self) -> bool {
        matches!(self, UpdateOp::Insert { .. } | UpdateOp::Update { .. })
    }
}

/// A batch of update operations.
#[derive(Debug, Clone, Default)]
pub struct UpdateBatch {
    ops: Vec<UpdateOp>,
}

impl UpdateBatch {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            ops: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, op: UpdateOp) {
        self.ops.push(op);
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &UpdateOp> {
        self.ops.iter()
    }
}

impl IntoIterator for UpdateBatch {
    type Item = UpdateOp;
    type IntoIter = std::vec::IntoIter<UpdateOp>;

    fn into_iter(self) -> Self::IntoIter {
        self.ops.into_iter()
    }
}

impl UpdateBatch {
    /// Count inserts in batch.
    pub fn insert_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, UpdateOp::Insert { .. }))
            .count()
    }

    /// Count deletes in batch.
    pub fn delete_count(&self) -> usize {
        self.ops.iter().filter(|op| op.is_delete()).count()
    }

    /// Count updates in batch.
    pub fn update_count(&self) -> usize {
        self.ops
            .iter()
            .filter(|op| matches!(op, UpdateOp::Update { .. }))
            .count()
    }
}

/// Statistics from applying updates.
#[derive(Debug, Clone, Default)]
pub struct UpdateStats {
    pub inserts_applied: usize,
    pub deletes_applied: usize,
    pub updates_applied: usize,
    pub errors: usize,
    /// Time spent applying updates (microseconds).
    pub duration_us: u64,
}

impl UpdateStats {
    pub fn merge(&mut self, other: &UpdateStats) {
        self.inserts_applied += other.inserts_applied;
        self.deletes_applied += other.deletes_applied;
        self.updates_applied += other.updates_applied;
        self.errors += other.errors;
        self.duration_us += other.duration_us;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch() {
        let mut batch = UpdateBatch::new();
        batch.push(UpdateOp::Insert {
            id: 0,
            vector: vec![1.0],
        });
        batch.push(UpdateOp::Delete { id: 1 });
        batch.push(UpdateOp::Update {
            id: 2,
            vector: vec![2.0],
        });

        assert_eq!(batch.len(), 3);
        assert_eq!(batch.insert_count(), 1);
        assert_eq!(batch.delete_count(), 1);
        assert_eq!(batch.update_count(), 1);
    }
}
