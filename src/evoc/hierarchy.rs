//! Cluster hierarchy for EVōC.
//!
//! Represents hierarchical clustering structure as a tree of cluster nodes.

use crate::RetrieveError;
use std::collections::HashMap;

/// Cluster hierarchy tree.
pub struct ClusterHierarchy {
    nodes: Vec<ClusterNode>,
    root: Option<usize>,
}

impl ClusterHierarchy {
    /// Build hierarchy from MST edges.
    pub fn from_mst(edges: Vec<(usize, usize, f32)>, num_vectors: usize) -> Self {
        // Create leaf nodes for each vector
        let mut nodes: Vec<ClusterNode> = (0..num_vectors)
            .map(|i| ClusterNode {
                id: i,
                children: Vec::new(),
                members: vec![i],
                distance: 0.0,
            })
            .collect();

        // Build tree by merging clusters based on MST edges
        // Sort edges by distance (already sorted from MST)
        let mut next_node_id = num_vectors;

        for (i, j, dist) in edges {
            // Find root nodes for i and j
            let root_i = Self::find_root(&nodes, i);
            let root_j = Self::find_root(&nodes, j);

            if root_i == root_j {
                continue;
            }

            // Create new internal node merging root_i and root_j
            let mut members = nodes[root_i].members.clone();
            members.extend_from_slice(&nodes[root_j].members);

            let new_node = ClusterNode {
                id: next_node_id,
                children: vec![root_i, root_j],
                members,
                distance: dist,
            };

            nodes.push(new_node);
            next_node_id += 1;
        }

        // Root is the last node (highest in hierarchy)
        let root = if nodes.len() > num_vectors {
            Some(nodes.len() - 1)
        } else {
            None
        };

        Self { nodes, root }
    }

    /// Find root node for a vector.
    fn find_root(nodes: &[ClusterNode], vector_idx: usize) -> usize {
        // Find the node that contains this vector and has no parent
        for (i, node) in nodes.iter().enumerate() {
            if node.members.contains(&vector_idx) && node.children.is_empty() {
                return i;
            }
        }
        vector_idx // Fallback: leaf node
    }

    /// Extract cluster layer at given distance threshold.
    pub fn extract_layer(
        &self,
        threshold: f32,
        min_cluster_size: usize,
    ) -> Result<crate::evoc::clustering::ClusterLayer, RetrieveError> {
        let num_vectors = self.nodes.iter().filter(|n| n.children.is_empty()).count();

        let mut assignments = vec![None; num_vectors];
        let mut clusters = HashMap::new();
        let mut cluster_id = 0;

        // Traverse tree and extract clusters at threshold
        if let Some(root) = self.root {
            self.extract_clusters_recursive(
                root,
                threshold,
                min_cluster_size,
                &mut assignments,
                &mut clusters,
                &mut cluster_id,
            )?;
        }

        Ok(crate::evoc::clustering::ClusterLayer {
            assignments,
            num_clusters: cluster_id,
            clusters,
        })
    }

    /// Recursively extract clusters.
    fn extract_clusters_recursive(
        &self,
        node_id: usize,
        threshold: f32,
        min_cluster_size: usize,
        assignments: &mut [Option<usize>],
        clusters: &mut HashMap<usize, Vec<usize>>,
        cluster_id: &mut usize,
    ) -> Result<(), RetrieveError> {
        let node = &self.nodes[node_id];

        // If node distance exceeds threshold, split into children
        if node.distance > threshold && !node.children.is_empty() {
            for &child_id in &node.children {
                self.extract_clusters_recursive(
                    child_id,
                    threshold,
                    min_cluster_size,
                    assignments,
                    clusters,
                    cluster_id,
                )?;
            }
        } else {
            // This is a cluster at this threshold
            if node.members.len() >= min_cluster_size {
                let current_id = *cluster_id;
                *cluster_id += 1;

                clusters.insert(current_id, node.members.clone());

                for &member in &node.members {
                    if member < assignments.len() {
                        assignments[member] = Some(current_id);
                    }
                }
            }
            // Otherwise, mark as noise (None assignment)
        }

        Ok(())
    }

    /// Get all distances in hierarchy.
    pub fn get_all_distances(&self) -> Vec<f32> {
        self.nodes
            .iter()
            .map(|n| n.distance)
            .filter(|&d| d > 0.0)
            .collect()
    }
}

/// Cluster node in hierarchy.
#[derive(Clone, Debug)]
pub struct ClusterNode {
    /// Node ID
    pub id: usize,

    /// Child node IDs
    pub children: Vec<usize>,

    /// Vector indices in this cluster
    pub members: Vec<usize>,

    /// Distance threshold for this merge
    pub distance: f32,
}
