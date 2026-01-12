//! HNSW graph persistence.
//!
//! Provides disk persistence for HNSW indexes, including:
//! - Graph structure serialization (layers, neighbors)
//! - Vector storage (reuses dense segment format)
//! - Layer assignments
//! - Parameters
//!
//! See `docs/PERSISTENCE_DESIGN_DENSE.md` for format specifications.

use crate::persistence::directory::Directory;
use crate::persistence::error::PersistenceResult;
use std::io::{Read, Write};

#[cfg(feature = "hnsw")]
use crate::hnsw::graph::Layer;
#[cfg(feature = "hnsw")]
use crate::hnsw::{
    HNSWIndex, HNSWParams, NeighborhoodDiversification, SeedSelectionStrategy,
};
#[cfg(feature = "hnsw")]
use smallvec::SmallVec;

/// HNSW segment writer for graph persistence.
#[cfg(feature = "hnsw")]
pub struct HNSWSegmentWriter {
    directory: Box<dyn Directory>,
    segment_id: u64,
}

#[cfg(feature = "hnsw")]
impl HNSWSegmentWriter {
    /// Create a new HNSW segment writer.
    pub fn new(directory: Box<dyn Directory>, segment_id: u64) -> Self {
        Self {
            directory,
            segment_id,
        }
    }

    /// Write an HNSW index to disk.
    ///
    /// Format:
    /// - `vectors.bin`: Vector data (SoA layout, same as dense segment)
    /// - `layers.bin`: Graph layers (serialized neighbor lists)
    /// - `layer_assignments.bin`: Layer assignment for each vector
    /// - `params.bin`: HNSW parameters
    /// - `metadata.bin`: Index metadata (dimension, num_vectors, etc.)
    pub fn write_hnsw_index(&mut self, index: &HNSWIndex) -> PersistenceResult<()> {
        let segment_dir = format!("segments/segment_hnsw_{}", self.segment_id);
        self.directory.create_dir_all(&segment_dir)?;

        // Write vectors (reuse dense segment format)
        let vectors_path = format!("{}/vectors.bin", segment_dir);
        let mut vectors_file = self.directory.create_file(&vectors_path)?;
        for &value in &index.vectors {
            vectors_file.write_all(&value.to_le_bytes())?;
        }
        vectors_file.flush()?;

        // Write layer assignments
        let assignments_path = format!("{}/layer_assignments.bin", segment_dir);
        let mut assignments_file = self.directory.create_file(&assignments_path)?;
        for &assignment in &index.layer_assignments {
            assignments_file.write_all(&[assignment])?;
        }
        assignments_file.flush()?;

        // Write graph layers
        let layers_path = format!("{}/layers.bin", segment_dir);
        let mut layers_file = self.directory.create_file(&layers_path)?;

        // Write number of layers
        layers_file.write_all(&(index.layers.len() as u32).to_le_bytes())?;

        // Write each layer
        for layer in &index.layers {
            // Get neighbors (only works for uncompressed layers)
            let neighbors = layer.get_all_neighbors().ok_or_else(|| {
                std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "Cannot persist compressed layers - decompress first",
                )
            })?;

            // Write number of neighbor lists
            layers_file.write_all(&(neighbors.len() as u32).to_le_bytes())?;

            // Write each neighbor list
            for neighbor_list in neighbors {
                // Write number of neighbors
                layers_file.write_all(&(neighbor_list.len() as u32).to_le_bytes())?;

                // Write neighbor IDs
                for &neighbor_id in neighbor_list.iter() {
                    layers_file.write_all(&neighbor_id.to_le_bytes())?;
                }
            }
        }
        layers_file.flush()?;

        // Write parameters
        let params_path = format!("{}/params.bin", segment_dir);
        let mut params_file = self.directory.create_file(&params_path)?;
        params_file.write_all(&(index.params.m as u32).to_le_bytes())?;
        params_file.write_all(&(index.params.m_max as u32).to_le_bytes())?;
        params_file.write_all(&index.params.m_l.to_le_bytes())?;
        params_file.write_all(&(index.params.ef_construction as u32).to_le_bytes())?;
        params_file.write_all(&(index.params.ef_search as u32).to_le_bytes())?;
        params_file.flush()?;

        // Write metadata
        let metadata_path = format!("{}/metadata.bin", segment_dir);
        let mut metadata_file = self.directory.create_file(&metadata_path)?;
        metadata_file.write_all(&(index.dimension as u32).to_le_bytes())?;
        metadata_file.write_all(&(index.num_vectors as u32).to_le_bytes())?;
        metadata_file.write_all(&[if index.is_built() { 1 } else { 0 }])?;
        metadata_file.flush()?;

        Ok(())
    }
}

/// HNSW segment reader for loading graphs from disk.
#[cfg(feature = "hnsw")]
pub struct HNSWSegmentReader {
    directory: Box<dyn Directory>,
    segment_id: u64,
    dimension: usize,
    num_vectors: usize,
    params: HNSWParams,
    built: bool,
}

#[cfg(feature = "hnsw")]
impl HNSWSegmentReader {
    /// Load an HNSW segment from disk.
    pub fn load(directory: Box<dyn Directory>, segment_id: u64) -> PersistenceResult<Self> {
        let segment_dir = format!("segments/segment_hnsw_{}", segment_id);

        // Load metadata
        let metadata_path = format!("{}/metadata.bin", segment_dir);
        let mut metadata_file = directory.open_file(&metadata_path)?;
        let mut dim_bytes = [0u8; 4];
        let mut num_vec_bytes = [0u8; 4];
        let mut built_byte = [0u8; 1];
        metadata_file.read_exact(&mut dim_bytes)?;
        metadata_file.read_exact(&mut num_vec_bytes)?;
        metadata_file.read_exact(&mut built_byte)?;

        let dimension = u32::from_le_bytes(dim_bytes) as usize;
        let num_vectors = u32::from_le_bytes(num_vec_bytes) as usize;
        let built = built_byte[0] != 0;

        // Load parameters
        let params_path = format!("{}/params.bin", segment_dir);
        let mut params_file = directory.open_file(&params_path)?;
        let mut m_bytes = [0u8; 4];
        let mut m_max_bytes = [0u8; 4];
        let mut m_l_bytes = [0u8; 8];
        let mut ef_construction_bytes = [0u8; 4];
        let mut ef_search_bytes = [0u8; 4];

        params_file.read_exact(&mut m_bytes)?;
        params_file.read_exact(&mut m_max_bytes)?;
        params_file.read_exact(&mut m_l_bytes)?;
        params_file.read_exact(&mut ef_construction_bytes)?;
        params_file.read_exact(&mut ef_search_bytes)?;

        let params = HNSWParams {
            m: u32::from_le_bytes(m_bytes) as usize,
            m_max: u32::from_le_bytes(m_max_bytes) as usize,
            m_l: f64::from_le_bytes(m_l_bytes),
            ef_construction: u32::from_le_bytes(ef_construction_bytes) as usize,
            ef_search: u32::from_le_bytes(ef_search_bytes) as usize,
            seed_selection: SeedSelectionStrategy::default(),
            neighborhood_diversification: NeighborhoodDiversification::default(),
            #[cfg(feature = "id-compression")]
            id_compression: None,
            #[cfg(feature = "id-compression")]
            compression_threshold: 100,
        };

        Ok(Self {
            directory,
            segment_id,
            dimension,
            num_vectors,
            params,
            built,
        })
    }

    /// Reconstruct the HNSW index from disk.
    ///
    /// This loads all data structures into memory.
    /// For large indexes, consider using memory mapping.
    ///
    /// Note: This is a placeholder implementation. Full reconstruction requires
    /// HNSWIndex to expose a constructor or builder that accepts all fields.
    pub fn load_index(&self) -> PersistenceResult<HNSWIndex> {
        let segment_dir = format!("segments/segment_hnsw_{}", self.segment_id);

        // Load vectors
        let vectors_path = format!("{}/vectors.bin", segment_dir);
        let mut vectors_file = self.directory.open_file(&vectors_path)?;
        let mut vectors = Vec::new();
        let mut value_bytes = [0u8; 4];
        for _ in 0..(self.num_vectors * self.dimension) {
            vectors_file.read_exact(&mut value_bytes)?;
            vectors.push(f32::from_le_bytes(value_bytes));
        }

        // Load layer assignments
        let assignments_path = format!("{}/layer_assignments.bin", segment_dir);
        let mut assignments_file = self.directory.open_file(&assignments_path)?;
        let mut layer_assignments = vec![0u8; self.num_vectors];
        assignments_file.read_exact(&mut layer_assignments)?;

        // Load graph layers
        let layers_path = format!("{}/layers.bin", segment_dir);
        let mut layers_file = self.directory.open_file(&layers_path)?;

        let mut num_layers_bytes = [0u8; 4];
        layers_file.read_exact(&mut num_layers_bytes)?;
        let num_layers = u32::from_le_bytes(num_layers_bytes) as usize;

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let mut num_lists_bytes = [0u8; 4];
            layers_file.read_exact(&mut num_lists_bytes)?;
            let num_lists = u32::from_le_bytes(num_lists_bytes) as usize;

            let mut neighbors_list = Vec::with_capacity(num_lists);
            for _ in 0..num_lists {
                let mut num_neighbors_bytes = [0u8; 4];
                layers_file.read_exact(&mut num_neighbors_bytes)?;
                let num_neighbors = u32::from_le_bytes(num_neighbors_bytes) as usize;

                let mut neighbors: SmallVec<[u32; 16]> = SmallVec::new();
                for _ in 0..num_neighbors {
                    let mut neighbor_bytes = [0u8; 4];
                    layers_file.read_exact(&mut neighbor_bytes)?;
                    neighbors.push(u32::from_le_bytes(neighbor_bytes));
                }
                neighbors_list.push(neighbors);
            }

            // Construct layer using crate-internal constructor
            layers.push(Layer::new_uncompressed(neighbors_list));
        }

        // Reconstruct index
        Ok(HNSWIndex::from_parts(
            vectors,
            self.dimension,
            self.num_vectors,
            layers,
            layer_assignments,
            self.params.clone(),
            self.built,
        ))
    }
}

#[cfg(test)]
#[cfg(feature = "hnsw")]
mod tests {
    use super::*;
    use crate::persistence::directory::MemoryDirectory;

    #[test]
    fn test_hnsw_segment_write_read() {
        // This test would require creating an HNSWIndex
        // For now, just test that the structure compiles
        let dir = Box::new(MemoryDirectory::new());
        let writer = HNSWSegmentWriter::new(dir, 1);
        // TODO: Add full test with actual HNSWIndex
    }
}
