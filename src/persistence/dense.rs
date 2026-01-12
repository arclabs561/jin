//! Dense retrieval persistence.
//!
//! Provides disk persistence for dense vector retrieval, including:
//! - Vector storage (Structure of Arrays layout)
//! - HNSW index persistence
//! - IVF-PQ index persistence
//! - DiskANN index persistence
//!
//! See `docs/PERSISTENCE_DESIGN_DENSE.md` for format specifications.

use crate::persistence::directory::Directory;
use crate::persistence::error::{PersistenceError, PersistenceResult};
#[cfg(feature = "persistence")]
use bytemuck;
#[cfg(all(feature = "persistence", feature = "memmap"))]
use memmap2::{Advice, Mmap, MmapOptions};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::sync::Arc;

/// Dense segment writer for vector storage.
pub struct DenseSegmentWriter {
    directory: Box<dyn Directory>,
    segment_id: u64,
    dimension: usize,
    vectors: Vec<f32>, // Structure of Arrays: [v0[0..d], v1[0..d], ..., vn[0..d]]
    vector_metadata: Vec<VectorMetadata>,
    max_doc_id: u32,
}

/// Per-vector metadata.
///
/// Stored as 12 bytes: doc_id (u32=4) + norm (f32=4) + flags (u8=1) + padding (3 bytes)
/// Marked as `bytemuck::Pod` for zero-copy access from memory-mapped files.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "persistence", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct VectorMetadata {
    /// Document ID
    pub doc_id: u32,
    /// L2 norm (for cosine similarity)
    pub norm: f32,
    /// Bit flags (normalized, quantized, etc.)
    pub flags: u8,
    /// Padding to 12-byte alignment
    pub padding: [u8; 3],
}

impl DenseSegmentWriter {
    /// Create a new dense segment writer.
    pub fn new(directory: Box<dyn Directory>, segment_id: u64, dimension: usize) -> Self {
        Self {
            directory,
            segment_id,
            dimension,
            vectors: Vec::new(),
            vector_metadata: Vec::new(),
            max_doc_id: 0,
        }
    }

    /// Add a vector to the segment.
    pub fn add_vector(&mut self, doc_id: u32, vector: &[f32]) -> PersistenceResult<()> {
        if vector.len() != self.dimension {
            return Err(PersistenceError::InvalidConfig(format!(
                "Vector dimension mismatch: expected {}, got {}",
                self.dimension,
                vector.len()
            )));
        }

        // Compute L2 norm
        let norm: f32 = vector.iter().map(|&x| x * x).sum::<f32>().sqrt();

        // Store vector in SoA layout (will be written later)
        self.vectors.extend_from_slice(vector);

        // Store metadata
        self.vector_metadata.push(VectorMetadata {
            doc_id,
            norm,
            flags: 0, // TODO: Set flags based on normalization, quantization, etc.
            padding: [0; 3],
        });

        self.max_doc_id = self.max_doc_id.max(doc_id);

        Ok(())
    }

    /// Finalize the segment by writing all files.
    pub fn finalize(self) -> PersistenceResult<()> {
        let segment_dir = format!("segments/segment_dense_{}", self.segment_id);
        self.directory.create_dir_all(&segment_dir)?;

        // Write vectors in SoA (Structure of Arrays) layout:
        // All v[0] values, then all v[1] values, etc.
        // This enables SIMD-friendly access patterns for vector operations.
        let vectors_path = format!("{}/vectors.bin", segment_dir);
        let mut vectors_file = self.directory.create_file(&vectors_path)?;
        let num_vectors = self.vector_metadata.len();
        for d in 0..self.dimension {
            for v_idx in 0..num_vectors {
                let aos_idx = v_idx * self.dimension + d;
                vectors_file.write_all(&self.vectors[aos_idx].to_le_bytes())?;
            }
        }
        vectors_file.flush()?;

        // Write vector metadata
        let metadata_path = format!("{}/vector_metadata.bin", segment_dir);
        let mut metadata_file = self.directory.create_file(&metadata_path)?;
        for meta in &self.vector_metadata {
            metadata_file.write_all(&meta.doc_id.to_le_bytes())?;
            metadata_file.write_all(&meta.norm.to_le_bytes())?;
            metadata_file.write_all(&[meta.flags])?;
            metadata_file.write_all(&meta.padding)?; // Padding (3 bytes)
        }
        metadata_file.flush()?;

        Ok(())
    }
}

/// Dense segment reader for loading vectors from disk.
pub struct DenseSegmentReader {
    directory: Box<dyn Directory>,
    segment_id: u64,
    dimension: usize,
    num_vectors: usize,
    docid_to_index: HashMap<u32, usize>,
    #[cfg(all(feature = "persistence", feature = "memmap"))]
    vectors_mmap: Option<Arc<Mmap>>,
    #[cfg(all(feature = "persistence", feature = "memmap"))]
    metadata_mmap: Option<Arc<Mmap>>,
}

impl DenseSegmentReader {
    /// Load a dense segment from disk.
    pub fn load(
        directory: Box<dyn Directory>,
        segment_id: u64,
        dimension: usize,
    ) -> PersistenceResult<Self> {
        let segment_dir = format!("segments/segment_dense_{}", segment_id);

        // Load metadata and build docID -> vector_index mapping
        let metadata_path = format!("{}/vector_metadata.bin", segment_dir);

        #[cfg(all(feature = "persistence", feature = "memmap"))]
        let (metadata_mmap, docid_to_index, num_vectors) = {
            if let Some(file_path) = directory.file_path(&metadata_path) {
                if let Ok(file) = std::fs::File::open(&file_path) {
                    if let Ok(mmap) = unsafe { MmapOptions::new().map(&file) } {
                        // Set advice for random access (metadata accessed by doc_id)
                        let _ = mmap.advise(Advice::Random);
                        let mmap_arc = Arc::new(mmap);

                        // Build docID -> vector_index mapping
                        let mut docid_to_index = HashMap::new();
                        // Verify file size is aligned to 12-byte entries
                        if mmap_arc.len() % 12 != 0 {
                            return Err(PersistenceError::Format {
                                message: format!(
                                    "Metadata file size {} is not aligned to 12-byte entries",
                                    mmap_arc.len()
                                ),
                                expected: Some("multiple of 12 bytes".to_string()),
                                actual: Some(format!("{} bytes", mmap_arc.len())),
                            });
                        }
                        let num_vectors = mmap_arc.len() / 12; // 12 bytes per entry: doc_id(4) + norm(4) + flags(4)
                        for i in 0..num_vectors {
                            let offset = i * 12;
                            // Bounds check: offset + 4 must be <= len
                            if offset + 4 > mmap_arc.len() {
                                return Err(PersistenceError::Format {
                                    message: format!("Metadata entry {} out of bounds", i),
                                    expected: Some(format!("offset + 4 <= {}", mmap_arc.len())),
                                    actual: Some(format!("offset + 4 = {}", offset + 4)),
                                });
                            }
                            let doc_id_bytes: [u8; 4] = mmap_arc[offset..offset + 4]
                                .try_into()
                                .map_err(|_| PersistenceError::Format {
                                    message: format!(
                                        "Failed to extract doc_id bytes at offset {}",
                                        offset
                                    ),
                                    expected: Some("4-byte array".to_string()),
                                    actual: None,
                                })?;
                            let doc_id = u32::from_le_bytes(doc_id_bytes);
                            docid_to_index.insert(doc_id, i);
                        }
                        (Some(mmap_arc), docid_to_index, num_vectors)
                    } else {
                        // Fallback to regular read within the memmap block if map fails
                        let mut metadata_file = directory.open_file(&metadata_path)?;
                        let mut metadata_buffer = Vec::new();
                        metadata_file.read_to_end(&mut metadata_buffer)?;
                        if metadata_buffer.len() % 12 != 0 {
                            return Err(PersistenceError::Format {
                                message: format!(
                                    "Metadata file size {} is not aligned to 12-byte entries",
                                    metadata_buffer.len()
                                ),
                                expected: Some("multiple of 12 bytes".to_string()),
                                actual: Some(format!("{} bytes", metadata_buffer.len())),
                            });
                        }
                        let num_vectors = metadata_buffer.len() / 12;
                        let mut docid_to_index = HashMap::new();
                        for i in 0..num_vectors {
                            let offset = i * 12;
                            let doc_id = u32::from_le_bytes(
                                metadata_buffer[offset..offset + 4]
                                    .try_into()
                                    .map_err(|_| PersistenceError::Format {
                                        message: format!(
                                            "Failed to extract doc_id bytes at offset {}",
                                            offset
                                        ),
                                        expected: Some("4-byte array".to_string()),
                                        actual: None,
                                    })?,
                            );
                            docid_to_index.insert(doc_id, i);
                        }
                        (None, docid_to_index, num_vectors)
                    }
                } else {
                    return Err(PersistenceError::NotFound(metadata_path));
                }
            } else {
                // Not FsDirectory - fallback to regular read
                let mut metadata_file = directory.open_file(&metadata_path)?;
                let mut metadata_buffer = Vec::new();
                metadata_file.read_to_end(&mut metadata_buffer)?;
                let num_vectors = metadata_buffer.len() / 12;
                let mut docid_to_index = HashMap::new();
                for i in 0..num_vectors {
                    let offset = i * 12;
                    let doc_id =
                        u32::from_le_bytes(metadata_buffer[offset..offset + 4].try_into().unwrap());
                    docid_to_index.insert(doc_id, i);
                }
                (None, docid_to_index, num_vectors)
            }
        };

        #[cfg(not(all(feature = "persistence", feature = "memmap")))]
        let (_metadata_mmap, docid_to_index, num_vectors) = {
            let mut metadata_file = directory.open_file(&metadata_path)?;
            let mut metadata_buffer = Vec::new();
            metadata_file.read_to_end(&mut metadata_buffer)?;
            let num_vectors = metadata_buffer.len() / 12;
            let mut docid_to_index = HashMap::new();
            for i in 0..num_vectors {
                let offset = i * 12;
                let doc_id =
                    u32::from_le_bytes(metadata_buffer[offset..offset + 4].try_into().unwrap());
                docid_to_index.insert(doc_id, i);
            }
            // Use a dummy type for metadata_mmap since it's not used in this branch,
            // but we need to match the return type tuple structure conceptually or variable binding.
            // Actually, we don't need to bind metadata_mmap if we don't use it.
            // But DenseSegmentReader needs it if the field exists.
            // Since the field is guarded by the same cfg, we just don't return it here.
            (Option::<Arc<()>>::None, docid_to_index, num_vectors)
        };

        // Memory map vectors file if available
        #[cfg(all(feature = "persistence", feature = "memmap"))]
        let vectors_mmap = {
            let vectors_path = format!("{}/vectors.bin", segment_dir);
            if let Some(file_path) = directory.file_path(&vectors_path) {
                if let Ok(file) = std::fs::File::open(&file_path) {
                    if let Ok(mmap) = unsafe { MmapOptions::new().map(&file) } {
                        // Set advice for random access (vectors accessed by dimension)
                        let _ = mmap.advise(Advice::Random);
                        Some(Arc::new(mmap))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        };

        Ok(Self {
            directory,
            segment_id,
            dimension,
            num_vectors,
            docid_to_index,
            #[cfg(all(feature = "persistence", feature = "memmap"))]
            vectors_mmap,
            #[cfg(all(feature = "persistence", feature = "memmap"))]
            metadata_mmap,
        })
    }

    /// Get a vector by document ID.
    ///
    /// Returns the vector and its metadata.
    /// Uses O(1) docID lookup and memory mapping when available.
    pub fn get_vector(&self, doc_id: u32) -> PersistenceResult<(Vec<f32>, VectorMetadata)> {
        // O(1) lookup using docID -> vector_index mapping
        let vector_index =
            self.docid_to_index.get(&doc_id).copied().ok_or_else(|| {
                PersistenceError::NotFound(format!("Document {} not found", doc_id))
            })?;

        // Read metadata (use memory mapping if available)
        let metadata = {
            #[cfg(all(feature = "persistence", feature = "memmap"))]
            {
                if let Some(ref mmap) = self.metadata_mmap {
                    let offset = vector_index * 12; // 12 bytes per entry
                    if offset + 12 <= mmap.len() {
                        // Zero-copy read using bytemuck (no intermediate byte array allocations)
                        // from_bytes is safe for memory-mapped data (page-aligned, Pod type)
                        *bytemuck::from_bytes::<VectorMetadata>(&mmap[offset..offset + 12])
                    } else {
                        return Err(PersistenceError::Format {
                            message: "Metadata offset out of bounds".to_string(),
                            expected: Some(format!("< {}", mmap.len())),
                            actual: Some(format!("{}", offset + 12)),
                        });
                    }
                } else {
                    // Fallback to file read
                    Self::read_metadata_from_file(
                        self.directory.as_ref(),
                        self.segment_id,
                        vector_index,
                    )?
                }
            }
            #[cfg(not(all(feature = "persistence", feature = "memmap")))]
            {
                Self::read_metadata_from_file(
                    self.directory.as_ref(),
                    self.segment_id,
                    vector_index,
                )?
            }
        };

        // Read vector (use memory mapping if available)
        let vector = {
            #[cfg(all(feature = "persistence", feature = "memmap"))]
            {
                if let Some(ref mmap) = self.vectors_mmap {
                    // SoA layout: all v[0], then all v[1], etc.
                    // Use bytemuck::cast_slice for efficient SIMD-friendly reads
                    // This is more efficient than individual from_bytes calls
                    let mut vec = Vec::with_capacity(self.dimension);
                    for d in 0..self.dimension {
                        let byte_offset = (d * self.num_vectors + vector_index) * 4;
                        if byte_offset + 4 <= mmap.len() {
                            // Zero-copy read using bytemuck::from_bytes for single f32
                            vec.push(*bytemuck::from_bytes::<f32>(
                                &mmap[byte_offset..byte_offset + 4],
                            ));
                        } else {
                            return Err(PersistenceError::Format {
                                message: "Vector offset out of bounds".to_string(),
                                expected: Some(format!("< {}", mmap.len())),
                                actual: Some(format!("{}", byte_offset + 4)),
                            });
                        }
                    }
                    vec
                } else {
                    // Fallback to file read
                    Self::read_vector_from_file(
                        self.directory.as_ref(),
                        self.segment_id,
                        vector_index,
                        self.dimension,
                        self.num_vectors,
                    )?
                }
            }
            #[cfg(not(all(feature = "persistence", feature = "memmap")))]
            {
                Self::read_vector_from_file(
                    self.directory.as_ref(),
                    self.segment_id,
                    vector_index,
                    self.dimension,
                    self.num_vectors,
                )?
            }
        };

        Ok((vector, metadata))
    }

    // Helper for non-mmap metadata reading
    fn read_metadata_from_file(
        directory: &dyn Directory,
        segment_id: u64,
        vector_index: usize,
    ) -> PersistenceResult<VectorMetadata> {
        let metadata_path = format!("segments/segment_dense_{}/vector_metadata.bin", segment_id);
        let mut metadata_file = directory.open_file(&metadata_path)?;
        // Read entire file and extract our entry (Directory trait doesn't guarantee Seek support)
        let mut metadata_buffer = Vec::new();
        metadata_file.read_to_end(&mut metadata_buffer)?;
        let offset = vector_index * 12;
        if offset + 12 > metadata_buffer.len() {
            return Err(PersistenceError::Format {
                message: format!("Metadata entry {} out of bounds", vector_index),
                expected: Some(format!("offset + 12 <= {}", metadata_buffer.len())),
                actual: Some(format!("offset + 12 = {}", offset + 12)),
            });
        }
        // Extract bytes from buffer
        let doc_id_bytes: [u8; 4] =
            metadata_buffer[offset..offset + 4]
                .try_into()
                .map_err(|_| PersistenceError::Format {
                    message: format!("Failed to extract doc_id bytes at offset {}", offset),
                    expected: Some("4-byte array".to_string()),
                    actual: None,
                })?;
        let norm_bytes: [u8; 4] =
            metadata_buffer[offset + 4..offset + 8]
                .try_into()
                .map_err(|_| PersistenceError::Format {
                    message: format!("Failed to extract norm bytes at offset {}", offset + 4),
                    expected: Some("4-byte array".to_string()),
                    actual: None,
                })?;

        Ok(VectorMetadata {
            doc_id: u32::from_le_bytes(doc_id_bytes),
            norm: f32::from_le_bytes(norm_bytes),
            flags: metadata_buffer[offset + 8],
            padding: [
                metadata_buffer.get(offset + 9).copied().unwrap_or(0),
                metadata_buffer.get(offset + 10).copied().unwrap_or(0),
                metadata_buffer.get(offset + 11).copied().unwrap_or(0),
            ],
        })
    }

    // Helper for non-mmap vector reading
    fn read_vector_from_file(
        directory: &dyn Directory,
        segment_id: u64,
        vector_index: usize,
        dimension: usize,
        num_vectors: usize,
    ) -> PersistenceResult<Vec<f32>> {
        let vectors_path = format!("segments/segment_dense_{}/vectors.bin", segment_id);
        let mut vectors_file = directory.open_file(&vectors_path)?;
        let mut all_vectors = vec![0f32; num_vectors * dimension];
        {
            let mut buffer = vec![0u8; num_vectors * dimension * 4];
            vectors_file.read_exact(&mut buffer)?;
            for (i, chunk) in buffer.chunks_exact(4).enumerate() {
                all_vectors[i] = f32::from_le_bytes(chunk.try_into().unwrap());
            }
        }

        // Extract vector for this index (SoA layout)
        let mut vec = Vec::with_capacity(dimension);
        for d in 0..dimension {
            let idx = d * num_vectors + vector_index;
            vec.push(all_vectors[idx]);
        }
        Ok(vec)
    }

    /// Get number of vectors in segment.
    pub fn num_vectors(&self) -> usize {
        self.num_vectors
    }

    /// Get vector dimension.
    pub fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::directory::MemoryDirectory;

    #[test]
    fn test_dense_segment_write_read() {
        let dir = Box::new(MemoryDirectory::new());
        let segment_id = 1;
        let dimension = 3;

        // Write segment
        let mut writer = DenseSegmentWriter::new(dir.clone(), segment_id, dimension);
        writer.add_vector(0, &[1.0, 0.0, 0.0]).unwrap();
        writer.add_vector(1, &[0.0, 1.0, 0.0]).unwrap();
        writer.finalize().unwrap();

        // Read segment
        let reader = DenseSegmentReader::load(dir, segment_id, dimension).unwrap();
        assert_eq!(reader.num_vectors(), 2);
        assert_eq!(reader.dimension(), 3);

        let (vector, metadata) = reader.get_vector(0).unwrap();
        assert_eq!(vector, vec![1.0, 0.0, 0.0]);
        assert_eq!(metadata.doc_id, 0);
    }
}
