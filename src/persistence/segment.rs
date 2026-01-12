//! Segment format and management.
//!
//! Segments are immutable units of storage containing indexed data.
//! This module handles segment creation, loading, and querying.
//!
//! See `docs/PERSISTENCE_DESIGN.md` for format specifications.

use crate::persistence::codec::{bitpack, delta, varint, BLOCK_SIZE};
use crate::persistence::directory::Directory;
use crate::persistence::error::{PersistenceError, PersistenceResult};
use crate::persistence::format::{SegmentFooter, SegmentOffsets};
#[cfg(feature = "persistence")]
use fst::{IntoStreamer, Map, MapBuilder, Streamer};
#[cfg(all(feature = "persistence", feature = "memmap"))]
use memmap2::{Advice, Mmap, MmapOptions};
use std::collections::HashMap;
use std::io::{Read, Write};
#[cfg(all(feature = "persistence", feature = "memmap"))]
use std::sync::Arc;

/// Term information stored in segment.
#[derive(Debug, Clone)]
pub struct TermInfo {
    /// Offset into postings.bin
    pub postings_offset: u64,
    /// Length of postings list in bytes
    pub postings_len: u64,
    /// Document frequency (number of documents containing this term)
    pub doc_frequency: u32,
    /// Collection frequency (total term frequency across all documents)
    pub collection_frequency: u64,
}

/// BM25 segment writer.
///
/// Builds a segment from an in-memory inverted index.
pub struct SegmentWriter {
    /// Directory for writing segment files
    directory: Box<dyn Directory>,
    /// Segment ID
    segment_id: u64,
    /// Current offset in postings file
    postings_offset: u64,
    /// Term dictionary (term -> ordinal)
    term_dict: Vec<(String, u64)>,
    /// Term info store (ordinal -> TermInfo)
    term_infos: Vec<TermInfo>,
    /// Document lengths
    doc_lengths: Vec<u32>,
    /// Document ID to user ID mapping (optional, for future use)
    #[allow(dead_code)]
    docid_to_userid: Vec<(u32, Vec<u8>)>,
    /// Maximum document ID
    max_doc_id: u32,
}

impl SegmentWriter {
    /// Create a new segment writer.
    pub fn new(directory: Box<dyn Directory>, segment_id: u64) -> Self {
        Self {
            directory,
            segment_id,
            postings_offset: 0,
            term_dict: Vec::new(),
            term_infos: Vec::new(),
            doc_lengths: Vec::new(),
            docid_to_userid: Vec::new(),
            max_doc_id: 0,
        }
    }

    /// Write a BM25 inverted index to a segment.
    ///
    /// This is the main entry point for creating a persistent segment from an in-memory index.
    pub fn write_bm25_index(
        &mut self,
        postings: &HashMap<String, HashMap<u32, u32>>,
        doc_lengths: &HashMap<u32, u32>,
        doc_frequencies: &HashMap<String, u32>,
    ) -> PersistenceResult<()> {
        // Sort terms for deterministic ordering
        let mut terms: Vec<&String> = postings.keys().collect();
        terms.sort();

        // Build document ID mapping (sorted for efficient encoding)
        let mut doc_ids: Vec<u32> = doc_lengths.keys().copied().collect();
        doc_ids.sort();
        self.max_doc_id = doc_ids.iter().max().copied().unwrap_or(0);

        // Initialize doc_lengths array (dense, indexed by doc_id)
        if !doc_ids.is_empty() {
            let max_id = self.max_doc_id as usize;
            self.doc_lengths = vec![0; max_id + 1];
            for (&doc_id, &length) in doc_lengths {
                if doc_id as usize <= max_id {
                    self.doc_lengths[doc_id as usize] = length;
                }
            }
        }

        // Write postings lists for each term
        for (ordinal, term) in terms.iter().enumerate() {
            let postings_list = postings.get(*term).unwrap();
            let doc_freq = doc_frequencies.get(*term).copied().unwrap_or(0);

            // Calculate collection frequency
            let collection_freq: u64 = postings_list.values().map(|&tf| tf as u64).sum();

            // Write postings list
            let (postings_len, _) = self.write_postings_list(postings_list)?;

            // Store term info
            let term_info = TermInfo {
                postings_offset: self.postings_offset,
                postings_len,
                doc_frequency: doc_freq,
                collection_frequency: collection_freq,
            };

            self.term_dict.push(((*term).clone(), ordinal as u64));
            self.term_infos.push(term_info);

            self.postings_offset += postings_len;
        }

        Ok(())
    }

    /// Write a postings list (doc_id -> term_frequency) to the postings file.
    ///
    /// Returns (bytes_written, term_info).
    fn write_postings_list(
        &mut self,
        postings: &HashMap<u32, u32>,
    ) -> PersistenceResult<(u64, TermInfo)> {
        // Sort document IDs for delta encoding
        let mut doc_ids: Vec<u32> = postings.keys().copied().collect();
        doc_ids.sort();

        // Collect term frequencies in same order
        let term_frequencies: Vec<u32> = doc_ids
            .iter()
            .map(|&doc_id| postings.get(&doc_id).copied().unwrap_or(0))
            .collect();

        // Encode postings list
        let encoded = self.encode_postings(&doc_ids, &term_frequencies)?;

        // Write to postings file (append to single file for all postings)
        let postings_path = format!("segments/segment_{}/postings.bin", self.segment_id);
        let mut file = if self.postings_offset == 0 {
            // First write - create file
            self.directory.create_file(&postings_path)?
        } else {
            // Subsequent writes - append to file
            self.directory.append_file(&postings_path)?
        };
        file.write_all(&encoded)?;
        file.flush()?;

        Ok((
            encoded.len() as u64,
            TermInfo {
                postings_offset: 0, // Will be set by caller
                postings_len: encoded.len() as u64,
                doc_frequency: doc_ids.len() as u32,
                collection_frequency: term_frequencies.iter().sum::<u32>() as u64,
            },
        ))
    }

    /// Encode a postings list using delta encoding + bitpacking or varint.
    ///
    /// Format:
    /// - For full blocks (128 docs): [bit_width_docids: u8][bitpacked_docid_deltas][bit_width_tfs: u8][bitpacked_tfs]
    /// - For partial blocks: [varint_docid_deltas...][varint_tfs...]
    fn encode_postings(
        &self,
        doc_ids: &[u32],
        term_frequencies: &[u32],
    ) -> PersistenceResult<Vec<u8>> {
        let mut encoded = Vec::new();

        // Delta encode document IDs
        let docid_deltas = delta::encode(doc_ids);

        // Process in blocks of BLOCK_SIZE
        let mut offset = 0;
        while offset < doc_ids.len() {
            let block_end = (offset + BLOCK_SIZE).min(doc_ids.len());
            let block_size = block_end - offset;

            let docid_block = &docid_deltas[offset..block_end];
            let tf_block = &term_frequencies[offset..block_end];

            if block_size == BLOCK_SIZE {
                // Full block: use bitpacking
                let docid_bit_width = bitpack::bit_width_many(docid_block);
                let tf_bit_width = bitpack::bit_width_many(tf_block);

                encoded.push(docid_bit_width);
                encoded.extend_from_slice(&bitpack::pack(docid_block, docid_bit_width));
                encoded.push(tf_bit_width);
                encoded.extend_from_slice(&bitpack::pack(tf_block, tf_bit_width));
            } else {
                // Partial block: use varint
                encoded.push(0); // Marker for varint encoding
                for &delta in docid_block {
                    encoded.extend_from_slice(&varint::encode(delta as u64));
                }
                for &tf in tf_block {
                    encoded.extend_from_slice(&varint::encode(tf as u64));
                }
            }

            offset = block_end;
        }

        Ok(encoded)
    }

    /// Finalize the segment by writing all files and footer.
    pub fn finalize(self) -> PersistenceResult<()> {
        let segment_dir = format!("segments/segment_{}", self.segment_id);
        self.directory.create_dir_all(&segment_dir)?;

        let mut offsets = SegmentOffsets::default();
        let mut current_offset = 0u64;

        // Write term dictionary using FST
        // FST requires keys to be inserted in lexicographic order for optimal compression
        // Reference: https://docs.rs/fst/latest/fst/struct.MapBuilder.html
        let term_dict_path = format!("{}/term_dict.fst", segment_dir);
        #[cfg(feature = "persistence")]
        {
            // Sort terms lexicographically before building FST
            // This is required for optimal FST compression and correctness
            let mut sorted_terms: Vec<_> = self.term_dict.iter().collect();
            sorted_terms.sort_by(|a, b| a.0.cmp(&b.0));

            let mut builder = MapBuilder::memory();
            for (term, ordinal) in sorted_terms {
                builder.insert(term.as_bytes(), *ordinal).map_err(|e| {
                    PersistenceError::Format {
                        message: format!("FST build error: {}", e),
                        expected: None,
                        actual: None,
                    }
                })?;
            }
            let fst_bytes = builder.into_inner().map_err(|e| PersistenceError::Format {
                message: format!("FST finalization error: {}", e),
                expected: None,
                actual: None,
            })?;

            // Write FST to file
            let mut term_dict_file = self.directory.create_file(&term_dict_path)?;
            term_dict_file.write_all(&fst_bytes)?;
            term_dict_file.flush()?;

            offsets.term_dict_offset = current_offset;
            offsets.term_dict_len = fst_bytes.len() as u64;
            current_offset += offsets.term_dict_len;
        }
        #[cfg(not(feature = "persistence"))]
        {
            offsets.term_dict_offset = current_offset;
            offsets.term_dict_len = 0;
        }

        // Write term info store
        let term_info_path = format!("{}/term_info.bin", segment_dir);
        let mut term_info_file = self.directory.create_file(&term_info_path)?;
        for term_info in &self.term_infos {
            term_info_file.write_all(&term_info.postings_offset.to_le_bytes())?;
            term_info_file.write_all(&term_info.postings_len.to_le_bytes())?;
            term_info_file.write_all(&term_info.doc_frequency.to_le_bytes())?;
            term_info_file.write_all(&term_info.collection_frequency.to_le_bytes())?;
        }
        term_info_file.flush()?;

        // Write document lengths
        let doc_lengths_path = format!("{}/doc_lengths.bin", segment_dir);
        let mut doc_lengths_file = self.directory.create_file(&doc_lengths_path)?;
        for &length in &self.doc_lengths {
            doc_lengths_file.write_all(&length.to_le_bytes())?;
        }
        doc_lengths_file.flush()?;

        // Update offsets for term info store
        offsets.term_info_offset = current_offset;
        offsets.term_info_len = (self.term_infos.len() * 28) as u64; // 8+8+4+8 bytes per TermInfo (postings_offset + postings_len + doc_frequency + collection_frequency)
        current_offset += offsets.term_info_len;

        // Update offsets for document lengths
        offsets.doc_lengths_offset = current_offset;
        offsets.doc_lengths_len = (self.doc_lengths.len() * 4) as u64;

        // Postings offset was tracked during writing
        offsets.postings_offset = 0; // Postings are in separate file, offset is relative to that file
        offsets.postings_len = self.postings_offset;

        // Write footer
        let footer_path = format!("{}/footer.bin", segment_dir);
        let mut footer_file = self.directory.create_file(&footer_path)?;

        let footer = SegmentFooter::new(self.doc_lengths.len() as u32, self.max_doc_id, offsets);
        footer.write(&mut footer_file)?;
        footer_file.flush()?;

        Ok(())
    }
}

/// BM25 segment reader.
///
/// Loads and queries a segment from disk.
pub struct SegmentReader {
    /// Directory for reading segment files (for future operations)
    #[allow(dead_code)]
    directory: Box<dyn Directory>,
    /// Segment ID (for identification)
    #[allow(dead_code)]
    segment_id: u64,
    /// Footer with offsets (for future operations)
    #[allow(dead_code)]
    footer: SegmentFooter,
    /// Term dictionary FST (term -> ordinal)
    /// Using FST directly for lookups is more memory-efficient than HashMap
    /// Reference: https://docs.rs/fst/latest/fst/struct.Map.html
    #[cfg(feature = "persistence")]
    term_dict_fst: Option<Map<Vec<u8>>>,
    /// Fallback HashMap for when FST is not available
    #[cfg(not(feature = "persistence"))]
    term_dict: HashMap<String, u64>,
    /// Term info store (ordinal -> TermInfo)
    term_infos: Vec<TermInfo>,
    /// Document lengths (memory-mapped if available)
    #[cfg(all(feature = "persistence", feature = "memmap"))]
    doc_lengths_mmap: Option<Arc<Mmap>>,
    /// Document lengths (fallback when memory mapping not available)
    doc_lengths: Vec<u32>,
    /// Postings file memory map (for efficient access)
    #[cfg(all(feature = "persistence", feature = "memmap"))]
    postings_mmap: Option<Arc<Mmap>>,
}

impl SegmentReader {
    /// Load a segment from disk.
    pub fn load(directory: Box<dyn Directory>, segment_id: u64) -> PersistenceResult<Self> {
        let segment_dir = format!("segments/segment_{}", segment_id);

        // Load footer
        let footer_path = format!("{}/footer.bin", segment_dir);
        let mut footer_file = directory.open_file(&footer_path)?;
        let footer = SegmentFooter::read(&mut footer_file)?;

        // Load term dictionary using FST
        // FST provides compact, memory-efficient term lookups
        // Reference: https://docs.rs/fst/latest/fst/struct.Map.html
        let term_dict_path = format!("{}/term_dict.fst", segment_dir);
        #[cfg(feature = "persistence")]
        let term_dict_fst: Option<Map<Vec<u8>>> = {
            if !directory.exists(&term_dict_path) {
                return Err(PersistenceError::NotFound(format!(
                    "FST file not found: {}",
                    term_dict_path
                )));
            }
            let mut term_dict_file = directory.open_file(&term_dict_path)?;
            let mut fst_buffer = Vec::new();
            term_dict_file.read_to_end(&mut fst_buffer)?;

            // Validate FST is not empty
            if fst_buffer.is_empty() {
                return Err(PersistenceError::Format {
                    message: "FST file is empty".to_string(),
                    expected: Some("non-empty FST data".to_string()),
                    actual: Some("0 bytes".to_string()),
                });
            }

            // Load FST from bytes
            // Use FST directly for lookups - more memory-efficient than HashMap
            // FST supports O(1) lookups and prefix searches
            Map::new(fst_buffer)
                .map_err(|e| PersistenceError::Format {
                    message: format!("FST load error: {}", e),
                    expected: None,
                    actual: None,
                })
                .ok()
        };
        #[cfg(not(feature = "persistence"))]
        let term_dict: HashMap<String, u64> = HashMap::new(); // Placeholder

        // Load term info store
        // Each TermInfo is 28 bytes: postings_offset (u64=8) + postings_len (u64=8) + doc_frequency (u32=4) + collection_frequency (u64=8)
        let term_info_path = format!("{}/term_info.bin", segment_dir);
        let mut term_info_file = directory.open_file(&term_info_path)?;
        let mut term_infos = Vec::new();
        let mut term_info_buffer = [0u8; 28]; // 8+8+4+8 bytes
        #[allow(clippy::while_let_loop)]
        loop {
            match term_info_file.read_exact(&mut term_info_buffer) {
                Ok(()) => {
                    let postings_offset =
                        u64::from_le_bytes(term_info_buffer[0..8].try_into().map_err(|_| {
                            PersistenceError::Format {
                                message: "Failed to extract postings_offset bytes".to_string(),
                                expected: Some("8-byte array".to_string()),
                                actual: None,
                            }
                        })?);
                    let postings_len =
                        u64::from_le_bytes(term_info_buffer[8..16].try_into().map_err(|_| {
                            PersistenceError::Format {
                                message: "Failed to extract postings_len bytes".to_string(),
                                expected: Some("8-byte array".to_string()),
                                actual: None,
                            }
                        })?);
                    let doc_frequency =
                        u32::from_le_bytes(term_info_buffer[16..20].try_into().map_err(|_| {
                            PersistenceError::Format {
                                message: "Failed to extract doc_frequency bytes".to_string(),
                                expected: Some("4-byte array".to_string()),
                                actual: None,
                            }
                        })?);
                    let collection_frequency =
                        u64::from_le_bytes(term_info_buffer[20..28].try_into().map_err(|_| {
                            PersistenceError::Format {
                                message: "Failed to extract collection_frequency bytes".to_string(),
                                expected: Some("8-byte array".to_string()),
                                actual: None,
                            }
                        })?);
                    term_infos.push(TermInfo {
                        postings_offset,
                        postings_len,
                        doc_frequency,
                        collection_frequency,
                    });
                }
                Err(_) => break,
            }
        }

        // Load document lengths (with memory mapping if enabled)
        let doc_lengths_path = format!("{}/doc_lengths.bin", segment_dir);
        #[cfg(all(feature = "persistence", feature = "memmap"))]
        let doc_lengths_mmap = {
            if let Some(file_path) = directory.file_path(&doc_lengths_path) {
                if let Ok(file) = std::fs::File::open(&file_path) {
                    if let Ok(mmap) = unsafe { MmapOptions::new().map(&file) } {
                        let _ = mmap.advise(Advice::Random); // Doc lengths are randomly accessed
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
        // Always load doc_lengths as fallback (even when mmap is available)
        // This ensures MemoryDirectory and other non-mmap scenarios work correctly
        let mut doc_lengths_file = directory.open_file(&doc_lengths_path)?;
        let mut doc_lengths_vec = Vec::new();
        let mut length_buffer = [0u8; 4];
        #[allow(clippy::while_let_loop)]
        loop {
            match doc_lengths_file.read_exact(&mut length_buffer) {
                Ok(()) => {
                    doc_lengths_vec.push(u32::from_le_bytes(length_buffer));
                }
                Err(_) => break,
            }
        }
        let doc_lengths = doc_lengths_vec;

        // Load postings file memory map
        #[cfg(all(feature = "persistence", feature = "memmap"))]
        let postings_mmap = {
            let postings_path = format!("segments/segment_{}/postings.bin", segment_id);
            if let Some(file_path) = directory.file_path(&postings_path) {
                if let Ok(file) = std::fs::File::open(&file_path) {
                    if let Ok(mmap) = unsafe { MmapOptions::new().map(&file) } {
                        let _ = mmap.advise(Advice::Sequential); // Postings are sequentially read
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
            footer,
            #[cfg(feature = "persistence")]
            term_dict_fst,
            #[cfg(not(feature = "persistence"))]
            term_dict,
            term_infos,
            #[cfg(all(feature = "persistence", feature = "memmap"))]
            doc_lengths_mmap,
            doc_lengths,
            #[cfg(all(feature = "persistence", feature = "memmap"))]
            postings_mmap,
        })
    }

    /// Get document length for a document ID.
    ///
    /// Uses memory mapping when available for efficient, zero-copy access.
    /// Optimized with `bytemuck` for zero-copy reads (no intermediate allocations).
    pub fn doc_length(&self, doc_id: u32) -> Option<u32> {
        #[cfg(all(feature = "persistence", feature = "memmap"))]
        {
            if let Some(ref mmap) = self.doc_lengths_mmap {
                let idx = doc_id as usize * 4;
                if idx + 4 <= mmap.len() {
                    // Zero-copy read using bytemuck (no intermediate byte array allocation)
                    // pod_read_unaligned is safe for memory-mapped data (page-aligned)
                    return Some(bytemuck::pod_read_unaligned::<u32>(&mmap[idx..idx + 4]));
                }
                return None;
            }
        }
        self.doc_lengths.get(doc_id as usize).copied()
    }

    /// Get postings list data for a term.
    ///
    /// Returns a slice of the postings list bytes, using memory mapping when available.
    #[cfg(all(feature = "persistence", feature = "memmap"))]
    pub fn get_postings_slice(&self, term_info: &TermInfo) -> Option<&[u8]> {
        if let Some(ref mmap) = self.postings_mmap {
            let start = term_info.postings_offset as usize;
            let end = start + term_info.postings_len as usize;
            if end <= mmap.len() {
                return Some(&mmap[start..end]);
            }
        }
        None
    }

    /// Get term info for a term.
    ///
    /// Uses FST directly for O(1) lookup without converting to HashMap.
    /// This is more memory-efficient, especially for large term dictionaries.
    pub fn term_info(&self, term: &str) -> Option<&TermInfo> {
        #[cfg(feature = "persistence")]
        {
            let ordinal = self.term_dict_fst.as_ref()?.get(term.as_bytes())?;
            self.term_infos.get(ordinal as usize)
        }
        #[cfg(not(feature = "persistence"))]
        {
            let ordinal = self.term_dict.get(term)?;
            self.term_infos.get(*ordinal as usize)
        }
    }

    /// Search for terms with a given prefix.
    ///
    /// Useful for query expansion and autocomplete.
    /// Returns all terms matching the prefix with their ordinals.
    ///
    /// Uses FST's efficient range search - O(prefix_length) to find start, then O(k) for k matches.
    /// Reference: https://docs.rs/fst/latest/fst/struct.Map.html#method.range
    #[cfg(feature = "persistence")]
    pub fn search_prefix(&self, prefix: &str) -> Vec<(String, u64)> {
        let Some(fst_map) = &self.term_dict_fst else {
            return Vec::new();
        };

        // Use range search for prefix matching
        // Find all terms >= prefix and < prefix with last byte incremented
        let prefix_bytes = prefix.as_bytes();
        let mut end_prefix = prefix_bytes.to_vec();
        if let Some(last) = end_prefix.last_mut() {
            *last = last.saturating_add(1);
        } else {
            // Empty prefix - search all terms
            let mut stream = fst_map.stream();
            let mut results = Vec::new();
            while let Some((term_bytes, ordinal)) = stream.next() {
                let term = String::from_utf8_lossy(term_bytes).to_string();
                results.push((term, ordinal));
            }
            return results;
        }

        let mut results = Vec::new();
        // Use range search for prefix matching
        // range() returns a StreamBuilder, into_stream() converts to Stream
        let mut stream = fst_map
            .range()
            .ge(prefix_bytes)
            .lt(&end_prefix)
            .into_stream();
        while let Some((term_bytes, ordinal)) = stream.next() {
            let term = String::from_utf8_lossy(term_bytes).to_string();
            results.push((term, ordinal));
        }
        results
    }

    /// Get the number of terms in the dictionary.
    pub fn term_count(&self) -> usize {
        #[cfg(feature = "persistence")]
        {
            self.term_dict_fst
                .as_ref()
                .map(|fst| fst.len())
                .unwrap_or(0)
        }
        #[cfg(not(feature = "persistence"))]
        {
            self.term_dict.len()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::directory::MemoryDirectory;
    use std::collections::HashMap;

    #[test]
    fn test_segment_write_read() {
        let dir = Box::new(MemoryDirectory::new());
        let segment_id = 1;

        // Create test index
        let mut postings = HashMap::new();
        let mut term_postings = HashMap::new();
        term_postings.insert(0u32, 2u32);
        term_postings.insert(1u32, 1u32);
        postings.insert("test".to_string(), term_postings);

        let mut doc_lengths = HashMap::new();
        doc_lengths.insert(0u32, 5u32);
        doc_lengths.insert(1u32, 3u32);

        let mut doc_frequencies = HashMap::new();
        doc_frequencies.insert("test".to_string(), 2u32);

        // Write segment
        let mut writer = SegmentWriter::new(dir.clone(), segment_id);
        writer
            .write_bm25_index(&postings, &doc_lengths, &doc_frequencies)
            .unwrap();
        writer.finalize().unwrap();

        // Read segment (using same directory)
        let reader = SegmentReader::load(dir, segment_id).unwrap();
        assert_eq!(reader.doc_length(0), Some(5));
        assert_eq!(reader.doc_length(1), Some(3));
        assert_eq!(reader.term_count(), 1);

        let term_info = reader.term_info("test");
        assert!(term_info.is_some());
        if let Some(info) = term_info {
            assert_eq!(info.doc_frequency, 2);
            assert_eq!(info.collection_frequency, 3);
        }

        // Test prefix search
        #[cfg(feature = "persistence")]
        {
            let results = reader.search_prefix("te");
            assert_eq!(results.len(), 1);
            assert_eq!(results[0].0, "test");
            assert_eq!(results[0].1, 0); // Ordinal of "test"
        }
        assert_eq!(reader.doc_length(0), Some(5));
        assert_eq!(reader.doc_length(1), Some(3));
        assert!(reader.term_info("test").is_some());
    }
}
