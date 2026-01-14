//! Write-ahead log (WAL) for incremental updates.
//!
//! The WAL records all mutations before they are applied to segments,
//! enabling crash recovery and incremental updates.
//!
//! See `docs/PERSISTENCE_DESIGN.md` for WAL format and recovery procedures.

use crate::persistence::directory::Directory;
use crate::persistence::error::{PersistenceError, PersistenceResult};
use crate::persistence::format::WAL_MAGIC;
use std::io::{Read, Write};
use std::sync::Arc;

#[cfg(feature = "persistence")]
use postcard;

/// WAL entry types.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum WalEntry {
    /// A new segment has been added to the index.
    AddSegment {
        entry_id: u64,
        segment_id: u64,
        doc_count: u32,
    },

    /// A merge has started.
    StartMerge {
        entry_id: u64,
        transaction_id: u64,
        segment_ids: Vec<u64>,
    },

    /// A merge has been cancelled.
    CancelMerge {
        entry_id: u64,
        transaction_id: u64,
        segment_ids: Vec<u64>,
    },

    /// A merge has completed.
    EndMerge {
        entry_id: u64,
        transaction_id: u64,
        new_segment_id: u64,
        old_segment_ids: Vec<u64>,
        /// Remapped deletes that occurred during merge: (segment_id, doc_id)
        remapped_deletes: Vec<(u64, u32)>,
    },

    /// Documents have been deleted.
    DeleteDocuments {
        entry_id: u64,
        /// Deletes: (segment_id, doc_id)
        deletes: Vec<(u64, u32)>,
    },

    /// A checkpoint has been created.
    Checkpoint {
        entry_id: u64,
        checkpoint_path: String,
        last_entry_id: u64,
    },
}

/// WAL segment header.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct WalSegmentHeader {
    /// Magic bytes: b"WAL\0"
    pub magic: [u8; 4],
    /// Format version
    pub version: u32,
    /// First entry ID in this segment
    pub start_entry_id: u64,
    /// Unique segment identifier
    pub segment_id: u64,
}

impl WalSegmentHeader {
    /// Size of WAL segment header in bytes.
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Write header to a writer (little-endian).
    pub fn write<W: Write>(&self, writer: &mut W) -> PersistenceResult<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        writer.write_all(&self.magic)?;
        writer.write_u32::<LittleEndian>(self.version)?;
        writer.write_u64::<LittleEndian>(self.start_entry_id)?;
        writer.write_u64::<LittleEndian>(self.segment_id)?;

        Ok(())
    }

    /// Read header from a reader (little-endian).
    pub fn read<R: Read>(reader: &mut R) -> PersistenceResult<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;

        if magic != WAL_MAGIC {
            return Err(PersistenceError::Format(format!(
                "Invalid WAL magic bytes (expected: {:?}, actual: {:?})",
                WAL_MAGIC, magic
            )));
        }

        let version = reader.read_u32::<LittleEndian>()?;
        let start_entry_id = reader.read_u64::<LittleEndian>()?;
        let segment_id = reader.read_u64::<LittleEndian>()?;

        Ok(Self {
            magic,
            version,
            start_entry_id,
            segment_id,
        })
    }
}

/// WAL entry on-disk format.
pub struct WalEntryOnDisk {
    /// Total length (including this header)
    pub length: u32,
    /// Entry type (0=AddSegment, 1=StartMerge, etc.)
    pub entry_type: u8,
    /// CRC32 checksum of payload
    pub checksum: u32,
    /// Serialized entry (postcard format - format-stable, smaller size)
    pub payload: Vec<u8>,
}

impl WalEntryOnDisk {
    /// Encode a WAL entry to on-disk format.
    ///
    /// Uses `postcard` for format-stable, space-efficient serialization.
    /// Postcard provides:
    /// - Documented wire format (long-term compatibility)
    /// - Varint encoding (10-15% size reduction)
    /// - Built-in CRC32 support (via use-crc feature)
    pub fn encode(entry: &WalEntry, _entry_id: u64) -> PersistenceResult<Vec<u8>> {
        use byteorder::{LittleEndian, WriteBytesExt};

        // Serialize entry using postcard (format-stable, smaller size)
        let payload = postcard::to_allocvec(entry).map_err(|e| {
            PersistenceError::Serialization(format!("Postcard serialization error: {}", e))
        })?;

        // Compute checksum (postcard's use-crc feature could be used, but we maintain explicit checksum for compatibility)
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&payload);
        let checksum = hasher.finalize();

        // Determine entry type
        let entry_type = match entry {
            WalEntry::AddSegment { .. } => 0,
            WalEntry::StartMerge { .. } => 1,
            WalEntry::CancelMerge { .. } => 2,
            WalEntry::EndMerge { .. } => 3,
            WalEntry::DeleteDocuments { .. } => 4,
            WalEntry::Checkpoint { .. } => 5,
        };

        // Write entry
        let mut encoded = Vec::new();
        let length = (4 + 1 + 4 + payload.len()) as u32; // length + type + checksum + payload
        encoded.write_u32::<LittleEndian>(length)?;
        encoded.write_u8(entry_type)?;
        encoded.write_u32::<LittleEndian>(checksum)?;
        encoded.extend_from_slice(&payload);

        Ok(encoded)
    }

    /// Decode a WAL entry from on-disk format.
    ///
    /// Supports both postcard (new format) and bincode (legacy format) for backward compatibility.
    /// Attempts postcard first, falls back to bincode if postcard fails (legacy data).
    pub fn decode<R: Read>(reader: &mut R) -> PersistenceResult<(WalEntry, u32)> {
        use byteorder::{LittleEndian, ReadBytesExt};

        let length = reader.read_u32::<LittleEndian>()?;
        let _entry_type = reader.read_u8()?;
        let checksum = reader.read_u32::<LittleEndian>()?;

        let payload_len = length as usize - 9; // length (4) + type (1) + checksum (4)
        let mut payload = vec![0u8; payload_len];
        reader.read_exact(&mut payload)?;

        // Verify checksum
        let mut hasher = crc32fast::Hasher::new();
        hasher.update(&payload);
        let computed_checksum = hasher.finalize();

        if computed_checksum != checksum {
            return Err(PersistenceError::ChecksumMismatch {
                expected: checksum,
                actual: computed_checksum,
            });
        }

        // Deserialize entry - try postcard first (new format), fall back to bincode (legacy)
        let entry: WalEntry = match postcard::from_bytes(&payload) {
            Ok(e) => e, // Postcard format (new)
            Err(_) => {
                // Fall back to bincode for legacy data
                #[cfg(feature = "persistence-bincode")]
                {
                    bincode::deserialize(&payload).map_err(|e| {
                        PersistenceError::Deserialization(format!(
                            "Failed to deserialize WAL entry (tried postcard and bincode): {}",
                            e
                        ))
                    })?
                }
                #[cfg(not(feature = "persistence-bincode"))]
                {
                    return Err(PersistenceError::Deserialization(
                        "WAL entry is in legacy bincode format, but persistence-bincode feature is not enabled".to_string()
                    ));
                }
            }
        };

        Ok((entry, length))
    }
}

/// WAL writer for appending entries.
pub struct WalWriter {
    directory: Arc<dyn Directory>,
    current_segment_id: u64,
    current_entry_id: u64,
    current_offset: u64,
    segment_size_limit: u64, // Maximum size per segment (e.g., 10MB)
}

impl WalWriter {
    /// Create a new WAL writer.
    pub fn new(directory: impl Into<Arc<dyn Directory>>) -> Self {
        Self {
            directory: directory.into(),
            current_segment_id: 1,
            current_entry_id: 1,
            current_offset: 0,
            segment_size_limit: 10 * 1024 * 1024, // 10MB default
        }
    }

    /// Append an entry to the WAL.
    pub fn append(&mut self, entry: WalEntry) -> PersistenceResult<u64> {
        let entry_id = self.current_entry_id;

        // Check if we need a new segment
        if self.current_offset > self.segment_size_limit {
            self.rotate_segment()?;
        }

        // Ensure WAL directory exists
        self.directory.create_dir_all("wal")?;

        // Encode entry
        let encoded = WalEntryOnDisk::encode(&entry, entry_id)?;

        // Write to current segment
        let wal_path = format!("wal/wal_{}.log", self.current_segment_id);

        // Write header if first write, then append entry
        if self.current_offset == 0 {
            // First write to segment - write header
            let mut file = self.directory.create_file(&wal_path)?;
            let header = WalSegmentHeader {
                magic: WAL_MAGIC,
                version: 1,
                start_entry_id: entry_id,
                segment_id: self.current_segment_id,
            };
            header.write(&mut file)?;
            file.flush()?;
            drop(file);
            self.current_offset = WalSegmentHeader::SIZE as u64;
        }

        // Append entry to file using append_file
        let mut file = self.directory.append_file(&wal_path)?;
        file.write_all(&encoded)?;
        file.flush()?;
        drop(file); // Ensure file is closed and flushed

        self.current_offset += encoded.len() as u64;
        self.current_entry_id += 1;

        Ok(entry_id)
    }

    /// Rotate to a new WAL segment.
    fn rotate_segment(&mut self) -> PersistenceResult<()> {
        self.current_segment_id += 1;
        self.current_offset = 0;
        Ok(())
    }
}

/// WAL reader for replaying entries.
pub struct WalReader {
    directory: Arc<dyn Directory>,
}

impl WalReader {
    /// Create a new WAL reader.
    pub fn new(directory: impl Into<Arc<dyn Directory>>) -> Self {
        Self {
            directory: directory.into(),
        }
    }

    /// Replay all WAL entries from disk.
    ///
    /// Returns entries in order and stops at first corruption.
    pub fn replay(&self) -> PersistenceResult<Vec<WalEntry>> {
        let mut entries = Vec::new();

        // List all WAL segments
        // For MemoryDirectory, directories don't "exist" as separate entities,
        // so we try to list the directory and check if we get any files
        let wal_dir = "wal";
        let mut wal_files = match self.directory.list_dir(wal_dir) {
            Ok(files) => files,
            Err(_) => {
                // Directory doesn't exist or is empty - no WAL yet
                return Ok(entries);
            }
        };

        if wal_files.is_empty() {
            return Ok(entries); // No WAL files
        }

        wal_files.sort(); // Sort by segment ID

        for wal_file in wal_files {
            if !wal_file.ends_with(".log") {
                continue;
            }

            let wal_path = format!("{}/{}", wal_dir, wal_file);
            let mut file = self.directory.open_file(&wal_path)?;

            // Read header
            let _header = WalSegmentHeader::read(&mut file)?;

            // Read entries until EOF or corruption
            loop {
                match WalEntryOnDisk::decode(&mut file) {
                    Ok((entry, _length)) => {
                        entries.push(entry);
                    }
                    Err(PersistenceError::Io(e)) => {
                        // EOF or read error - stop at this segment
                        if e.kind() == std::io::ErrorKind::UnexpectedEof {
                            // Normal EOF - we've read all entries
                            break;
                        } else {
                            // Other I/O error - propagate
                            return Err(PersistenceError::Io(e));
                        }
                    }
                    Err(e) => {
                        // Corruption detected - stop replay
                        return Err(e);
                    }
                }
            }
        }

        Ok(entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::directory::MemoryDirectory;

    #[test]
    fn test_wal_entry_encode_decode() {
        let entry = WalEntry::AddSegment {
            entry_id: 1,
            segment_id: 100,
            doc_count: 1000,
        };

        let encoded = WalEntryOnDisk::encode(&entry, 1).unwrap();
        let mut reader = std::io::Cursor::new(&encoded);
        let (decoded, _) = WalEntryOnDisk::decode(&mut reader).unwrap();

        match (entry, decoded) {
            (
                WalEntry::AddSegment {
                    entry_id: e1,
                    segment_id: s1,
                    doc_count: d1,
                },
                WalEntry::AddSegment {
                    entry_id: e2,
                    segment_id: s2,
                    doc_count: d2,
                },
            ) => {
                assert_eq!(e1, e2);
                assert_eq!(s1, s2);
                assert_eq!(d1, d2);
            }
            _ => panic!("Entry type mismatch"),
        }
    }

    #[test]
    fn test_wal_write_read() {
        use std::sync::Arc;
        let mem_dir = MemoryDirectory::new();
        let dir: Arc<dyn Directory> = Arc::new(mem_dir) as Arc<dyn Directory>;
        dir.create_dir_all("wal").unwrap();

        let mut writer = WalWriter::new(dir.clone());

        let entry1 = WalEntry::AddSegment {
            entry_id: 1,
            segment_id: 1,
            doc_count: 100,
        };

        let entry2 = WalEntry::AddSegment {
            entry_id: 2,
            segment_id: 2,
            doc_count: 200,
        };

        writer.append(entry1).unwrap();
        writer.append(entry2).unwrap();

        // Debug: Check if WAL file exists
        let wal_path = "wal/wal_1.log";
        assert!(dir.exists(wal_path), "WAL file should exist");

        let reader = WalReader::new(dir);
        let entries = reader.replay().unwrap();
        assert_eq!(entries.len(), 2);
    }
}
