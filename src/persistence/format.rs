//! Binary format definitions for persistence.
//!
//! Defines the on-disk format for all persistent data structures:
//! - Segment footers
//! - WAL entries
//! - Checkpoint headers
//! - Format versioning

use crate::persistence::error::{PersistenceError, PersistenceResult};

/// Magic bytes for ordino-retrieve format identification.
pub const MAGIC_BYTES: [u8; 4] = *b"RANK";

/// Current format version.
pub const FORMAT_VERSION: u32 = 1;

/// Segment footer magic bytes.
pub const SEGMENT_MAGIC: [u8; 4] = *b"RANK";

/// WAL segment magic bytes.
pub const WAL_MAGIC: [u8; 4] = *b"WAL\0";

/// Checkpoint magic bytes.
pub const CHECKPOINT_MAGIC: [u8; 4] = *b"CHKP";

/// Transaction log magic bytes.
pub const TRANSACTION_LOG_MAGIC: [u8; 4] = *b"TXLO";

/// Segment footer (fixed size, 48 bytes).
///
/// Stored at the end of each segment directory.
/// Contains offsets to all segment data files.
/// Marked as `bytemuck::Pod` for zero-copy access from memory-mapped files.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "persistence", derive(bytemuck::Pod, bytemuck::Zeroable))]
pub struct SegmentFooter {
    /// Magic bytes: b"RANK"
    pub magic: [u8; 4],

    /// Format version
    pub format_version: u32,

    /// Offset to term dictionary (FST)
    pub term_dict_offset: u64,
    pub term_dict_len: u64,

    /// Offset to term info store
    pub term_info_offset: u64,
    pub term_info_len: u64,

    /// Offset to postings lists
    pub postings_offset: u64,
    pub postings_len: u64,

    /// Offset to document lengths
    pub doc_lengths_offset: u64,
    pub doc_lengths_len: u64,

    /// Offset to docID → userID mapping (FST, optional)
    pub docid_to_userid_offset: u64,
    pub docid_to_userid_len: u64,

    /// Offset to userID → docID mapping (FST, optional)
    pub userid_to_docid_offset: u64,
    pub userid_to_docid_len: u64,

    /// Offset to tombstones (deleted documents)
    pub tombstones_offset: u64,
    pub tombstones_len: u64,

    /// Number of documents in segment
    pub doc_count: u32,

    /// Maximum document ID in segment
    pub max_doc_id: u32,

    /// CRC32 checksum of all data (excluding footer)
    pub checksum: u32,

    /// Padding to 64-byte alignment
    pub padding: [u8; 4],
}

impl SegmentFooter {
    /// Size of segment footer in bytes.
    pub const SIZE: usize = std::mem::size_of::<Self>();

    /// Create a new segment footer.
    pub fn new(doc_count: u32, max_doc_id: u32, offsets: SegmentOffsets) -> Self {
        Self {
            magic: SEGMENT_MAGIC,
            format_version: FORMAT_VERSION,
            term_dict_offset: offsets.term_dict_offset,
            term_dict_len: offsets.term_dict_len,
            term_info_offset: offsets.term_info_offset,
            term_info_len: offsets.term_info_len,
            postings_offset: offsets.postings_offset,
            postings_len: offsets.postings_len,
            doc_lengths_offset: offsets.doc_lengths_offset,
            doc_lengths_len: offsets.doc_lengths_len,
            docid_to_userid_offset: offsets.docid_to_userid_offset,
            docid_to_userid_len: offsets.docid_to_userid_len,
            userid_to_docid_offset: offsets.userid_to_docid_offset,
            userid_to_docid_len: offsets.userid_to_docid_len,
            tombstones_offset: offsets.tombstones_offset,
            tombstones_len: offsets.tombstones_len,
            doc_count,
            max_doc_id,
            checksum: 0, // Computed after writing data
            padding: [0; 4],
        }
    }

    /// Validate footer magic and version.
    pub fn validate(&self) -> PersistenceResult<()> {
        if self.magic != SEGMENT_MAGIC {
            return Err(PersistenceError::Format {
                message: "Invalid segment magic bytes".to_string(),
                expected: Some(format!("{:?}", SEGMENT_MAGIC)),
                actual: Some(format!("{:?}", self.magic)),
            });
        }

        if self.format_version != FORMAT_VERSION {
            return Err(PersistenceError::Format {
                message: "Format version mismatch".to_string(),
                expected: Some(FORMAT_VERSION.to_string()),
                actual: Some(self.format_version.to_string()),
            });
        }

        Ok(())
    }

    /// Write footer to a writer (little-endian).
    pub fn write<W: std::io::Write>(&self, writer: &mut W) -> PersistenceResult<()> {
        use byteorder::{LittleEndian, WriteBytesExt};

        writer.write_all(&self.magic)?;
        writer.write_u32::<LittleEndian>(self.format_version)?;
        writer.write_u64::<LittleEndian>(self.term_dict_offset)?;
        writer.write_u64::<LittleEndian>(self.term_dict_len)?;
        writer.write_u64::<LittleEndian>(self.term_info_offset)?;
        writer.write_u64::<LittleEndian>(self.term_info_len)?;
        writer.write_u64::<LittleEndian>(self.postings_offset)?;
        writer.write_u64::<LittleEndian>(self.postings_len)?;
        writer.write_u64::<LittleEndian>(self.doc_lengths_offset)?;
        writer.write_u64::<LittleEndian>(self.doc_lengths_len)?;
        writer.write_u64::<LittleEndian>(self.docid_to_userid_offset)?;
        writer.write_u64::<LittleEndian>(self.docid_to_userid_len)?;
        writer.write_u64::<LittleEndian>(self.userid_to_docid_offset)?;
        writer.write_u64::<LittleEndian>(self.userid_to_docid_len)?;
        writer.write_u64::<LittleEndian>(self.tombstones_offset)?;
        writer.write_u64::<LittleEndian>(self.tombstones_len)?;
        writer.write_u32::<LittleEndian>(self.doc_count)?;
        writer.write_u32::<LittleEndian>(self.max_doc_id)?;
        writer.write_u32::<LittleEndian>(self.checksum)?;
        writer.write_all(&self.padding)?;

        Ok(())
    }

    /// Read footer from a reader (little-endian).
    pub fn read<R: std::io::Read>(reader: &mut R) -> PersistenceResult<Self> {
        use byteorder::{LittleEndian, ReadBytesExt};

        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        let format_version = reader.read_u32::<LittleEndian>()?;
        let term_dict_offset = reader.read_u64::<LittleEndian>()?;
        let term_dict_len = reader.read_u64::<LittleEndian>()?;
        let term_info_offset = reader.read_u64::<LittleEndian>()?;
        let term_info_len = reader.read_u64::<LittleEndian>()?;
        let postings_offset = reader.read_u64::<LittleEndian>()?;
        let postings_len = reader.read_u64::<LittleEndian>()?;
        let doc_lengths_offset = reader.read_u64::<LittleEndian>()?;
        let doc_lengths_len = reader.read_u64::<LittleEndian>()?;
        let docid_to_userid_offset = reader.read_u64::<LittleEndian>()?;
        let docid_to_userid_len = reader.read_u64::<LittleEndian>()?;
        let userid_to_docid_offset = reader.read_u64::<LittleEndian>()?;
        let userid_to_docid_len = reader.read_u64::<LittleEndian>()?;
        let tombstones_offset = reader.read_u64::<LittleEndian>()?;
        let tombstones_len = reader.read_u64::<LittleEndian>()?;
        let doc_count = reader.read_u32::<LittleEndian>()?;
        let max_doc_id = reader.read_u32::<LittleEndian>()?;
        let checksum = reader.read_u32::<LittleEndian>()?;
        let mut padding = [0u8; 4];
        reader.read_exact(&mut padding)?;

        let footer = Self {
            magic,
            format_version,
            term_dict_offset,
            term_dict_len,
            term_info_offset,
            term_info_len,
            postings_offset,
            postings_len,
            doc_lengths_offset,
            doc_lengths_len,
            docid_to_userid_offset,
            docid_to_userid_len,
            userid_to_docid_offset,
            userid_to_docid_len,
            tombstones_offset,
            tombstones_len,
            doc_count,
            max_doc_id,
            checksum,
            padding,
        };

        footer.validate()?;
        Ok(footer)
    }
}

/// Segment file offsets (for constructing footer).
#[derive(Debug, Clone, Default)]
pub struct SegmentOffsets {
    pub term_dict_offset: u64,
    pub term_dict_len: u64,
    pub term_info_offset: u64,
    pub term_info_len: u64,
    pub postings_offset: u64,
    pub postings_len: u64,
    pub doc_lengths_offset: u64,
    pub doc_lengths_len: u64,
    pub docid_to_userid_offset: u64,
    pub docid_to_userid_len: u64,
    pub userid_to_docid_offset: u64,
    pub userid_to_docid_len: u64,
    pub tombstones_offset: u64,
    pub tombstones_len: u64,
}

/// Format version information.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatVersion {
    pub major: u16,
    pub minor: u16,
    pub patch: u16,
}

impl FormatVersion {
    /// Check if two versions are compatible (same major version).
    pub fn is_compatible(&self, other: &FormatVersion) -> bool {
        self.major == other.major
    }
}

impl From<u32> for FormatVersion {
    fn from(v: u32) -> Self {
        Self {
            major: ((v >> 16) & 0xFFFF) as u16,
            minor: ((v >> 8) & 0xFF) as u16,
            patch: (v & 0xFF) as u16,
        }
    }
}

impl From<FormatVersion> for u32 {
    fn from(v: FormatVersion) -> Self {
        ((v.major as u32) << 16) | ((v.minor as u32) << 8) | (v.patch as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_segment_footer_roundtrip() {
        let offsets = SegmentOffsets {
            term_dict_offset: 0,
            term_dict_len: 100,
            term_info_offset: 100,
            term_info_len: 200,
            postings_offset: 300,
            postings_len: 1000,
            doc_lengths_offset: 1300,
            doc_lengths_len: 400,
            docid_to_userid_offset: 0,
            docid_to_userid_len: 0,
            userid_to_docid_offset: 0,
            userid_to_docid_len: 0,
            tombstones_offset: 1700,
            tombstones_len: 100,
        };

        let mut footer = SegmentFooter::new(1000, 999, offsets);
        footer.checksum = 12345;

        let mut buffer = Vec::new();
        footer.write(&mut buffer).unwrap();

        assert_eq!(buffer.len(), SegmentFooter::SIZE);

        let mut reader = std::io::Cursor::new(&buffer);
        let read_footer = SegmentFooter::read(&mut reader).unwrap();

        assert_eq!(read_footer.magic, footer.magic);
        assert_eq!(read_footer.format_version, footer.format_version);
        assert_eq!(read_footer.doc_count, footer.doc_count);
        assert_eq!(read_footer.checksum, footer.checksum);
    }

    #[test]
    fn test_format_version() {
        let v1 = FormatVersion {
            major: 1,
            minor: 0,
            patch: 0,
        };
        let v2 = FormatVersion {
            major: 1,
            minor: 1,
            patch: 0,
        };
        let v3 = FormatVersion {
            major: 2,
            minor: 0,
            patch: 0,
        };

        assert!(v1.is_compatible(&v2));
        assert!(!v1.is_compatible(&v3));

        let v1_u32: u32 = v1.into();
        let v1_back = FormatVersion::from(v1_u32);
        assert_eq!(v1, v1_back);
    }
}
