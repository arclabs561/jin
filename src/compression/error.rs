//! Error types for compression operations.

use thiserror::Error;

/// Errors that can occur during compression operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum CompressionError {
    /// Invalid input (e.g., unsorted IDs, empty universe).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Compression operation failed.
    #[error("compression failed: {0}")]
    CompressionFailed(String),

    /// Decompression operation failed.
    #[error("decompression failed: {0}")]
    DecompressionFailed(String),

    /// ANS encoding/decoding error.
    #[error("ANS encoding error: {0}")]
    AnsError(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(String),
}

impl From<std::io::Error> for CompressionError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
