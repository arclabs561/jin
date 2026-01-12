//! Error types for compression operations.

use std::fmt;

/// Errors that can occur during compression operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CompressionError {
    /// Invalid input (e.g., unsorted IDs, empty universe).
    InvalidInput(String),

    /// Compression operation failed.
    CompressionFailed(String),

    /// Decompression operation failed.
    DecompressionFailed(String),

    /// ANS encoding/decoding error.
    AnsError(String),

    /// I/O error.
    Io(String),
}

impl fmt::Display for CompressionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressionError::InvalidInput(msg) => {
                write!(f, "Invalid input: {}", msg)
            }
            CompressionError::CompressionFailed(msg) => {
                write!(f, "Compression failed: {}", msg)
            }
            CompressionError::DecompressionFailed(msg) => {
                write!(f, "Decompression failed: {}", msg)
            }
            CompressionError::AnsError(msg) => {
                write!(f, "ANS encoding error: {}", msg)
            }
            CompressionError::Io(msg) => {
                write!(f, "I/O error: {}", msg)
            }
        }
    }
}

impl std::error::Error for CompressionError {}

impl From<std::io::Error> for CompressionError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.to_string())
    }
}
