//! Error types for persistence operations.

use thiserror::Error;

/// Errors that can occur during persistence operations.
#[derive(Debug, Error)]
pub enum PersistenceError {
    /// I/O error (file operations, disk I/O)
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Format error (invalid magic bytes, version mismatch, corruption)
    #[error("format error: {0}")]
    Format(String),

    /// Serialization error (bincode, serde)
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Deserialization error
    #[error("deserialization error: {0}")]
    Deserialization(String),

    /// Checksum mismatch (data corruption detected)
    #[error("checksum mismatch: expected {expected}, got {actual}")]
    ChecksumMismatch { expected: u32, actual: u32 },

    /// Lock acquisition failed (concurrent access conflict)
    #[error("failed to acquire lock on {resource}: {reason}")]
    LockFailed { resource: String, reason: String },

    /// Invalid state (e.g., operation not allowed in current state)
    #[error("invalid state: {0}")]
    InvalidState(String),

    /// Resource not found (file, segment, etc.)
    #[error("resource not found: {0}")]
    NotFound(String),

    /// Invalid configuration
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),

    /// Operation not supported
    #[error("operation not supported: {0}")]
    NotSupported(String),
}

/// Helper to format expected/actual values for error messages.
fn format_expected_actual(expected: &Option<String>, actual: &Option<String>) -> String {
    match (expected, actual) {
        (Some(e), Some(a)) => format!(" (expected: {}, actual: {})", e, a),
        (Some(e), None) => format!(" (expected: {})", e),
        (None, Some(a)) => format!(" (actual: {})", a),
        (None, None) => String::new(),
    }
}

#[cfg(feature = "persistence")]
impl From<postcard::Error> for PersistenceError {
    fn from(e: postcard::Error) -> Self {
        Self::Serialization(format!("postcard error: {}", e))
    }
}

#[cfg(all(feature = "persistence", feature = "persistence-bincode"))]
impl From<bincode::Error> for PersistenceError {
    fn from(e: bincode::Error) -> Self {
        Self::Serialization(format!("bincode error (legacy): {}", e))
    }
}

/// Result type for persistence operations.
pub type PersistenceResult<T> = Result<T, PersistenceError>;
