//! Error types for persistence operations.

use std::fmt;

/// Errors that can occur during persistence operations.
#[derive(Debug)]
pub enum PersistenceError {
    /// I/O error (file operations, disk I/O)
    Io(std::io::Error),

    /// Format error (invalid magic bytes, version mismatch, corruption)
    Format {
        message: String,
        expected: Option<String>,
        actual: Option<String>,
    },

    /// Serialization error (bincode, serde)
    Serialization(String),

    /// Deserialization error
    Deserialization(String),

    /// Checksum mismatch (data corruption detected)
    ChecksumMismatch { expected: u32, actual: u32 },

    /// Lock acquisition failed (concurrent access conflict)
    LockFailed { resource: String, reason: String },

    /// Invalid state (e.g., operation not allowed in current state)
    InvalidState(String),

    /// Resource not found (file, segment, etc.)
    NotFound(String),

    /// Invalid configuration
    InvalidConfig(String),

    /// Operation not supported
    NotSupported(String),
}

impl fmt::Display for PersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O error: {}", e),
            Self::Format {
                message,
                expected,
                actual,
            } => {
                write!(f, "Format error: {}", message)?;
                if let Some(e) = expected {
                    write!(f, " (expected: {})", e)?;
                }
                if let Some(a) = actual {
                    write!(f, " (actual: {})", a)?;
                }
                Ok(())
            }
            Self::Serialization(msg) => write!(f, "Serialization error: {}", msg),
            Self::Deserialization(msg) => write!(f, "Deserialization error: {}", msg),
            Self::ChecksumMismatch { expected, actual } => {
                write!(
                    f,
                    "Checksum mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            Self::LockFailed { resource, reason } => {
                write!(f, "Failed to acquire lock on {}: {}", resource, reason)
            }
            Self::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
            Self::NotFound(resource) => write!(f, "Resource not found: {}", resource),
            Self::InvalidConfig(msg) => write!(f, "Invalid configuration: {}", msg),
            Self::NotSupported(msg) => write!(f, "Operation not supported: {}", msg),
        }
    }
}

impl std::error::Error for PersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for PersistenceError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

#[cfg(feature = "persistence")]
impl From<postcard::Error> for PersistenceError {
    fn from(e: postcard::Error) -> Self {
        Self::Serialization(format!("Postcard error: {}", e))
    }
}

#[cfg(all(feature = "persistence", feature = "persistence-bincode"))]
impl From<bincode::Error> for PersistenceError {
    fn from(e: bincode::Error) -> Self {
        Self::Serialization(format!("Bincode error (legacy): {}", e))
    }
}

/// Result type for persistence operations.
pub type PersistenceResult<T> = Result<T, PersistenceError>;
