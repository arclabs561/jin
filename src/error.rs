//! Error types for vicinity.

use thiserror::Error;

/// Errors that can occur during indexing/search operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum RetrieveError {
    /// Empty query provided.
    #[error("query is empty")]
    EmptyQuery,

    /// Empty index (no documents indexed).
    #[error("index is empty")]
    EmptyIndex,

    /// Invalid parameter value.
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Dimension mismatch between query and documents.
    #[error("dimension mismatch: query has {query_dim} dimensions, document has {doc_dim}")]
    DimensionMismatch { query_dim: usize, doc_dim: usize },

    /// Invalid sparse vector (empty or malformed).
    #[error("invalid sparse vector: {0}")]
    InvalidSparseVector(String),

    /// Other error (for extensibility).
    #[error("{0}")]
    Other(String),
}

/// Result type alias for vicinity operations.
pub type Result<T> = std::result::Result<T, RetrieveError>;
