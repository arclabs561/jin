//! ANS (Asymmetric Numeral Systems) encoder/decoder wrapper.
//!
//! This module provides a simplified interface to constriction's ANS coder
//! for use in ROC compression.

use crate::compression::error::CompressionError;
#[cfg(feature = "id-compression")]
use constriction::stream::stack::DefaultAnsCoder;

/// ANS coder wrapper for entropy coding.
///
/// Uses constriction's `DefaultAnsCoder` which provides stack-based ANS
/// encoding/decoding suitable for bits-back coding.
#[allow(dead_code)] // Future ID compression feature
pub struct AnsCoder {
    #[cfg(feature = "id-compression")]
    coder: DefaultAnsCoder,
}

impl AnsCoder {
    /// Create a new ANS coder.
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "id-compression")]
            coder: DefaultAnsCoder::new(),
        }
    }

    /// Create ANS coder from compressed data (for decoding).
    pub fn from_compressed(compressed: Vec<u32>) -> Result<Self, CompressionError> {
        Ok(Self {
            #[cfg(feature = "id-compression")]
            coder: DefaultAnsCoder::from_compressed(compressed).map_err(|e| {
                CompressionError::AnsError(format!("Failed to create decoder: {:?}", e))
            })?,
        })
    }

    /// Encode a symbol with uniform probability.
    ///
    /// # Arguments
    ///
    /// * `symbol` - Symbol to encode (must be in range [0, alphabet_size))
    /// * `alphabet_size` - Size of alphabet (for uniform probability: 1/alphabet_size)
    ///
    /// # Note
    ///
    /// ANS encodes in reverse order (stack-like), so symbols should be encoded
    /// in reverse order of how they will be decoded.
    pub fn encode_uniform(
        &mut self,
        symbol: u32,
        alphabet_size: u32,
    ) -> Result<(), CompressionError> {
        #[cfg(feature = "id-compression")]
        {
            // Create uniform probability model
            // For now, use a simple uniform model
            // TODO: Use constriction's uniform model properly
            if symbol >= alphabet_size {
                return Err(CompressionError::InvalidInput(format!(
                    "Symbol {} out of range [0, {})",
                    symbol, alphabet_size
                )));
            }

            // For uniform distribution, we need to create a quantized model
            // This is a simplified version - full implementation would use proper quantizer
            // For now, return error indicating this needs proper implementation
            Err(CompressionError::AnsError(
                "Uniform encoding not yet fully implemented - needs quantizer setup".to_string(),
            ))
        }

        #[cfg(not(feature = "id-compression"))]
        {
            let _ = (symbol, alphabet_size);
            Err(CompressionError::AnsError(
                "id-compression feature not enabled".to_string(),
            ))
        }
    }

    /// Decode a symbol with uniform probability.
    ///
    /// # Arguments
    ///
    /// * `alphabet_size` - Size of alphabet
    ///
    /// # Returns
    ///
    /// Decoded symbol in range [0, alphabet_size)
    pub fn decode_uniform(&mut self, _alphabet_size: u32) -> Result<u32, CompressionError> {
        #[cfg(feature = "id-compression")]
        {
            // Similar to encode - needs proper quantizer setup
            Err(CompressionError::AnsError(
                "Uniform decoding not yet fully implemented - needs quantizer setup".to_string(),
            ))
        }

        #[cfg(not(feature = "id-compression"))]
        {
            let _ = alphabet_size;
            Err(CompressionError::AnsError(
                "id-compression feature not enabled".to_string(),
            ))
        }
    }

    /// Get compressed data as bytes.
    ///
    /// # Returns
    ///
    /// Compressed representation as byte vector.
    pub fn into_bytes(self) -> Result<Vec<u8>, CompressionError> {
        #[cfg(feature = "id-compression")]
        {
            let compressed = self.coder.into_compressed().map_err(|e| {
                CompressionError::AnsError(format!("Failed to get compressed: {:?}", e))
            })?;

            // Convert Vec<u32> to Vec<u8> (little-endian)
            let mut bytes = Vec::with_capacity(compressed.len() * 4);
            for word in compressed {
                bytes.extend_from_slice(&word.to_le_bytes());
            }
            Ok(bytes)
        }

        #[cfg(not(feature = "id-compression"))]
        {
            Err(CompressionError::AnsError(
                "id-compression feature not enabled".to_string(),
            ))
        }
    }

    /// Create from byte vector (for decoding).
    pub fn from_bytes(bytes: Vec<u8>) -> Result<Self, CompressionError> {
        #[cfg(feature = "id-compression")]
        {
            // Convert Vec<u8> to Vec<u32> (little-endian)
            if bytes.len() % 4 != 0 {
                return Err(CompressionError::InvalidInput(
                    "Byte vector length must be multiple of 4".to_string(),
                ));
            }

            let mut words = Vec::with_capacity(bytes.len() / 4);
            for chunk in bytes.chunks_exact(4) {
                let word = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                words.push(word);
            }

            Self::from_compressed(words)
        }

        #[cfg(not(feature = "id-compression"))]
        {
            let _ = bytes;
            Err(CompressionError::AnsError(
                "id-compression feature not enabled".to_string(),
            ))
        }
    }
}

impl Default for AnsCoder {
    fn default() -> Self {
        Self::new()
    }
}
