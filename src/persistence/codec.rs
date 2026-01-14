//! Compression codecs for persistence.
//!
//! Provides efficient encoding/decoding for:
//! - Varint encoding (for small integers)
//! - Delta encoding (for sorted sequences)
//! - Bitpacking (for SIMD-accelerated compression)
//!
//! See `docs/PERSISTENCE_DESIGN.md` for format details.

use crate::persistence::error::{PersistenceError, PersistenceResult};

/// Block size for bitpacking (matches Tantivy's 128-doc blocks).
pub const BLOCK_SIZE: usize = 128;

pub mod varint {
    //! Varint encoding for variable-length integers.

    use crate::persistence::error::{PersistenceError, PersistenceResult};

    /// Encode a u64 value as varint.
    ///
    /// Format: Each byte contains 7 bits of data and 1 continuation bit.
    /// The MSB (most significant bit) is set to 1 if more bytes follow, 0 for the last byte.
    pub fn encode(value: u64) -> Vec<u8> {
        let mut result = Vec::new();
        let mut v = value;
        loop {
            let byte = (v & 0x7F) as u8;
            v >>= 7;
            if v == 0 {
                result.push(byte | 0x80); // Last byte has MSB set
                break;
            } else {
                result.push(byte);
            }
        }
        result
    }

    /// Decode a varint from a byte slice.
    ///
    /// Returns the decoded value and the number of bytes consumed.
    pub fn decode(data: &[u8]) -> PersistenceResult<(u64, usize)> {
        let mut result = 0u64;
        let mut shift = 0;
        let mut bytes_read = 0;

        for &byte in data {
            bytes_read += 1;
            result |= ((byte & 0x7F) as u64) << shift;
            if (byte & 0x80) != 0 {
                return Ok((result, bytes_read));
            }
            shift += 7;
            if shift >= 64 {
                return Err(PersistenceError::Format(
                    "Varint overflow (64 bits)".to_string(),
                ));
            }
        }

        Err(PersistenceError::Format("Incomplete varint".to_string()))
    }

    /// Encode multiple values as varints.
    pub fn encode_many(values: &[u64]) -> Vec<u8> {
        let mut result = Vec::new();
        for &value in values {
            result.extend_from_slice(&encode(value));
        }
        result
    }

    /// Decode multiple varints from a byte slice.
    ///
    /// Returns the decoded values and the number of bytes consumed.
    pub fn decode_many(data: &[u8], count: usize) -> PersistenceResult<(Vec<u64>, usize)> {
        let mut values = Vec::with_capacity(count);
        let mut offset = 0;

        for _ in 0..count {
            let (value, bytes_read) = decode(&data[offset..])?;
            values.push(value);
            offset += bytes_read;
        }

        Ok((values, offset))
    }
}

pub mod delta {
    //! Delta encoding for sorted sequences.

    /// Encode a sorted sequence using delta encoding.
    ///
    /// First value is stored as-is, subsequent values are stored as differences.
    /// Example: [5, 7, 9, 12] -> [5, 2, 2, 3]
    pub fn encode(sorted_values: &[u32]) -> Vec<u32> {
        if sorted_values.is_empty() {
            return Vec::new();
        }

        let mut deltas = Vec::with_capacity(sorted_values.len());
        deltas.push(sorted_values[0]); // First value stored as-is

        for i in 1..sorted_values.len() {
            deltas.push(sorted_values[i] - sorted_values[i - 1]);
        }

        deltas
    }

    /// Decode a delta-encoded sequence.
    pub fn decode(deltas: &[u32]) -> Vec<u32> {
        if deltas.is_empty() {
            return Vec::new();
        }

        let mut values = Vec::with_capacity(deltas.len());
        values.push(deltas[0]);

        for i in 1..deltas.len() {
            values.push(values[i - 1] + deltas[i]);
        }

        values
    }
}

pub mod bitpack {
    //! Bitpacking for fixed-width integer compression.

    use super::*;

    /// Calculate the minimum number of bits needed to represent a value.
    pub fn bit_width(value: u32) -> u8 {
        if value == 0 {
            return 1;
        }
        (32 - value.leading_zeros()) as u8
    }

    /// Calculate the minimum number of bits needed for all values in a slice.
    pub fn bit_width_many(values: &[u32]) -> u8 {
        let max = values.iter().max().copied().unwrap_or(0);
        bit_width(max)
    }

    /// Pack values into bytes using the specified bit width.
    ///
    /// Values are packed in little-endian order, with the first value starting at bit 0.
    /// Returns the packed bytes.
    pub fn pack(values: &[u32], bit_width: u8) -> Vec<u8> {
        if values.is_empty() || bit_width == 0 {
            return Vec::new();
        }

        let bits_needed = values.len() * bit_width as usize;
        let bytes_needed = bits_needed.div_ceil(8);
        let mut result = vec![0u8; bytes_needed];

        let mut bit_offset = 0usize;
        for &value in values {
            let mut remaining = value;
            let mut bits_to_write = bit_width as usize;

            while bits_to_write > 0 {
                let byte_idx = bit_offset / 8;
                let bit_in_byte = bit_offset % 8;
                let bits_available = 8 - bit_in_byte;
                let bits_to_take = bits_to_write.min(bits_available);

                let mask = (1u32 << bits_to_take) - 1;
                let bits = remaining & mask;
                remaining >>= bits_to_take;

                result[byte_idx] |= (bits << bit_in_byte) as u8;

                bit_offset += bits_to_take;
                bits_to_write -= bits_to_take;
            }
        }

        result
    }

    /// Unpack values from bytes using the specified bit width.
    ///
    /// Returns the unpacked values.
    pub fn unpack(data: &[u8], count: usize, bit_width: u8) -> PersistenceResult<Vec<u32>> {
        if count == 0 || bit_width == 0 {
            return Ok(Vec::new());
        }

        let mut result = Vec::with_capacity(count);
        let mut bit_offset = 0usize;

        for _ in 0..count {
            let mut value = 0u32;
            let mut bits_remaining = bit_width as usize;
            let mut bits_read = 0usize;

            while bits_remaining > 0 {
                let byte_idx = bit_offset / 8;
                if byte_idx >= data.len() {
                    return Err(PersistenceError::Format(
                        "Incomplete bitpacked data".to_string(),
                    ));
                }

                let bit_in_byte = bit_offset % 8;
                let bits_available = 8 - bit_in_byte;
                let bits_to_take = bits_remaining.min(bits_available);

                let mask = ((1u32 << bits_to_take) - 1) << bit_in_byte;
                let bits = ((data[byte_idx] as u32) & mask) >> bit_in_byte;
                value |= bits << bits_read;

                bit_offset += bits_to_take;
                bits_remaining -= bits_to_take;
                bits_read += bits_to_take;
            }

            result.push(value);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encode_decode() {
        let values = vec![
            0,
            1,
            127,
            128,
            255,
            256,
            65535,
            65536,
            u32::MAX as u64,
            u64::MAX,
        ];

        for &value in &values {
            let encoded = varint::encode(value);
            let (decoded, bytes_read) = varint::decode(&encoded).unwrap();
            assert_eq!(decoded, value);
            assert_eq!(bytes_read, encoded.len());
        }
    }

    #[test]
    fn test_varint_encode_many() {
        let values = vec![1u64, 2, 3, 100, 1000];
        let encoded = varint::encode_many(&values);
        let (decoded, _) = varint::decode_many(&encoded, values.len()).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn test_delta_encode_decode() {
        let sorted = vec![5, 7, 9, 12, 15, 20];
        let deltas = delta::encode(&sorted);
        assert_eq!(deltas, vec![5, 2, 2, 3, 3, 5]);

        let decoded = delta::decode(&deltas);
        assert_eq!(decoded, sorted);
    }

    #[test]
    fn test_bitpack_pack_unpack() {
        let values = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
        let bit_width = bitpack::bit_width_many(&values);
        assert_eq!(bit_width, 4); // Need 4 bits for max value 8

        let packed = bitpack::pack(&values, bit_width);
        let unpacked = bitpack::unpack(&packed, values.len(), bit_width).unwrap();
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_bitpack_block_size() {
        // Test with exactly BLOCK_SIZE values
        let values: Vec<u32> = (0..BLOCK_SIZE as u32).collect();
        let bit_width = bitpack::bit_width_many(&values);
        let packed = bitpack::pack(&values, bit_width);
        let unpacked = bitpack::unpack(&packed, values.len(), bit_width).unwrap();
        assert_eq!(unpacked, values);
    }

    #[test]
    fn test_bitpack_various_widths() {
        for width in 1..=8 {
            // Create values that fit in the specified bit width
            let max_value = (1u32 << width) - 1;
            let values: Vec<u32> = (0..10).map(|i| (i % max_value as usize) as u32).collect();
            let bit_width = bitpack::bit_width_many(&values);
            let packed = bitpack::pack(&values, bit_width);
            let unpacked = bitpack::unpack(&packed, values.len(), bit_width).unwrap();
            assert_eq!(unpacked, values, "Failed for width {}", width);
        }
    }
}
