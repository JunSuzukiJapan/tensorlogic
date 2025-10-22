//! Type conversion utilities for model formats
//!
//! All external formats are converted to/from TensorLogic's native f16 format.

use half::f16;

/// Trait for type conversion to/from f16
pub trait TypeConverter {
    /// Convert to f16
    fn to_f16_vec(&self) -> Vec<f16>;

    /// Convert from f16
    fn from_f16_vec(data: &[f16]) -> Self;
}

/// Convert f32 slice to f16 vec
pub fn f32_to_f16(data: &[f32]) -> Vec<f16> {
    data.iter().map(|&x| f16::from_f32(x)).collect()
}

/// Convert f16 slice to f32 vec
pub fn f16_to_f32(data: &[f16]) -> Vec<f32> {
    data.iter().map(|&x| x.to_f32()).collect()
}

/// Convert f64 slice to f16 vec
pub fn f64_to_f16(data: &[f64]) -> Vec<f16> {
    data.iter().map(|&x| f16::from_f64(x)).collect()
}

/// Convert f16 slice to f64 vec
pub fn f16_to_f64(data: &[f16]) -> Vec<f64> {
    data.iter().map(|&x| x.to_f64()).collect()
}

/// Quantize f16 to 8-bit (simple linear quantization)
///
/// Maps f16 range to [-128, 127] int8 range
pub fn quantize_to_8bit(data: &[f16]) -> (Vec<i8>, f32, f32) {
    if data.is_empty() {
        return (vec![], 0.0, 1.0);
    }

    // Find min/max for scaling
    let min_val = data.iter().map(|&x| x.to_f32()).fold(f32::INFINITY, f32::min);
    let max_val = data.iter().map(|&x| x.to_f32()).fold(f32::NEG_INFINITY, f32::max);

    let scale = if (max_val - min_val).abs() < 1e-6 {
        1.0
    } else {
        255.0 / (max_val - min_val)
    };

    let quantized = data.iter().map(|&x| {
        let val = x.to_f32();
        let q = ((val - min_val) * scale - 128.0).round();
        q.clamp(-128.0, 127.0) as i8
    }).collect();

    (quantized, min_val, scale)
}

/// Dequantize 8-bit to f16
pub fn dequantize_from_8bit(data: &[i8], min_val: f32, scale: f32) -> Vec<f16> {
    data.iter().map(|&q| {
        let val = (q as f32 + 128.0) / scale + min_val;
        f16::from_f32(val)
    }).collect()
}

/// Quantize f16 to 4-bit (simple linear quantization)
///
/// Maps f16 range to [0, 15] 4-bit range
/// Packs two 4-bit values into one u8
pub fn quantize_to_4bit(data: &[f16]) -> (Vec<u8>, f32, f32) {
    if data.is_empty() {
        return (vec![], 0.0, 1.0);
    }

    // Find min/max for scaling
    let min_val = data.iter().map(|&x| x.to_f32()).fold(f32::INFINITY, f32::min);
    let max_val = data.iter().map(|&x| x.to_f32()).fold(f32::NEG_INFINITY, f32::max);

    let scale = if (max_val - min_val).abs() < 1e-6 {
        1.0
    } else {
        15.0 / (max_val - min_val)
    };

    // Quantize and pack
    let mut packed = Vec::with_capacity((data.len() + 1) / 2);

    for chunk in data.chunks(2) {
        let q1 = {
            let val = chunk[0].to_f32();
            let q = ((val - min_val) * scale).round();
            q.clamp(0.0, 15.0) as u8
        };

        let q2 = if chunk.len() > 1 {
            let val = chunk[1].to_f32();
            let q = ((val - min_val) * scale).round();
            q.clamp(0.0, 15.0) as u8
        } else {
            0
        };

        // Pack two 4-bit values into one byte
        packed.push((q1 << 4) | q2);
    }

    (packed, min_val, scale)
}

/// Dequantize 4-bit to f16
pub fn dequantize_from_4bit(packed: &[u8], original_len: usize, min_val: f32, scale: f32) -> Vec<f16> {
    let mut result = Vec::with_capacity(original_len);

    for &byte in packed {
        let q1 = (byte >> 4) & 0x0F;
        let q2 = byte & 0x0F;

        let val1 = (q1 as f32) / scale + min_val;
        result.push(f16::from_f32(val1));

        if result.len() < original_len {
            let val2 = (q2 as f32) / scale + min_val;
            result.push(f16::from_f32(val2));
        }
    }

    result.truncate(original_len);
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f32_to_f16_conversion() {
        let f32_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let f16_data = f32_to_f16(&f32_data);
        let back = f16_to_f32(&f16_data);

        for (orig, converted) in f32_data.iter().zip(back.iter()) {
            assert!((orig - converted).abs() < 0.01);
        }
    }

    #[test]
    fn test_f64_to_f16_conversion() {
        let f64_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let f16_data = f64_to_f16(&f64_data);
        let back = f16_to_f64(&f16_data);

        for (orig, converted) in f64_data.iter().zip(back.iter()) {
            assert!((orig - converted).abs() < 0.01);
        }
    }

    #[test]
    fn test_8bit_quantization() {
        let data = vec![
            f16::from_f32(0.0),
            f16::from_f32(0.5),
            f16::from_f32(1.0),
            f16::from_f32(1.5),
        ];

        let (quantized, min_val, scale) = quantize_to_8bit(&data);
        let dequantized = dequantize_from_8bit(&quantized, min_val, scale);

        assert_eq!(quantized.len(), data.len());
        assert_eq!(dequantized.len(), data.len());

        // Check approximate equality (quantization loses precision)
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig.to_f32() - deq.to_f32()).abs() < 0.1);
        }
    }

    #[test]
    fn test_4bit_quantization() {
        let data = vec![
            f16::from_f32(0.0),
            f16::from_f32(0.5),
            f16::from_f32(1.0),
            f16::from_f32(1.5),
        ];

        let (packed, min_val, scale) = quantize_to_4bit(&data);
        let dequantized = dequantize_from_4bit(&packed, data.len(), min_val, scale);

        assert_eq!(packed.len(), 2); // 4 values packed into 2 bytes
        assert_eq!(dequantized.len(), data.len());

        // Check approximate equality (quantization loses precision)
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            assert!((orig.to_f32() - deq.to_f32()).abs() < 0.2);
        }
    }
}
