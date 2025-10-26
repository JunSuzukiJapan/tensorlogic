//! GGUF format loader
//!
//! Loads quantized GGUF models (llama.cpp format) and dequantizes to f16.

use crate::tensor::Tensor;
use crate::device::MetalDevice;
use crate::error::TensorError;
use crate::model::{Model, ModelMetadata, ModelFormat, QuantizationType};
use crate::model::convert::f32_to_f16;
use gguf_rs_lib::prelude::*;
use gguf_rs_lib::tensor::quantization::blocks::{Q4_0Block, Q6_KBlock, Q8_0Block};
use std::collections::HashMap;
use std::path::Path;
use std::fs::File;

pub struct GGUFLoader;

impl GGUFLoader {
    /// Dequantize Q8_0 format to f16
    /// Q8_0: 32 int8 values per block with 1 f16 scale
    fn dequantize_q8_0(data: &[u8], num_elements: usize) -> Vec<half::f16> {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = std::mem::size_of::<Q8_0Block>();

        let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut result = Vec::with_capacity(num_elements);

        for block_idx in 0..num_blocks {
            let block_offset = block_idx * BLOCK_BYTES;
            if block_offset + BLOCK_BYTES > data.len() {
                break;
            }

            // Read scale (f16)
            let scale_bytes = [data[block_offset], data[block_offset + 1]];
            let scale = half::f16::from_le_bytes(scale_bytes);

            // Read 32 int8 values
            let values_offset = block_offset + 2;
            for i in 0..BLOCK_SIZE {
                if result.len() >= num_elements {
                    break;
                }
                let q_value = data[values_offset + i] as i8;
                let f_value = half::f16::from_f32(q_value as f32) * scale;
                result.push(f_value);
            }
        }

        result.truncate(num_elements);
        result
    }

    /// Dequantize Q4_0 format to f16
    /// Q4_0: 32 4-bit values per block with 1 f16 scale
    ///
    /// Reference: https://github.com/huggingface/candle/blob/main/candle-core/src/quantized/k_quants.rs
    /// Block structure:
    ///   - 2 bytes: f16 scale (d)
    ///   - 16 bytes: 32 packed 4-bit values (qs)
    ///
    /// Dequantization:
    ///   for each byte:
    ///     lower 4 bits → output[j]
    ///     upper 4 bits → output[j + 16]
    ///   value = (nibble - 8) * scale
    fn dequantize_q4_0(data: &[u8], num_elements: usize) -> Vec<half::f16> {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = std::mem::size_of::<Q4_0Block>();

        let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut result = vec![half::f16::ZERO; num_elements];

        for block_idx in 0..num_blocks {
            let block_offset = block_idx * BLOCK_BYTES;
            if block_offset + BLOCK_BYTES > data.len() {
                break;
            }

            // Read scale (f16)
            let scale_bytes = [data[block_offset], data[block_offset + 1]];
            let scale = half::f16::from_le_bytes(scale_bytes);
            let scale_f32 = scale.to_f32();

            // Read 16 bytes of 4-bit values (32 values total)
            // Layout: [low0-15, high0-15] not [low0, high0, low1, high1, ...]
            let values_offset = block_offset + 2;
            let base_idx = block_idx * BLOCK_SIZE;

            for j in 0..16 {
                if base_idx + j >= num_elements {
                    break;
                }

                let byte = data[values_offset + j];

                // Lower 4 bits → first half of block
                let x0 = ((byte & 0x0F) as i8 - 8) as f32;
                result[base_idx + j] = half::f16::from_f32(x0 * scale_f32);

                // Upper 4 bits → second half of block
                let second_idx = base_idx + j + 16;
                if second_idx < num_elements {
                    let x1 = ((byte >> 4) as i8 - 8) as f32;
                    result[second_idx] = half::f16::from_f32(x1 * scale_f32);
                }
            }
        }

        result
    }

    /// Dequantize Q6_K format to f16
    /// Q6_K: 256 6-bit values per block with 16 f16 scales
    /// Dequantize Q6_K format to f16
    /// Q6_K: 256 values per block, 6-bit quantization with scales
    ///
    /// Reference: llama.cpp dequantize_row_q6_K
    /// Block structure (210 bytes total):
    ///   - ql[128]: lower 4 bits of quantized values
    ///   - qh[64]: upper 2 bits of quantized values
    ///   - scales[16]: int8 scales
    ///   - d: f16 super-block scale
    ///
    /// Each 6-bit value = (4 bits from ql) | (2 bits from qh << 4)
    /// value = d * scale[i] * (q - 32)
    fn dequantize_q6_k(data: &[u8], num_elements: usize) -> Vec<half::f16> {
        const QK_K: usize = 256;  // Block size
        const BLOCK_BYTES: usize = 210;  // 128 + 64 + 16 + 2

        // Debug output
        if std::env::var("TL_DEBUG_Q6K").is_ok() {
            eprintln!("\n=== Q6_K Dequantization Debug ===");
            eprintln!("  Elements: {}", num_elements);
            eprintln!("  Data bytes: {}", data.len());
            eprintln!("  Expected blocks: {}", (num_elements + QK_K - 1) / QK_K);
            eprintln!("  Expected bytes: {}", ((num_elements + QK_K - 1) / QK_K) * BLOCK_BYTES);
            eprintln!("==================================\n");
        }

        let num_blocks = (num_elements + QK_K - 1) / QK_K;
        let mut result = vec![half::f16::ZERO; num_elements];

        for block_idx in 0..num_blocks {
            let block_offset = block_idx * BLOCK_BYTES;
            if block_offset + BLOCK_BYTES > data.len() {
                break;
            }

            // Read block components
            let ql = &data[block_offset..block_offset + 128];        // ql[128]
            let qh = &data[block_offset + 128..block_offset + 192];  // qh[64]
            let sc = &data[block_offset + 192..block_offset + 208];  // scales[16]
            let d_bytes = [data[block_offset + 208], data[block_offset + 209]];
            let d = half::f16::from_le_bytes(d_bytes).to_f32();

            // Convert scales from int8 to i8
            let scales: Vec<i8> = sc.iter().map(|&b| b as i8).collect();

            let base_idx = block_idx * QK_K;

            // Process 2 groups of 128 values each (total 256)
            // Each group processes 32 iterations, writing 4 values each = 128 values
            let mut ql_offset = 0;
            let mut qh_offset = 0;
            let mut sc_offset = 0;

            for n in (0..QK_K).step_by(128) {
                // Process 32 iterations, each producing 4 values (total 128)
                for l in 0..32 {
                    let out_base = base_idx + n + l;
                    if out_base >= num_elements {
                        break;
                    }

                    let is = l / 16;  // Scale index within current group: 0 or 1

                    // Reconstruct 4 6-bit values from ql and qh
                    // Following llama.cpp dequantize_row_q6_K exactly

                    // q1: ql[l+0] lower 4 bits + qh[l] bits 0-1
                    let q1 = ((ql[ql_offset + l] & 0xF) | (((qh[qh_offset + l] >> 0) & 0x3) << 4)) as i8 - 32;

                    // q2: ql[l+32] lower 4 bits + qh[l] bits 2-3
                    let q2 = ((ql[ql_offset + l + 32] & 0xF) | (((qh[qh_offset + l] >> 2) & 0x3) << 4)) as i8 - 32;

                    // q3: ql[l+0] upper 4 bits + qh[l] bits 4-5
                    let q3 = ((ql[ql_offset + l] >> 4) | (((qh[qh_offset + l] >> 4) & 0x3) << 4)) as i8 - 32;

                    // q4: ql[l+32] upper 4 bits + qh[l] bits 6-7
                    let q4 = ((ql[ql_offset + l + 32] >> 4) | (((qh[qh_offset + l] >> 6) & 0x3) << 4)) as i8 - 32;

                    // Write 4 values with corresponding scales
                    // y[l + 0] = d * sc[is + 0] * q1
                    if out_base + 0 < num_elements {
                        result[out_base + 0] = half::f16::from_f32(d * scales[sc_offset + is + 0] as f32 * q1 as f32);
                    }
                    // y[l + 32] = d * sc[is + 2] * q2
                    if out_base + 32 < num_elements {
                        result[out_base + 32] = half::f16::from_f32(d * scales[sc_offset + is + 2] as f32 * q2 as f32);
                    }
                    // y[l + 64] = d * sc[is + 4] * q3
                    if out_base + 64 < num_elements {
                        result[out_base + 64] = half::f16::from_f32(d * scales[sc_offset + is + 4] as f32 * q3 as f32);
                    }
                    // y[l + 96] = d * sc[is + 6] * q4
                    if out_base + 96 < num_elements {
                        result[out_base + 96] = half::f16::from_f32(d * scales[sc_offset + is + 6] as f32 * q4 as f32);
                    }
                }

                // Advance pointers for next 128-element group (llama.cpp: y+=128, ql+=64, qh+=32, sc+=8)
                ql_offset += 64;
                qh_offset += 32;
                sc_offset += 8;
            }
        }

        // Debug: Check output values
        if std::env::var("TL_DEBUG_Q6K").is_ok() {
            eprintln!("\n=== Q6_K Dequantization Output ===");
            eprintln!("  First 10 values:");
            for i in 0..10.min(result.len()) {
                eprintln!("    [{}]: {}", i, result[i].to_f32());
            }
            eprintln!("  Last 10 values:");
            let start = result.len().saturating_sub(10);
            for i in start..result.len() {
                eprintln!("    [{}]: {}", i, result[i].to_f32());
            }

            // Check for abnormal values
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            let mut sum = 0.0f32;
            for &val in &result[..1000.min(result.len())] {
                let f = val.to_f32();
                if f.is_finite() {
                    min_val = min_val.min(f);
                    max_val = max_val.max(f);
                    sum += f;
                }
            }
            eprintln!("  Stats (first 1000):");
            eprintln!("    Min: {}", min_val);
            eprintln!("    Max: {}", max_val);
            eprintln!("    Avg: {}", sum / 1000.0);
            eprintln!("===================================\n");
        }

        result
    }

    /// Load a GGUF file and dequantize to f16 (tensors loaded to Metal GPU)
    pub fn load<P: AsRef<Path>>(path: P, device: &MetalDevice) -> std::result::Result<Model, TensorError> {
        let path = path.as_ref();

        // Load GGUF file using file reader
        let file = File::open(path)
            .map_err(|e| TensorError::InvalidOperation(format!("Failed to open file: {}", e)))?;

        let mut reader = GGUFFileReader::new(file)
            .map_err(|e| TensorError::InvalidOperation(format!("Failed to read GGUF: {}", e)))?;

        let mut tensors = HashMap::new();
        let mut quantization_type = QuantizationType::None;

        // Get tensor infos
        let tensor_infos = reader.tensor_infos().to_vec();

        // Iterate through tensor info
        for tensor_info in &tensor_infos {
            let name = tensor_info.name.clone();
            let mut shape: Vec<usize> = tensor_info.shape.dimensions.iter().map(|&d| d as usize).collect();

            // GGUF dimensions handling
            // gguf-rs returns dimensions as-is from file (which are in reverse order per GGUF spec)
            // We need to reverse to get PyTorch-style dimensions:
            // - All 2D+ tensors: REVERSE (weight matrices stored as transposed in GGUF)
            // - 1D tensors: NO reverse (normalization weights, stored correctly)
            //
            // Examples:
            // - token_embd.weight: [2048, 32000] → [32000, 2048] for embedding()
            // - attn_k.weight: [2048, 256] → [256, 2048] for linear() which transposes internally
            // - attn_norm.weight: [2048] → [2048] (1D, no change)

            let needs_reverse = shape.len() > 1;  // Reverse all multi-dimensional tensors
            if needs_reverse {
                shape.reverse();
            }

            // Load and dequantize based on tensor type
            let f16_data = match tensor_info.tensor_type {
                GGUFTensorType::F32 => {
                    quantization_type = QuantizationType::F32;

                    // Load tensor data
                    let data = reader.load_tensor_data(&tensor_info.name)
                        .map_err(|e| TensorError::InvalidOperation(format!("Failed to load tensor data: {}", e)))?
                        .ok_or_else(|| TensorError::InvalidOperation(format!("Tensor data not found for {}", tensor_info.name)))?;

                    // Get bytes from TensorData enum
                    let bytes = match data {
                        gguf_rs_lib::tensor::TensorData::Owned(ref v) => v.as_slice(),
                        gguf_rs_lib::tensor::TensorData::Borrowed(b) => b,
                        gguf_rs_lib::tensor::TensorData::Shared(ref arc) => arc.as_slice(),
                        _ => return Err(TensorError::InvalidOperation("Unexpected tensor data type".to_string())),
                    };

                    // Debug: Check data size for critical tensors
                    if tensor_info.name == "output.weight" || std::env::var("TL_DEBUG_GGUF").is_ok() {
                        let num_elements: usize = shape.iter().product();
                        let expected_bytes = num_elements * 4; // F32 = 4 bytes
                        eprintln!("\n=== GGUF F32 Tensor Debug ===");
                        eprintln!("  Tensor: {}", tensor_info.name);
                        eprintln!("  Shape: {:?} ({} elements)", shape, num_elements);
                        eprintln!("  Expected bytes: {}", expected_bytes);
                        eprintln!("  Actual bytes: {}", bytes.len());
                        if bytes.len() != expected_bytes {
                            eprintln!("  ❌ SIZE MISMATCH!");
                        }
                        eprintln!("=============================\n");
                    }

                    // Convert f32 bytes to f32 values
                    let f32_data: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();

                    // Convert to f16
                    f32_to_f16(&f32_data)
                }
                GGUFTensorType::F16 => {
                    quantization_type = QuantizationType::F16;

                    // Load tensor data
                    let data = reader.load_tensor_data(&tensor_info.name)
                        .map_err(|e| TensorError::InvalidOperation(format!("Failed to load tensor data: {}", e)))?
                        .ok_or_else(|| TensorError::InvalidOperation(format!("Tensor data not found for {}", tensor_info.name)))?;

                    // Get bytes from TensorData enum
                    let bytes = match data {
                        gguf_rs_lib::tensor::TensorData::Owned(ref v) => v.as_slice(),
                        gguf_rs_lib::tensor::TensorData::Borrowed(b) => b,
                        gguf_rs_lib::tensor::TensorData::Shared(ref arc) => arc.as_slice(),
                        _ => return Err(TensorError::InvalidOperation("Unexpected tensor data type".to_string())),
                    };

                    // Already f16, just convert bytes
                    bytes
                        .chunks_exact(2)
                        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect()
                }
                GGUFTensorType::Q8_0 => {
                    quantization_type = QuantizationType::Q8;

                    // Load tensor data
                    let data = reader.load_tensor_data(&tensor_info.name)
                        .map_err(|e| TensorError::InvalidOperation(format!("Failed to load tensor data: {}", e)))?
                        .ok_or_else(|| TensorError::InvalidOperation(format!("Tensor data not found for {}", tensor_info.name)))?;

                    // Get bytes from TensorData enum
                    let bytes = match data {
                        gguf_rs_lib::tensor::TensorData::Owned(ref v) => v.as_slice(),
                        gguf_rs_lib::tensor::TensorData::Borrowed(b) => b,
                        gguf_rs_lib::tensor::TensorData::Shared(ref arc) => arc.as_slice(),
                        _ => return Err(TensorError::InvalidOperation("Unexpected tensor data type".to_string())),
                    };

                    // Dequantize Q8_0 to f16
                    let num_elements: usize = shape.iter().product();
                    Self::dequantize_q8_0(bytes, num_elements)
                }
                GGUFTensorType::Q4_0 => {
                    quantization_type = QuantizationType::Q4;

                    // Load tensor data
                    let data = reader.load_tensor_data(&tensor_info.name)
                        .map_err(|e| TensorError::InvalidOperation(format!("Failed to load tensor data: {}", e)))?
                        .ok_or_else(|| TensorError::InvalidOperation(format!("Tensor data not found for {}", tensor_info.name)))?;

                    // Get bytes from TensorData enum
                    let bytes = match data {
                        gguf_rs_lib::tensor::TensorData::Owned(ref v) => v.as_slice(),
                        gguf_rs_lib::tensor::TensorData::Borrowed(b) => b,
                        gguf_rs_lib::tensor::TensorData::Shared(ref arc) => arc.as_slice(),
                        _ => return Err(TensorError::InvalidOperation("Unexpected tensor data type".to_string())),
                    };

                    // Dequantize Q4_0 to f16
                    let num_elements: usize = shape.iter().product();
                    Self::dequantize_q4_0(bytes, num_elements)
                }
                GGUFTensorType::Q6_K => {
                    quantization_type = QuantizationType::Q6;

                    // Debug: Q6_K loading
                    if tensor_info.name == "output.weight" || std::env::var("TL_DEBUG_Q6K").is_ok() {
                        eprintln!("\n=== Loading Q6_K Tensor ===");
                        eprintln!("  Name: {}", tensor_info.name);
                        eprintln!("  Shape: {:?}", shape);
                        eprintln!("===========================\n");
                    }

                    // Load tensor data
                    let data = reader.load_tensor_data(&tensor_info.name)
                        .map_err(|e| TensorError::InvalidOperation(format!("Failed to load tensor data: {}", e)))?
                        .ok_or_else(|| TensorError::InvalidOperation(format!("Tensor data not found for {}", tensor_info.name)))?;

                    // Get bytes from TensorData enum
                    let bytes = match data {
                        gguf_rs_lib::tensor::TensorData::Owned(ref v) => v.as_slice(),
                        gguf_rs_lib::tensor::TensorData::Borrowed(b) => b,
                        gguf_rs_lib::tensor::TensorData::Shared(ref arc) => arc.as_slice(),
                        _ => return Err(TensorError::InvalidOperation("Unexpected tensor data type".to_string())),
                    };

                    // Dequantize Q6_K to f16
                    let num_elements: usize = shape.iter().product();
                    Self::dequantize_q6_k(bytes, num_elements)
                }
                _ => {
                    // Other quantized formats (Q4_1, Q5_0, Q5_1, Q8_1, other K-quants, etc.)
                    return Err(TensorError::InvalidOperation(
                        format!("Quantized tensor type {:?} not yet supported. Currently supporting: F32, F16, Q4_0, Q6_K, Q8_0", tensor_info.tensor_type)
                    ));
                }
            };

            // Create TensorLogic tensor (on Metal GPU)
            let tensor = Tensor::from_vec_metal(device, f16_data, shape)?;
            tensors.insert(name, tensor);
        }

        let metadata = ModelMetadata {
            name: path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            format: ModelFormat::GGUF,
            quantization: Some(quantization_type),
        };

        Ok(Model { tensors, metadata })
    }

    /// Save to GGUF file (with optional quantization)
    pub fn save<P: AsRef<Path>>(_model: &Model, _path: P) -> std::result::Result<(), TensorError> {
        // GGUF writing is complex and requires proper quantization
        // For now, return an error suggesting to use SafeTensors for saving
        Err(TensorError::InvalidOperation(
            "GGUF saving not yet implemented. Use SafeTensors format for saving models.".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q8_0_dequantization() {
        // Create mock Q8_0 data (1 block = 34 bytes: 2 bytes scale + 32 bytes data)
        let mut data = vec![0u8; 34];

        // Set scale to 1.0 in f16
        let scale = half::f16::from_f32(1.0);
        data[0..2].copy_from_slice(&scale.to_le_bytes());

        // Set some int8 values
        for i in 0..32 {
            data[2 + i] = i as u8; // 0, 1, 2, ..., 31
        }

        let result = GGUFLoader::dequantize_q8_0(&data, 32);
        assert_eq!(result.len(), 32);

        // Check first few values
        assert_eq!(result[0].to_f32(), 0.0);
        assert_eq!(result[1].to_f32(), 1.0);
        assert_eq!(result[2].to_f32(), 2.0);
    }

    #[test]
    fn test_q4_0_dequantization() {
        // Create mock Q4_0 data (1 block = 18 bytes: 2 bytes scale + 16 bytes data)
        let mut data = vec![0u8; 18];

        // Set scale to 1.0 in f16
        let scale = half::f16::from_f32(1.0);
        data[0..2].copy_from_slice(&scale.to_le_bytes());

        // Set some 4-bit values (each byte contains 2 values)
        // Byte 0x01 = low nibble 0x1, high nibble 0x0
        // 4-bit signed: values are offset by -8, so 0x1 = -7, 0x0 = -8
        data[2] = 0x01; // First two 4-bit values

        let result = GGUFLoader::dequantize_q4_0(&data, 32);
        assert_eq!(result.len(), 32);

        // Check that dequantization runs without panic
        // Actual values depend on the quantization formula
    }

    #[test]
    fn test_q8_0_multiple_blocks() {
        // Create 2 blocks of Q8_0 data
        let mut data = vec![0u8; 68]; // 2 blocks * 34 bytes

        // Block 1: scale = 1.0
        let scale1 = half::f16::from_f32(1.0);
        data[0..2].copy_from_slice(&scale1.to_le_bytes());
        for i in 0..32 {
            data[2 + i] = i as u8;
        }

        // Block 2: scale = 2.0
        let scale2 = half::f16::from_f32(2.0);
        data[34..36].copy_from_slice(&scale2.to_le_bytes());
        for i in 0..32 {
            data[36 + i] = i as u8;
        }

        let result = GGUFLoader::dequantize_q8_0(&data, 64);
        assert_eq!(result.len(), 64);

        // First block values should be i * 1.0
        assert_eq!(result[0].to_f32(), 0.0);
        assert_eq!(result[10].to_f32(), 10.0);

        // Second block values should be i * 2.0
        assert_eq!(result[32].to_f32(), 0.0);
        assert_eq!(result[42].to_f32(), 20.0);
    }

    #[test]
    fn test_q4_0_scale_effect() {
        // Test that scale factor affects the output
        let mut data = vec![0u8; 18];

        // Set scale to 0.5
        let scale = half::f16::from_f32(0.5);
        data[0..2].copy_from_slice(&scale.to_le_bytes());

        // Set all nibbles to max value (0xF = 15, which is 15-8=7 in signed 4-bit)
        for i in 2..18 {
            data[i] = 0xFF;
        }

        let result = GGUFLoader::dequantize_q4_0(&data, 32);
        assert_eq!(result.len(), 32);

        // All values should be 7 * 0.5 = 3.5
        for val in result {
            assert!((val.to_f32() - 3.5).abs() < 0.01);
        }
    }

    #[test]
    #[ignore] // Requires actual GGUF file
    fn test_gguf_load() {
        // This would require a real GGUF file for testing
        // Example usage:
        // let model = GGUFLoader::load("path/to/model.gguf").unwrap();
        // assert!(!model.tensors.is_empty());
        // assert!(model.metadata.quantization.is_some());
    }
}
