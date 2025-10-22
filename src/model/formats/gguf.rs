//! GGUF format loader
//!
//! Loads quantized GGUF models (llama.cpp format) and dequantizes to f16.

use crate::tensor::Tensor;
use crate::device::MetalDevice;
use crate::error::TensorError;
use crate::model::{Model, ModelMetadata, ModelFormat, QuantizationType};
use crate::model::convert::f32_to_f16;
use gguf_rs_lib::prelude::*;
use gguf_rs_lib::tensor::quantization::blocks::{Q4_0Block, Q8_0Block};
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
    fn dequantize_q4_0(data: &[u8], num_elements: usize) -> Vec<half::f16> {
        const BLOCK_SIZE: usize = 32;
        const BLOCK_BYTES: usize = std::mem::size_of::<Q4_0Block>();

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

            // Read 16 bytes of 4-bit values (32 values total)
            let values_offset = block_offset + 2;
            for i in 0..16 {
                if result.len() >= num_elements {
                    break;
                }

                let byte = data[values_offset + i];

                // Lower 4 bits
                let q_low = (byte & 0x0F) as i8 - 8;  // 4-bit signed: -8 to 7
                let f_low = half::f16::from_f32(q_low as f32) * scale;
                result.push(f_low);

                if result.len() >= num_elements {
                    break;
                }

                // Upper 4 bits
                let q_high = ((byte >> 4) & 0x0F) as i8 - 8;
                let f_high = half::f16::from_f32(q_high as f32) * scale;
                result.push(f_high);
            }
        }

        result.truncate(num_elements);
        result
    }

    /// Load a GGUF file and dequantize to f16 (tensors loaded to Metal GPU)
    pub fn load<P: AsRef<Path>>(path: P) -> std::result::Result<Model, TensorError> {
        let device = MetalDevice::new()?;
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
            let shape: Vec<usize> = tensor_info.shape.dimensions.iter().map(|&d| d as usize).collect();

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
                _ => {
                    // Other quantized formats (Q4_1, Q5_0, Q5_1, Q8_1, K-quants, etc.)
                    return Err(TensorError::InvalidOperation(
                        format!("Quantized tensor type {:?} not yet supported. Currently supporting: F32, F16, Q4_0, Q8_0", tensor_info.tensor_type)
                    ));
                }
            };

            // Create TensorLogic tensor (on Metal GPU)
            let tensor = Tensor::from_vec_metal(&device, f16_data, shape)?;
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
