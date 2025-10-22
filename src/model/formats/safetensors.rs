//! SafeTensors format loader
//!
//! Loads PyTorch-compatible SafeTensors files and converts to f16 format.

use crate::tensor::Tensor;
use crate::device::{Device, MetalDevice};
use crate::error::TensorError;
use crate::model::{Model, ModelMetadata, ModelFormat, QuantizationType};
use crate::model::convert::{f32_to_f16, f16_to_f32};
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

pub struct SafeTensorsLoader;

impl SafeTensorsLoader {
    /// Load a SafeTensors file (tensors loaded to Metal GPU)
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Model, TensorError> {
        let device = MetalDevice::new()?;
        let path = path.as_ref();
        let buffer = fs::read(path)
            .map_err(|e| TensorError::InvalidOperation(format!("Failed to read file: {}", e)))?;

        let safe_tensors = SafeTensors::deserialize(&buffer)
            .map_err(|e| TensorError::InvalidOperation(format!("Failed to parse SafeTensors: {}", e)))?;

        let mut tensors = HashMap::new();

        for (name, tensor_view) in safe_tensors.tensors() {
            // SafeTensors stores data in various formats, convert to f16
            let shape = tensor_view.shape();
            let dtype = tensor_view.dtype();
            let data = tensor_view.data();

            // Get raw data and convert to f16 based on dtype
            let f16_data = match dtype {
                safetensors::Dtype::F32 => {
                    // Convert f32 bytes to f32 values, then to f16
                    let f32_vec: Vec<f32> = data
                        .chunks_exact(4)
                        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                        .collect();
                    f32_to_f16(&f32_vec)
                }
                safetensors::Dtype::F64 => {
                    // Convert f64 bytes to f64 values, then to f16
                    let f64_vec: Vec<f64> = data
                        .chunks_exact(8)
                        .map(|chunk| f64::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                            chunk[4], chunk[5], chunk[6], chunk[7]
                        ]))
                        .collect();
                    crate::model::convert::f64_to_f16(&f64_vec)
                }
                safetensors::Dtype::F16 => {
                    // Already f16, just convert bytes
                    data.chunks_exact(2)
                        .map(|chunk| half::f16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect()
                }
                safetensors::Dtype::BF16 => {
                    // BF16 to f16 conversion (bfloat16 format)
                    let bf16_vec: Vec<half::bf16> = data
                        .chunks_exact(2)
                        .map(|chunk| half::bf16::from_le_bytes([chunk[0], chunk[1]]))
                        .collect();
                    bf16_vec.iter().map(|&bf| half::f16::from_f32(bf.to_f32())).collect()
                }
                safetensors::Dtype::I8 | safetensors::Dtype::I16 |
                safetensors::Dtype::I32 | safetensors::Dtype::I64 |
                safetensors::Dtype::U8 | safetensors::Dtype::U16 |
                safetensors::Dtype::U32 | safetensors::Dtype::U64 => {
                    return Err(TensorError::InvalidOperation(
                        format!("Integer dtype {:?} not supported for tensor '{}'. TensorLogic uses f16 only.", dtype, name)
                    ));
                }
                _ => {
                    return Err(TensorError::InvalidOperation(
                        format!("Unsupported dtype {:?} for tensor '{}'", dtype, name)
                    ));
                }
            };

            // Create TensorLogic tensor (on Metal GPU)
            let tensor = Tensor::from_vec_metal(&device, f16_data, shape.to_vec())?;
            tensors.insert(name.to_string(), tensor);
        }

        let metadata = ModelMetadata {
            name: path.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown")
                .to_string(),
            format: ModelFormat::SafeTensors,
            quantization: Some(QuantizationType::F16),
        };

        Ok(Model { tensors, metadata })
    }

    /// Save to SafeTensors file
    pub fn save<P: AsRef<Path>>(model: &Model, path: P) -> Result<(), TensorError> {
        use safetensors::tensor::{Dtype, TensorView};

        // Convert all tensors to f32 first (need owned data for lifetimes)
        let mut f32_tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();

        for (name, tensor) in &model.tensors {
            let f16_data = tensor.to_vec();
            let f32_data = f16_to_f32(&f16_data);
            let shape: Vec<usize> = tensor.shape().dims().to_vec();
            f32_tensors.push((name.clone(), shape, f32_data));
        }

        // Create tensor views from owned data
        let tensor_views: Vec<(String, TensorView)> = f32_tensors.iter()
            .map(|(name, shape, data)| {
                let view = TensorView::new(
                    Dtype::F32,
                    shape.clone(),
                    bytemuck::cast_slice(data)
                ).map_err(|e| TensorError::InvalidOperation(format!("Failed to create tensor view: {}", e)))?;
                Ok((name.clone(), view))
            })
            .collect::<Result<Vec<_>, TensorError>>()?;

        // Serialize
        let data = safetensors::tensor::serialize(tensor_views, &None)
            .map_err(|e| TensorError::InvalidOperation(format!("Failed to serialize: {}", e)))?;

        // Write to file
        fs::write(path, data)
            .map_err(|e| TensorError::InvalidOperation(format!("Failed to write file: {}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_roundtrip() {
        use crate::device::MetalDevice;

        // Create a simple model
        let device = MetalDevice::new().unwrap();
        let mut tensors = HashMap::new();
        let data = vec![half::f16::from_f32(1.0), half::f16::from_f32(2.0)];
        let tensor = Tensor::from_vec_metal(&device, data, vec![2]).unwrap();
        tensors.insert("test_tensor".to_string(), tensor);

        let metadata = ModelMetadata {
            name: "test".to_string(),
            format: ModelFormat::SafeTensors,
            quantization: Some(QuantizationType::F16),
        };

        let model = Model { tensors, metadata };

        // Save and load
        let path = "/tmp/test_model.safetensors";
        SafeTensorsLoader::save(&model, path).unwrap();
        let loaded = SafeTensorsLoader::load(path).unwrap();

        assert_eq!(loaded.tensors.len(), 1);
        assert!(loaded.tensors.contains_key("test_tensor"));
    }

    #[test]
    fn test_safetensors_multiple_tensors() {
        use crate::device::MetalDevice;

        let device = MetalDevice::new().unwrap();
        let mut tensors = HashMap::new();

        // Add multiple tensors with different shapes
        tensors.insert(
            "weights".to_string(),
            Tensor::from_vec_metal(&device, vec![half::f16::from_f32(0.5); 12], vec![3, 4]).unwrap()
        );
        tensors.insert(
            "bias".to_string(),
            Tensor::from_vec_metal(&device, vec![half::f16::from_f32(0.1); 4], vec![4]).unwrap()
        );
        tensors.insert(
            "scale".to_string(),
            Tensor::from_vec_metal(&device, vec![half::f16::from_f32(1.0)], vec![1]).unwrap()
        );

        let metadata = ModelMetadata {
            name: "multi_tensor_model".to_string(),
            format: ModelFormat::SafeTensors,
            quantization: Some(QuantizationType::F16),
        };

        let model = Model { tensors, metadata };

        // Save and load
        let path = "/tmp/test_multi_tensor.safetensors";
        SafeTensorsLoader::save(&model, path).unwrap();
        let loaded = SafeTensorsLoader::load(path).unwrap();

        assert_eq!(loaded.tensors.len(), 3);
        assert!(loaded.tensors.contains_key("weights"));
        assert!(loaded.tensors.contains_key("bias"));
        assert!(loaded.tensors.contains_key("scale"));

        // Verify shapes
        assert_eq!(loaded.get_tensor("weights").unwrap().shape().dims(), &[3, 4]);
        assert_eq!(loaded.get_tensor("bias").unwrap().shape().dims(), &[4]);
        assert_eq!(loaded.get_tensor("scale").unwrap().shape().dims(), &[1]);
    }

    #[test]
    fn test_safetensors_large_tensor() {
        use crate::device::MetalDevice;

        let device = MetalDevice::new().unwrap();
        let mut tensors = HashMap::new();

        // Create a larger tensor (1000 elements)
        let size = 1000;
        let data: Vec<half::f16> = (0..size)
            .map(|i| half::f16::from_f32(i as f32 / size as f32))
            .collect();

        tensors.insert(
            "large_tensor".to_string(),
            Tensor::from_vec_metal(&device, data, vec![size]).unwrap()
        );

        let metadata = ModelMetadata {
            name: "large_model".to_string(),
            format: ModelFormat::SafeTensors,
            quantization: Some(QuantizationType::F16),
        };

        let model = Model { tensors, metadata };

        // Save and load
        let path = "/tmp/test_large_tensor.safetensors";
        SafeTensorsLoader::save(&model, path).unwrap();
        let loaded = SafeTensorsLoader::load(path).unwrap();

        assert_eq!(loaded.tensors.len(), 1);
        assert_eq!(loaded.get_tensor("large_tensor").unwrap().numel(), size);
    }
}
