//! GGUF weight cache with lazy loading
//!
//! Implements lazy loading for GGUF quantized models with LRU cache.
//! Weights are dequantized on-demand and evicted when not recently used.

use crate::device::MetalDevice;
use crate::error::{TensorError, TensorResult};
use crate::model::QuantizationType;
use crate::tensor::{FloatType, Tensor, TensorCreation};
use gguf_rs_lib::prelude::*;
use gguf_rs_lib::tensor::quantization::blocks::{Q4_0Block, Q8_0Block};
use half::f16;
use lru::LruCache;
use std::collections::HashMap;
use std::fs::File;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// Metadata for a weight tensor in the GGUF file
#[derive(Debug, Clone)]
struct GGUFWeightMetadata {
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Quantization type
    quantization: GGUFTensorType,
}

/// GGUF weight cache with lazy loading
///
/// Uses LRU eviction to keep only recently-used weights in memory.
/// Weights are dequantized on-demand from GGUF format.
///
/// # Example
/// ```rust,ignore
/// let cache = GGUFWeightCache::new("model.gguf", device, 10)?;
///
/// // Load weight on-demand (dequantized if not cached)
/// let weight = cache.get_weight("layer.0.weight")?;
///
/// // Weight is cached for subsequent access
/// let weight2 = cache.get_weight("layer.0.weight")?; // Fast: from cache
/// ```
#[derive(Debug)]
pub struct GGUFWeightCache<T: FloatType> {
    /// Path to the GGUF file
    path: PathBuf,

    /// GGUF file reader (wrapped in Arc<Mutex> for thread-safe access)
    reader: Arc<Mutex<GGUFFileReader<File>>>,

    /// LRU cache of loaded and dequantized tensors
    cache: Arc<Mutex<LruCache<String, Tensor<T>>>>,

    /// Metadata: weight name -> (shape, quantization)
    metadata: HashMap<String, GGUFWeightMetadata>,

    /// Metal device for GPU tensor creation
    device: MetalDevice,
}

impl<T: FloatType> GGUFWeightCache<T> {
    /// Create a new GGUF weight cache from a GGUF file
    ///
    /// # Arguments
    /// * `path` - Path to GGUF file
    /// * `device` - Metal device for tensor creation
    /// * `cache_capacity` - Maximum number of weights to keep in memory (LRU)
    ///
    /// # Returns
    /// Weight cache ready for lazy loading
    pub fn new<P: AsRef<Path>>(
        path: P,
        device: MetalDevice,
        cache_capacity: usize,
    ) -> TensorResult<Self> {
        let path = path.as_ref();

        // Open file and create reader to get metadata
        let file = File::open(path).map_err(|e| {
            TensorError::LoadError(format!("Failed to open GGUF file: {}", e))
        })?;

        let reader = GGUFFileReader::new(file).map_err(|e| {
            TensorError::LoadError(format!("Failed to create GGUF reader: {}", e))
        })?;

        // Extract metadata for each tensor
        let mut metadata = HashMap::new();
        let tensor_infos = reader.tensor_infos().to_vec();

        for tensor_info in tensor_infos {
            // GGUF stores dimensions in reverse order compared to TensorLogic
            // GGUF: [dim0, dim1, ...] → TensorLogic expects: [..., dim1, dim0]
            let mut shape: Vec<usize> = tensor_info.shape.dimensions.iter().map(|&d| d as usize).collect();
            shape.reverse();

            metadata.insert(
                tensor_info.name.clone(),
                GGUFWeightMetadata {
                    shape,
                    quantization: tensor_info.tensor_type,
                },
            );
        }

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[GGUFWeightCache] Loaded metadata for {} tensors", metadata.len());
        }

        Ok(Self {
            path: path.to_path_buf(),
            reader: Arc::new(Mutex::new(reader)),
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_capacity).unwrap(),
            ))),
            metadata,
            device,
        })
    }

    /// Get a weight tensor, dequantizing if not cached
    ///
    /// # Arguments
    /// * `name` - Weight name (e.g., "model.layers.0.self_attn.q_proj.weight")
    ///
    /// # Returns
    /// Tensor loaded from cache or dequantized from file
    pub fn get_weight(&self, name: &str) -> TensorResult<Tensor<T>> {
        // Check cache first
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(tensor) = cache.get(name) {
                if std::env::var("TL_DEBUG").is_ok() {
                    eprintln!("[GGUFWeightCache] Cache HIT: {}", name);
                }
                return Ok(tensor.clone());
            }
        }

        // Cache miss - load and dequantize from file
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[GGUFWeightCache] Cache MISS: {} (loading from file)", name);
        }

        let meta = self.metadata.get(name).ok_or_else(|| {
            TensorError::LoadError(format!("Weight not found: {}", name))
        })?;

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[GGUFWeightCache]   Shape: {:?}, Quantization: {:?}", meta.shape, meta.quantization);
        }

        // Load tensor data using the shared reader
        let data = {
            let mut reader = self.reader.lock().unwrap();
            reader.load_tensor_data(name)
                .map_err(|e| TensorError::LoadError(format!("Failed to load tensor data: {}", e)))?
                .ok_or_else(|| TensorError::LoadError(format!("Tensor data not found for {}", name)))?
        };

        // Get bytes from TensorData enum
        let bytes = match data {
            gguf_rs_lib::tensor::TensorData::Owned(ref v) => v.as_slice(),
            gguf_rs_lib::tensor::TensorData::Borrowed(b) => b,
            gguf_rs_lib::tensor::TensorData::Shared(ref arc) => arc.as_slice(),
            _ => return Err(TensorError::LoadError("Unexpected tensor data type".to_string())),
        };

        // Dequantize based on type
        let num_elements: usize = meta.shape.iter().product();

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[GGUFWeightCache]   Data bytes: {}, Expected elements: {}", bytes.len(), num_elements);
        }

        let values: Vec<T> = match meta.quantization {
            GGUFTensorType::Q4_0 => {
                let f16_values = Self::dequantize_q4_0(bytes, num_elements);
                if T::is_f16() {
                    // Convert f16 to T (which is f16)
                    unsafe { std::mem::transmute(f16_values) }
                } else {
                    // Convert f16 to f32
                    f16_values.iter().map(|&v| T::from_f32(v.to_f32())).collect()
                }
            }
            GGUFTensorType::Q8_0 => {
                let f16_values = Self::dequantize_q8_0(bytes, num_elements);
                if T::is_f16() {
                    unsafe { std::mem::transmute(f16_values) }
                } else {
                    f16_values.iter().map(|&v| T::from_f32(v.to_f32())).collect()
                }
            }
            GGUFTensorType::Q6_K => {
                let f16_values = Self::dequantize_q6_k(bytes, num_elements);
                if T::is_f16() {
                    unsafe { std::mem::transmute(f16_values) }
                } else {
                    f16_values.iter().map(|&v| T::from_f32(v.to_f32())).collect()
                }
            }
            GGUFTensorType::F16 => {
                // Direct f16 data
                let f16_values: Vec<f16> = bytes.chunks_exact(2)
                    .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]))
                    .collect();
                if T::is_f16() {
                    unsafe { std::mem::transmute(f16_values) }
                } else {
                    f16_values.iter().map(|&v| T::from_f32(v.to_f32())).collect()
                }
            }
            GGUFTensorType::F32 => {
                // Direct f32 data
                let f32_values: Vec<f32> = bytes.chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                if T::is_f32() {
                    unsafe { std::mem::transmute(f32_values) }
                } else {
                    f32_values.iter().map(|&v| T::from_f32(v)).collect()
                }
            }
            _ => return Err(TensorError::LoadError(
                format!("Unsupported quantization type: {:?}", meta.quantization)
            )),
        };

        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[GGUFWeightCache]   Dequantized to {} elements", values.len());
        }

        // Create tensor on GPU
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[GGUFWeightCache]   Creating tensor with shape: {:?}", meta.shape);
        }
        let tensor = Tensor::from_vec_gpu(&self.device, values, meta.shape.clone())?;
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[GGUFWeightCache]   Created tensor shape: {:?}", tensor.shape);
        }

        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(name.to_string(), tensor.clone());

            if std::env::var("TL_DEBUG").is_ok() {
                eprintln!(
                    "[GGUFWeightCache] Cached {} ({} elements, cache size: {}/{})",
                    name,
                    num_elements,
                    cache.len(),
                    cache.cap()
                );
            }
        }

        Ok(tensor)
    }

    /// Dequantize Q4_0 format to f16
    /// Q4_0: 32 4-bit values per block with 1 f16 scale
    fn dequantize_q4_0(data: &[u8], num_elements: usize) -> Vec<f16> {
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
            let scale = f16::from_le_bytes(scale_bytes);

            // Read 16 bytes containing 32 4-bit values
            let values_offset = block_offset + 2;
            for i in 0..16 {
                if values_offset + i >= data.len() {
                    break;
                }

                let byte = data[values_offset + i];
                let v1 = (byte & 0x0F) as i8 - 8;
                let v2 = ((byte >> 4) & 0x0F) as i8 - 8;

                result.push(scale * f16::from_f32(v1 as f32));
                result.push(scale * f16::from_f32(v2 as f32));
            }
        }

        result.truncate(num_elements);
        result
    }

    /// Dequantize Q8_0 format to f16
    /// Q8_0: 32 int8 values per block with 1 f16 scale
    fn dequantize_q8_0(data: &[u8], num_elements: usize) -> Vec<f16> {
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
            let scale = f16::from_le_bytes(scale_bytes);

            // Read 32 int8 values
            let values_offset = block_offset + 2;
            for i in 0..BLOCK_SIZE {
                if values_offset + i >= data.len() {
                    break;
                }

                let v = data[values_offset + i] as i8;
                result.push(scale * f16::from_f32(v as f32));
            }
        }

        result.truncate(num_elements);
        result
    }

    /// Dequantize Q6_K format to f16
    /// Q6_K: Complex k-quant format with multiple scales
    fn dequantize_q6_k(data: &[u8], num_elements: usize) -> Vec<f16> {
        // Q6_K block structure:
        // - 256 6-bit values (packed)
        // - 16 scales (int8_t, quantized)
        // - 1 global scale d (f16)
        const BLOCK_SIZE: usize = 256;
        const SCALES_COUNT: usize = 16;
        const VALUES_PER_SCALE: usize = BLOCK_SIZE / SCALES_COUNT; // 16

        // Block layout:
        // ql[128]  : lower 4 bits of 6-bit values
        // qh[64]   : upper 2 bits of 6-bit values
        // scales[16]: 16 int8_t quantized scales
        // d[2]     : 1 f16 global scale
        const BLOCK_BYTES: usize = 128 + 64 + 16 + 2; // = 210 bytes

        let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
        let mut result = Vec::with_capacity(num_elements);

        for block_idx in 0..num_blocks {
            let block_offset = block_idx * BLOCK_BYTES;
            if block_offset + BLOCK_BYTES > data.len() {
                break;
            }

            // Read ql (lower 4 bits)
            let ql_offset = block_offset;
            // Read qh (upper 2 bits)
            let qh_offset = block_offset + 128;
            // Read scales (int8_t, quantized)
            let scales_offset = block_offset + 128 + 64;
            // Read global scale d (f16)
            let d_offset = block_offset + 128 + 64 + 16;

            let d_bytes = [data[d_offset], data[d_offset + 1]];
            let d = f16::from_le_bytes(d_bytes);

            // Read 16 quantized scales (int8_t)
            let mut scales = [0i8; SCALES_COUNT];
            for i in 0..SCALES_COUNT {
                scales[i] = data[scales_offset + i] as i8;
            }

            // Dequantize 256 values
            for i in 0..BLOCK_SIZE {
                let ql_idx = i / 2; // 2 values per byte
                let qh_idx = i / 4; // 4 values per byte
                let scale_idx = i / VALUES_PER_SCALE;

                // Get lower 4 bits
                let ql_byte = data[ql_offset + ql_idx];
                let ql = if i % 2 == 0 {
                    (ql_byte & 0x0F) as i8
                } else {
                    ((ql_byte >> 4) & 0x0F) as i8
                };

                // Get upper 2 bits
                let qh_byte = data[qh_offset + qh_idx];
                let shift = (i % 4) * 2;
                let qh = ((qh_byte >> shift) & 0x03) as i8;

                // Combine to 6-bit value (0..63) and center to (-32..31)
                let q6 = ((qh << 4) | ql) - 32;

                // Dequantize: value = d * (scale[i/16] as f32) * q6
                // Scales are quantized int8_t values
                let scale = f16::from_f32(scales[scale_idx] as f32);
                let value = d * scale * f16::from_f32(q6 as f32);
                result.push(value);
            }
        }

        result.truncate(num_elements);
        result
    }

    /// Clear the cache (free all cached weights)
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[GGUFWeightCache] Cache cleared");
        }
    }

    /// Get cache statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        let cache = self.cache.lock().unwrap();
        (cache.len(), cache.cap().get())
    }

    /// Check if a weight exists in the file
    pub fn contains(&self, name: &str) -> bool {
        self.metadata.contains_key(name)
    }

    /// List all available weight names
    pub fn weight_names(&self) -> Vec<String> {
        self.metadata.keys().cloned().collect()
    }
}

impl<T: FloatType> Clone for GGUFWeightCache<T> {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            reader: Arc::clone(&self.reader),
            cache: Arc::clone(&self.cache),
            metadata: self.metadata.clone(),
            device: self.device.clone(),
        }
    }
}

impl<T: FloatType> crate::model::LazyWeightLoader<T> for GGUFWeightCache<T> {
    fn get_weight(&self, name: &str) -> TensorResult<Tensor<T>> {
        self.get_weight(name)
    }

    fn cache_stats(&self) -> (usize, usize) {
        self.cache_stats()
    }

    fn contains(&self, name: &str) -> bool {
        self.contains(name)
    }

    fn weight_names(&self) -> Vec<String> {
        self.weight_names()
    }

    fn clear_cache(&self) {
        self.clear_cache()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::MetalDevice;

    #[test]
    #[ignore] // Requires GGUF file
    fn test_gguf_weight_cache_lazy_loading() {
        let device = MetalDevice::new().unwrap();
        let home = std::env::var("HOME").unwrap();
        let path = format!("{}/.llm/models/tinyllama-1.1b-chat-q4_0.gguf", home);
        let cache = GGUFWeightCache::<half::f16>::new(
            &path,
            device,
            5, // Only cache 5 weights
        ).unwrap();

        // First access: cache miss
        let w1 = cache.get_weight("blk.0.attn_q.weight").unwrap();
        assert_eq!(cache.cache_stats(), (1, 5));

        // Second access: cache hit
        let w2 = cache.get_weight("blk.0.attn_q.weight").unwrap();
        assert_eq!(cache.cache_stats(), (1, 5));

        // Load more weights until LRU eviction
        for i in 0..6 {
            cache.get_weight(&format!("blk.{}.attn_q.weight", i)).ok();
        }

        // Cache should be at capacity with LRU eviction
        assert_eq!(cache.cache_stats().0, 5);
    }
}
