//! Weight cache with lazy loading using memory-mapped files
//!
//! Implements llama.cpp-style mmap + LRU cache for efficient memory usage.
//! Weights are loaded on-demand and evicted when not recently used.

use crate::device::MetalDevice;
use crate::error::{TensorError, TensorResult};
use crate::tensor::{FloatType, Tensor, TensorCreation};
use lru::LruCache;
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Metadata for a weight tensor in the memory-mapped file
#[derive(Debug, Clone)]
struct WeightMetadata {
    /// Offset in bytes from start of mmap
    offset: usize,
    /// Size in bytes
    size: usize,
    /// Shape of the tensor
    shape: Vec<usize>,
    /// Data type name (e.g., "F32", "F16")
    dtype: String,
}

/// Weight cache with lazy loading from memory-mapped safetensors file
///
/// Uses LRU eviction to keep only recently-used weights in memory.
/// This allows running large models (e.g., f32 weights) that don't fit
/// entirely in RAM.
///
/// # Example
/// ```rust,ignore
/// let cache = WeightCache::new("model.safetensors", device, 10)?;
///
/// // Load weight on-demand (from mmap if not cached)
/// let weight = cache.get_weight("layer.0.weight")?;
///
/// // Weight is cached for subsequent access
/// let weight2 = cache.get_weight("layer.0.weight")?; // Fast: from cache
/// ```
#[derive(Debug)]
pub struct WeightCache<T: FloatType> {
    /// Memory-mapped safetensors file (zero-copy, OS managed)
    mmap: Arc<Mmap>,

    /// LRU cache of loaded tensors (only keeps recently used)
    cache: Arc<Mutex<LruCache<String, Tensor<T>>>>,

    /// Metadata: weight name -> (offset, size, shape)
    metadata: HashMap<String, WeightMetadata>,

    /// Metal device for GPU tensor creation
    device: MetalDevice,

    /// SafeTensors header offset (data starts after this)
    data_offset: usize,
}

impl<T: FloatType> WeightCache<T> {
    /// Create a new weight cache from a safetensors file
    ///
    /// # Arguments
    /// * `path` - Path to safetensors file
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
        // Memory-map the safetensors file (zero-copy, OS managed)
        let file = File::open(path.as_ref()).map_err(|e| {
            TensorError::LoadError(format!("Failed to open safetensors file: {}", e))
        })?;

        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| {
                TensorError::LoadError(format!("Failed to mmap safetensors file: {}", e))
            })?
        };

        // Parse safetensors header to get metadata
        let safetensors = SafeTensors::deserialize(&mmap).map_err(|e| {
            TensorError::LoadError(format!("Failed to parse safetensors: {}", e))
        })?;

        // Extract metadata for each tensor
        let mut metadata = HashMap::new();
        for name in safetensors.names() {
            let tensor_view = safetensors.tensor(&name).map_err(|e| {
                TensorError::LoadError(format!("Failed to get tensor {}: {}", name, e))
            })?;

            let data_slice = tensor_view.data();
            let offset = data_slice.as_ptr() as usize - mmap.as_ptr() as usize;

            metadata.insert(
                name.to_string(),
                WeightMetadata {
                    offset,
                    size: data_slice.len(),
                    shape: tensor_view.shape().to_vec(),
                    dtype: format!("{:?}", tensor_view.dtype()),
                },
            );
        }

        // Data offset (after header) - use first tensor's offset as reference
        let data_offset = if let Some(meta) = metadata.values().next() {
            meta.offset
        } else {
            0
        };

        Ok(Self {
            mmap: Arc::new(mmap),
            cache: Arc::new(Mutex::new(LruCache::new(
                NonZeroUsize::new(cache_capacity).unwrap(),
            ))),
            metadata,
            device,
            data_offset,
        })
    }

    /// Get a weight tensor, loading from mmap if not cached
    ///
    /// # Arguments
    /// * `name` - Weight name (e.g., "model.layers.0.self_attn.q_proj.weight")
    ///
    /// # Returns
    /// Tensor loaded from cache or mmap
    ///
    /// # Performance
    /// - Cache hit: O(1) - returns cached tensor
    /// - Cache miss: O(n) - loads from mmap and copies to GPU
    pub fn get_weight(&self, name: &str) -> TensorResult<Tensor<T>> {
        // Check cache first
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(tensor) = cache.get(name) {
                if std::env::var("TL_DEBUG").is_ok() {
                    eprintln!("[WeightCache] Cache HIT: {}", name);
                }
                return Ok(tensor.clone());
            }
        }

        // Cache miss - load from mmap
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[WeightCache] Cache MISS: {} (loading from mmap)", name);
        }

        let meta = self.metadata.get(name).ok_or_else(|| {
            TensorError::LoadError(format!("Weight not found: {}", name))
        })?;

        // Verify dtype matches T
        if !Self::dtype_matches(&meta.dtype) {
            return Err(TensorError::LoadError(format!(
                "Weight {} has dtype {} but expected {}",
                name, meta.dtype, std::any::type_name::<T>()
            )));
        }

        // Read data from mmap (zero-copy, just get slice)
        let data_slice = &self.mmap[meta.offset..meta.offset + meta.size];

        // Convert bytes to T values
        let num_elements: usize = meta.shape.iter().product();
        let values: Vec<T> = unsafe {
            let ptr = data_slice.as_ptr() as *const T;
            std::slice::from_raw_parts(ptr, num_elements).to_vec()
        };

        // Create tensor on GPU
        let tensor = Tensor::from_vec_gpu(&self.device, values, meta.shape.clone())?;

        // Store in cache
        {
            let mut cache = self.cache.lock().unwrap();
            cache.put(name.to_string(), tensor.clone());

            if std::env::var("TL_DEBUG").is_ok() {
                eprintln!(
                    "[WeightCache] Cached {} ({} elements, cache size: {}/{})",
                    name,
                    num_elements,
                    cache.len(),
                    cache.cap()
                );
            }
        }

        Ok(tensor)
    }

    /// Check if dtype string matches T
    fn dtype_matches(dtype: &str) -> bool {
        if T::is_f32() {
            dtype == "F32"
        } else if T::is_f16() {
            dtype == "F16"
        } else {
            false
        }
    }

    /// Clear the cache (free all cached weights)
    pub fn clear_cache(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
        if std::env::var("TL_DEBUG").is_ok() {
            eprintln!("[WeightCache] Cache cleared");
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

impl<T: FloatType> Clone for WeightCache<T> {
    fn clone(&self) -> Self {
        Self {
            mmap: Arc::clone(&self.mmap),
            cache: Arc::clone(&self.cache),
            metadata: self.metadata.clone(),
            device: self.device.clone(),
            data_offset: self.data_offset,
        }
    }
}

impl<T: FloatType> crate::model::LazyWeightLoader<T> for WeightCache<T> {
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
    use half::f16;

    #[test]
    #[ignore] // Requires safetensors file
    fn test_weight_cache_lazy_loading() {
        let device = MetalDevice::new().unwrap();
        let cache = WeightCache::<f16>::new(
            "models/TinyLlama_f16.safetensors",
            device,
            5, // Only cache 5 weights
        ).unwrap();

        // First access: cache miss
        let w1 = cache.get_weight("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(cache.cache_stats(), (1, 5));

        // Second access: cache hit
        let w2 = cache.get_weight("model.layers.0.self_attn.q_proj.weight").unwrap();
        assert_eq!(cache.cache_stats(), (1, 5));

        // Load more weights until LRU eviction
        for i in 0..6 {
            cache.get_weight(&format!("model.layers.{}.self_attn.q_proj.weight", i)).ok();
        }

        // Cache should be at capacity with LRU eviction
        assert_eq!(cache.cache_stats().0, 5);
    }
}
