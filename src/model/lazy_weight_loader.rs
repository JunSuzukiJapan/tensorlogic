//! Common trait for lazy weight loading with caching

use crate::error::TensorResult;
use crate::tensor::{FloatType, Tensor};
use std::sync::Arc;

/// Trait for lazy weight loading with LRU caching
///
/// This trait defines a common interface for loading model weights on-demand
/// from various storage formats (SafeTensors, GGUF, etc.) with caching.
pub trait LazyWeightLoader<T: FloatType>: Clone + Send + Sync {
    /// Get a weight tensor by name, loading from storage if not cached
    ///
    /// # Arguments
    /// * `name` - Weight name (e.g., "model.layers.0.self_attn.q_proj.weight")
    ///
    /// # Returns
    /// Arc<Tensor> loaded from cache or storage (Arc avoids cloning GPU data)
    fn get_weight(&self, name: &str) -> TensorResult<Arc<Tensor<T>>>;

    /// Get cache statistics (cached_count, capacity)
    ///
    /// # Returns
    /// Tuple of (number of cached weights, cache capacity)
    fn cache_stats(&self) -> (usize, usize);

    /// Check if a weight exists in the storage
    ///
    /// # Arguments
    /// * `name` - Weight name to check
    ///
    /// # Returns
    /// True if the weight exists, false otherwise
    fn contains(&self, name: &str) -> bool;

    /// List all available weight names in storage
    ///
    /// # Returns
    /// Vector of all weight names
    fn weight_names(&self) -> Vec<String>;

    /// Clear all cached weights from memory
    fn clear_cache(&self);
}
