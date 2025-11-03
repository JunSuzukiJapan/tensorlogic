//! Runtime value types for TensorLogic interpreter

use crate::tensor::{Tensor, TokenIdArray, TensorAccessors};
use crate::model::{Model, WeightCache, GGUFWeightCache};
use half::f16;
use super::{RuntimeError, RuntimeResult, DISPLAY_LIMIT};

/// Model layer collection (e.g., model.blk returns this)
#[derive(Debug, Clone)]
pub struct ModelLayerCollection<T: crate::tensor::FloatType> {
    pub layers: std::collections::HashMap<usize, std::collections::HashMap<String, Tensor<T>>>,
    pub model_metadata: crate::model::ModelMetadata,
}

impl<T: crate::tensor::FloatType> ModelLayerCollection<T> {
    /// Get a specific layer by index
    pub fn get_layer(&self, index: usize) -> Option<ModelLayer<T>> {
        self.layers.get(&index).map(|features_map| {
            // Group features by name (e.g., "attn_norm.weight" -> feature "attn_norm", property "weight")
            let mut features: std::collections::HashMap<String, ModelFeature<T>> = std::collections::HashMap::new();
            
            for (full_path, tensor) in features_map {
                let parts: Vec<&str> = full_path.split('.').collect();
                if parts.len() >= 2 {
                    let feature_name = parts[0].to_string();
                    let property_name = parts[1..].join(".");
                    
                    features.entry(feature_name.clone())
                        .or_insert_with(|| ModelFeature {
                            name: feature_name,
                            properties: std::collections::HashMap::new(),
                        })
                        .properties.insert(property_name, tensor.clone());
                } else if parts.len() == 1 {
                    // Single property without sub-path
                    let feature_name = parts[0].to_string();
                    features.insert(feature_name.clone(), ModelFeature {
                        name: feature_name,
                        properties: {
                            let mut props = std::collections::HashMap::new();
                            props.insert("".to_string(), tensor.clone());
                            props
                        },
                    });
                }
            }
            
            ModelLayer {
                index,
                features,
            }
        })
    }
}

/// Model layer (e.g., model.blk[0] returns this)
#[derive(Debug, Clone)]
pub struct ModelLayer<T: crate::tensor::FloatType> {
    pub index: usize,
    pub features: std::collections::HashMap<String, ModelFeature<T>>,
}

impl<T: crate::tensor::FloatType> ModelLayer<T> {
    /// Get a specific feature by name (e.g., "attn_norm")
    pub fn get_feature(&self, feature_name: &str) -> Option<&ModelFeature<T>> {
        self.features.get(feature_name)
    }
}

/// Model feature (e.g., model.blk[0].attn_norm returns this)
#[derive(Debug, Clone)]
pub struct ModelFeature<T: crate::tensor::FloatType> {
    pub name: String,
    pub properties: std::collections::HashMap<String, Tensor<T>>,
}

impl<T: crate::tensor::FloatType> ModelFeature<T> {
    /// Get a specific property tensor (e.g., "weight", "bias")
    pub fn get_property(&self, property_name: &str) -> Option<&Tensor<T>> {
        self.properties.get(property_name)
    }
}

/// Runtime value
#[derive(Debug, Clone)]
pub enum Value {
    /// Tensor with f16 precision (float16)
    TensorF16(Tensor<f16>),
    /// Tensor with f32 precision (float32)
    TensorF32(Tensor<f32>),
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    /// Model with f16 tensors
    ModelF16(Model<f16>),
    /// Model with f32 tensors
    ModelF32(Model<f32>),
    /// Model layer collection (f16)
    ModelLayerCollectionF16(ModelLayerCollection<f16>),
    /// Model layer collection (f32)
    ModelLayerCollectionF32(ModelLayerCollection<f32>),
    /// Model layer (f16)
    ModelLayerF16(ModelLayer<f16>),
    /// Model layer (f32)
    ModelLayerF32(ModelLayer<f32>),
    /// Model feature (f16)
    ModelFeatureF16(ModelFeature<f16>),
    /// Model feature (f32)
    ModelFeatureF32(ModelFeature<f32>),
    Tokenizer(std::sync::Arc<crate::tokenizer::Tokenizer>),
    TokenIds(Vec<u32>),
    /// Token ID array with integer precision (no f16 precision loss)
    TokenIdArray(TokenIdArray),
    /// Meta-type: represents an entity type
    Type(String),
    /// KV Cache for transformer attention layers (f16)
    KVCacheF16(std::sync::Arc<std::sync::Mutex<crate::model::llama::Cache<half::f16>>>),
    /// KV Cache for transformer attention layers (f32)
    KVCacheF32(std::sync::Arc<std::sync::Mutex<crate::model::llama::Cache<f32>>>),
    /// Weight cache for lazy loading (f16)
    WeightCacheF16(WeightCache<f16>),
    /// Weight cache for lazy loading (f32)
    WeightCacheF32(WeightCache<f32>),
    /// GGUF weight cache for lazy loading (f16)
    GGUFWeightCacheF16(GGUFWeightCache<f16>),
    /// GGUF weight cache for lazy loading (f32)
    GGUFWeightCacheF32(GGUFWeightCache<f32>),
    Void,
}

impl Value {
    /// Get the type name of this value
    pub fn type_name(&self) -> &'static str {
        match self {
            Value::TensorF16(_) => "TensorF16",
            Value::TensorF32(_) => "TensorF32",
            Value::Boolean(_) => "Boolean",
            Value::Integer(_) => "Integer",
            Value::Float(_) => "Float",
            Value::String(_) => "String",
            Value::ModelF16(_) => "ModelF16",
            Value::ModelF32(_) => "ModelF32",
            Value::ModelLayerCollectionF16(_) => "ModelLayerCollectionF16",
            Value::ModelLayerCollectionF32(_) => "ModelLayerCollectionF32",
            Value::ModelLayerF16(_) => "ModelLayerF16",
            Value::ModelLayerF32(_) => "ModelLayerF32",
            Value::ModelFeatureF16(_) => "ModelFeatureF16",
            Value::ModelFeatureF32(_) => "ModelFeatureF32",
            Value::Tokenizer(_) => "Tokenizer",
            Value::TokenIds(_) => "TokenIds",
            Value::TokenIdArray(_) => "TokenIdArray",
            Value::Type(_) => "Type",
            Value::KVCacheF16(_) => "KVCache",
            Value::KVCacheF32(_) => "KVCache",
            Value::WeightCacheF16(_) => "WeightCacheF16",
            Value::WeightCacheF32(_) => "WeightCacheF32",
            Value::GGUFWeightCacheF16(_) => "GGUFWeightCacheF16",
            Value::GGUFWeightCacheF32(_) => "GGUFWeightCacheF32",
            Value::Void => "Void",
        }
    }

    /// Convert to f16 tensor if possible
    pub fn as_tensor_f16(&self) -> RuntimeResult<&Tensor<f16>> {
        match self {
            Value::TensorF16(t) => Ok(t),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected f16 tensor, found {:?}",
                self
            ))),
        }
    }

    /// Convert to f32 tensor if possible
    pub fn as_tensor_f32(&self) -> RuntimeResult<&Tensor<f32>> {
        match self {
            Value::TensorF32(t) => Ok(t),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected f32 tensor, found {:?}",
                self
            ))),
        }
    }

    /// Convert to tensor (f16) - backward compatibility helper
    /// **WARNING**: This assumes f16. Prefer as_tensor_f16() or as_tensor_f32() for clarity.
    /// Functions using this need to be converted to support both f16 and f32.
    #[deprecated(note = "Use as_tensor_f16() or as_tensor_f32() for explicit type handling")]
    // pub fn as_tensor(&self) -> RuntimeResult<&Tensor<f16>> {
    //     self.as_tensor_f16()
    // }

    /// Convert to float if possible
    pub fn as_float(&self) -> RuntimeResult<f64> {
        match self {
            Value::Float(f) => Ok(*f),
            Value::Integer(i) => Ok(*i as f64),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected float, found {:?}",
                self
            ))),
        }
    }

    /// Convert to boolean if possible
    pub fn as_bool(&self) -> RuntimeResult<bool> {
        match self {
            Value::Boolean(b) => Ok(*b),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected boolean, found {:?}",
                self
            ))),
        }
    }

    /// Convert to integer if possible
    pub fn as_integer(&self) -> RuntimeResult<i64> {
        match self {
            Value::Integer(i) => Ok(*i),
            Value::Float(f) => Ok(*f as i64),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected integer, found {:?}",
                self
            ))),
        }
    }

    /// Convert to token ID array if possible
    pub fn as_token_id_array(&self) -> RuntimeResult<&TokenIdArray> {
        match self {
            Value::TokenIdArray(arr) => Ok(arr),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected TokenIdArray, found {:?}",
                self
            ))),
        }
    }

    /// Convert to mutable token ID array if possible
    pub fn as_token_id_array_mut(&mut self) -> RuntimeResult<&mut TokenIdArray> {
        match self {
            Value::TokenIdArray(arr) => Ok(arr),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected TokenIdArray, found {:?}",
                self
            ))),
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::TensorF16(t) => {
                // Display tensor shape only to avoid GPU->CPU transfer
                write!(f, "Tensor<f16>(shape={:?})", t.dims())
            }
            Value::TensorF32(t) => {
                // Display tensor shape only to avoid GPU->CPU transfer
                write!(f, "Tensor<f32>(shape={:?})", t.dims())
            }
            Value::Boolean(b) => write!(f, "{}", b),
            Value::Integer(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::String(s) => write!(f, "{}", s),
            Value::ModelF16(m) => write!(f, "Model<f16>({:?})", m.metadata.format),
            Value::ModelF32(m) => write!(f, "Model({:?})", m.metadata.format),
            Value::ModelLayerCollectionF16(c) => write!(f, "ModelLayerCollection<f16>(layers={})", c.layers.len()),
            Value::ModelLayerCollectionF32(c) => write!(f, "ModelLayerCollection<f32>(layers={})", c.layers.len()),
            Value::ModelLayerF16(l) => write!(f, "ModelLayer<f16>[{}](features={})", l.index, l.features.len()),
            Value::ModelLayerF32(l) => write!(f, "ModelLayer<f32>[{}](features={})", l.index, l.features.len()),
            Value::ModelFeatureF16(feat) => write!(f, "ModelFeature<f16>({}, props={})", feat.name, feat.properties.len()),
            Value::ModelFeatureF32(feat) => write!(f, "ModelFeature<f32>({}, props={})", feat.name, feat.properties.len()),
            Value::Tokenizer(_) => write!(f, "Tokenizer"),
            Value::TokenIds(ids) => write!(f, "TokenIds({:?})", ids),
            Value::TokenIdArray(arr) => {
                let data = arr.data();
                if data.len() <= DISPLAY_LIMIT {
                    write!(f, "[")?;
                    for (i, val) in data.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:.4}", *val as f64)?;
                    }
                    write!(f, "]")
                } else {
                    write!(f, "[{:.4}, {:.4}, ..., {:.4}] (len={})",
                        data[0] as f64, data[1] as f64, data[data.len()-1] as f64, data.len())
                }
            }
            Value::Type(type_name) => write!(f, "Type({})", type_name),
            Value::KVCacheF16(cache) => {
                let c = cache.lock().unwrap();
                write!(f, "KVCache(layers={})", c.kvs.len())
            }
            Value::KVCacheF32(cache) => {
                let c = cache.lock().unwrap();
                write!(f, "KVCache(layers={})", c.kvs.len())
            }
            Value::WeightCacheF16(cache) => {
                let (cached, capacity) = cache.cache_stats();
                write!(f, "WeightCache<f16>(cached={}/{}, weights={})",
                       cached, capacity, cache.weight_names().len())
            }
            Value::WeightCacheF32(cache) => {
                let (cached, capacity) = cache.cache_stats();
                write!(f, "WeightCache<f32>(cached={}/{}, weights={})",
                       cached, capacity, cache.weight_names().len())
            }
            Value::GGUFWeightCacheF16(cache) => {
                let (cached, capacity) = cache.cache_stats();
                write!(f, "GGUFWeightCache<f16>(cached={}/{}, weights={})",
                       cached, capacity, cache.weight_names().len())
            }
            Value::GGUFWeightCacheF32(cache) => {
                let (cached, capacity) = cache.cache_stats();
                write!(f, "GGUFWeightCache<f32>(cached={}/{}, weights={})",
                       cached, capacity, cache.weight_names().len())
            }
            Value::Void => write!(f, "()"),
        }
    }
}

/// Trait for converting tensor results to Value
pub trait ToValue {
    fn to_value(self) -> Value;
}

impl ToValue for Tensor<f16> {
    fn to_value(self) -> Value {
        Value::TensorF16(self)
    }
}

impl ToValue for Tensor<f32> {
    fn to_value(self) -> Value {
        Value::TensorF32(self)
    }
}
