//! Runtime value types for TensorLogic interpreter

use crate::tensor::{Tensor, TokenIdArray, TensorIO};
use crate::model::Model;
use crate::ast::StructType;
use half::f16;
use std::collections::HashMap;
use super::{RuntimeError, RuntimeResult, DISPLAY_LIMIT};

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
    Model(Model),
    Tokenizer(std::sync::Arc<crate::tokenizer::Tokenizer>),
    TokenIds(Vec<u32>),
    /// Token ID array with integer precision (no f16 precision loss)
    TokenIdArray(TokenIdArray),
    /// Meta-type: represents an entity type
    Type(String),
    /// Struct instance with type and field values
    Struct {
        struct_type: StructType,
        fields: HashMap<String, Box<Value>>,
    },
    Void,
}

impl Value {
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
    /// Prefer as_tensor_f16() or as_tensor_f32() for clarity
    pub fn as_tensor(&self) -> RuntimeResult<&Tensor<f16>> {
        self.as_tensor_f16()
    }

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

    /// Convert to struct if possible
    pub fn as_struct(&self) -> RuntimeResult<(&StructType, &HashMap<String, Box<Value>>)> {
        match self {
            Value::Struct { struct_type, fields } => Ok((struct_type, fields)),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected struct, found {:?}",
                self
            ))),
        }
    }

    /// Get field from struct
    pub fn get_field(&self, field_name: &str) -> RuntimeResult<&Value> {
        match self {
            Value::Struct { fields, .. } => {
                fields.get(field_name)
                    .map(|v| v.as_ref())
                    .ok_or_else(|| RuntimeError::TypeError(format!(
                        "Field '{}' not found in struct",
                        field_name
                    )))
            }
            _ => Err(RuntimeError::TypeError(format!(
                "Cannot access field '{}' on non-struct value",
                field_name
            ))),
        }
    }
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::TensorF16(t) => {
                // Display f16 tensor in a compact format
                let data = t.to_vec();
                if data.len() <= DISPLAY_LIMIT {
                    write!(f, "[")?;
                    for (i, val) in data.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:.4}", val.to_f32())?;
                    }
                    write!(f, "]")
                } else {
                    write!(f, "[{:.4}, {:.4}, ..., {:.4}] (len={})",
                        data[0].to_f32(), data[1].to_f32(), data[data.len()-1].to_f32(), data.len())
                }
            }
            Value::TensorF32(t) => {
                // Display f32 tensor in a compact format
                let data = t.to_vec();
                if data.len() <= DISPLAY_LIMIT {
                    write!(f, "[")?;
                    for (i, val) in data.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{:.4}", val)?;
                    }
                    write!(f, "]")
                } else {
                    write!(f, "[{:.4}, {:.4}, ..., {:.4}] (len={})",
                        data[0], data[1], data[data.len()-1], data.len())
                }
            }
            Value::Boolean(b) => write!(f, "{}", b),
            Value::Integer(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::String(s) => write!(f, "{}", s),
            Value::Model(m) => write!(f, "Model({:?})", m.metadata.format),
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
            Value::Struct { struct_type, fields } => {
                write!(f, "{} {{", struct_type.name.as_str())?;
                let mut first = true;
                for (name, value) in fields.iter() {
                    if !first {
                        write!(f, ", ")?;
                    }
                    first = false;
                    write!(f, "{}: {}", name, value)?;
                }
                write!(f, "}}")
            }
            Value::Void => write!(f, "()"),
        }
    }
}
