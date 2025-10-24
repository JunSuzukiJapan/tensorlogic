//! Runtime value types for TensorLogic interpreter

use crate::tensor::Tensor;
use crate::model::Model;
use super::{RuntimeError, RuntimeResult, DISPLAY_LIMIT};

/// Runtime value
#[derive(Debug, Clone)]
pub enum Value {
    Tensor(Tensor),
    Boolean(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Model(Model),
    Tokenizer(std::sync::Arc<crate::tokenizer::Tokenizer>),
    TokenIds(Vec<u32>),
    /// Meta-type: represents an entity type
    Type(String),
    Void,
}

impl Value {
    /// Convert to tensor if possible
    pub fn as_tensor(&self) -> RuntimeResult<&Tensor> {
        match self {
            Value::Tensor(t) => Ok(t),
            _ => Err(RuntimeError::TypeError(format!(
                "Expected tensor, found {:?}",
                self
            ))),
        }
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
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Value::Tensor(t) => {
                // Display tensor in a compact format
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
            Value::Boolean(b) => write!(f, "{}", b),
            Value::Integer(i) => write!(f, "{}", i),
            Value::Float(fl) => write!(f, "{}", fl),
            Value::String(s) => write!(f, "{}", s),
            Value::Model(m) => write!(f, "Model({:?})", m.metadata.format),
            Value::Tokenizer(_) => write!(f, "Tokenizer"),
            Value::TokenIds(ids) => write!(f, "TokenIds({:?})", ids),
            Value::Type(type_name) => write!(f, "Type({})", type_name),
            Value::Void => write!(f, "()"),
        }
    }
}
