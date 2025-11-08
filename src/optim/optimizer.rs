//! Base optimizer trait and common data structures

use crate::error::TensorResult;
use crate::tensor::{FloatType, TensorAccessors, TensorCreation, TensorIO};
use crate::tensor::Tensor;
use crate::autograd::AutogradContext;
use std::collections::HashMap;

/// Parameter group with learning rate and regularization settings
#[derive(Clone, Debug)]
pub struct ParamGroup {
    /// Parameters to optimize
    pub params: Vec<Tensor>,

    /// Learning rate for this group
    pub lr: f32,

    /// Weight decay (L2 regularization)
    pub weight_decay: f32,

    /// Custom options for specific optimizers
    pub options: HashMap<String, f32>,
}

impl ParamGroup {
    /// Create a new parameter group with default settings
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self {
            params,
            lr,
            weight_decay: 0.0,
            options: HashMap::new(),
        }
    }

    /// Create a parameter group with weight decay
    pub fn with_weight_decay(params: Vec<Tensor>, lr: f32, weight_decay: f32) -> Self {
        Self {
            params,
            lr,
            weight_decay,
            options: HashMap::new(),
        }
    }
}

/// Optimizer state for saving/loading
#[derive(Clone, Debug)]
pub struct OptimizerState {
    /// Step count
    pub step: usize,

    /// Per-parameter state (e.g., momentum buffers)
    pub param_state: HashMap<usize, HashMap<String, Vec<f32>>>,
}

impl OptimizerState {
    pub fn new() -> Self {
        Self {
            step: 0,
            param_state: HashMap::new(),
        }
    }
}

/// Base optimizer trait
pub trait Optimizer {
    /// Perform a single optimization step (parameter update)
    fn step(&mut self) -> TensorResult<()>;

    /// Zero out all parameter gradients
    fn zero_grad(&mut self);

    /// Get current learning rate
    fn get_lr(&self) -> f32;

    /// Set learning rate for all parameter groups
    fn set_lr(&mut self, lr: f32);

    /// Get optimizer state for saving
    fn state_dict(&self) -> OptimizerState;

    /// Load optimizer state
    fn load_state_dict(&mut self, state: OptimizerState) -> TensorResult<()>;

    /// Add a parameter group
    fn add_param_group(&mut self, group: ParamGroup);

    /// Get number of parameter groups
    fn num_param_groups(&self) -> usize;

    /// Get reference to parameters
    fn params(&self) -> &[Tensor];
}

/// Helper function to create a scalar tensor
#[allow(dead_code)]
pub(crate) fn scalar_tensor(value: f32) -> TensorResult<Tensor> {
    Tensor::from_vec(vec![half::f16::from_f32(value)], vec![1])
}

/// Helper function to multiply tensor by scalar (returns new tensor)
pub(crate) fn mul_scalar(tensor: &Tensor, scalar: f32) -> TensorResult<Tensor> {
    // Broadcast scalar to tensor shape
    let mut result = tensor.clone();
    for i in 0..tensor.numel() {
        let mut data = result.sync_and_read();
        data[i] = half::f16::from_f32(data[i].to_f32() * scalar);
        result = Tensor::from_vec(data, tensor.dims().to_vec())?;
    }
    Ok(result)
}

/// Helper function to update parameter from registry
pub(crate) fn update_param_from_registry(param: &mut Tensor) -> TensorResult<()> {
    if let Some(node_id) = param.grad_node() {
        if let Some(updated) = AutogradContext::get_tensor_generic::<half::f16>(node_id) {
            *param = updated;
        }
    }
    Ok(())
}

/// Helper function to register parameter after modification
pub(crate) fn register_param(param: &Tensor) -> TensorResult<()> {
    if let Some(node_id) = param.grad_node() {
        AutogradContext::register_tensor_generic(node_id, param.clone());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn test_param_group_creation() {
        let param = Tensor::from_vec(vec![f16::ONE; 10], vec![10]).unwrap();
        let group = ParamGroup::new(vec![param], 0.01);

        assert_eq!(group.lr, 0.01);
        assert_eq!(group.weight_decay, 0.0);
        assert_eq!(group.params.len(), 1);
    }

    #[test]
    fn test_param_group_with_weight_decay() {
        let param = Tensor::from_vec(vec![f16::ONE; 10], vec![10]).unwrap();
        let group = ParamGroup::with_weight_decay(vec![param], 0.01, 0.0001);

        assert_eq!(group.lr, 0.01);
        assert_eq!(group.weight_decay, 0.0001);
    }

    #[test]
    fn test_optimizer_state() {
        let state = OptimizerState::new();
        assert_eq!(state.step, 0);
        assert!(state.param_state.is_empty());
    }
}
