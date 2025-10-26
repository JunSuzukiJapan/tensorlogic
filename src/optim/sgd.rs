//! Stochastic Gradient Descent (SGD) optimizer with momentum support

use crate::error::{TensorError, TensorResult};
use crate::tensor::{FloatType, TensorAccessors, TensorCreation, TensorIO, TensorAutograd};
use crate::tensor::Tensor;
use super::optimizer::{Optimizer, ParamGroup, OptimizerState, mul_scalar, update_param_from_registry, register_param};
use std::collections::HashMap;

/// SGD optimizer with optional momentum
pub struct SGD {
    /// Parameter groups
    param_groups: Vec<ParamGroup>,

    /// Momentum coefficient (0.0 for no momentum)
    momentum: f32,

    /// Dampening for momentum
    dampening: f32,

    /// Use Nesterov momentum
    nesterov: bool,

    /// Velocity buffers for momentum (param_index -> velocity)
    velocity_buffers: HashMap<usize, Tensor>,

    /// Parameter index counter
    param_counter: usize,
}

impl SGD {
    /// Get mutable access to parameters (for direct manipulation in training loop)
    pub fn get_params_mut(&mut self) -> &mut Vec<Tensor> {
        &mut self.param_groups[0].params
    }

    /// Create a new SGD optimizer with default settings
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        Self {
            param_groups: vec![ParamGroup::new(params, lr)],
            momentum: 0.0,
            dampening: 0.0,
            nesterov: false,
            velocity_buffers: HashMap::new(),
            param_counter: 0,
        }
    }

    /// Create SGD with momentum
    pub fn with_momentum(params: Vec<Tensor>, lr: f32, momentum: f32) -> Self {
        Self {
            param_groups: vec![ParamGroup::new(params, lr)],
            momentum,
            dampening: 0.0,
            nesterov: false,
            velocity_buffers: HashMap::new(),
            param_counter: 0,
        }
    }

    /// Create SGD with full options
    pub fn with_options(
        params: Vec<Tensor>,
        lr: f32,
        momentum: f32,
        dampening: f32,
        nesterov: bool,
        weight_decay: f32,
    ) -> Self {
        Self {
            param_groups: vec![ParamGroup::with_weight_decay(params, lr, weight_decay)],
            momentum,
            dampening,
            nesterov,
            velocity_buffers: HashMap::new(),
            param_counter: 0,
        }
    }

}

impl Optimizer for SGD {
    fn step(&mut self) -> TensorResult<()> {
        let mut param_idx = 0;
        let momentum = self.momentum;
        let dampening = self.dampening;
        let nesterov = self.nesterov;

        for group in &mut self.param_groups {
            let lr = group.lr;
            let weight_decay = group.weight_decay;

            for param in &mut group.params {
                if !param.requires_grad() {
                    continue;
                }

                // Get gradient from registry
                update_param_from_registry(param)?;

                let grad = param.grad().ok_or_else(|| {
                    TensorError::InvalidOperation("No gradient available for parameter".to_string())
                })?;

                // Apply weight decay if needed
                let mut d_p = grad.clone();
                if weight_decay != 0.0 {
                    let wd = mul_scalar(param, weight_decay)?;
                    d_p = d_p.add(&wd)?;
                }

                // Apply momentum if configured
                if momentum != 0.0 {
                    if let Some(buf) = self.velocity_buffers.get_mut(&param_idx) {
                        // v_{t+1} = momentum * v_t + (1 - dampening) * d_p
                        let momentum_buf = mul_scalar(buf, momentum)?;
                        let grad_contrib = mul_scalar(&d_p, 1.0 - dampening)?;
                        *buf = momentum_buf.add(&grad_contrib)?;
                    } else {
                        // First time: initialize velocity buffer
                        self.velocity_buffers.insert(param_idx, d_p.clone());
                    }

                    let buf = self.velocity_buffers.get(&param_idx).unwrap();

                    if nesterov {
                        // Nesterov momentum: d_p = d_p + momentum * buf
                        let momentum_term = mul_scalar(buf, momentum)?;
                        d_p = d_p.add(&momentum_term)?;
                    } else {
                        // Standard momentum: use buf directly
                        d_p = buf.clone();
                    }
                }

                // Update parameter: param -= lr * d_p
                let update = mul_scalar(&d_p, lr)?;
                param.sub_(&update)?;

                // Re-register parameter in AutogradContext
                register_param(param)?;

                param_idx += 1;
            }
        }

        self.param_counter = param_idx;
        Ok(())
    }

    fn zero_grad(&mut self) {
        for group in &mut self.param_groups {
            for param in &mut group.params {
                param.zero_grad();
                // Re-register after modifying
                if let Some(node_id) = param.grad_node() {
                    crate::autograd::AutogradContext::register_tensor_generic(node_id, param.clone());
                }
            }
        }
    }

    fn get_lr(&self) -> f32 {
        self.param_groups.first().map(|g| g.lr).unwrap_or(0.0)
    }

    fn set_lr(&mut self, lr: f32) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn state_dict(&self) -> OptimizerState {
        let mut state = OptimizerState::new();

        for (param_idx, velocity) in &self.velocity_buffers {
            let mut param_state = HashMap::new();
            param_state.insert("velocity".to_string(), velocity.to_vec().iter().map(|&v| v.to_f32()).collect());
            state.param_state.insert(*param_idx, param_state);
        }

        state
    }

    fn load_state_dict(&mut self, state: OptimizerState) -> TensorResult<()> {
        for (param_idx, param_state) in state.param_state {
            if let Some(velocity_data) = param_state.get("velocity") {
                let velocity_f16: Vec<_> = velocity_data.iter().map(|&v| half::f16::from_f32(v)).collect();

                // Find the shape of this parameter
                let mut found_shape = None;
                let mut current_idx = 0;
                for group in &self.param_groups {
                    for param in &group.params {
                        if current_idx == param_idx {
                            found_shape = Some(param.dims().to_vec());
                            break;
                        }
                        current_idx += 1;
                    }
                    if found_shape.is_some() {
                        break;
                    }
                }

                if let Some(shape) = found_shape {
                    let velocity = Tensor::from_vec(velocity_f16, shape)?;
                    self.velocity_buffers.insert(param_idx, velocity);
                }
            }
        }

        Ok(())
    }

    fn add_param_group(&mut self, group: ParamGroup) {
        self.param_groups.push(group);
    }

    fn num_param_groups(&self) -> usize {
        self.param_groups.len()
    }

    fn params(&self) -> &[Tensor] {
        // Return parameters from the first param group
        if !self.param_groups.is_empty() {
            &self.param_groups[0].params
        } else {
            &[]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use crate::autograd::AutogradContext;

    #[test]
    fn test_sgd_creation() {
        let param = Tensor::from_vec(vec![f16::ONE; 10], vec![10]).unwrap();
        let optimizer = SGD::new(vec![param], 0.01);

        assert_eq!(optimizer.get_lr(), 0.01);
        assert_eq!(optimizer.momentum, 0.0);
        assert_eq!(optimizer.num_param_groups(), 1);
    }

    #[test]
    fn test_sgd_with_momentum() {
        let param = Tensor::from_vec(vec![f16::ONE; 10], vec![10]).unwrap();
        let optimizer = SGD::with_momentum(vec![param], 0.01, 0.9);

        assert_eq!(optimizer.momentum, 0.9);
    }

    #[test]
    fn test_sgd_zero_grad() {
        AutogradContext::clear();

        let mut param = Tensor::from_vec(vec![f16::from_f32(2.0); 5], vec![5]).unwrap();
        param.set_requires_grad(true);

        // Simulate gradient
        let grad = Tensor::from_vec(vec![f16::from_f32(1.0); 5], vec![5]).unwrap();
        param.set_grad(grad);

        let mut optimizer = SGD::new(vec![param.clone()], 0.1);
        optimizer.zero_grad();

        // Gradient should be cleared
        let updated_param = &optimizer.param_groups[0].params[0];
        assert!(updated_param.grad().is_none());
    }

    #[test]
    fn test_sgd_step_simple() {
        AutogradContext::clear();

        // Create parameter: x = [2.0, 2.0]
        let mut x = Tensor::from_vec(vec![f16::from_f32(2.0), f16::from_f32(2.0)], vec![2]).unwrap();
        x.set_requires_grad(true);

        // Set gradient: grad = [1.0, 1.0]
        let grad = Tensor::from_vec(vec![f16::from_f32(1.0), f16::from_f32(1.0)], vec![2]).unwrap();
        x.set_grad(grad);

        // Register in context
        AutogradContext::register_tensor_generic(x.grad_node().unwrap(), x.clone());

        // Create optimizer with lr=0.1
        let mut optimizer = SGD::new(vec![x.clone()], 0.1);

        // Step: x_new = x - lr * grad = 2.0 - 0.1 * 1.0 = 1.9
        optimizer.step().unwrap();

        // Get updated parameter
        let updated_x = &optimizer.param_groups[0].params[0];
        let values = updated_x.to_vec();

        // Check values are approximately 1.9
        assert!((values[0].to_f32() - 1.9).abs() < 0.01);
        assert!((values[1].to_f32() - 1.9).abs() < 0.01);
    }
}
