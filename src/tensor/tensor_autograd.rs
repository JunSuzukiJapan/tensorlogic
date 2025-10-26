//! Tensor automatic differentiation methods

use crate::autograd::{AutogradContext, NodeId, TensorVariant};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{FloatType, Tensor};
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO, TensorTransform};
use half::f16;

/// Trait for automatic differentiation operations
pub trait TensorAutograd<T: FloatType>: Sized {
    /// Get the gradient
    fn grad(&self) -> Option<&Tensor<T>>;

    /// Set gradient (public for optimizer use)
    fn set_grad(&mut self, grad: Tensor<T>);

    /// Zero out the gradient
    fn zero_grad(&mut self);

    /// Set whether gradient is required
    fn set_requires_grad(&mut self, requires: bool);

    /// Set computation graph node ID (internal use)
    fn set_grad_node(&mut self, node_id: NodeId);

    /// Perform backward pass (for scalar tensors)
    fn backward(&mut self) -> TensorResult<()>;

    /// Perform backward pass with specified gradient
    fn backward_with_grad(&mut self, grad: Tensor<T>) -> TensorResult<()>;

    /// Perform backward pass with computation graph creation (for higher-order derivatives)
    fn backward_create_graph(&mut self) -> TensorResult<()>;
}

impl TensorAutograd<f16> for Tensor<f16> {
    fn grad(&self) -> Option<&Tensor<f16>> {
        self.grad.as_ref().map(|g| g.as_ref())
    }

    fn set_grad(&mut self, grad: Tensor<f16>) {
        self.grad = Some(Box::new(grad));
    }

    fn zero_grad(&mut self) {
        self.grad = None;
        self.version += 1;
    }

    fn set_requires_grad(&mut self, requires: bool) {
        self.requires_grad = requires;

        // Allocate a node ID for this tensor if it doesn't have one and requires_grad is true
        if requires && self.grad_node.is_none() {
            let node_id = AutogradContext::allocate_id();
            self.grad_node = Some(node_id);
            AutogradContext::register_tensor(node_id, TensorVariant::F16(self.clone()));
        }
    }

    fn set_grad_node(&mut self, node_id: NodeId) {
        self.grad_node = Some(node_id);
    }

    fn backward(&mut self) -> TensorResult<()> {
        if !self.requires_grad {
            return Err(TensorError::InvalidOperation(
                "Cannot call backward on tensor with requires_grad=False".to_string(),
            ));
        }

        if self.numel() != 1 {
            return Err(TensorError::InvalidOperation(
                "backward() can only be called on scalar tensors. Use backward_with_grad() for non-scalar tensors."
                    .to_string(),
            ));
        }

        // 初期勾配は1.0
        let grad = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;
        self.backward_with_grad(grad)
    }

    fn backward_with_grad(&mut self, grad: Tensor<f16>) -> TensorResult<()> {
        self.backward_impl(grad, false)
    }

    fn backward_create_graph(&mut self) -> TensorResult<()> {
        if !self.requires_grad {
            return Err(TensorError::InvalidOperation(
                "Cannot call backward on tensor with requires_grad=False".to_string(),
            ));
        }

        if self.numel() != 1 {
            return Err(TensorError::InvalidOperation(
                "backward_create_graph() can only be called on scalar tensors."
                    .to_string(),
            ));
        }

        let grad = Tensor::from_vec(vec![f16::from_f32(1.0)], vec![1])?;
        self.backward_impl(grad, true)
    }
}

impl TensorAutograd<f32> for Tensor<f32> {
    fn grad(&self) -> Option<&Tensor<f32>> {
        self.grad.as_ref().map(|g| g.as_ref())
    }

    fn set_grad(&mut self, grad: Tensor<f32>) {
        self.grad = Some(Box::new(grad));
    }

    fn zero_grad(&mut self) {
        self.grad = None;
        self.version += 1;
    }

    fn set_requires_grad(&mut self, requires: bool) {
        self.requires_grad = requires;

        // Allocate a node ID for this tensor if it doesn't have one and requires_grad is true
        if requires && self.grad_node.is_none() {
            let node_id = AutogradContext::allocate_id();
            self.grad_node = Some(node_id);
            AutogradContext::register_tensor(node_id, TensorVariant::F32(self.clone()));
        }
    }

    fn set_grad_node(&mut self, node_id: NodeId) {
        self.grad_node = Some(node_id);
    }

    fn backward(&mut self) -> TensorResult<()> {
        if !self.requires_grad {
            return Err(TensorError::InvalidOperation(
                "Cannot call backward on tensor with requires_grad=False".to_string(),
            ));
        }

        if self.numel() != 1 {
            return Err(TensorError::InvalidOperation(
                "backward() can only be called on scalar tensors. Use backward_with_grad() for non-scalar tensors."
                    .to_string(),
            ));
        }

        // 初期勾配は1.0
        let grad = Tensor::from_vec(vec![1.0f32], vec![1])?;
        self.backward_with_grad(grad)
    }

    fn backward_with_grad(&mut self, grad: Tensor<f32>) -> TensorResult<()> {
        self.backward_impl(grad, false)
    }

    fn backward_create_graph(&mut self) -> TensorResult<()> {
        if !self.requires_grad {
            return Err(TensorError::InvalidOperation(
                "Cannot call backward on tensor with requires_grad=False".to_string(),
            ));
        }

        if self.numel() != 1 {
            return Err(TensorError::InvalidOperation(
                "backward_create_graph() can only be called on scalar tensors."
                    .to_string(),
            ));
        }

        let grad = Tensor::from_vec(vec![1.0f32], vec![1])?;
        self.backward_impl(grad, true)
    }
}

// Internal implementation methods (not part of trait)
impl<T: FloatType> Tensor<T> {
    /// Internal backward implementation
    pub(crate) fn backward_impl(&mut self, grad: Tensor<T>, create_graph: bool) -> TensorResult<()>
    where
        Tensor<T>: TensorAutograd<T>,
    {
        use crate::autograd::AutogradContext;

        // Get node ID for this tensor
        let node_id = self.grad_node.ok_or_else(|| {
            TensorError::InvalidOperation(
                "Cannot call backward on tensor without computation graph node".to_string(),
            )
        })?;

        // Perform backward pass through computation graph
        let gradients = if create_graph {
            AutogradContext::backward_with_graph_generic::<T>(node_id, grad)?
        } else {
            AutogradContext::backward_generic::<T>(node_id, grad)?
        };

        // Distribute gradients to all tensors in the graph
        // If create_graph is true, enable gradient recording for gradient accumulation
        if create_graph {
            AutogradContext::set_enabled(true);
        }

        for (tensor_node_id, gradient) in gradients {
            if let Some(mut tensor) = AutogradContext::get_tensor_generic::<T>(tensor_node_id) {
                // Accumulate gradient if it already exists
                if let Some(existing_grad) = tensor.grad() {
                    let accumulated_grad = if create_graph {
                        // With create_graph, gradient accumulation should be recorded
                        existing_grad.add(&gradient).unwrap()
                    } else {
                        // Without create_graph, use no_grad for accumulation
                        AutogradContext::no_grad(|| existing_grad.add(&gradient).unwrap())
                    };
                    tensor.set_grad(accumulated_grad);
                } else {
                    // If create_graph is true, make gradient tensor require grad
                    let mut grad_tensor = gradient;
                    if create_graph {
                        grad_tensor.set_requires_grad(true);
                    }
                    tensor.set_grad(grad_tensor);
                }

                // Re-register the updated tensor
                AutogradContext::register_tensor_generic::<T>(tensor_node_id, tensor);
            }
        }

        // Disable gradient recording after distribution
        if create_graph {
            AutogradContext::set_enabled(false);
        }

        Ok(())
    }
}
