use crate::autograd::{ComputationGraph, GradNode, GradientFunction, NodeId, Operation, TensorVariant};
use crate::tensor::{FloatType, Tensor};
use crate::error::TensorResult;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// Thread-local computation graph
    static COMPUTATION_GRAPH: RefCell<ComputationGraph> = RefCell::new(ComputationGraph::new());

    /// Thread-local tensor registry (NodeId -> TensorVariant)
    /// Stores both f16 and f32 tensors using TensorVariant enum
    static TENSOR_REGISTRY: RefCell<HashMap<NodeId, TensorVariant>> = RefCell::new(HashMap::new());

    /// Flag for creating computation graph during backward pass (for higher-order derivatives)
    static CREATE_GRAPH: RefCell<bool> = RefCell::new(false);
}

/// Autograd context for managing computation graph
pub struct AutogradContext;

impl AutogradContext {
    /// Add a node to the computation graph, returns the newly created node ID
    pub fn add_node(
        operation: Operation,
        inputs: Vec<NodeId>,
        grad_fn: Option<Box<dyn GradientFunction>>,
    ) -> NodeId {
        COMPUTATION_GRAPH.with(|graph| {
            let mut g = graph.borrow_mut();
            let id = g.allocate_id();
            let node = GradNode::new(id, operation, inputs, grad_fn);
            g.add_node(node);
            id
        })
    }

    /// Allocate a new node ID (for leaf tensors that don't have operations)
    pub fn allocate_id() -> NodeId {
        COMPUTATION_GRAPH.with(|graph| {
            let mut g = graph.borrow_mut();
            g.allocate_id()
        })
    }

    /// Register a tensor with its node ID (using TensorVariant)
    pub fn register_tensor(node_id: NodeId, tensor: TensorVariant) {
        TENSOR_REGISTRY.with(|registry| {
            registry.borrow_mut().insert(node_id, tensor);
        });
    }

    /// Register a tensor with its node ID (generic version)
    pub fn register_tensor_generic<T: FloatType>(node_id: NodeId, tensor: Tensor<T>) {
        TENSOR_REGISTRY.with(|registry| {
            registry.borrow_mut().insert(node_id, tensor.into());
        });
    }

    /// Get a tensor by node ID (generic version)
    pub fn get_tensor_generic<T: FloatType>(node_id: NodeId) -> Option<Tensor<T>> {
        TENSOR_REGISTRY.with(|registry| {
            if let Some(variant) = registry.borrow().get(&node_id) {
                if T::is_f16() {
                    variant.clone_f16().map(|t| unsafe {
                        // Safety: We checked T::is_f16(), so T = f16
                        std::mem::transmute_copy(&t)
                    })
                } else if T::is_f32() {
                    variant.clone_f32().map(|t| unsafe {
                        // Safety: We checked T::is_f32(), so T = f32
                        std::mem::transmute_copy(&t)
                    })
                } else {
                    None
                }
            } else {
                None
            }
        })
    }

    /// Perform backward pass from a node (generic version)
    pub fn backward_generic<T: FloatType>(node_id: NodeId, grad: Tensor<T>) -> TensorResult<HashMap<NodeId, Tensor<T>>> {
        Self::backward_impl_generic::<T>(node_id, grad, false)
    }

    /// Perform backward pass with computation graph creation for higher-order derivatives (generic version)
    pub fn backward_with_graph_generic<T: FloatType>(node_id: NodeId, grad: Tensor<T>) -> TensorResult<HashMap<NodeId, Tensor<T>>> {
        Self::backward_impl_generic::<T>(node_id, grad, true)
    }

    /// Internal backward implementation (generic version)
    fn backward_impl_generic<T: FloatType>(node_id: NodeId, grad: Tensor<T>, create_graph: bool) -> TensorResult<HashMap<NodeId, Tensor<T>>> {
        // Save current states
        let prev_enabled = Self::is_enabled();
        let prev_create_graph = Self::is_create_graph();

        // Set create_graph mode
        Self::set_create_graph(create_graph);

        // Disable gradient recording during backward computation
        // (will be re-enabled during gradient distribution if create_graph=true)
        Self::set_enabled(false);

        // Convert Tensor<T> to TensorVariant for backward computation
        let grad_variant: TensorVariant = grad.into();

        let result = COMPUTATION_GRAPH.with(|graph| {
            let g = graph.borrow();
            // backward returns HashMap<NodeId, TensorVariant> internally
            // We need to convert it back to HashMap<NodeId, Tensor<T>>
            let variant_result = g.backward_variant(node_id, grad_variant, prev_enabled)?;

            // Convert each TensorVariant back to Tensor<T>
            let mut tensor_result = HashMap::new();
            for (nid, variant) in variant_result {
                if let Some(tensor) = Self::extract_tensor_from_variant::<T>(&variant) {
                    tensor_result.insert(nid, tensor);
                }
            }
            Ok(tensor_result)
        });

        // Restore previous states
        Self::set_enabled(prev_enabled);
        Self::set_create_graph(prev_create_graph);
        result
    }

    /// Helper to extract Tensor<T> from TensorVariant
    fn extract_tensor_from_variant<T: FloatType>(variant: &TensorVariant) -> Option<Tensor<T>> {
        if T::is_f16() {
            variant.clone_f16().map(|t| unsafe {
                // Safety: We checked T::is_f16(), so T = f16
                std::mem::transmute_copy(&t)
            })
        } else if T::is_f32() {
            variant.clone_f32().map(|t| unsafe {
                // Safety: We checked T::is_f32(), so T = f32
                std::mem::transmute_copy(&t)
            })
        } else {
            None
        }
    }

    /// Clear the computation graph and tensor registry
    pub fn clear() {
        COMPUTATION_GRAPH.with(|graph| {
            graph.borrow_mut().clear();
        });
        TENSOR_REGISTRY.with(|registry| {
            registry.borrow_mut().clear();
        });
    }

    /// Check if gradient computation is enabled
    pub fn is_enabled() -> bool {
        COMPUTATION_GRAPH.with(|graph| graph.borrow().is_enabled())
    }

    /// Set gradient computation enabled/disabled
    pub fn set_enabled(enabled: bool) {
        COMPUTATION_GRAPH.with(|graph| {
            graph.borrow_mut().set_enabled(enabled);
        });
    }

    /// Execute a closure with gradient computation disabled
    pub fn no_grad<F, R>(f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let prev_enabled = Self::is_enabled();
        Self::set_enabled(false);

        let result = f();

        Self::set_enabled(prev_enabled);
        result
    }

    /// Check if create_graph mode is enabled
    pub fn is_create_graph() -> bool {
        CREATE_GRAPH.with(|flag| *flag.borrow())
    }

    /// Set create_graph mode (for higher-order derivatives)
    pub fn set_create_graph(create_graph: bool) {
        CREATE_GRAPH.with(|flag| {
            *flag.borrow_mut() = create_graph;
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_id() {
        AutogradContext::clear();

        let id1 = AutogradContext::allocate_id();
        let id2 = AutogradContext::allocate_id();
        let id3 = AutogradContext::allocate_id();

        assert_eq!(id1, 0);
        assert_eq!(id2, 1);
        assert_eq!(id3, 2);
    }

    #[test]
    fn test_add_node() {
        AutogradContext::clear();

        let id = AutogradContext::add_node(Operation::Add, vec![], None);

        assert_eq!(id, 0);
    }

    #[test]
    fn test_no_grad() {
        AutogradContext::clear();
        AutogradContext::set_enabled(true);

        assert!(AutogradContext::is_enabled());

        AutogradContext::no_grad(|| {
            assert!(!AutogradContext::is_enabled());
        });

        assert!(AutogradContext::is_enabled());
    }
}
