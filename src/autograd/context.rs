use crate::autograd::{ComputationGraph, GradNode, GradientFunction, NodeId, Operation};
use crate::tensor::Tensor;
use std::cell::RefCell;
use std::collections::HashMap;

thread_local! {
    /// Thread-local computation graph
    static COMPUTATION_GRAPH: RefCell<ComputationGraph> = RefCell::new(ComputationGraph::new());

    /// Thread-local tensor registry (NodeId -> Tensor)
    static TENSOR_REGISTRY: RefCell<HashMap<NodeId, Tensor>> = RefCell::new(HashMap::new());
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

    /// Register a tensor with its node ID
    pub fn register_tensor(node_id: NodeId, tensor: Tensor) {
        TENSOR_REGISTRY.with(|registry| {
            registry.borrow_mut().insert(node_id, tensor);
        });
    }

    /// Get a tensor by node ID
    pub fn get_tensor(node_id: NodeId) -> Option<Tensor> {
        TENSOR_REGISTRY.with(|registry| registry.borrow().get(&node_id).cloned())
    }

    /// Perform backward pass from a node
    pub fn backward(node_id: NodeId, grad: Tensor) -> crate::error::TensorResult<HashMap<NodeId, Tensor>> {
        // Save current enabled state and disable gradient recording during backward
        let prev_enabled = Self::is_enabled();
        Self::set_enabled(false);

        let result = COMPUTATION_GRAPH.with(|graph| {
            let g = graph.borrow();
            g.backward(node_id, grad, prev_enabled)
        });

        // Restore previous enabled state
        Self::set_enabled(prev_enabled);
        result
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
