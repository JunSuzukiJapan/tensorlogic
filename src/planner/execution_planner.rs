//! Automatic device placement and execution planning

use crate::device::{Device, MetalDevice};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Operation type for device selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    ReLU,
    GELU,
    Softmax,
    Sum,
    Mean,
    Max,
    Min,
}

/// Device selection strategy
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    /// Use heuristic rules only (default)
    Heuristic,

    /// Always use specified device
    Fixed(Device),
}

/// Key for caching device selection decisions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct DecisionKey {
    operation: OperationType,
    total_elements: usize,
}

/// Execution planner for automatic device placement
pub struct ExecutionPlanner {
    /// Device selection strategy
    strategy: SelectionStrategy,

    /// Decision cache (operation + size → device)
    decision_cache: Arc<Mutex<HashMap<DecisionKey, Device>>>,

    /// Metal device (if available)
    metal_device: Option<MetalDevice>,

    /// Whether Neural Engine is available
    neural_engine_available: bool,
}

impl ExecutionPlanner {
    /// Create a new execution planner
    pub fn new() -> Self {
        let metal_device = MetalDevice::new().ok();
        let neural_engine_available = cfg!(target_os = "macos") || cfg!(target_os = "ios");

        Self {
            strategy: SelectionStrategy::Heuristic,
            decision_cache: Arc::new(Mutex::new(HashMap::new())),
            metal_device,
            neural_engine_available,
        }
    }

    /// Get the global execution planner instance
    pub fn global() -> &'static Mutex<ExecutionPlanner> {
        static INSTANCE: std::sync::OnceLock<Mutex<ExecutionPlanner>> = std::sync::OnceLock::new();
        INSTANCE.get_or_init(|| Mutex::new(ExecutionPlanner::new()))
    }

    /// Set the device selection strategy
    pub fn set_strategy(&mut self, strategy: SelectionStrategy) {
        self.strategy = strategy;
        // Clear cache when strategy changes
        self.decision_cache.lock().unwrap().clear();
    }

    /// Clear the decision cache
    pub fn clear_cache(&mut self) {
        self.decision_cache.lock().unwrap().clear();
    }

    /// Select the best device for an operation
    ///
    /// # Arguments
    /// * `operation` - Type of operation
    /// * `shapes` - Input tensor shapes
    ///
    /// # Returns
    /// * Recommended device for this operation
    pub fn select_device(&self, operation: OperationType, shapes: &[&[usize]]) -> Device {
        match &self.strategy {
            SelectionStrategy::Fixed(device) => device.clone(),
            SelectionStrategy::Heuristic => {
                // Calculate total elements
                let total_elements: usize = shapes.iter().map(|s| s.iter().product::<usize>()).sum();

                // Check cache
                let key = DecisionKey {
                    operation,
                    total_elements,
                };

                if let Some(cached_device) = self.decision_cache.lock().unwrap().get(&key) {
                    return cached_device.clone();
                }

                // Apply heuristic rules
                let device = self.select_device_heuristic(operation, total_elements, shapes);

                // Cache the decision
                self.decision_cache.lock().unwrap().insert(key, device.clone());

                device
            }
        }
    }

    /// Heuristic-based device selection
    fn select_device_heuristic(
        &self,
        operation: OperationType,
        total_elements: usize,
        _shapes: &[&[usize]],
    ) -> Device {
        // Size thresholds
        const METAL_MIN_SIZE: usize = 1000;
        const NEURAL_ENGINE_MATMUL_MIN: usize = 64 * 64;
        const REDUCTION_GPU_MIN: usize = 10000;

        match operation {
            // Large matrix multiplication → Neural Engine (if available)
            OperationType::MatMul => {
                if self.neural_engine_available && total_elements >= NEURAL_ENGINE_MATMUL_MIN {
                    // For now, we use Metal since Neural Engine integration is not complete
                    // return Device::NeuralEngine;
                    if let Some(ref device) = self.metal_device {
                        return Device::Metal(device.clone());
                    }
                }

                // Medium-sized matmul → Metal
                if total_elements >= METAL_MIN_SIZE {
                    if let Some(ref device) = self.metal_device {
                        return Device::Metal(device.clone());
                    }
                }

                // Small matmul → CPU
                Device::CPU
            }

            // Element-wise operations → Metal for large tensors
            OperationType::Add
            | OperationType::Sub
            | OperationType::Mul
            | OperationType::Div
            | OperationType::ReLU
            | OperationType::GELU => {
                if total_elements >= METAL_MIN_SIZE {
                    if let Some(ref device) = self.metal_device {
                        return Device::Metal(device.clone());
                    }
                }
                Device::CPU
            }

            // Reduction operations → Metal for large tensors
            OperationType::Sum | OperationType::Mean | OperationType::Max | OperationType::Min => {
                if total_elements >= REDUCTION_GPU_MIN {
                    if let Some(ref device) = self.metal_device {
                        return Device::Metal(device.clone());
                    }
                }
                Device::CPU
            }

            // Softmax → Metal for medium-large tensors
            OperationType::Softmax => {
                if total_elements >= METAL_MIN_SIZE {
                    if let Some(ref device) = self.metal_device {
                        return Device::Metal(device.clone());
                    }
                }
                Device::CPU
            }
        }
    }

    /// Select device for element-wise binary operation
    pub fn select_device_for_elementwise(&self, a_shape: &[usize], b_shape: &[usize], op: OperationType) -> Device {
        self.select_device(op, &[a_shape, b_shape])
    }

    /// Select device for unary operation
    pub fn select_device_for_unary(&self, shape: &[usize], op: OperationType) -> Device {
        self.select_device(op, &[shape])
    }

    /// Select device for matrix multiplication
    pub fn select_device_for_matmul(&self, a_shape: &[usize], b_shape: &[usize]) -> Device {
        self.select_device(OperationType::MatMul, &[a_shape, b_shape])
    }

    /// Get statistics about cached decisions
    pub fn get_cache_stats(&self) -> CacheStats {
        let cache = self.decision_cache.lock().unwrap();
        let mut device_counts = HashMap::new();

        for device in cache.values() {
            let device_type = match device {
                Device::Metal(_) => "Metal",
                Device::CPU => "CPU",
                Device::NeuralEngine => "NeuralEngine",
            };
            *device_counts.entry(device_type).or_insert(0) += 1;
        }

        CacheStats {
            total_cached: cache.len(),
            device_counts,
        }
    }
}

impl Default for ExecutionPlanner {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about cached device selection decisions
#[derive(Debug)]
pub struct CacheStats {
    pub total_cached: usize,
    pub device_counts: HashMap<&'static str, usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planner_creation() {
        let planner = ExecutionPlanner::new();
        assert!(matches!(planner.strategy, SelectionStrategy::Heuristic));
    }

    #[test]
    fn test_small_tensor_uses_cpu() {
        let planner = ExecutionPlanner::new();

        // Small tensor (100 elements) should use CPU
        let device = planner.select_device(OperationType::Add, &[&[10, 10]]);
        assert!(matches!(device, Device::CPU));
    }

    #[test]
    fn test_large_tensor_uses_metal() {
        let planner = ExecutionPlanner::new();

        // Large tensor (10000 elements) should use Metal if available
        let device = planner.select_device(OperationType::Add, &[&[100, 100]]);

        if planner.metal_device.is_some() {
            assert!(matches!(device, Device::Metal(_)));
        } else {
            assert!(matches!(device, Device::CPU));
        }
    }

    #[test]
    fn test_matmul_device_selection() {
        let planner = ExecutionPlanner::new();

        // Small matmul → CPU
        let device_small = planner.select_device_for_matmul(&[10, 10], &[10, 10]);
        assert!(matches!(device_small, Device::CPU));

        // Large matmul → Metal (or Neural Engine when fully integrated)
        let device_large = planner.select_device_for_matmul(&[128, 128], &[128, 128]);
        if planner.metal_device.is_some() {
            assert!(matches!(device_large, Device::Metal(_)));
        }
    }

    #[test]
    fn test_fixed_strategy() {
        let mut planner = ExecutionPlanner::new();
        planner.set_strategy(SelectionStrategy::Fixed(Device::CPU));

        // Should always return CPU regardless of size
        let device = planner.select_device(OperationType::Add, &[&[1000, 1000]]);
        assert!(matches!(device, Device::CPU));
    }

    #[test]
    fn test_cache_functionality() {
        let planner = ExecutionPlanner::new();

        // First call - not cached
        let device1 = planner.select_device(OperationType::Add, &[&[100, 100]]);

        // Second call - should be cached
        let device2 = planner.select_device(OperationType::Add, &[&[100, 100]]);

        // Devices should be the same
        match (&device1, &device2) {
            (Device::CPU, Device::CPU) => (),
            (Device::Metal(_), Device::Metal(_)) => (),
            (Device::NeuralEngine, Device::NeuralEngine) => (),
            _ => panic!("Devices should match"),
        }

        let stats = planner.get_cache_stats();
        assert!(stats.total_cached > 0);
    }

    #[test]
    fn test_reduction_ops_threshold() {
        let planner = ExecutionPlanner::new();

        // Small reduction → CPU
        let device_small = planner.select_device(OperationType::Sum, &[&[100]]);
        assert!(matches!(device_small, Device::CPU));

        // Large reduction → Metal
        let device_large = planner.select_device(OperationType::Sum, &[&[20000]]);
        if planner.metal_device.is_some() {
            assert!(matches!(device_large, Device::Metal(_)));
        }
    }
}
