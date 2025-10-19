# Phase 8.4: Automatic Operator Fusion Design

## Overview

Implement automatic detection and application of operator fusion opportunities in computation graphs to reduce memory access and kernel launch overhead.

## Design Goals

1. **Pattern Recognition**: Automatically detect fusible operation sequences
2. **Performance Optimization**: Reduce kernel launches and memory transfers
3. **Transparency**: Seamless integration with existing autograd system
4. **Configurability**: User control over fusion behavior
5. **Maintainability**: Clean separation from core tensor operations

## Architecture

```
┌─────────────────────────────────────────────────┐
│         FusionOptimizer (Singleton)             │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Pattern Matcher                         │  │
│  │  - Detect fusible sequences              │  │
│  │  - add+relu, mul+relu, matmul+bias+relu  │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Fusion Applicator                       │  │
│  │  - Replace ops with fused versions       │  │
│  │  - Update computation graph              │  │
│  └──────────────────────────────────────────┘  │
│                                                 │
│  ┌──────────────────────────────────────────┐  │
│  │  Performance Tracker                     │  │
│  │  - Measure fusion benefits               │  │
│  │  - Adaptive fusion decisions             │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────┐
│        AutogradContext (Enhanced)               │
│  - Track operation sequences                    │
│  - Apply fusion optimization                    │
└─────────────────────────────────────────────────┘
```

## Fusion Patterns

### Pattern 1: Element-wise + Activation
```
Input → Add → ReLU → Output
      ↓
Input → FusedAddReLU → Output

Benefits:
- 1 kernel launch instead of 2
- No intermediate buffer allocation
- 50% reduction in memory bandwidth
```

### Pattern 2: MatMul + Bias + Activation
```
Input → MatMul → Add(bias) → ReLU → Output
      ↓
Input → FusedLinear(bias, relu) → Output

Benefits:
- 1 kernel launch instead of 3
- No intermediate allocations
- 66% reduction in memory transfers
```

### Pattern 3: Scalar + Activation
```
Input → MulScalar → ReLU → Output
      ↓
Input → FusedMulScalarReLU → Output

Benefits:
- Combined memory access pattern
- Reduced overhead for small tensors
```

## Data Structures

```rust
/// Fusion pattern identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// Binary op + activation: (add/sub/mul/div) + (relu/gelu)
    BinaryActivation { binary_op: BinaryOp, activation: Activation },

    /// MatMul + bias + optional activation
    LinearLayer { has_bias: bool, activation: Option<Activation> },

    /// Scalar op + activation
    ScalarActivation { scalar_op: ScalarOp, activation: Activation },
}

/// Fusion configuration
pub struct FusionConfig {
    /// Enable automatic fusion
    pub enabled: bool,

    /// Minimum tensor size for fusion (avoid overhead for small tensors)
    pub min_tensor_size: usize,

    /// Device-specific fusion settings
    pub metal_fusion: bool,
    pub cpu_fusion: bool,
    pub neural_engine_fusion: bool,

    /// Pattern-specific enables
    pub enabled_patterns: HashSet<FusionPattern>,
}

/// Fusion opportunity detected in computation graph
pub struct FusionOpportunity {
    /// Pattern type
    pub pattern: FusionPattern,

    /// Node IDs involved in this fusion
    pub node_ids: Vec<usize>,

    /// Estimated speedup (ratio)
    pub estimated_speedup: f32,

    /// Memory savings (bytes)
    pub memory_savings: usize,
}

/// Fusion optimizer
pub struct FusionOptimizer {
    /// Configuration
    config: FusionConfig,

    /// Performance statistics (pattern → average speedup)
    performance_stats: HashMap<FusionPattern, f32>,
}
```

## Implementation Strategy

### 1. Pattern Detection

```rust
impl FusionOptimizer {
    /// Detect fusion opportunities in computation graph
    pub fn detect_opportunities(
        &self,
        graph: &ComputationGraph,
    ) -> Vec<FusionOpportunity> {
        let mut opportunities = Vec::new();

        // Iterate through graph in topological order
        for node_id in graph.topological_order() {
            let node = &graph.nodes[node_id];

            // Pattern 1: Binary + Activation
            if let Some(opp) = self.detect_binary_activation(graph, node_id) {
                opportunities.push(opp);
            }

            // Pattern 2: MatMul + Bias + Activation
            if let Some(opp) = self.detect_linear_layer(graph, node_id) {
                opportunities.push(opp);
            }

            // Pattern 3: Scalar + Activation
            if let Some(opp) = self.detect_scalar_activation(graph, node_id) {
                opportunities.push(opp);
            }
        }

        // Sort by estimated benefit
        opportunities.sort_by(|a, b| {
            b.estimated_speedup.partial_cmp(&a.estimated_speedup).unwrap()
        });

        opportunities
    }

    /// Detect binary op + activation pattern
    fn detect_binary_activation(
        &self,
        graph: &ComputationGraph,
        node_id: usize,
    ) -> Option<FusionOpportunity> {
        let node = &graph.nodes[node_id];

        // Check if this is an activation
        let activation = match &node.operation {
            Operation::ReLU => Activation::ReLU,
            Operation::GELU => Activation::GELU,
            _ => return None,
        };

        // Check if input is a binary operation
        if node.inputs.len() != 1 {
            return None;
        }

        let input_id = node.inputs[0];
        let input_node = &graph.nodes[input_id];

        let binary_op = match &input_node.operation {
            Operation::Add => BinaryOp::Add,
            Operation::Mul => BinaryOp::Mul,
            Operation::Sub => BinaryOp::Sub,
            Operation::Div => BinaryOp::Div,
            _ => return None,
        };

        // Check if fusion is beneficial
        let tensor_size = node.output_shape.iter().product::<usize>();
        if tensor_size < self.config.min_tensor_size {
            return None;
        }

        Some(FusionOpportunity {
            pattern: FusionPattern::BinaryActivation { binary_op, activation },
            node_ids: vec![input_id, node_id],
            estimated_speedup: 1.5, // 2 ops → 1 op
            memory_savings: tensor_size * std::mem::size_of::<f16>(),
        })
    }
}
```

### 2. Fusion Application

```rust
impl FusionOptimizer {
    /// Apply fusion opportunities to computation graph
    pub fn apply_fusions(
        &mut self,
        graph: &mut ComputationGraph,
        opportunities: Vec<FusionOpportunity>,
    ) -> FusionStats {
        let mut stats = FusionStats::default();

        for opp in opportunities {
            match self.apply_fusion(graph, &opp) {
                Ok(()) => {
                    stats.applied_count += 1;
                    stats.total_memory_saved += opp.memory_savings;
                },
                Err(e) => {
                    stats.failed_count += 1;
                    eprintln!("Failed to apply fusion: {:?}", e);
                },
            }
        }

        stats
    }

    fn apply_fusion(
        &mut self,
        graph: &mut ComputationGraph,
        opportunity: &FusionOpportunity,
    ) -> TensorResult<()> {
        match opportunity.pattern {
            FusionPattern::BinaryActivation { binary_op, activation } => {
                self.apply_binary_activation_fusion(
                    graph,
                    &opportunity.node_ids,
                    binary_op,
                    activation,
                )
            },

            FusionPattern::LinearLayer { has_bias, activation } => {
                self.apply_linear_layer_fusion(
                    graph,
                    &opportunity.node_ids,
                    has_bias,
                    activation,
                )
            },

            FusionPattern::ScalarActivation { scalar_op, activation } => {
                self.apply_scalar_activation_fusion(
                    graph,
                    &opportunity.node_ids,
                    scalar_op,
                    activation,
                )
            },
        }
    }
}
```

### 3. Integration with AutogradContext

```rust
impl AutogradContext {
    /// Apply automatic fusion optimization
    pub fn optimize_with_fusion(&mut self) -> TensorResult<FusionStats> {
        if !FusionOptimizer::global().lock().unwrap().config.enabled {
            return Ok(FusionStats::default());
        }

        let optimizer = FusionOptimizer::global();
        let mut optimizer = optimizer.lock().unwrap();

        // Detect fusion opportunities
        let opportunities = optimizer.detect_opportunities(&self.graph);

        // Apply fusions
        let stats = optimizer.apply_fusions(&mut self.graph, opportunities);

        Ok(stats)
    }

    /// Record operation with automatic fusion check
    pub fn record_operation_with_fusion<F>(
        &mut self,
        operation: Operation,
        inputs: Vec<usize>,
        output_id: usize,
        gradient_fn: F,
    ) -> TensorResult<()>
    where
        F: GradientFunction + 'static,
    {
        // Record operation normally
        self.record_operation(operation, inputs, output_id, gradient_fn)?;

        // Check for immediate fusion opportunities
        self.check_immediate_fusion(output_id)?;

        Ok(())
    }
}
```

## Configuration API

```rust
impl FusionOptimizer {
    /// Get the global fusion optimizer
    pub fn global() -> &'static Mutex<FusionOptimizer> {
        static INSTANCE: OnceLock<Mutex<FusionOptimizer>> = OnceLock::new();
        INSTANCE.get_or_init(|| Mutex::new(FusionOptimizer::new()))
    }

    /// Enable/disable automatic fusion
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
    }

    /// Set minimum tensor size for fusion
    pub fn set_min_tensor_size(&mut self, size: usize) {
        self.config.min_tensor_size = size;
    }

    /// Enable specific fusion pattern
    pub fn enable_pattern(&mut self, pattern: FusionPattern) {
        self.config.enabled_patterns.insert(pattern);
    }

    /// Get fusion statistics
    pub fn get_stats(&self) -> FusionStatsSummary {
        FusionStatsSummary {
            total_fusions: self.performance_stats.len(),
            average_speedup: self.performance_stats.values().sum::<f32>()
                / self.performance_stats.len() as f32,
            patterns_used: self.config.enabled_patterns.clone(),
        }
    }
}
```

## User API

```rust
// Enable automatic fusion globally
FusionOptimizer::global().lock().unwrap().set_enabled(true);

// Configure fusion
let mut optimizer = FusionOptimizer::global().lock().unwrap();
optimizer.set_min_tensor_size(1000);
optimizer.enable_pattern(FusionPattern::BinaryActivation {
    binary_op: BinaryOp::Add,
    activation: Activation::ReLU,
});

// Operations are automatically fused
let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3])?;
let y = Tensor::from_vec(vec![4.0, 5.0, 6.0], vec![3])?;
let z = x.add(&y)?.relu()?; // Automatically uses fused_add_relu

// Get fusion statistics
let stats = FusionOptimizer::global().lock().unwrap().get_stats();
println!("Total fusions applied: {}", stats.total_fusions);
println!("Average speedup: {:.2}x", stats.average_speedup);
```

## Performance Tracking

```rust
/// Performance measurement for fusion patterns
pub struct PerformanceTracker {
    /// Benchmarks for each pattern
    benchmarks: HashMap<FusionPattern, BenchmarkResult>,
}

impl PerformanceTracker {
    /// Benchmark a fusion pattern
    pub fn benchmark_pattern(
        &mut self,
        pattern: FusionPattern,
        tensor_size: usize,
    ) -> BenchmarkResult {
        // Run unfused version
        let unfused_time = self.run_unfused(pattern, tensor_size);

        // Run fused version
        let fused_time = self.run_fused(pattern, tensor_size);

        let speedup = unfused_time / fused_time;

        BenchmarkResult {
            pattern,
            tensor_size,
            unfused_time_ms: unfused_time,
            fused_time_ms: fused_time,
            speedup,
        }
    }
}
```

## Testing Strategy

### Unit Tests
1. **Pattern Detection**: Test each pattern detector independently
2. **Fusion Application**: Test graph transformation correctness
3. **Configuration**: Test enable/disable and settings
4. **Performance**: Verify fused ops produce same results

### Integration Tests
1. **End-to-End**: Complete workflow with auto-fusion enabled
2. **Backward Pass**: Verify gradients match for fused operations
3. **Multi-Pattern**: Test multiple fusion patterns together
4. **Device Support**: Test on CPU, Metal, Neural Engine

### Performance Tests
1. **Speedup Measurement**: Benchmark fused vs unfused
2. **Memory Usage**: Verify memory savings
3. **Kernel Launch Overhead**: Measure reduction in launches

## Implementation Phases

### Phase 8.4.1: Pattern Detection ✅ (Target)
- [x] FusionPattern enum
- [x] FusionOpportunity structure
- [x] Pattern detection for binary+activation
- [x] Pattern detection for linear layers

### Phase 8.4.2: Fusion Application
- [ ] Graph transformation logic
- [ ] Fused operation replacement
- [ ] Gradient function updates

### Phase 8.4.3: Performance Tracking
- [ ] Benchmark infrastructure
- [ ] Adaptive fusion decisions
- [ ] Statistics collection

### Phase 8.4.4: Integration & Testing
- [ ] AutogradContext integration
- [ ] Comprehensive test suite
- [ ] Performance benchmarks

## Expected Performance Improvements

| Pattern | Kernel Launches | Memory Transfers | Expected Speedup |
|---------|----------------|------------------|------------------|
| Add + ReLU | 2 → 1 | 3 → 2 | 1.5x |
| Mul + ReLU | 2 → 1 | 3 → 2 | 1.5x |
| MatMul + Bias + ReLU | 3 → 1 | 5 → 3 | 2.0x |
| Affine + ReLU | 2 → 1 | 3 → 2 | 1.4x |

## Future Enhancements

1. **ML-Based Fusion**: Learn optimal fusion patterns from profiling
2. **Multi-Tensor Fusion**: Fuse operations across multiple tensors
3. **Custom Fusion Patterns**: User-defined fusion rules
4. **JIT Compilation**: Generate fused kernels on-the-fly
5. **Cross-Device Fusion**: Optimize data movement between devices

## Notes

- Fusion should preserve numerical equivalence (same results as unfused)
- Gradient computation must work correctly with fused operations
- Performance benefits depend on tensor size and operation complexity
- Some patterns may not benefit all devices equally
