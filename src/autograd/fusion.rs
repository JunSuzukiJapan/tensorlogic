//! Automatic operator fusion for computation graph optimization

use std::collections::{HashMap, HashSet};
use std::sync::{Mutex, OnceLock};
use crate::autograd::ComputationGraph;
use crate::ops::Activation;
use crate::error::TensorResult;

/// Binary operation types for fusion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Scalar operation types for fusion
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarOp {
    AddScalar,
    MulScalar,
}

/// Fusion pattern identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusionPattern {
    /// Binary op + activation: (add/sub/mul/div) + (relu/gelu)
    BinaryActivation {
        binary_op: BinaryOp,
        activation: Activation,
    },

    /// MatMul + bias + optional activation
    LinearLayer {
        has_bias: bool,
        activation: Option<Activation>,
    },

    /// Scalar op + activation
    ScalarActivation {
        scalar_op: ScalarOp,
        activation: Activation,
    },
}

/// Fusion opportunity detected in computation graph
#[derive(Debug, Clone)]
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

/// Fusion configuration
#[derive(Debug, Clone)]
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

impl Default for FusionConfig {
    fn default() -> Self {
        let mut enabled_patterns = HashSet::new();

        // Enable common fusion patterns by default
        enabled_patterns.insert(FusionPattern::BinaryActivation {
            binary_op: BinaryOp::Add,
            activation: Activation::ReLU,
        });
        enabled_patterns.insert(FusionPattern::BinaryActivation {
            binary_op: BinaryOp::Mul,
            activation: Activation::ReLU,
        });

        Self {
            enabled: true,
            min_tensor_size: 1000,
            metal_fusion: true,
            cpu_fusion: true,
            neural_engine_fusion: true,
            enabled_patterns,
        }
    }
}

/// Fusion statistics
#[derive(Debug, Clone, Default)]
pub struct FusionStats {
    pub applied_count: usize,
    pub failed_count: usize,
    pub total_memory_saved: usize,
}

/// Statistics summary
#[derive(Debug, Clone)]
pub struct FusionStatsSummary {
    pub total_fusions: usize,
    pub average_speedup: f32,
    pub patterns_used: HashSet<FusionPattern>,
}

/// Fusion optimizer for automatic operator fusion
pub struct FusionOptimizer {
    /// Configuration
    config: FusionConfig,

    /// Performance statistics (pattern → average speedup)
    performance_stats: HashMap<FusionPattern, f32>,
}

impl FusionOptimizer {
    /// Create a new fusion optimizer
    pub fn new() -> Self {
        Self {
            config: FusionConfig::default(),
            performance_stats: HashMap::new(),
        }
    }

    /// Get the global fusion optimizer instance
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

    /// Disable specific fusion pattern
    pub fn disable_pattern(&mut self, pattern: FusionPattern) {
        self.config.enabled_patterns.remove(&pattern);
    }

    /// Get fusion statistics
    pub fn get_stats(&self) -> FusionStatsSummary {
        FusionStatsSummary {
            total_fusions: self.performance_stats.len(),
            average_speedup: if self.performance_stats.is_empty() {
                1.0
            } else {
                self.performance_stats.values().sum::<f32>()
                    / self.performance_stats.len() as f32
            },
            patterns_used: self.config.enabled_patterns.clone(),
        }
    }

    /// Detect fusion opportunities in computation graph
    pub fn detect_opportunities(
        &self,
        _graph: &ComputationGraph,
    ) -> Vec<FusionOpportunity> {
        if !self.config.enabled {
            return Vec::new();
        }

        let mut opportunities = Vec::new();

        // Pattern detection would go here
        // For now, return empty (Phase 8.4.1 implementation)

        // Sort by estimated benefit
        opportunities.sort_by(|a: &FusionOpportunity, b: &FusionOpportunity| {
            b.estimated_speedup.partial_cmp(&a.estimated_speedup).unwrap()
        });

        opportunities
    }

    /// Apply fusion opportunities to computation graph
    pub fn apply_fusions(
        &mut self,
        _graph: &mut ComputationGraph,
        opportunities: Vec<FusionOpportunity>,
    ) -> FusionStats {
        let mut stats = FusionStats::default();

        for opp in opportunities {
            match self.apply_fusion(&opp) {
                Ok(()) => {
                    stats.applied_count += 1;
                    stats.total_memory_saved += opp.memory_savings;

                    // Update performance stats
                    self.performance_stats
                        .entry(opp.pattern)
                        .and_modify(|e| *e = (*e + opp.estimated_speedup) / 2.0)
                        .or_insert(opp.estimated_speedup);
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
        _opportunity: &FusionOpportunity,
    ) -> TensorResult<()> {
        // Fusion application logic would go here
        // For now, placeholder (Phase 8.4.2 implementation)
        Ok(())
    }

    /// Check if a pattern is enabled
    pub fn is_pattern_enabled(&self, pattern: &FusionPattern) -> bool {
        self.config.enabled && self.config.enabled_patterns.contains(pattern)
    }

    /// Get current configuration
    pub fn config(&self) -> &FusionConfig {
        &self.config
    }
}

impl Default for FusionOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimizer_creation() {
        let optimizer = FusionOptimizer::new();
        assert!(optimizer.config.enabled);
        assert_eq!(optimizer.config.min_tensor_size, 1000);
    }

    #[test]
    fn test_enable_disable() {
        let mut optimizer = FusionOptimizer::new();
        optimizer.set_enabled(false);
        assert!(!optimizer.config.enabled);
        optimizer.set_enabled(true);
        assert!(optimizer.config.enabled);
    }

    #[test]
    fn test_min_tensor_size() {
        let mut optimizer = FusionOptimizer::new();
        optimizer.set_min_tensor_size(5000);
        assert_eq!(optimizer.config.min_tensor_size, 5000);
    }

    #[test]
    fn test_pattern_enable_disable() {
        let mut optimizer = FusionOptimizer::new();
        let pattern = FusionPattern::BinaryActivation {
            binary_op: BinaryOp::Add,
            activation: Activation::GELU,
        };

        optimizer.enable_pattern(pattern);
        assert!(optimizer.is_pattern_enabled(&pattern));

        optimizer.disable_pattern(pattern);
        assert!(!optimizer.is_pattern_enabled(&pattern));
    }

    #[test]
    fn test_stats_summary() {
        let optimizer = FusionOptimizer::new();
        let stats = optimizer.get_stats();
        assert_eq!(stats.total_fusions, 0);
        assert_eq!(stats.average_speedup, 1.0);
    }

    #[test]
    fn test_default_patterns_enabled() {
        let optimizer = FusionOptimizer::new();

        // Default: Add+ReLU should be enabled
        let add_relu = FusionPattern::BinaryActivation {
            binary_op: BinaryOp::Add,
            activation: Activation::ReLU,
        };
        assert!(optimizer.is_pattern_enabled(&add_relu));

        // Default: Mul+ReLU should be enabled
        let mul_relu = FusionPattern::BinaryActivation {
            binary_op: BinaryOp::Mul,
            activation: Activation::ReLU,
        };
        assert!(optimizer.is_pattern_enabled(&mul_relu));
    }

    #[test]
    fn test_detect_opportunities_disabled() {
        let mut optimizer = FusionOptimizer::new();
        optimizer.set_enabled(false);

        let graph = ComputationGraph::new();
        let opportunities = optimizer.detect_opportunities(&graph);
        assert_eq!(opportunities.len(), 0);
    }
}
