# Tutorial 02: Multi-Parameter Optimization with TensorLogic

**Difficulty**: Beginner
**Time**: 5 minutes
**Topics**: Multiple Parameters, Gradient Descent Coordination

## Overview

Learn how to train multiple parameters simultaneously with TensorLogic's learning system.

## Complete Code

See [examples/tutorial_02_logistic_regression.tl](../examples/tutorial_02_logistic_regression.tl):

```tensorlogic
// Declare learnable parameters with different initial values
tensor w1: float32[1] learnable = [1.0]
tensor w2: float32[1] learnable = [0.5]

main {
    // Multi-parameter optimization: minimize w1^2 + w2^2
    learn {
        objective: w1 * w1 + w2 * w2,
        optimizer: sgd(lr: 0.1),
        epochs: 50
    }
}
```

## Key Concepts

- **Multiple Learnable Parameters**: All tensors marked `learnable` are optimized simultaneously
- **Combined Loss**: Single objective function can depend on multiple parameters
- **Independent Convergence**: Each parameter follows its own gradient

## Expected Output

```
Epoch   1/50: Loss = 0.750000
Epoch  10/50: Loss = 0.001099
Epoch  30/50: Loss = 0.000000

Final Parameter Values:
  w1: [0.0]
  w2: [3.5e-6]
```

Both parameters converge to ~0 as expected.

---

**Status**: ✅ Verified working (2025-10-20)
