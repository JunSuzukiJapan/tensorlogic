# Tutorial 03: Neural Network Weights with TensorLogic

**Difficulty**: Beginner
**Time**: 5 minutes
**Topics**: L2 Regularization, Weight Decay, Multiple Parameters

## Overview

Learn how to optimize multiple network weights simultaneously, demonstrating weight regularization (L2 penalty).

## Complete Code

See [examples/tutorial_03_neural_network.tl](../examples/tutorial_03_neural_network.tl)

## Key Concepts

- **L2 Regularization**: Penalizes large weight values by minimizing sum of squares
- **Weight Decay**: Encourages smaller weights for better generalization
- **Simultaneous Optimization**: All 3 weights updated together

## Usage

```bash
./target/release/tensorlogic run examples/tutorial_03_neural_network.tl
```

---

**Status**: ✅ Verified working (2025-10-20)
