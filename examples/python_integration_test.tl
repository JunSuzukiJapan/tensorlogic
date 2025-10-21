// Python Integration Test Example
// Tests python import and python.call() syntax

main {
    // Python import statement
    python import numpy as np
    python import torch

    // Create a tensor
    tensor x: float16[3] = [1.0, 2.0, 3.0]

    // Python function call (will be implemented in Phase 2)
    // tensor y = python.call("np.sum", x)

    print("Python integration test loaded successfully")
}
