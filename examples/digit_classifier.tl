// Simple Digit Classification Example
// Demonstrates training and inference with a neural network
//
// Task: Classify 3x3 pixel patterns into 3 classes (0, 1, 2)
// Architecture: Linear model with learned weights

// ============================================================================
// Model Parameters
// ============================================================================

// Weights for each class (3 classes x 9 input pixels)
tensor W: float16[9, 3] learnable = [
    0.1, -0.1,  0.05,
   -0.05,  0.15, -0.1,
    0.08, -0.12,  0.1,
   -0.1,  0.1, -0.05,
    0.12, -0.08,  0.15,
   -0.15,  0.1, -0.1,
    0.1, -0.05,  0.12,
   -0.08,  0.12, -0.15,
    0.15, -0.1,  0.1
]

tensor b: float16[3] learnable = [0.0, 0.0, 0.0]

// ============================================================================
// Training Data
// ============================================================================

// Digit 0: Circle pattern
tensor x0: float16[9] = [1.0, 1.0, 1.0,  1.0, 0.0, 1.0,  1.0, 1.0, 1.0]
tensor y0: float16[3] = [1.0, 0.0, 0.0]

// Digit 1: Vertical line
tensor x1: float16[9] = [0.0, 1.0, 0.0,  0.0, 1.0, 0.0,  0.0, 1.0, 0.0]
tensor y1: float16[3] = [0.0, 1.0, 0.0]

// Digit 2: S-shape
tensor x2: float16[9] = [1.0, 1.0, 1.0,  0.0, 0.0, 1.0,  1.0, 1.0, 1.0]
tensor y2: float16[3] = [0.0, 0.0, 1.0]

main {
    print("============================================================")
    print("Digit Classification: Training + Inference Demo")
    print("============================================================")
    print("")
    print("Task: Classify 3x3 pixel patterns")
    print("Classes: 0 (circle), 1 (line), 2 (S-shape)")
    print("Model: Linear classifier (9 inputs -> 3 outputs)")
    print("")

    // ========================================================================
    // Training Phase
    // ========================================================================
    print("------------------------------------------------------------")
    print("TRAINING PHASE")
    print("------------------------------------------------------------")
    print("")

    learn {
        // Predictions for all training samples
        pred0 := x0 @ W + b
        pred1 := x1 @ W + b
        pred2 := x2 @ W + b

        // Mean squared error loss
        loss0 := sum((pred0 - y0) * (pred0 - y0))
        loss1 := sum((pred1 - y1) * (pred1 - y1))
        loss2 := sum((pred2 - y2) * (pred2 - y2))

        total_loss := loss0 + loss1 + loss2

        objective: total_loss,
        optimizer: adam(lr: 0.05),
        epochs: 50
    }

    print("")
    print("Training completed!")
    print("")

    // ========================================================================
    // Inference Phase
    // ========================================================================
    print("------------------------------------------------------------")
    print("INFERENCE PHASE")
    print("------------------------------------------------------------")
    print("")

    // Test on training data
    print("Test 1: Digit 0 (circle)")
    print("  Pattern: [1 1 1]")
    print("           [1 0 1]")
    print("           [1 1 1]")
    test0 := x0 @ W + b
    print("  Scores:", test0)
    print("")

    print("Test 2: Digit 1 (line)")
    print("  Pattern: [0 1 0]")
    print("           [0 1 0]")
    print("           [0 1 0]")
    test1 := x1 @ W + b
    print("  Scores:", test1)
    print("")

    print("Test 3: Digit 2 (S-shape)")
    print("  Pattern: [1 1 1]")
    print("           [0 0 1]")
    print("           [1 1 1]")
    test2 := x2 @ W + b
    print("  Scores:", test2)
    print("")

    // Test on new data (slight variations)
    tensor x0_noisy: float16[9] = [0.9, 1.0, 0.9,  1.0, 0.1, 1.0,  0.9, 1.0, 0.9]
    print("Test 4: Noisy digit 0")
    print("  Pattern: [0.9 1.0 0.9]")
    print("           [1.0 0.1 1.0]")
    print("           [0.9 1.0 0.9]")
    test0_noisy := x0_noisy @ W + b
    print("  Scores:", test0_noisy)
    print("")

    tensor x1_tilted: float16[9] = [0.2, 1.0, 0.0,  0.0, 1.0, 0.1,  0.0, 1.0, 0.2]
    print("Test 5: Tilted digit 1")
    print("  Pattern: [0.2 1.0 0.0]")
    print("           [0.0 1.0 0.1]")
    print("           [0.0 1.0 0.2]")
    test1_tilted := x1_tilted @ W + b
    print("  Scores:", test1_tilted)
    print("")

    tensor x_mixed: float16[9] = [0.5, 1.0, 0.5,  0.5, 0.5, 0.5,  0.5, 1.0, 0.5]
    print("Test 6: Ambiguous pattern (mix of 0 and 1)")
    print("  Pattern: [0.5 1.0 0.5]")
    print("           [0.5 0.5 0.5]")
    print("           [0.5 1.0 0.5]")
    test_mixed := x_mixed @ W + b
    print("  Scores:", test_mixed)
    print("")

    print("============================================================")
    print("INTERPRETATION")
    print("============================================================")
    print("Scores: [class_0, class_1, class_2]")
    print("Higher value = stronger prediction for that class")
    print("")
    print("Expected results:")
    print("  Tests 1-3: Clear predictions for each class")
    print("  Test 4: Should predict class 0 despite noise")
    print("  Test 5: Should predict class 1 despite tilt")
    print("  Test 6: May show uncertainty between classes 0 and 1")
    print("============================================================")
}
