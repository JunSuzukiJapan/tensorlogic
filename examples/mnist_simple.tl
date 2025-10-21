// Simple Handwritten Digit Recognition (MNIST-like)
// 3x3 pixel images, 3 classes (digits 0, 1, 2)
//
// This example demonstrates:
// - Training a neural network classifier
// - Making predictions with the trained model
// - Practical pattern recognition task

// ============================================================================
// Model Parameters (Neural Network: 9 -> 6 -> 3)
// ============================================================================

// Input layer to hidden layer weights (9 input pixels -> 6 hidden neurons)
tensor W1: float16[9, 6] learnable = [
    0.1, -0.1,  0.2, -0.2,  0.1, -0.1,
   -0.1,  0.2, -0.1,  0.1, -0.2,  0.1,
    0.2, -0.1,  0.1, -0.1,  0.2, -0.2,
   -0.2,  0.1, -0.2,  0.2, -0.1,  0.1,
    0.1, -0.2,  0.1, -0.1,  0.2, -0.2,
   -0.1,  0.1, -0.1,  0.2, -0.1,  0.1,
    0.2, -0.2,  0.2, -0.1,  0.1, -0.1,
   -0.1,  0.1, -0.1,  0.1, -0.2,  0.2,
    0.1, -0.1,  0.2, -0.2,  0.1, -0.1
]

tensor b1: float16[6] learnable = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

// Hidden layer to output layer weights (6 hidden -> 3 classes)
tensor W2: float16[6, 3] learnable = [
    0.2, -0.1,  0.1,
   -0.1,  0.2, -0.1,
    0.1, -0.1,  0.2,
   -0.2,  0.1, -0.1,
    0.1, -0.2,  0.1,
   -0.1,  0.1, -0.2
]

tensor b2: float16[3] learnable = [0.0, 0.0, 0.0]

// ============================================================================
// Training Data: 3x3 pixel images (flattened to 9 values)
// ============================================================================
// Pixel values: 0.0 = black (background), 1.0 = white (foreground)

// Digit 0 patterns (4 samples)
tensor train_0_1: float16[9] = [1.0, 1.0, 1.0,  1.0, 0.0, 1.0,  1.0, 1.0, 1.0]
tensor train_0_2: float16[9] = [0.8, 1.0, 0.8,  1.0, 0.0, 1.0,  0.8, 1.0, 0.8]
tensor train_0_3: float16[9] = [1.0, 1.0, 1.0,  1.0, 0.2, 1.0,  1.0, 1.0, 1.0]
tensor train_0_4: float16[9] = [0.9, 1.0, 0.9,  1.0, 0.1, 1.0,  0.9, 1.0, 0.9]

// Digit 1 patterns (4 samples)
tensor train_1_1: float16[9] = [0.0, 1.0, 0.0,  0.0, 1.0, 0.0,  0.0, 1.0, 0.0]
tensor train_1_2: float16[9] = [0.1, 1.0, 0.0,  0.0, 1.0, 0.1,  0.0, 1.0, 0.0]
tensor train_1_3: float16[9] = [0.0, 1.0, 0.1,  0.0, 1.0, 0.0,  0.1, 1.0, 0.0]
tensor train_1_4: float16[9] = [0.2, 1.0, 0.0,  0.0, 1.0, 0.0,  0.0, 1.0, 0.2]

// Digit 2 patterns (4 samples)
tensor train_2_1: float16[9] = [1.0, 1.0, 1.0,  0.0, 0.0, 1.0,  1.0, 1.0, 1.0]
tensor train_2_2: float16[9] = [1.0, 1.0, 0.9,  0.1, 0.0, 1.0,  1.0, 1.0, 1.0]
tensor train_2_3: float16[9] = [1.0, 1.0, 1.0,  0.0, 0.1, 1.0,  0.9, 1.0, 1.0]
tensor train_2_4: float16[9] = [0.9, 1.0, 1.0,  0.0, 0.0, 1.0,  1.0, 1.0, 0.9]

// One-hot encoded labels
tensor label_0: float16[3] = [1.0, 0.0, 0.0]  // Class 0
tensor label_1: float16[3] = [0.0, 1.0, 0.0]  // Class 1
tensor label_2: float16[3] = [0.0, 0.0, 1.0]  // Class 2

// ============================================================================
// Helper: Forward pass (neural network inference)
// ============================================================================
// Input: x (9 pixels)
// Output: logits (3 class scores)
//
// Architecture:
//   x (9) -> W1 @ x + b1 -> ReLU -> (6) -> W2 @ hidden + b2 -> (3)

main {
    print("=" * 60)
    print("MNIST-like Digit Recognition: Training + Inference")
    print("=" * 60)
    print("")
    print("Architecture: 9 inputs -> 6 hidden -> 3 outputs")
    print("Classes: 0 (circle), 1 (vertical line), 2 (S-shape)")
    print("")

    // ========================================================================
    // Phase 1: Training
    // ========================================================================
    print("=" * 60)
    print("PHASE 1: TRAINING")
    print("=" * 60)
    print("")
    print("Training samples: 12 (4 per class)")
    print("Optimizer: Adam (lr=0.01)")
    print("Epochs: 100")
    print("")

    // Training loop: iterate over all samples
    learn {
        // Forward pass for all training samples
        // Digit 0 samples
        h1_0_1 := relu(train_0_1 @ W1 + b1)
        logits_0_1 := h1_0_1 @ W2 + b2
        loss_0_1 := sum((logits_0_1 - label_0) * (logits_0_1 - label_0))

        h1_0_2 := relu(train_0_2 @ W1 + b1)
        logits_0_2 := h1_0_2 @ W2 + b2
        loss_0_2 := sum((logits_0_2 - label_0) * (logits_0_2 - label_0))

        h1_0_3 := relu(train_0_3 @ W1 + b1)
        logits_0_3 := h1_0_3 @ W2 + b2
        loss_0_3 := sum((logits_0_3 - label_0) * (logits_0_3 - label_0))

        h1_0_4 := relu(train_0_4 @ W1 + b1)
        logits_0_4 := h1_0_4 @ W2 + b2
        loss_0_4 := sum((logits_0_4 - label_0) * (logits_0_4 - label_0))

        // Digit 1 samples
        h1_1_1 := relu(train_1_1 @ W1 + b1)
        logits_1_1 := h1_1_1 @ W2 + b2
        loss_1_1 := sum((logits_1_1 - label_1) * (logits_1_1 - label_1))

        h1_1_2 := relu(train_1_2 @ W1 + b1)
        logits_1_2 := h1_1_2 @ W2 + b2
        loss_1_2 := sum((logits_1_2 - label_1) * (logits_1_2 - label_1))

        h1_1_3 := relu(train_1_3 @ W1 + b1)
        logits_1_3 := h1_1_3 @ W2 + b2
        loss_1_3 := sum((logits_1_3 - label_1) * (logits_1_3 - label_1))

        h1_1_4 := relu(train_1_4 @ W1 + b1)
        logits_1_4 := h1_1_4 @ W2 + b2
        loss_1_4 := sum((logits_1_4 - label_1) * (logits_1_4 - label_1))

        // Digit 2 samples
        h1_2_1 := relu(train_2_1 @ W1 + b1)
        logits_2_1 := h1_2_1 @ W2 + b2
        loss_2_1 := sum((logits_2_1 - label_2) * (logits_2_1 - label_2))

        h1_2_2 := relu(train_2_2 @ W1 + b1)
        logits_2_2 := h1_2_2 @ W2 + b2
        loss_2_2 := sum((logits_2_2 - label_2) * (logits_2_2 - label_2))

        h1_2_3 := relu(train_2_3 @ W1 + b1)
        logits_2_3 := h1_2_3 @ W2 + b2
        loss_2_3 := sum((logits_2_3 - label_2) * (logits_2_3 - label_2))

        h1_2_4 := relu(train_2_4 @ W1 + b1)
        logits_2_4 := h1_2_4 @ W2 + b2
        loss_2_4 := sum((logits_2_4 - label_2) * (logits_2_4 - label_2))

        // Total loss (mean squared error across all samples)
        total_loss := loss_0_1 + loss_0_2 + loss_0_3 + loss_0_4 +
                      loss_1_1 + loss_1_2 + loss_1_3 + loss_1_4 +
                      loss_2_1 + loss_2_2 + loss_2_3 + loss_2_4

        objective: total_loss,
        optimizer: adam(lr: 0.01),
        epochs: 100
    }

    print("")
    print("Training completed!")
    print("")

    // ========================================================================
    // Phase 2: Inference on Test Data
    // ========================================================================
    print("=" * 60)
    print("PHASE 2: INFERENCE")
    print("=" * 60)
    print("")

    // Test sample 1: Clear digit 0
    tensor test_0: float16[9] = [1.0, 1.0, 1.0,  1.0, 0.0, 1.0,  1.0, 1.0, 1.0]
    h_test_0 := relu(test_0 @ W1 + b1)
    pred_0 := h_test_0 @ W2 + b2
    print("Test 1 - Digit 0 pattern:")
    print("  Input:  [1 1 1]")
    print("          [1 0 1]")
    print("          [1 1 1]")
    print("  Scores:", pred_0)
    print("")

    // Test sample 2: Clear digit 1
    tensor test_1: float16[9] = [0.0, 1.0, 0.0,  0.0, 1.0, 0.0,  0.0, 1.0, 0.0]
    h_test_1 := relu(test_1 @ W1 + b1)
    pred_1 := h_test_1 @ W2 + b2
    print("Test 2 - Digit 1 pattern:")
    print("  Input:  [0 1 0]")
    print("          [0 1 0]")
    print("          [0 1 0]")
    print("  Scores:", pred_1)
    print("")

    // Test sample 3: Clear digit 2
    tensor test_2: float16[9] = [1.0, 1.0, 1.0,  0.0, 0.0, 1.0,  1.0, 1.0, 1.0]
    h_test_2 := relu(test_2 @ W1 + b1)
    pred_2 := h_test_2 @ W2 + b2
    print("Test 3 - Digit 2 pattern:")
    print("  Input:  [1 1 1]")
    print("          [0 0 1]")
    print("          [1 1 1]")
    print("  Scores:", pred_2)
    print("")

    // Test sample 4: Noisy digit 0 (with slight variation)
    tensor test_0_noisy: float16[9] = [0.9, 1.0, 0.8,  1.0, 0.1, 0.9,  0.8, 1.0, 0.9]
    h_test_0_noisy := relu(test_0_noisy @ W1 + b1)
    pred_0_noisy := h_test_0_noisy @ W2 + b2
    print("Test 4 - Noisy Digit 0:")
    print("  Input:  [0.9 1.0 0.8]")
    print("          [1.0 0.1 0.9]")
    print("          [0.8 1.0 0.9]")
    print("  Scores:", pred_0_noisy)
    print("")

    // Test sample 5: Ambiguous pattern (between 1 and 0)
    tensor test_ambiguous: float16[9] = [0.5, 1.0, 0.5,  0.5, 0.5, 0.5,  0.5, 1.0, 0.5]
    h_test_amb := relu(test_ambiguous @ W1 + b1)
    pred_amb := h_test_amb @ W2 + b2
    print("Test 5 - Ambiguous pattern:")
    print("  Input:  [0.5 1.0 0.5]")
    print("          [0.5 0.5 0.5]")
    print("          [0.5 1.0 0.5]")
    print("  Scores:", pred_amb)
    print("")

    print("=" * 60)
    print("INTERPRETATION:")
    print("=" * 60)
    print("Scores format: [class_0, class_1, class_2]")
    print("Higher score = higher confidence for that class")
    print("")
    print("Expected results:")
    print("  Test 1 (digit 0): Highest score at index 0")
    print("  Test 2 (digit 1): Highest score at index 1")
    print("  Test 3 (digit 2): Highest score at index 2")
    print("  Test 4 (noisy 0): Should still predict 0 (index 0)")
    print("  Test 5 (ambiguous): May show mixed scores")
    print("=" * 60)
}
