// Simple Linear Model: y = Wx (no bias)
// Training and Inference Example

// Model parameter (learnable weight)
tensor W: float16[1] learnable = [0.5]

main {
    print("=== Simple Linear Model: y = 3x ===")
    print("Training model to learn W ≈ 3.0")
    print()

    // Training data (single samples): y = 3x
    tensor x1: float16[1] = [1.0]
    tensor y1: float16[1] = [3.0]
    tensor x2: float16[1] = [2.0]
    tensor y2: float16[1] = [6.0]
    tensor x3: float16[1] = [3.0]
    tensor y3: float16[1] = [9.0]
    tensor x4: float16[1] = [4.0]
    tensor y4: float16[1] = [12.0]
    tensor x5: float16[1] = [5.0]
    tensor y5: float16[1] = [15.0]

    // Train the model
    learn {
        // Forward pass for all samples
        pred1 := x1 * W
        pred2 := x2 * W
        pred3 := x3 * W
        pred4 := x4 * W
        pred5 := x5 * W

        // Compute errors
        err1 := pred1 - y1
        err2 := pred2 - y2
        err3 := pred3 - y3
        err4 := pred4 - y4
        err5 := pred5 - y5

        // Sum of squared errors
        loss1 := err1 * err1
        loss2 := err2 * err2
        loss3 := err3 * err3
        loss4 := err4 * err4
        loss5 := err5 * err5

        total_loss := loss1 + loss2 + loss3 + loss4 + loss5

        objective: total_loss,
        optimizer: sgd(lr: 0.01),
        epochs: 200
    }

    print()
    print("=== Training Complete ===")
    print("Learned parameter: W =", W, " (expected: 3.0)")
    print()

    // Inference: Test on new data
    print("=== Inference on Test Data ===")

    tensor x_test_1: float16[1] = [6.0]
    tensor y_pred_1: float16[1] = x_test_1 * W
    print("  x = 6.0  =>  y =", y_pred_1, " (expected: 18.0)")

    tensor x_test_2: float16[1] = [10.0]
    tensor y_pred_2: float16[1] = x_test_2 * W
    print("  x = 10.0 =>  y =", y_pred_2, " (expected: 30.0)")

    tensor x_test_3: float16[1] = [-2.0]
    tensor y_pred_3: float16[1] = x_test_3 * W
    print("  x = -2.0 =>  y =", y_pred_3, " (expected: -6.0)")
}
