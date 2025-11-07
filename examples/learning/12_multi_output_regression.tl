// Multi-Output Regression
//
// Predict multiple targets simultaneously

main {
    print("=== Multi-Output Regression ===")

    // Input: single feature
    // Outputs: [2x, 3x, x^2]
    let X = [[1.0], [2.0], [3.0], [4.0]]
    let y = [
        [2.0, 3.0, 1.0],    // x=1: [2, 3, 1]
        [4.0, 6.0, 4.0],    // x=2: [4, 6, 4]
        [6.0, 9.0, 9.0],    // x=3: [6, 9, 9]
        [8.0, 12.0, 16.0]   // x=4: [8, 12, 16]
    ]

    // Model: 1 -> 8 -> 3
    let W1 = learnable(rand([1, 8]))
    let b1 = learnable(zeros([8]))
    let W2 = learnable(rand([8, 3]))
    let b2 = learnable(zeros([3]))

    learn {
        optimizer: adam(lr=0.1)
        epochs: 200

        for epoch in range(epochs) {
            let h1 = relu(matmul(X, W1) + b1)
            let y_pred = matmul(h1, W2) + b2

            let diff = y_pred - y
            let loss = mean(diff * diff)

            backward(loss)

            if epoch % 40 == 0 {
                print("Epoch", epoch, "Loss:", loss)
            }
        }
    }

    // Test
    let test_x = [[5.0]]
    let h1 = relu(matmul(test_x, W1) + b1)
    let pred = matmul(h1, W2) + b2

    print("\nTest x=5:")
    print("  Predicted:", pred)
    print("  Expected: [10, 15, 25]")
    print("✅ Multi-output model trained!")
}
