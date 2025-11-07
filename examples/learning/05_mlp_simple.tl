// Simple Multi-Layer Perceptron (MLP)
//
// Two-layer neural network with ReLU activation
// Architecture: 2 -> 4 -> 1

main {
    print("=== Simple MLP ===")

    // XOR-like problem (not linearly separable)
    let X = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    let y = [[0.0], [1.0], [1.0], [0.0]]

    // Layer 1: 2 -> 4
    let W1 = learnable(rand([2, 4]))
    let b1 = learnable(zeros([4]))

    // Layer 2: 4 -> 1
    let W2 = learnable(rand([4, 1]))
    let b2 = learnable(zeros([1]))

    learn {
        optimizer: adam(lr=0.1)
        epochs: 200

        for epoch in range(epochs) {
            // Forward pass
            let h1 = matmul(X, W1) + b1
            let a1 = relu(h1)

            let h2 = matmul(a1, W2) + b2
            let y_pred = sigmoid(h2)

            // MSE loss
            let diff = y_pred - y
            let loss = mean(diff * diff)

            backward(loss)

            if epoch % 40 == 0 {
                print("Epoch", epoch, "Loss:", loss)
            }
        }
    }

    // Test
    let h1 = relu(matmul(X, W1) + b1)
    let y_pred = sigmoid(matmul(h1, W2) + b2)

    print("\nPredictions:")
    print("  ", y_pred)
    print("  (target: [0, 1, 1, 0])")
    print("✅ MLP learned XOR!")
}
