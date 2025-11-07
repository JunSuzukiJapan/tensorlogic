// Deep Neural Network (4 layers)
//
// Demonstrates training a deeper architecture

main {
    print("=== Deep Neural Network (4 layers) ===")

    let X = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    let y = [[0.3], [0.7], [1.1], [1.5]]

    // Architecture: 2 -> 16 -> 8 -> 4 -> 1
    let W1 = learnable(rand([2, 16]) * 0.1)
    let b1 = learnable(zeros([16]))

    let W2 = learnable(rand([16, 8]) * 0.1)
    let b2 = learnable(zeros([8]))

    let W3 = learnable(rand([8, 4]) * 0.1)
    let b3 = learnable(zeros([4]))

    let W4 = learnable(rand([4, 1]) * 0.1)
    let b4 = learnable(zeros([1]))

    learn {
        optimizer: adam(lr=0.01)
        epochs: 200

        for epoch in range(epochs) {
            // Layer 1
            let h1 = relu(matmul(X, W1) + b1)

            // Layer 2
            let h2 = relu(matmul(h1, W2) + b2)

            // Layer 3
            let h3 = relu(matmul(h2, W3) + b3)

            // Output layer
            let y_pred = matmul(h3, W4) + b4

            let diff = y_pred - y
            let loss = mean(diff * diff)

            backward(loss)

            if epoch % 40 == 0 {
                print("Epoch", epoch, "Loss:", loss)
            }
        }
    }

    print("\n✅ Deep network trained!")
    print("4 layers successfully optimized")
}
