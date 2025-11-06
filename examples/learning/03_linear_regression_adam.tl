// Linear Regression with Adam
//
// Adam optimizer adapts learning rate per parameter

main {
    print("=== Linear Regression with Adam ===")

    let X_train = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    let y_train = [[3.0], [5.0], [7.0], [9.0], [11.0]]

    let W = learnable(rand([1, 1]))  // Random initialization
    let b = learnable(rand([1]))

    print("Random initialization:")
    print("  W:", W)
    print("  b:", b)

    learn {
        optimizer: adam(lr=0.1)
        epochs: 50

        for epoch in range(epochs) {
            let y_pred = matmul(X_train, W) + b
            let diff = y_pred - y_train
            let loss = mean(diff * diff)

            backward(loss)

            if epoch % 10 == 0 {
                print("Epoch", epoch, "Loss:", loss)
            }
        }
    }

    print("\nAdam converged faster:")
    print("  W:", W, "(target: 2.0)")
    print("  b:", b, "(target: 1.0)")
    print("✅ Adam optimization complete!")
}
