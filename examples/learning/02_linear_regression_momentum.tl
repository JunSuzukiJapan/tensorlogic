// Linear Regression with SGD + Momentum
//
// Same as 01 but uses momentum for faster convergence

main {
    print("=== Linear Regression with Momentum ===")

    let X_train = [[1.0], [2.0], [3.0], [4.0], [5.0]]
    let y_train = [[3.0], [5.0], [7.0], [9.0], [11.0]]

    let W = learnable(ones([1, 1]))
    let b = learnable(zeros([1]))

    learn {
        optimizer: sgd(lr=0.01, momentum=0.9)
        epochs: 100

        for epoch in range(epochs) {
            let y_pred = matmul(X_train, W) + b
            let diff = y_pred - y_train
            let loss = mean(diff * diff)

            backward(loss)

            if epoch % 20 == 0 {
                print("Epoch", epoch, "Loss:", loss)
            }
        }
    }

    print("\nWith momentum:")
    print("  W:", W)
    print("  b:", b)
    print("✅ Momentum accelerated convergence!")
}
