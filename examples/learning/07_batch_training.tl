// Mini-Batch Training
//
// Process data in small batches for efficiency

main {
    print("=== Mini-Batch Training ===")

    // Larger dataset
    let X = [[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]]
    let y = [[2.0], [4.0], [6.0], [8.0], [10.0], [12.0], [14.0], [16.0]]

    let W = learnable(rand([1, 1]))
    let b = learnable(zeros([1]))

    let batch_size = 2
    let n_samples = 8
    let n_batches = n_samples / batch_size

    learn {
        optimizer: sgd(lr=0.01, momentum=0.9)
        epochs: 50

        for epoch in range(epochs) {
            let total_loss = 0.0

            // Process each batch
            for batch_idx in range(n_batches) {
                let start = batch_idx * batch_size
                let end = start + batch_size

                // Get batch (simplified indexing)
                let X_batch = X[start:end]
                let y_batch = y[start:end]

                let y_pred = matmul(X_batch, W) + b
                let diff = y_pred - y_batch
                let loss = mean(diff * diff)

                backward(loss)
                total_loss := total_loss + loss
            }

            if epoch % 10 == 0 {
                print("Epoch", epoch, "Avg Loss:", total_loss / n_batches)
            }
        }
    }

    print("\nMini-batch training complete!")
    print("W:", W, "(target: ~2.0)")
    print("b:", b, "(target: ~0.0)")
    print("✅ Batched training efficient!")
}
