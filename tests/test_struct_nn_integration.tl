// Integration test: Neural Network with structs

struct Layer<W, B> {
    weights: W learnable,
    bias: B learnable,
    input_dim: int32,
    output_dim: int32,
    name: string,
}

impl Layer<float16[?, ?], float16[?]> {
    fn new(name: string, input_dim: int32, output_dim: int32) -> Layer<float16[?, ?], float16[?]> {
        tensor w: float16[input_dim, output_dim] learnable = random_normal(0.0, 0.1)
        tensor b: float16[output_dim] learnable = zeros()

        return Layer<float16[?, ?], float16[?]> {
            weights: w,
            bias: b,
            input_dim: input_dim,
            output_dim: output_dim,
            name: name,
        }
    }

    fn forward(self, input: float16[?, ?]) -> float16[?, ?] {
        result := (input @ self.weights) + self.bias
    }
}

struct Network {
    layer1_dim: int32,
    layer2_dim: int32,
    output_dim: int32,
}

impl Network {
    fn new(input_dim: int32, hidden_dim: int32, output_dim: int32) -> Network {
        return Network {
            layer1_dim: input_dim,
            layer2_dim: hidden_dim,
            output_dim: output_dim,
        }
    }
}

main {
    print("=== Neural Network Integration Test ===")
    print("")

    // Test 1: Create network configuration
    print("Test 1: Create network config")
    let config = Network::new(784, 128, 10)
    print("Network config: 784 -> 128 -> 10")

    // Test 2: Create first layer
    print("")
    print("Test 2: Create input layer")
    let layer1 = Layer<float16[?, ?], float16[?]>::new("input", 784, 128)
    print("Created layer: 784 -> 128")

    // Test 3: Create output layer
    print("")
    print("Test 3: Create output layer")
    let layer2 = Layer<float16[?, ?], float16[?]>::new("output", 128, 10)
    print("Created layer: 128 -> 10")

    // Test 4: Forward pass through layer1
    print("")
    print("Test 4: Forward pass (layer 1)")
    tensor input: float16[32, 784] = random_normal(0.0, 1.0)
    let h1 = layer1.forward(input)
    print("Computed hidden layer output")

    // Test 5: Activation
    print("")
    print("Test 5: Apply ReLU activation")
    let h1_activated = relu(h1)
    print("Applied ReLU")

    // Test 6: Forward pass through layer2
    print("")
    print("Test 6: Forward pass (layer 2)")
    let output = layer2.forward(h1_activated)
    print("Computed network output")

    // Test 7: Softmax
    print("")
    print("Test 7: Apply softmax")
    let probabilities = softmax(output)
    print("Computed probabilities")

    print("")
    print("=== Integration test completed successfully! ===")
    print("Successfully created a 2-layer neural network using structs")
}
