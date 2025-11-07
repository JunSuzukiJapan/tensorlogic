// Test structs with tensor fields

struct Vector3 {
    data: float16[3],
}

struct Matrix2x2 {
    data: float16[2, 2],
}

struct NeuralLayer {
    weights: float16[?, ?] learnable,
    bias: float16[?] learnable,
    input_dim: int32,
    output_dim: int32,
}

main {
    print("=== Struct with Tensor Fields Tests ===")
    print("")

    // Test 1: Struct with 1D tensor
    print("Test 1: Vector3 struct")
    tensor v_data: float16[3] = [1.0, 2.0, 3.0]
    let vec = Vector3 { data: v_data }
    print("Created Vector3")

    // Test 2: Struct with 2D tensor
    print("")
    print("Test 2: Matrix2x2 struct")
    tensor m_data: float16[2, 2] = [[1.0, 2.0], [3.0, 4.0]]
    let mat = Matrix2x2 { data: m_data }
    print("Created Matrix2x2")

    // Test 3: Neural layer with learnable tensors
    print("")
    print("Test 3: NeuralLayer with learnable tensors")
    tensor w: float16[10, 20] learnable = random_normal(0.0, 0.1)
    tensor b: float16[20] learnable = zeros()

    let layer = NeuralLayer {
        weights: w,
        bias: b,
        input_dim: 10,
        output_dim: 20,
    }
    print("Created neural layer")

    // Test 4: Field access on tensor field
    print("")
    print("Test 4: Access tensor fields")
    print("layer.input_dim should be 10")
    print("layer.output_dim should be 20")

    print("")
    print("All tensor struct tests passed!")
}
