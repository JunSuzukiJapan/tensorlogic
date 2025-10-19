use tensorlogic::{MetalDevice, Tensor};
use tensorlogic::tensor::TensorShape;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== TensorLogic f16 Demo ===\n");

    // Initialize Metal device
    let device = MetalDevice::new()?;
    println!("Metal Device: {}", device.name());
    println!("Supports f16: {}\n", device.supports_f16());

    // Create tensors
    println!("Creating tensors...");
    let a = Tensor::ones(&device, vec![3])?;
    let b = Tensor::from_vec_metal(
        &device,
        vec![
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0),
        ],
        vec![3],
    )?;

    println!("a = {:?}", a.to_vec_f32());
    println!("b = {:?}", b.to_vec_f32());

    // Perform operations
    println!("\nPerforming operations...");

    let c = a.add(&b)?;
    println!("a + b = {:?}", c.to_vec_f32());

    let d = b.mul(&b)?;
    println!("b * b = {:?}", d.to_vec_f32());

    // Test activation functions
    println!("\nTesting activation functions...");
    let x = Tensor::from_vec_metal(
        &device,
        vec![
            half::f16::from_f32(-2.0),
            half::f16::from_f32(-1.0),
            half::f16::from_f32(0.0),
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
        ],
        vec![5],
    )?;

    let relu_result = x.relu()?;
    println!("ReLU({:?}) = {:?}", x.to_vec_f32(), relu_result.to_vec_f32());

    let gelu_result = x.gelu()?;
    println!("GELU({:?}) = {:?}", x.to_vec_f32(), gelu_result.to_vec_f32());

    let softmax_input = Tensor::from_vec_metal(
        &device,
        vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
        ],
        vec![3],
    )?;
    let softmax_result = softmax_input.softmax()?;
    println!("Softmax({:?}) = {:?}", softmax_input.to_vec_f32(), softmax_result.to_vec_f32());

    // Test matrix multiplication
    println!("\nTesting matrix multiplication...");
    let mat_a = Tensor::from_vec_metal(
        &device,
        vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0),
            half::f16::from_f32(5.0),
            half::f16::from_f32(6.0),
        ],
        vec![2, 3],
    )?;

    let mat_b = Tensor::from_vec_metal(
        &device,
        vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0),
            half::f16::from_f32(5.0),
            half::f16::from_f32(6.0),
        ],
        vec![3, 2],
    )?;

    let mat_c = mat_a.matmul(&mat_b)?;
    println!("Matrix A [2x3]: {:?}", mat_a.to_vec_f32());
    println!("Matrix B [3x2]: {:?}", mat_b.to_vec_f32());
    println!("A @ B [2x2]: {:?}", mat_c.to_vec_f32());

    // Test broadcasting
    println!("\nTesting broadcasting...");
    let vec = Tensor::from_vec_metal(
        &device,
        vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
        ],
        vec![3],
    )?;

    let broadcast_shape = TensorShape::new(vec![2, 3]);
    let broadcasted = vec.broadcast_to(&broadcast_shape)?;
    println!("Vector [3]: {:?}", vec.to_vec_f32());
    println!("Broadcasted to [2x3]: {:?}", broadcasted.to_vec_f32());

    // Test reduction operations
    println!("\nTesting reduction operations...");
    let matrix = Tensor::from_vec_metal(
        &device,
        vec![
            half::f16::from_f32(1.0),
            half::f16::from_f32(2.0),
            half::f16::from_f32(3.0),
            half::f16::from_f32(4.0),
            half::f16::from_f32(5.0),
            half::f16::from_f32(6.0),
        ],
        vec![2, 3],
    )?;

    println!("Matrix [2x3]: {:?}", matrix.to_vec_f32());
    println!("Sum: {}", matrix.sum()?.to_f32());
    println!("Mean: {}", matrix.mean()?.to_f32());
    println!("Max: {}", matrix.max()?.to_f32());
    println!("Min: {}", matrix.min()?.to_f32());

    let sum_rows = matrix.sum_dim(0, false)?;
    println!("Sum along rows: {:?}", sum_rows.to_vec_f32());

    let sum_cols = matrix.sum_dim(1, false)?;
    println!("Sum along columns: {:?}", sum_cols.to_vec_f32());

    println!("\n✅ All operations completed successfully!");
    println!("   - Element-wise: add, sub, mul, div");
    println!("   - Activations: ReLU, GELU, Softmax");
    println!("   - Matrix ops: matmul");
    println!("   - Broadcasting: broadcast_to");
    println!("   - Reductions: sum, mean, max, min, sum_dim");
    println!("   - All running on Metal GPU with f16 precision! 🚀");

    Ok(())
}
