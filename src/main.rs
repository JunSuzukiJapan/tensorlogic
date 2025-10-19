use tensorlogic::{MetalDevice, Tensor};

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

    // Create 2D tensors
    println!("\nCreating 2D tensors...");
    let matrix = Tensor::zeros(&device, vec![2, 3])?;
    println!("Zeros matrix shape: {:?}", matrix.dims());

    let flat = matrix.flatten()?;
    println!("Flattened shape: {:?}", flat.dims());

    println!("\n✅ All operations completed successfully!");

    Ok(())
}
