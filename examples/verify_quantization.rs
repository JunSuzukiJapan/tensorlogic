//! Verify quantization dequantization matches llama.cpp
//! Focus on Q4_0 (most weights) and Q6_K (output.weight)

use tensorlogic::model::formats::gguf::GGUFFile;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Quantization Verification ===\n");

    let model_path = Path::new("/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf");
    let gguf = GGUFFile::load(model_path)?;

    // Test 1: Verify Q4_0 weight (most common)
    println!("Test 1: Q4_0 Weight Verification");
    println!("  Loading blk.0.attn_q.weight (Q4_0)...");

    let q_weight = gguf.get_tensor_by_name("blk.0.attn_q.weight")
        .ok_or("Tensor not found")?;

    println!("  Shape: {:?}", q_weight.shape);
    println!("  Type: {:?}", q_weight.tensor_type);

    if let Some(data) = &q_weight.data {
        let data_bytes = data.len();
        println!("  Data size: {} bytes", data_bytes);

        // Q4_0 block structure: 20 bytes per block (2 bytes scale + 16 bytes quants + 2 padding)
        // For [2048, 2048] weight: 2048 * 2048 = 4,194,304 elements
        // With Q4_0: 4 bits per element = 0.5 bytes per element
        // Block size: 32 elements per block
        // Number of blocks: 4,194,304 / 32 = 131,072 blocks
        // Total size: 131,072 * 18 = 2,359,296 bytes (18 bytes per block in Q4_0)

        let expected_elements = q_weight.shape.iter().product::<usize>();
        let block_size = 32;
        let num_blocks = (expected_elements + block_size - 1) / block_size;

        println!("  Expected elements: {}", expected_elements);
        println!("  Blocks: {}", num_blocks);
        println!("  Expected size: {} bytes (18 bytes/block for Q4_0)", num_blocks * 18);

        // Show first few bytes
        println!("  First 32 bytes (hex): {:02X?}", &data[..32.min(data.len())]);
    }

    // Test 2: Verify Q6_K weight (output.weight)
    println!("\nTest 2: Q6_K Weight Verification");
    println!("  Loading output.weight (Q6_K)...");

    let output_weight = gguf.get_tensor_by_name("output.weight")
        .ok_or("Tensor not found")?;

    println!("  Shape: {:?}", output_weight.shape);
    println!("  Type: {:?}", output_weight.tensor_type);

    if let Some(data) = &output_weight.data {
        let data_bytes = data.len();
        println!("  Data size: {} bytes", data_bytes);

        // Q6_K block structure: 256 elements per block, 210 bytes per block
        // For [32000, 2048] weight: 65,536,000 elements
        // Number of blocks: 65,536,000 / 256 = 256,000 blocks
        // Total size: 256,000 * 210 = 53,760,000 bytes

        let expected_elements = output_weight.shape.iter().product::<usize>();
        let block_size = 256;
        let num_blocks = (expected_elements + block_size - 1) / block_size;

        println!("  Expected elements: {}", expected_elements);
        println!("  Blocks: {}", num_blocks);
        println!("  Expected size: {} bytes (210 bytes/block for Q6_K)", num_blocks * 210);

        println!("  First 32 bytes (hex): {:02X?}", &data[..32.min(data.len())]);
    }

    // Test 3: Dequantize and check value ranges
    println!("\nTest 3: Dequantization Value Range Check");

    use tensorlogic::tensor::Tensor;
    use tensorlogic::device::MetalDevice;

    let device = MetalDevice::new()?;

    // Load and dequantize a Q4_0 weight
    println!("  Dequantizing blk.0.attn_q.weight...");
    let q_tensor = Tensor::from_gguf_tensor(&device, &q_weight)?;
    let q_values = q_tensor.to_vec();

    // Compute statistics
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    let mut sum = 0.0f32;
    let mut num_zeros = 0;

    for val in &q_values {
        let f = val.to_f32();
        if f < min_val { min_val = f; }
        if f > max_val { max_val = f; }
        sum += f;
        if f == 0.0 { num_zeros += 1; }
    }

    let mean = sum / q_values.len() as f32;

    println!("  Min: {:.6}", min_val);
    println!("  Max: {:.6}", max_val);
    println!("  Mean: {:.6}", mean);
    println!("  Num zeros: {} ({:.2}%)", num_zeros, (num_zeros as f64 / q_values.len() as f64) * 100.0);
    println!("  Total values: {}", q_values.len());

    // Expected for Q4_0: values should be in reasonable range like -0.5 to 0.5
    if min_val < -5.0 || max_val > 5.0 {
        println!("  ⚠️  WARNING: Values outside expected range for weights!");
    } else if min_val.abs() < 0.00001 && max_val.abs() < 0.00001 {
        println!("  ⚠️  WARNING: All values near zero - possible dequantization issue!");
    } else {
        println!("  ✅ Values in reasonable range");
    }

    // Test Q6_K dequantization
    println!("\n  Dequantizing output.weight (Q6_K)...");
    let out_tensor = Tensor::from_gguf_tensor(&device, &output_weight)?;
    let out_values = out_tensor.to_vec();

    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    let mut sum = 0.0f32;
    let mut num_zeros = 0;

    for val in &out_values {
        let f = val.to_f32();
        if f < min_val { min_val = f; }
        if f > max_val { max_val = f; }
        sum += f;
        if f == 0.0 { num_zeros += 1; }
    }

    let mean = sum / out_values.len() as f32;

    println!("  Min: {:.6}", min_val);
    println!("  Max: {:.6}", max_val);
    println!("  Mean: {:.6}", mean);
    println!("  Num zeros: {} ({:.2}%)", num_zeros, (num_zeros as f64 / out_values.len() as f64) * 100.0);
    println!("  Total values: {}", out_values.len());

    if min_val < -5.0 || max_val > 5.0 {
        println!("  ⚠️  WARNING: Values outside expected range!");
    } else if min_val.abs() < 0.00001 && max_val.abs() < 0.00001 {
        println!("  ⚠️  WARNING: All values near zero!");
    } else {
        println!("  ✅ Values in reasonable range");
    }

    println!("\n=== Verification Complete ===");

    Ok(())
}
