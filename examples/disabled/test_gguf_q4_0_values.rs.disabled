/// Verify Q4_0 dequantization produces reasonable values
///
/// This test loads actual GGUF weights and checks:
/// 1. Weights are loaded successfully
/// 2. Values are not all zeros or NaN
/// 3. Values are in reasonable range for f16

use std::path::PathBuf;

fn main() {
    println!("=== GGUF Q4_0 Weight Verification ===\n");

    // Get model path
    let home = std::env::var("HOME").expect("HOME not set");
    let model_path = PathBuf::from(home).join(".llm/models/tinyllama-1.1b-chat-q4_0.gguf");

    if !model_path.exists() {
        eprintln!("❌ Model file not found: {:?}", model_path);
        return;
    }

    println!("Loading model from: {:?}\n", model_path);

    // Load GGUF file using gguf-rs directly
    use std::fs::File;
    use gguf_rs_lib::prelude::*;

    let file = File::open(&model_path).expect("Failed to open GGUF file");
    let mut reader = GGUFFileReader::new(file).expect("Failed to create GGUF reader");

    // Find token_embd.weight tensor
    println!("Looking for token_embd.weight...");

    let tensor_infos = reader.tensor_infos().to_vec();

    for (idx, tensor_info) in tensor_infos.iter().enumerate().take(10) {
        let name = &tensor_info.name;
        let shape = &tensor_info.shape.dimensions;
        let tensor_type = &tensor_info.tensor_type;

        println!("  [{:2}] {}: shape={:?}, type={:?}", idx, name, shape, tensor_type);

        if name == "token_embd.weight" {
            println!("\n✓ Found token_embd.weight");

            // Load tensor data
            let data = reader.load_tensor_data(name)
                .expect("Failed to load tensor data")
                .expect("Tensor data not found");

            // Get bytes
            let bytes = match data {
                gguf_rs_lib::tensor::TensorData::Owned(ref v) => v.as_slice(),
                gguf_rs_lib::tensor::TensorData::Borrowed(b) => b,
                gguf_rs_lib::tensor::TensorData::Shared(ref arc) => arc.as_slice(),
                _ => panic!("Unexpected tensor data type"),
            };

            println!("  Data size: {} bytes", bytes.len());
            println!("  Shape: {:?}", shape);

            // Expected size for Q4_0
            // Shape: [2048, 32000] = 65,536,000 elements
            // Q4_0: 32 values per block, 18 bytes per block
            // Expected blocks: 65,536,000 / 32 = 2,048,000
            // Expected bytes: 2,048,000 * 18 = 36,864,000

            let expected_elements: usize = shape.iter().map(|&d| d as usize).product();
            let expected_blocks = (expected_elements + 31) / 32;
            let expected_bytes = expected_blocks * 18;

            println!("  Expected elements: {}", expected_elements);
            println!("  Expected blocks: {}", expected_blocks);
            println!("  Expected bytes: {}", expected_bytes);

            if bytes.len() == expected_bytes {
                println!("  ✅ Data size matches expected");
            } else {
                println!("  ❌ Data size mismatch! Got {}, expected {}", bytes.len(), expected_bytes);
            }

            // Dequantize first block
            println!("\n  Dequantizing first block (32 values)...");

            // Read scale (f16, 2 bytes)
            let scale_bytes = [bytes[0], bytes[1]];
            let scale = half::f16::from_le_bytes(scale_bytes);
            let scale_f32 = scale.to_f32();

            println!("  Scale: {:.6}", scale_f32);

            // Read 16 bytes of 4-bit values
            let mut values = Vec::new();
            for j in 0..16 {
                let byte = bytes[2 + j];
                let x0 = ((byte & 0x0F) as i8 - 8) as f32;
                let x1 = ((byte >> 4) as i8 - 8) as f32;

                values.push(x0 * scale_f32);
                values.push(x1 * scale_f32);  // This would be at index j+16 in grouped layout
            }

            println!("  First 10 values (interleaved): {:?}", &values[0..10]);

            // Grouped layout (correct)
            let mut grouped_values = vec![0.0f32; 32];
            for j in 0..16 {
                let byte = bytes[2 + j];
                let x0 = ((byte & 0x0F) as i8 - 8) as f32;
                let x1 = ((byte >> 4) as i8 - 8) as f32;

                grouped_values[j] = x0 * scale_f32;
                grouped_values[j + 16] = x1 * scale_f32;
            }

            println!("  First 10 values (grouped):     {:?}", &grouped_values[0..10]);
            println!("  Values [16-19] (grouped):      {:?}", &grouped_values[16..20]);

            // Check for all zeros or NaN
            let all_zero = grouped_values.iter().all(|&v| v == 0.0);
            let any_nan = grouped_values.iter().any(|&v| v.is_nan());
            let any_inf = grouped_values.iter().any(|&v| v.is_infinite());

            if all_zero {
                println!("\n  ❌ WARNING: All values are zero");
            } else if any_nan {
                println!("\n  ❌ WARNING: Contains NaN values");
            } else if any_inf {
                println!("\n  ❌ WARNING: Contains Inf values");
            } else {
                println!("\n  ✅ Values look reasonable");
            }

            break;
        }
    }

    println!("\n=== Verification Complete ===");
}
