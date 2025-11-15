/// Extract reference values using Candle's GGUF loader
///
/// This program loads the TinyLlama Q4_0 model using Candle and extracts
/// key tensor values for comparison with TensorLogic.

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_core::quantized::QTensor;
use std::path::PathBuf;

fn print_qtensor_stats(name: &str, qtensor: &QTensor, num_values: usize) -> Result<Tensor> {
    println!("\n{}", "=".repeat(80));
    println!("Tensor: {}", name);
    println!("{}", "=".repeat(80));

    // Dequantize to regular tensor
    let tensor = qtensor.dequantize(&Device::Cpu)?;

    println!("Shape: {:?}", tensor.dims());
    println!("DType: {:?}", tensor.dtype());

    // Convert to f32 for statistics
    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    let flat = tensor_f32.flatten_all()?;
    let data = flat.to_vec1::<f32>()?;

    println!("\nFirst {} values:", num_values.min(data.len()));
    for (i, val) in data.iter().take(num_values).enumerate() {
        println!("[{}]: {:.10}", i, val);
    }

    // Statistics
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!("\nStatistics:");
    println!("  Sum:  {:.10}", sum);
    println!("  Mean: {:.10}", mean);
    println!("  Min:  {:.10}", min);
    println!("  Max:  {:.10}", max);

    Ok(tensor)
}

fn print_tensor_stats_only(name: &str, tensor: &Tensor, num_values: usize) -> Result<()> {
    println!("\n{}", "=".repeat(80));
    println!("Tensor: {}", name);
    println!("{}", "=".repeat(80));
    println!("Shape: {:?}", tensor.dims());
    println!("DType: {:?}", tensor.dtype());

    // Convert to f32 for statistics
    let tensor_f32 = tensor.to_dtype(DType::F32)?;
    let flat = tensor_f32.flatten_all()?;
    let data = flat.to_vec1::<f32>()?;

    println!("\nFirst {} values:", num_values.min(data.len()));
    for (i, val) in data.iter().take(num_values).enumerate() {
        println!("[{}]: {:.10}", i, val);
    }

    // Statistics
    let sum: f32 = data.iter().sum();
    let mean = sum / data.len() as f32;
    let min = data.iter().copied().fold(f32::INFINITY, f32::min);
    let max = data.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    println!("\nStatistics:");
    println!("  Sum:  {:.10}", sum);
    println!("  Mean: {:.10}", mean);
    println!("  Min:  {:.10}", min);
    println!("  Max:  {:.10}", max);

    Ok(())
}

fn main() -> Result<()> {
    println!("================================================================================");
    println!("CANDLE REFERENCE VALUE EXTRACTOR");
    println!("================================================================================\n");

    // Get model path from command line or use default
    let model_path = if let Some(path_arg) = std::env::args().nth(1) {
        PathBuf::from(path_arg)
    } else {
        PathBuf::from(std::env::var("HOME")?)
            .join(".llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    };

    if !model_path.exists() {
        eprintln!("❌ Error: Model file not found: {}", model_path.display());
        eprintln!("\nUsage: {} [model_path.gguf]", std::env::args().next().unwrap());
        std::process::exit(1);
    }

    println!("Model: {}", model_path.display());
    println!("Device: CPU (for reference extraction)\n");

    // Load model using Candle's quantized GGUF loader
    println!("Loading model with Candle...");
    let device = Device::Cpu;

    let mut file = std::fs::File::open(&model_path)?;
    let model_content = candle_core::quantized::gguf_file::Content::read(&mut file)?;

    println!("Model loaded successfully!");
    println!("Number of tensors: {}", model_content.tensor_infos.len());

    // Extract key tensors
    let key_tensors = [
        "token_embd.weight",
        "blk.0.attn_norm.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "output_norm.weight",
    ];

    println!("\n## Extracting Key Tensors ##\n");

    for tensor_name in &key_tensors {
        // Find tensor in model
        if let Some((name, tensor_info)) = model_content.tensor_infos
            .iter()
            .find(|(n, _)| n.as_str() == *tensor_name)
        {
            println!("\nProcessing: {}", name);
            println!("Original GGUF shape: {:?}", tensor_info.shape);
            println!("GGUF dtype: {:?}", tensor_info.ggml_dtype);

            // Get tensor data (returns QTensor for quantized models)
            let qtensor = model_content.tensor(&mut file, name, &device)?;

            // Check if it's F32 (not quantized) or quantized
            if matches!(tensor_info.ggml_dtype, candle_core::quantized::GgmlDType::F32) {
                // F32 tensor - dequantize and print
                let deq_tensor = qtensor.dequantize(&device)?;
                print_tensor_stats_only(name, &deq_tensor, 10)?;
            } else {
                // Quantized tensor - dequantize and print, keep tensor for BOS extraction
                let deq_tensor = print_qtensor_stats(name, &qtensor, 10)?;

                // Special handling for token_embd.weight BOS token
                if *tensor_name == "token_embd.weight" {
                    let shape = deq_tensor.dims();
                    if shape.len() == 2 {
                        // Extract BOS token (ID=1) embedding using narrow
                        // narrow(dim, start, len)
                        let bos_embedding = deq_tensor.narrow(0, 1, 1)?.squeeze(0)?;
                        let bos_data = bos_embedding.to_dtype(DType::F32)?.to_vec1::<f32>()?;
                        let bos_sum: f32 = bos_data.iter().sum();

                        println!("\n**BOS Token (ID=1) Embedding:**");
                        println!("  Sum: {:.10}", bos_sum);
                        println!("  Mean: {:.10}", bos_sum / bos_data.len() as f32);
                    }
                }
            }
        } else {
            println!("\n⚠️  Tensor '{}' not found", tensor_name);
        }
    }

    println!("\n================================================================================");
    println!("✅ Reference extraction complete");
    println!("================================================================================");

    Ok(())
}
