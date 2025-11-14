// Inspect GGUF file structure and tensor metadata

use std::path::Path;
use std::fs::File;
use gguf_rs_lib::prelude::*;

fn main() -> std::io::Result<()> {
    let gguf_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap();
            format!("{}/.llm/models/tinyllama-1.1b-chat-f16.gguf", home)
        });

    println!("================================================================================");
    println!("GGUF File Inspector");
    println!("================================================================================");
    println!();
    println!("File: {}", gguf_path);
    println!();

    // Open GGUF file
    let path = Path::new(&gguf_path);
    let file = File::open(path)?;
    let mut reader = GGUFFileReader::new(file)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?;

    // Print tensor information
    println!("[Tensors]");
    println!();

    let tensor_infos = reader.tensor_infos().to_vec();
    for tensor_info in &tensor_infos {
        println!("Tensor: {}", tensor_info.name);
        println!("  Type: {:?}", tensor_info.tensor_type);
        println!("  Shape: {:?}", tensor_info.shape);

        // Load tensor data for token_embd.weight
        if tensor_info.name == "token_embd.weight" {
            println!("  [Loading data for analysis...]");

            let data = reader.load_tensor_data(&tensor_info.name)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e.to_string()))?
                .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "Tensor data not found"))?;

            let bytes = match data {
                gguf_rs_lib::tensor::TensorData::Owned(ref v) => v.as_slice(),
                gguf_rs_lib::tensor::TensorData::Borrowed(b) => b,
                gguf_rs_lib::tensor::TensorData::Shared(ref arc) => arc.as_slice(),
                _ => return Err(std::io::Error::new(std::io::ErrorKind::Other, "Unexpected tensor data type")),
            };

            println!("  Data size: {} bytes", bytes.len());

            // Convert first values to f16
            println!("  First 10 f16 values:");
            for i in 0..10 {
                if i * 2 + 1 < bytes.len() {
                    let f16_val = half::f16::from_le_bytes([bytes[i * 2], bytes[i * 2 + 1]]);
                    println!("    [{}]: {}", i, f16_val.to_f32());
                }
            }

            // Calculate BOS token (ID=1) position
            if tensor_info.shape.dimensions.len() == 2 {
                let dim0 = tensor_info.shape.dimensions[0] as usize;
                let dim1 = tensor_info.shape.dimensions[1] as usize;

                println!("  Tensor dimensions: [{}, {}]", dim0, dim1);

                // GGUF stores in row-major order
                // Try both [vocab, emb_dim] and [emb_dim, vocab] interpretations
                println!("\n  Interpretation 1: shape = [vocab_size={}, emb_dim={}]", dim0, dim1);
                let bos_start_1 = dim1 * 2; // Token ID=1, skip token 0
                println!("    BOS token (ID=1) starts at byte {}", bos_start_1);
                println!("    BOS embedding (first 10 dims):");
                for i in 0..10 {
                    let byte_pos = bos_start_1 + i * 2;
                    if byte_pos + 1 < bytes.len() {
                        let f16_val = half::f16::from_le_bytes([bytes[byte_pos], bytes[byte_pos + 1]]);
                        println!("      [{}]: {}", i, f16_val.to_f32());
                    }
                }

                println!("\n  Interpretation 2: shape = [emb_dim={}, vocab_size={}]", dim0, dim1);
                println!("    (transposed layout - token embeddings are columns)");
                println!("    BOS token (ID=1) first 10 dims (strided access):");
                for i in 0..10 {
                    // Token 1, dimension i in column-major: i * vocab_size + 1
                    let elem_index = i * dim1 + 1;
                    let byte_pos = elem_index * 2;
                    if byte_pos + 1 < bytes.len() {
                        let f16_val = half::f16::from_le_bytes([bytes[byte_pos], bytes[byte_pos + 1]]);
                        println!("      [{}]: {}", i, f16_val.to_f32());
                    }
                }
            }
        }

        println!();
    }

    println!("================================================================================");

    Ok(())
}
