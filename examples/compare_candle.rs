//! Compare TensorLogic vs Candle logits for the same input
//!
//! Usage:
//!   cargo run --release --example compare_candle

use candle_core::{DType, Device, Tensor};
use candle_transformers::models::quantized_llama as qllama;
use candle_transformers::quantized_var_builder::VarBuilder;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Candle vs TensorLogic Comparison ===\n");

    // Paths
    let model_path = std::env::var("HOME")? + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    let tokenizer_path = std::env::var("HOME")? + "/.llm/tokenizers/tinyllama-tokenizer.json";

    println!("[1/4] Loading tokenizer...");
    let tokenizer = Tokenizer::from_file(&tokenizer_path)
        .map_err(|e| format!("Failed to load tokenizer: {}", e))?;
    println!("  ✓ Tokenizer loaded\n");

    println!("[2/4] Tokenizing input 'Hello'...");
    let prompt = "Hello";
    let encoding = tokenizer.encode(prompt, false)
        .map_err(|e| format!("Tokenization failed: {}", e))?;
    let input_tokens: Vec<u32> = encoding.get_ids().to_vec();
    println!("  Prompt: \"{}\"", prompt);
    println!("  Tokens: {:?}", input_tokens);
    println!("  ✓ Tokenized\n");

    println!("[3/4] Loading Candle GGUF model...");
    let device = Device::new_metal(0)?;

    let mut file = std::fs::File::open(&model_path)?;
    let vb = VarBuilder::from_gguf(&model_path, &device)?;

    // Create model config for TinyLlama
    let config = qllama::Config {
        hidden_size: 2048,
        intermediate_size: 5632,
        vocab_size: 32000,
        num_hidden_layers: 22,
        num_attention_heads: 32,
        num_key_value_heads: 4, // GQA
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        use_flash_attn: false,
        bos_token_id: Some(1),
        eos_token_id: Some(2),
    };

    let model = qllama::ModelWeights::from_gguf(vb, &config)?;
    println!("  ✓ Model loaded (device: {:?})\n", device);

    println!("[4/4] Running forward pass...");

    // Create input tensor
    let input_tensor = Tensor::new(&input_tokens[..], &device)?
        .to_dtype(DType::U32)?
        .unsqueeze(0)?; // Add batch dimension [1, seq_len]

    println!("  Input shape: {:?}", input_tensor.shape());

    // Forward pass (single token only for simplicity)
    let logits = model.forward(&input_tensor, 0)?;

    println!("  Logits shape: {:?}", logits.shape());

    // Get top 5 tokens
    let logits_vec = logits.squeeze(0)?.to_vec1::<f32>()?;
    let mut top_tokens: Vec<(usize, f32)> = logits_vec
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v))
        .collect();
    top_tokens.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    println!("\n  Top 5 tokens:");
    for (i, (token_id, logit_val)) in top_tokens.iter().take(5).enumerate() {
        let token_str = tokenizer.decode(&[*token_id as u32], false)
            .unwrap_or_else(|_| format!("[{}]", token_id));
        println!("    {}: token={}, logit={:.4}, text=\"{}\"",
                 i+1, token_id, logit_val, token_str);
    }

    // Argmax token
    let max_token = top_tokens[0].0;
    let max_token_str = tokenizer.decode(&[max_token as u32], false)
        .unwrap_or_else(|_| format!("[{}]", max_token));
    println!("\n  Greedy (argmax) token: {} (\"{}\")", max_token, max_token_str);

    println!("\n✅ Candle forward pass complete");
    println!("\n📝 Compare with TensorLogic output:");
    println!("   Run: ./target/release/tl run examples/tests/simple_forward.tl");

    Ok(())
}
