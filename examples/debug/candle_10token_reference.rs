// Candle reference: Extract intermediate values at 10 token generation
// Matches the same prompt and settings as TensorLogic's debug_zero_logits_issue.tl

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::llama as model;
use tokenizers::Tokenizer;

struct GenerationState {
    token_count: usize,
    embedding_sum: f32,
    layer0_sum: f32,
    all_layers_sum: f32,
    last_hidden_sum: f32,
    normed_sum: f32,
    logits_sum: f32,
    logits_max: f32,
    logits_min: f32,
}

fn main() -> Result<()> {
    println!("================================================================================");
    println!("Candle Reference: 10 Token Generation Debug");
    println!("================================================================================");
    println!();

    // Use GPU if available
    let device = Device::cuda_if_available(0)?;
    println!("Device: {:?}", device);
    println!();

    // Load model
    let model_path = std::env::var("HOME").unwrap() + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf";
    println!("Loading model: {}", model_path);

    let mut file = std::fs::File::open(&model_path)?;
    let model_content = candle_transformers::models::quantized_llama::ModelWeights::from_gguf(
        &mut file,
        &device,
    )?;
    println!("✓ Model loaded");
    println!();

    // Load tokenizer
    let tokenizer_path = std::env::var("HOME").unwrap() + "/.llm/tokenizers/tinyllama-tokenizer.json";
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    println!("✓ Tokenizer loaded");
    println!();

    // Prepare prompt (same as TensorLogic)
    let system_prompt = "You are a friendly and helpful AI assistant.";
    let user_message = "Hello";
    let formatted_prompt = format!("<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n", system_prompt, user_message);

    let encoding = tokenizer.encode(formatted_prompt.clone(), false)
        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
    let prompt_tokens: Vec<u32> = encoding.get_ids().iter().map(|&id| id).collect();
    let prompt_len = prompt_tokens.len();

    println!("Prompt: {}", formatted_prompt.replace("\n", "\\n"));
    println!("Prompt tokens: {:?}", prompt_tokens);
    println!("Prompt length: {}", prompt_len);
    println!();

    // Generation settings
    let temperature = 0.7f32;
    let eos_token_id = 2u32;

    println!("================================================================================");
    println!("Generating 10 tokens with intermediate value tracking...");
    println!("================================================================================");
    println!();

    let mut gen_tokens = prompt_tokens.clone();
    let mut states: Vec<GenerationState> = Vec::new();

    // Generate 10 tokens
    for token_step in 0..10 {
        println!("--------------------------------------------------------------------------------");
        println!("Token {} - Sequence length: {}", token_step + 1, gen_tokens.len());
        println!("--------------------------------------------------------------------------------");

        // Convert tokens to tensor
        let input_ids = Tensor::new(gen_tokens.clone(), &device)?.unsqueeze(0)?;

        // Get embeddings
        let embeddings = model_content.embed(&input_ids)?;
        let emb_sum = embeddings.sum_all()?.to_scalar::<f32>()?;
        println!("  Embedding sum: {}", emb_sum);

        // Process through model (this is simplified - actual implementation depends on Candle's API)
        // We need to track intermediate values through the forward pass
        let logits = model_content.forward(&input_ids, 0)?;

        // Extract last token logits
        let (_b_size, seq_len, _vocab_size) = logits.dims3()?;
        let last_logits = logits.i((.., seq_len - 1, ..))?;

        let logits_sum = last_logits.sum_all()?.to_scalar::<f32>()?;
        let logits_max = last_logits.max(1)?.max_keepdim(0)?.to_scalar::<f32>()?;
        let logits_min = last_logits.min(1)?.min_keepdim(0)?.to_scalar::<f32>()?;

        println!("  Logits sum: {}", logits_sum);
        println!("  Logits max: {}", logits_max);
        println!("  Logits min: {}", logits_min);

        // Sample next token
        let next_token = sample_token(&last_logits, temperature)?;
        println!("  Sampled token ID: {}", next_token);
        println!();

        // Store state for token 10
        if token_step == 9 {
            println!("*** TOKEN 10 DETAILED VALUES ***");
            println!("  Embedding sum: {}", emb_sum);
            println!("  Logits sum: {}", logits_sum);
            println!("  Logits max: {}", logits_max);
            println!("  Logits min: {}", logits_min);
            println!();
        }

        if next_token == eos_token_id {
            println!("EOS reached after {} tokens", token_step + 1);
            break;
        }

        gen_tokens.push(next_token);
    }

    println!("================================================================================");
    println!("Reference extraction complete");
    println!("================================================================================");

    Ok(())
}

fn sample_token(logits: &Tensor, temperature: f32) -> Result<u32> {
    let logits = (logits / temperature as f64)?;
    let probs = candle_nn::ops::softmax_last_dim(&logits)?;

    // Sample from distribution
    let probs_vec = probs.to_vec1::<f32>()?;
    let mut rng = rand::thread_rng();
    use rand::Rng;

    let sum: f32 = probs_vec.iter().sum();
    let mut cumsum = 0.0f32;
    let random_val: f32 = rng.gen_range(0.0..sum);

    for (idx, &prob) in probs_vec.iter().enumerate() {
        cumsum += prob;
        if cumsum >= random_val {
            return Ok(idx as u32);
        }
    }

    Ok((probs_vec.len() - 1) as u32)
}
