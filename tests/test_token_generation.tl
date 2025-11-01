// ============================================================================
// Token Generation Tests
// ============================================================================
//
// Tests for validating the token generation pipeline step by step
//

// Test 1: Load model and tokenizer
test load_model {
    print("Test: Load model and tokenizer")

    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    print("  ✓ Model loaded successfully")
    print("  ✓ Tokenizer loaded successfully")
}

// Test 2: Tokenize and encode simple text
test tokenize_simple {
    print("Test: Tokenize simple text")

    let home = env("HOME")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let text = "Hello"
    let tokens = tokenizer.tokenize(text, false)
    print("  Input: '", text, "'")
    print("  Tokens: ", tokens)

    let decoded = tokenizer.detokenize(tokens, false)
    print("  Decoded: '", decoded, "'")
    print("  ✓ Tokenization successful")
}

// Test 3: Get embedding for a single token
test single_token_embedding {
    print("Test: Single token embedding")

    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")

    let tok_embd = model.token_embd.weight
    let tok_shape = shape(tok_embd)

    print("  Token embedding shape: [", tok_shape[0], ", ", tok_shape[1], "]")

    // Get embedding for token ID 1 (should be BOS token)
    let token_ids = int_to_tokenids(1)
    let emb = embedding(tok_embd, token_ids)
    let emb_shape = shape(emb)

    print("  Single token embedding shape: [", emb_shape[0], ", ", emb_shape[1], "]")
    print("  ✓ Embedding extraction successful")
}

// Test 4: Forward pass through first layer
test first_layer_forward {
    print("Test: Forward pass through first layer")

    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")

    let tok_embd = model.token_embd.weight
    let L0 = model.blk[0]

    // Get embedding for BOS token
    let token_ids = int_to_tokenids(1)
    let x = embedding(tok_embd, token_ids)

    // Apply layer norm
    let normed = rms_norm(x, L0.attn_norm.weight)
    let normed_shape = shape(normed)

    print("  Normalized shape: [", normed_shape[0], ", ", normed_shape[1], "]")

    // Compute Q, K, V
    let Q = linear(normed, L0.attn_q.weight)
    let K = linear(normed, L0.attn_k.weight)
    let V = linear(normed, L0.attn_v.weight)

    let q_shape = shape(Q)
    print("  Q shape: [", q_shape[0], ", ", q_shape[1], "]")

    print("  ✓ First layer forward pass successful")
}

// Test 5: Generate logits for single token
test generate_logits {
    print("Test: Generate logits for single token")

    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight
    let L0 = model.blk[0]

    // Simple forward pass through layer 0 only
    let token_ids = int_to_tokenids(1)
    let x = embedding(tok_embd, token_ids)

    // Attention
    let normed = rms_norm(x, L0.attn_norm.weight)
    let Q = linear(normed, L0.attn_q.weight)
    let K = linear(normed, L0.attn_k.weight)
    let V = linear(normed, L0.attn_v.weight)
    let attn_out = attention_with_cache(Q, K, V, L0.attn_output.weight)
    let after_attn = x + attn_out

    // FFN (skipping for speed)
    // let ffn_norm = rms_norm(after_attn, L0.ffn_norm.weight)
    // Use after_attn directly

    // Output
    let final_norm = rms_norm(after_attn, output_norm)
    let logits = linear(final_norm, output)
    let logits_shape = shape(logits)

    print("  Logits shape: [", logits_shape[0], ", ", logits_shape[1], "]")
    print("  Expected: [1, 32000] (vocab size)")

    // Sample a token
    let token_id = temperature_sample(logits, 1.0)
    print("  Sampled token ID: ", token_id)

    print("  ✓ Logits generation successful")
}

// Test 6: Two-token generation
test two_token_generation {
    print("Test: Two-token generation")

    let home = env("HOME")
    let model = load_model_f16(home + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer(home + "/.llm/tokenizers/tinyllama-tokenizer.json")

    let tok_embd = model.token_embd.weight
    let output_norm = model.output_norm.weight
    let output = model.output.weight
    let L0 = model.blk[0]

    // Start with BOS token
    let bos_token = 1
    let token_ids = int_to_tokenids(bos_token)

    print("  Starting with BOS token (", bos_token, ")")

    // Generate first token
    let x = embedding(tok_embd, token_ids)
    let normed = rms_norm(x, L0.attn_norm.weight)
    let Q = linear(normed, L0.attn_q.weight)
    let K = linear(normed, L0.attn_k.weight)
    let V = linear(normed, L0.attn_v.weight)
    let attn_out = attention_with_cache(Q, K, V, L0.attn_output.weight)
    let after_attn = x + attn_out
    let final_norm = rms_norm(after_attn, output_norm)
    let logits = linear(final_norm, output)

    let token1 = temperature_sample(logits, 0.7)
    print("  Token 1: ", token1)

    // Decode
    let tokens = int_to_tokenids(bos_token)
    let tokens2 = tokens.append_token(token1)
    let text = tokenizer.detokenize(tokens2, false)
    print("  Text so far: '", text, "'")

    // Generate second token
    let token_ids2 = int_to_tokenids(token1)
    let x2 = embedding(tok_embd, token_ids2)
    let normed2 = rms_norm(x2, L0.attn_norm.weight)
    let Q2 = linear(normed2, L0.attn_q.weight)
    let K2 = linear(normed2, L0.attn_k.weight)
    let V2 = linear(normed2, L0.attn_v.weight)

    // Update KV cache (concat)
    let K_cache = concat(K, K2, 0.0)
    let V_cache = concat(V, V2, 0.0)

    let attn_out2 = attention_with_cache(Q2, K_cache, V_cache, L0.attn_output.weight)
    let after_attn2 = x2 + attn_out2
    let final_norm2 = rms_norm(after_attn2, output_norm)
    let logits2 = linear(final_norm2, output)

    let token2 = temperature_sample(logits2, 0.7)
    print("  Token 2: ", token2)

    // Decode final
    let tokens3 = tokens2.append_token(token2)
    let final_text = tokenizer.detokenize(tokens3, false)
    print("  Final text: '", final_text, "'")

    print("  ✓ Two-token generation successful")
}
