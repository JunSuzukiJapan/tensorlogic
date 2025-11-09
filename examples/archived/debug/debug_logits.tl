// Debug: Print top-10 logits to compare with llama.cpp

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    x * sig
}

fn gqa_attention(
    x: float16[?, ?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let x_shape = shape(x)
    let seq_len = x_shape[0]

    let Q = matmul(x, W_q)
    let K = matmul(x, W_k)
    let V = matmul(x, W_v)

    let Q_heads = reshape(Q, [seq_len, 32.0, 64.0])
    let K_heads = reshape(K, [seq_len, 4.0, 64.0])
    let V_heads = reshape(V, [seq_len, 4.0, 64.0])

    let K_with_group = reshape(K_heads, [seq_len, 4.0, 1.0, 64.0])
    let V_with_group = reshape(V_heads, [seq_len, 4.0, 1.0, 64.0])

    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4.0, 8.0, 64.0])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4.0, 8.0, 64.0])

    let K_expanded = reshape(K_broadcast, [seq_len, 32.0, 64.0])
    let V_expanded = reshape(V_broadcast, [seq_len, 32.0, 64.0])

    let scores = einsum("ihd,jhd->ihj", Q_heads, K_expanded)
    let scaled_scores = scores * 0.125
    let attn_weights = softmax(scaled_scores, 2)

    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    let attn_reshaped = reshape(attn_output, [seq_len, 2048.0])
    matmul(attn_reshaped, W_o)
}

fn transformer_layer(
    x: float16[?, ?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?],
    attn_norm: float16[?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?],
    ffn_norm: float16[?]
) -> float16[?, ?] {
    let x_norm = rms_norm(x, attn_norm)
    let attn_out = gqa_attention(x_norm, W_q, W_k, W_v, W_o)
    let h1 = x + attn_out

    let h1_norm = rms_norm(h1, ffn_norm)
    let gate = matmul(h1_norm, W_gate)
    let up = matmul(h1_norm, W_up)
    let gate_act = silu(gate)
    let ffn_hidden = gate_act * up
    let ffn_out = matmul(ffn_hidden, W_down)

    h1 + ffn_out
}

main {
    print("=== Logits Debug Test ===")
    print("")

    // Load
    print("[1/3] Loading model and tokenizer...")
    let model = load_model("/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-q4_0.gguf")
    let tokenizer = load_tokenizer("/Users/junsuzuki/.llm/tokenizers/tinyllama-tokenizer.json")

    let embed_table = get_tensor(model, "token_embd.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    // Load layer 0 only
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")

    print("      ✓ Loaded layer 0")

    // Tokenize
    print("[2/3] Tokenizing...")
    let prompt = "<|system|>\nYou are a helpful assistant.\n<|user|>\nHello\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, true)
    print("      Tokens:", tokens)

    // Inference
    print("[3/3] Running inference...")
    let embeddings = embedding(embed_table, tokens)
    print("  Embedding shape:", shape(embeddings))

    let h0 = transformer_layer(embeddings, W_q_0, W_k_0, W_v_0, W_o_0, attn_norm_0, W_gate_0, W_up_0, W_down_0, ffn_norm_0)
    print("  After layer 0:", shape(h0))

    let final_norm = rms_norm(h0, output_norm)
    let logits = matmul(final_norm, output_weight)
    print("  Logits shape:", shape(logits))

    // Get last position logits
    let logits_shape = shape(logits)
    let seq_len = logits_shape[0]
    let vocab_size = logits_shape[1]
    print("  Sequence length:", seq_len)
    print("  Vocab size:", vocab_size)

    // Print top-10 logits
    print("")
    print("Expected from llama.cpp (temp=0.0, 'Hello' prompt):")
    print("  Top token: 'Write' or similar")
    print("")
    print("TensorLogic output:")

    // Sample
    let next_token = temperature_sample(logits, 0.0)
    print("  First token ID:", next_token)


    print("")
    print("If this doesn't match llama.cpp, the problem is in:")
    print("  - Model weight loading/quantization")
    print("  - RMSNorm implementation")
    print("  - SwiGLU FFN implementation")
    print("  - Or another operation")
}
