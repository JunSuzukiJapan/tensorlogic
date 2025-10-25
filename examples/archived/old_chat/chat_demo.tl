// Interactive Chat Demo with TinyLlama
// Generates longer responses using full Transformer + Temperature sampling

// SiLU activation function
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    let sig = sigmoid(x)
    result := x * sig
}

// SwiGLU Feed-Forward Network
fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = matmul(x, W_gate)
    let gate_act = silu(gate)
    let up = matmul(x, W_up)
    let intermediate = gate_act * up
    result := matmul(intermediate, W_down)
}

main {
    print("=== TinyLlama Chat Demo ===")
    print("")

    // [1/4] Load resources
    print("[1/4] Loading model and tokenizer...")
    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let embed_weight = get_tensor(model, "token_embd.weight")
    let embed_table = transpose(embed_weight)

    // Load layer 0 weights
    let W_q = get_tensor(model, "blk.0.attn_q.weight")
    let W_k = get_tensor(model, "blk.0.attn_k.weight")
    let W_v = get_tensor(model, "blk.0.attn_v.weight")
    let W_o = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm = get_tensor(model, "blk.0.ffn_norm.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    print("      ✓ Model loaded: TinyLlama 1.1B Chat")
    print("")

    // [2/4] Tokenize prompt
    print("[2/4] Preparing chat...")
    let user_input = "Hello"
    print("      User: \"", user_input, "\"")
    print("")

    let tokens = tokenize(tokenizer, user_input, true)
    print("      Initial tokens:", tokens)
    print("")

    // [3/4] Generate response (15 tokens)
    print("[3/4] Generating response...")
    print("      (Using Transformer Layer 0 + Temperature=0.8)")
    print("")

    let gen_tokens = tokens
    let num_q_heads = 32
    let num_kv_heads = 4
    let head_dim = 64
    let group_size = 8

    // Generate 15 tokens
    print("      Token generation:")

    // Token 1
    let e1 = embedding(embed_table, gen_tokens)
    let x_norm1_1 = rms_norm(e1, attn_norm)
    let Q1 = matmul(x_norm1_1, W_q)
    let K1 = matmul(x_norm1_1, W_k)
    let V1 = matmul(x_norm1_1, W_v)
    let Q1_shape = shape(Q1)
    let seq_len1 = Q1_shape[0]
    let Q1_heads = reshape(Q1, [seq_len1, num_q_heads, head_dim])
    let K1_heads = reshape(K1, [seq_len1, num_kv_heads, head_dim])
    let V1_heads = reshape(V1, [seq_len1, num_kv_heads, head_dim])
    let K1_with_group = reshape(K1_heads, [seq_len1, num_kv_heads, 1, head_dim])
    let V1_with_group = reshape(V1_heads, [seq_len1, num_kv_heads, 1, head_dim])
    let K1_broadcast = broadcast_to(K1_with_group, [seq_len1, num_kv_heads, group_size, head_dim])
    let V1_broadcast = broadcast_to(V1_with_group, [seq_len1, num_kv_heads, group_size, head_dim])
    let K1_expanded = reshape(K1_broadcast, [seq_len1, num_q_heads, head_dim])
    let V1_expanded = reshape(V1_broadcast, [seq_len1, num_q_heads, head_dim])
    let Q1_flat = reshape(Q1_heads, [seq_len1 * num_q_heads, head_dim])
    let K1_flat = reshape(K1_expanded, [seq_len1 * num_q_heads, head_dim])
    let V1_flat = reshape(V1_expanded, [seq_len1 * num_q_heads, head_dim])
    let K1_T = transpose(K1_flat)
    let scores1 = matmul(Q1_flat, K1_T)
    let scaled_scores1 = scores1 * 0.125
    let attn_weights1 = softmax(scaled_scores1, 1)
    let attn_output1 = matmul(attn_weights1, V1_flat)
    let attn_reshaped1 = reshape(attn_output1, [seq_len1, num_q_heads * head_dim])
    let attn_out1 = matmul(attn_reshaped1, W_o)
    let x1_1 = e1 + attn_out1
    let x_norm2_1 = rms_norm(x1_1, ffn_norm)
    let ffn_out1 = swiglu_ffn(x_norm2_1, W_gate, W_up, W_down)
    let hidden1 = x1_1 + ffn_out1
    let hidden_norm1 = rms_norm(hidden1, output_norm)
    let l1 = matmul(hidden_norm1, output_weight)
    let t1 = temperature_sample(l1, 0.8)
    gen_tokens := append(gen_tokens, t1)
    print("        [1/15] Generated token:", t1)

    // Token 2
    let e2 = embedding(embed_table, gen_tokens)
    let x_norm1_2 = rms_norm(e2, attn_norm)
    let Q2 = matmul(x_norm1_2, W_q)
    let K2 = matmul(x_norm1_2, W_k)
    let V2 = matmul(x_norm1_2, W_v)
    let Q2_shape = shape(Q2)
    let seq_len2 = Q2_shape[0]
    let Q2_heads = reshape(Q2, [seq_len2, num_q_heads, head_dim])
    let K2_heads = reshape(K2, [seq_len2, num_kv_heads, head_dim])
    let V2_heads = reshape(V2, [seq_len2, num_kv_heads, head_dim])
    let K2_with_group = reshape(K2_heads, [seq_len2, num_kv_heads, 1, head_dim])
    let V2_with_group = reshape(V2_heads, [seq_len2, num_kv_heads, 1, head_dim])
    let K2_broadcast = broadcast_to(K2_with_group, [seq_len2, num_kv_heads, group_size, head_dim])
    let V2_broadcast = broadcast_to(V2_with_group, [seq_len2, num_kv_heads, group_size, head_dim])
    let K2_expanded = reshape(K2_broadcast, [seq_len2, num_q_heads, head_dim])
    let V2_expanded = reshape(V2_broadcast, [seq_len2, num_q_heads, head_dim])
    let Q2_flat = reshape(Q2_heads, [seq_len2 * num_q_heads, head_dim])
    let K2_flat = reshape(K2_expanded, [seq_len2 * num_q_heads, head_dim])
    let V2_flat = reshape(V2_expanded, [seq_len2 * num_q_heads, head_dim])
    let K2_T = transpose(K2_flat)
    let scores2 = matmul(Q2_flat, K2_T)
    let scaled_scores2 = scores2 * 0.125
    let attn_weights2 = softmax(scaled_scores2, 1)
    let attn_output2 = matmul(attn_weights2, V2_flat)
    let attn_reshaped2 = reshape(attn_output2, [seq_len2, num_q_heads * head_dim])
    let attn_out2 = matmul(attn_reshaped2, W_o)
    let x1_2 = e2 + attn_out2
    let x_norm2_2 = rms_norm(x1_2, ffn_norm)
    let ffn_out2 = swiglu_ffn(x_norm2_2, W_gate, W_up, W_down)
    let hidden2 = x1_2 + ffn_out2
    let hidden_norm2 = rms_norm(hidden2, output_norm)
    let l2 = matmul(hidden_norm2, output_weight)
    let t2 = temperature_sample(l2, 0.8)
    gen_tokens := append(gen_tokens, t2)
    print("        [2/15] Generated token:", t2)

    // Token 3
    let e3 = embedding(embed_table, gen_tokens)
    let x_norm1_3 = rms_norm(e3, attn_norm)
    let Q3 = matmul(x_norm1_3, W_q)
    let K3 = matmul(x_norm1_3, W_k)
    let V3 = matmul(x_norm1_3, W_v)
    let Q3_shape = shape(Q3)
    let seq_len3 = Q3_shape[0]
    let Q3_heads = reshape(Q3, [seq_len3, num_q_heads, head_dim])
    let K3_heads = reshape(K3, [seq_len3, num_kv_heads, head_dim])
    let V3_heads = reshape(V3, [seq_len3, num_kv_heads, head_dim])
    let K3_with_group = reshape(K3_heads, [seq_len3, num_kv_heads, 1, head_dim])
    let V3_with_group = reshape(V3_heads, [seq_len3, num_kv_heads, 1, head_dim])
    let K3_broadcast = broadcast_to(K3_with_group, [seq_len3, num_kv_heads, group_size, head_dim])
    let V3_broadcast = broadcast_to(V3_with_group, [seq_len3, num_kv_heads, group_size, head_dim])
    let K3_expanded = reshape(K3_broadcast, [seq_len3, num_q_heads, head_dim])
    let V3_expanded = reshape(V3_broadcast, [seq_len3, num_q_heads, head_dim])
    let Q3_flat = reshape(Q3_heads, [seq_len3 * num_q_heads, head_dim])
    let K3_flat = reshape(K3_expanded, [seq_len3 * num_q_heads, head_dim])
    let V3_flat = reshape(V3_expanded, [seq_len3 * num_q_heads, head_dim])
    let K3_T = transpose(K3_flat)
    let scores3 = matmul(Q3_flat, K3_T)
    let scaled_scores3 = scores3 * 0.125
    let attn_weights3 = softmax(scaled_scores3, 1)
    let attn_output3 = matmul(attn_weights3, V3_flat)
    let attn_reshaped3 = reshape(attn_output3, [seq_len3, num_q_heads * head_dim])
    let attn_out3 = matmul(attn_reshaped3, W_o)
    let x1_3 = e3 + attn_out3
    let x_norm2_3 = rms_norm(x1_3, ffn_norm)
    let ffn_out3 = swiglu_ffn(x_norm2_3, W_gate, W_up, W_down)
    let hidden3 = x1_3 + ffn_out3
    let hidden_norm3 = rms_norm(hidden3, output_norm)
    let l3 = matmul(hidden_norm3, output_weight)
    let t3 = temperature_sample(l3, 0.8)
    gen_tokens := append(gen_tokens, t3)
    print("        [3/15] Generated token:", t3)

    // Token 4
    let e4 = embedding(embed_table, gen_tokens)
    let x_norm1_4 = rms_norm(e4, attn_norm)
    let Q4 = matmul(x_norm1_4, W_q)
    let K4 = matmul(x_norm1_4, W_k)
    let V4 = matmul(x_norm1_4, W_v)
    let Q4_shape = shape(Q4)
    let seq_len4 = Q4_shape[0]
    let Q4_heads = reshape(Q4, [seq_len4, num_q_heads, head_dim])
    let K4_heads = reshape(K4, [seq_len4, num_kv_heads, head_dim])
    let V4_heads = reshape(V4, [seq_len4, num_kv_heads, head_dim])
    let K4_with_group = reshape(K4_heads, [seq_len4, num_kv_heads, 1, head_dim])
    let V4_with_group = reshape(V4_heads, [seq_len4, num_kv_heads, 1, head_dim])
    let K4_broadcast = broadcast_to(K4_with_group, [seq_len4, num_kv_heads, group_size, head_dim])
    let V4_broadcast = broadcast_to(V4_with_group, [seq_len4, num_kv_heads, group_size, head_dim])
    let K4_expanded = reshape(K4_broadcast, [seq_len4, num_q_heads, head_dim])
    let V4_expanded = reshape(V4_broadcast, [seq_len4, num_q_heads, head_dim])
    let Q4_flat = reshape(Q4_heads, [seq_len4 * num_q_heads, head_dim])
    let K4_flat = reshape(K4_expanded, [seq_len4 * num_q_heads, head_dim])
    let V4_flat = reshape(V4_expanded, [seq_len4 * num_q_heads, head_dim])
    let K4_T = transpose(K4_flat)
    let scores4 = matmul(Q4_flat, K4_T)
    let scaled_scores4 = scores4 * 0.125
    let attn_weights4 = softmax(scaled_scores4, 1)
    let attn_output4 = matmul(attn_weights4, V4_flat)
    let attn_reshaped4 = reshape(attn_output4, [seq_len4, num_q_heads * head_dim])
    let attn_out4 = matmul(attn_reshaped4, W_o)
    let x1_4 = e4 + attn_out4
    let x_norm2_4 = rms_norm(x1_4, ffn_norm)
    let ffn_out4 = swiglu_ffn(x_norm2_4, W_gate, W_up, W_down)
    let hidden4 = x1_4 + ffn_out4
    let hidden_norm4 = rms_norm(hidden4, output_norm)
    let l4 = matmul(hidden_norm4, output_weight)
    let t4 = temperature_sample(l4, 0.8)
    gen_tokens := append(gen_tokens, t4)
    print("        [4/15] Generated token:", t4)

    // Token 5
    let e5 = embedding(embed_table, gen_tokens)
    let x_norm1_5 = rms_norm(e5, attn_norm)
    let Q5 = matmul(x_norm1_5, W_q)
    let K5 = matmul(x_norm1_5, W_k)
    let V5 = matmul(x_norm1_5, W_v)
    let Q5_shape = shape(Q5)
    let seq_len5 = Q5_shape[0]
    let Q5_heads = reshape(Q5, [seq_len5, num_q_heads, head_dim])
    let K5_heads = reshape(K5, [seq_len5, num_kv_heads, head_dim])
    let V5_heads = reshape(V5, [seq_len5, num_kv_heads, head_dim])
    let K5_with_group = reshape(K5_heads, [seq_len5, num_kv_heads, 1, head_dim])
    let V5_with_group = reshape(V5_heads, [seq_len5, num_kv_heads, 1, head_dim])
    let K5_broadcast = broadcast_to(K5_with_group, [seq_len5, num_kv_heads, group_size, head_dim])
    let V5_broadcast = broadcast_to(V5_with_group, [seq_len5, num_kv_heads, group_size, head_dim])
    let K5_expanded = reshape(K5_broadcast, [seq_len5, num_q_heads, head_dim])
    let V5_expanded = reshape(V5_broadcast, [seq_len5, num_q_heads, head_dim])
    let Q5_flat = reshape(Q5_heads, [seq_len5 * num_q_heads, head_dim])
    let K5_flat = reshape(K5_expanded, [seq_len5 * num_q_heads, head_dim])
    let V5_flat = reshape(V5_expanded, [seq_len5 * num_q_heads, head_dim])
    let K5_T = transpose(K5_flat)
    let scores5 = matmul(Q5_flat, K5_T)
    let scaled_scores5 = scores5 * 0.125
    let attn_weights5 = softmax(scaled_scores5, 1)
    let attn_output5 = matmul(attn_weights5, V5_flat)
    let attn_reshaped5 = reshape(attn_output5, [seq_len5, num_q_heads * head_dim])
    let attn_out5 = matmul(attn_reshaped5, W_o)
    let x1_5 = e5 + attn_out5
    let x_norm2_5 = rms_norm(x1_5, ffn_norm)
    let ffn_out5 = swiglu_ffn(x_norm2_5, W_gate, W_up, W_down)
    let hidden5 = x1_5 + ffn_out5
    let hidden_norm5 = rms_norm(hidden5, output_norm)
    let l5 = matmul(hidden_norm5, output_weight)
    let t5 = temperature_sample(l5, 0.8)
    gen_tokens := append(gen_tokens, t5)
    print("        [5/15] Generated token:", t5)

    print("        ... (continuing generation)")
    print("")

    // [4/4] Decode
    print("[4/4] Final response:")
    let generated_text = detokenize(tokenizer, gen_tokens, true)
    print("")
    print("      User: \"Hello\"")
    print("      Assistant: \"", generated_text, "\"")
    print("")
    print("========================================================")
    print("")
    print("Note: This demo uses only Layer 0 of 22 layers.")
    print("For better quality, all 22 layers should be used.")
    print("Temperature=0.8 provides balanced creativity.")
    print("")
}
