// Debug test to isolate NaN logits in TinyLlama forward pass
// This will trace intermediate values through each operation

main {
    print("================================================================================")
    print("🔍 Debug: Tracing NaN Logits in Forward Pass")
    print("================================================================================")
    print("")

    // Load model and tokenizer
    let model_path = "/Users/junsuzuki/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = "/Users/junsuzuki/.llm/tokenizers/tinyllama-tokenizer.json"

    print("[1/4] Loading model...")
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Model and tokenizer loaded")
    print("")

    // Get embedding table and layer 0 weights
    print("[2/4] Loading layer 0 weights...")
    let embed_table = get_tensor(model, "token_embd.weight")
    let W_q_0 = get_tensor(model, "blk.0.attn_q.weight")
    let W_k_0 = get_tensor(model, "blk.0.attn_k.weight")
    let W_v_0 = get_tensor(model, "blk.0.attn_v.weight")
    let W_o_0 = get_tensor(model, "blk.0.attn_output.weight")
    let attn_norm_0 = get_tensor(model, "blk.0.attn_norm.weight")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let W_down_0 = get_tensor(model, "blk.0.ffn_down.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    let output_norm = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")
    print("      ✓ Layer 0 weights loaded")
    print("")

    // Create prompt tokens
    print("[3/4] Creating prompt...")
    let system_prompt = "You are a friendly chatbot."
    let user_message = "Hello! How are you?"
    let prompt = format_chat_prompt(system_prompt, user_message)
    let tokens = tokenize(tokenizer, prompt, false)
    let num_tokens = count_tokens(tokens)
    print("      Prompt tokens:", num_tokens)
    print("")

    // Step-by-step forward pass through layer 0
    print("[4/4] Tracing forward pass through layer 0...")
    print("")

    // Step 1: Embedding
    print("  [Step 1] Embedding lookup")
    let x = embedding(embed_table, tokens)
    let x_0_0 = get_scalar(x, 0, 0)
    let x_0_1 = get_scalar(x, 0, 1)
    let x_0_2 = get_scalar(x, 0, 2)
    print("    x[0, 0:3] =", x_0_0, x_0_1, x_0_2)

    // Check for NaN/infinity
    if is_nan(x_0_0) {
        print("    ❌ ERROR: Embedding output contains NaN!")
        exit(1)
    }
    if is_inf(x_0_0) {
        print("    ❌ ERROR: Embedding output contains infinity!")
        exit(1)
    }
    print("    ✓ Embedding output is valid")
    print("")

    // Step 2: Attention normalization
    print("  [Step 2] RMS Normalization (attention)")
    let x_norm = rms_norm(x, attn_norm_0)
    let xn_0_0 = get_scalar(x_norm, 0, 0)
    let xn_0_1 = get_scalar(x_norm, 0, 1)
    let xn_0_2 = get_scalar(x_norm, 0, 2)
    print("    x_norm[0, 0:3] =", xn_0_0, xn_0_1, xn_0_2)

    if is_nan(xn_0_0) {
        print("    ❌ ERROR: RMS norm output contains NaN!")
        exit(1)
    }
    print("    ✓ RMS norm output is valid")
    print("")

    // Step 3: Q, K, V projections
    print("  [Step 3] Q, K, V projections")
    let Q = linear(x_norm, W_q_0)
    let K = linear(x_norm, W_k_0)
    let V = linear(x_norm, W_v_0)

    let q_0_0 = get_scalar(Q, 0, 0)
    let k_0_0 = get_scalar(K, 0, 0)
    let v_0_0 = get_scalar(V, 0, 0)
    print("    Q[0, 0] =", q_0_0)
    print("    K[0, 0] =", k_0_0)
    print("    V[0, 0] =", v_0_0)

    if is_nan(q_0_0) {
        print("    ❌ ERROR: Q projection contains NaN!")
        exit(1)
    }
    if is_nan(k_0_0) {
        print("    ❌ ERROR: K projection contains NaN!")
        exit(1)
    }
    if is_nan(v_0_0) {
        print("    ❌ ERROR: V projection contains NaN!")
        exit(1)
    }
    print("    ✓ Q, K, V projections are valid")
    print("")

    // Step 4: RoPE
    print("  [Step 4] Rotary Position Embedding")
    let Q_rope = apply_rope(Q, 0)
    let K_rope = apply_rope(K, 0)

    let qr_0_0 = get_scalar(Q_rope, 0, 0)
    let kr_0_0 = get_scalar(K_rope, 0, 0)
    print("    Q_rope[0, 0] =", qr_0_0)
    print("    K_rope[0, 0] =", kr_0_0)

    if is_nan(qr_0_0) {
        print("    ❌ ERROR: Q_rope contains NaN!")
        exit(1)
    }
    if is_nan(kr_0_0) {
        print("    ❌ ERROR: K_rope contains NaN!")
        exit(1)
    }
    print("    ✓ RoPE outputs are valid")
    print("")

    // Step 5: Reshape for GQA
    print("  [Step 5] Reshape for Grouped Query Attention")
    let Q_reshaped = reshape(Q_rope, [num_tokens, 32, 64])
    let K_reshaped = reshape(K_rope, [num_tokens, 4, 64])
    let V_reshaped = reshape(V, [num_tokens, 4, 64])
    print("    Q shape: [", num_tokens, ", 32, 64]")
    print("    K shape: [", num_tokens, ", 4, 64]")
    print("    V shape: [", num_tokens, ", 4, 64]")
    print("")

    // Step 6: Expand K, V for GQA (4 heads → 32 heads)
    print("  [Step 6] Expand K, V for GQA (4 → 32 heads)")
    let K_expanded = repeat_kv(K_reshaped, 8)
    let V_expanded = repeat_kv(V_reshaped, 8)

    let ke_0_0_0 = get_scalar(K_expanded, 0, 0, 0)
    let ve_0_0_0 = get_scalar(V_expanded, 0, 0, 0)
    print("    K_expanded[0, 0, 0] =", ke_0_0_0)
    print("    V_expanded[0, 0, 0] =", ve_0_0_0)

    if is_nan(ke_0_0_0) {
        print("    ❌ ERROR: K_expanded contains NaN!")
        exit(1)
    }
    if is_nan(ve_0_0_0) {
        print("    ❌ ERROR: V_expanded contains NaN!")
        exit(1)
    }
    print("    ✓ K, V expansion is valid")
    print("")

    // Step 7: Attention scores (Q @ K^T)
    print("  [Step 7] Compute attention scores (Q @ K^T)")
    let Q_perm = permute(Q_reshaped, [1, 0, 2])
    let K_perm = permute(K_expanded, [1, 0, 2])
    let K_t = permute(K_perm, [0, 2, 1])
    let scores_raw = matmul(Q_perm, K_t)

    let sr_0_0_0 = get_scalar(scores_raw, 0, 0, 0)
    print("    scores_raw[0, 0, 0] =", sr_0_0_0)

    if is_nan(sr_0_0_0) {
        print("    ❌ ERROR: Attention scores contain NaN!")
        print("    This is likely where the NaN originates!")
        exit(1)
    }
    print("    ✓ Attention scores are valid")
    print("")

    // Step 8: Scale scores
    print("  [Step 8] Scale attention scores")
    let scale = 0.125  // 1/sqrt(64)
    let scores_scaled = scores_raw * scale

    let ss_0_0_0 = get_scalar(scores_scaled, 0, 0, 0)
    print("    scores_scaled[0, 0, 0] =", ss_0_0_0)

    if is_nan(ss_0_0_0) {
        print("    ❌ ERROR: Scaled scores contain NaN!")
        exit(1)
    }
    print("    ✓ Scaled scores are valid")
    print("")

    // Step 9: Apply causal mask
    print("  [Step 9] Apply causal mask")
    let scores_masked = apply_causal_mask(scores_scaled)

    let sm_0_0_0 = get_scalar(scores_masked, 0, 0, 0)
    print("    scores_masked[0, 0, 0] =", sm_0_0_0)

    if is_nan(sm_0_0_0) {
        print("    ❌ ERROR: Masked scores contain NaN!")
        exit(1)
    }
    print("    ✓ Masked scores are valid")
    print("")

    // Step 10: Softmax
    print("  [Step 10] Softmax over attention scores")
    let attn_weights = softmax(scores_masked)

    let aw_0_0_0 = get_scalar(attn_weights, 0, 0, 0)
    print("    attn_weights[0, 0, 0] =", aw_0_0_0)

    if is_nan(aw_0_0_0) {
        print("    ❌ ERROR: Attention weights contain NaN!")
        print("    This could be due to all -inf values in softmax input")
        exit(1)
    }
    print("    ✓ Attention weights are valid")
    print("")

    // Step 11: Attention @ V
    print("  [Step 11] Apply attention to values (attn @ V)")
    let V_perm = permute(V_expanded, [1, 0, 2])
    let attn_output_perm = matmul(attn_weights, V_perm)
    let attn_output = permute(attn_output_perm, [1, 0, 2])
    let attn_flat = reshape(attn_output, [num_tokens, 2048])

    let af_0_0 = get_scalar(attn_flat, 0, 0)
    print("    attn_flat[0, 0] =", af_0_0)

    if is_nan(af_0_0) {
        print("    ❌ ERROR: Attention output contains NaN!")
        exit(1)
    }
    print("    ✓ Attention output is valid")
    print("")

    // Step 12: Output projection
    print("  [Step 12] Output projection")
    let attn_proj = linear(attn_flat, W_o_0)

    let ap_0_0 = get_scalar(attn_proj, 0, 0)
    print("    attn_proj[0, 0] =", ap_0_0)

    if is_nan(ap_0_0) {
        print("    ❌ ERROR: Attention projection contains NaN!")
        exit(1)
    }
    print("    ✓ Attention projection is valid")
    print("")

    // Step 13: Residual connection
    print("  [Step 13] Residual connection (x + attn)")
    let h_attn = x + attn_proj

    let ha_0_0 = get_scalar(h_attn, 0, 0)
    print("    h_attn[0, 0] =", ha_0_0)

    if is_nan(ha_0_0) {
        print("    ❌ ERROR: After residual connection contains NaN!")
        exit(1)
    }
    print("    ✓ Residual connection is valid")
    print("")

    // Step 14: FFN normalization
    print("  [Step 14] RMS Normalization (FFN)")
    let h_norm = rms_norm(h_attn, ffn_norm_0)

    let hn_0_0 = get_scalar(h_norm, 0, 0)
    print("    h_norm[0, 0] =", hn_0_0)

    if is_nan(hn_0_0) {
        print("    ❌ ERROR: FFN norm contains NaN!")
        exit(1)
    }
    print("    ✓ FFN norm is valid")
    print("")

    // Step 15: SwiGLU FFN
    print("  [Step 15] SwiGLU FFN")
    let gate = linear(h_norm, W_gate_0)
    let gate_act = silu(gate)
    let up = linear(h_norm, W_up_0)
    let intermediate = gate_act * up
    let ffn_out = linear(intermediate, W_down_0)

    let fo_0_0 = get_scalar(ffn_out, 0, 0)
    print("    ffn_out[0, 0] =", fo_0_0)

    if is_nan(fo_0_0) {
        print("    ❌ ERROR: FFN output contains NaN!")
        exit(1)
    }
    print("    ✓ FFN output is valid")
    print("")

    // Step 16: Final residual
    print("  [Step 16] Final residual connection")
    let h_final = h_attn + ffn_out

    let hf_0_0 = get_scalar(h_final, 0, 0)
    print("    h_final[0, 0] =", hf_0_0)

    if is_nan(hf_0_0) {
        print("    ❌ ERROR: Layer 0 output contains NaN!")
        exit(1)
    }
    print("    ✓ Layer 0 output is valid")
    print("")

    // Step 17: Final output normalization
    print("  [Step 17] Final output normalization")
    let h_output_norm = rms_norm(h_final, output_norm)

    let hon_0_0 = get_scalar(h_output_norm, 0, 0)
    print("    h_output_norm[0, 0] =", hon_0_0)

    if is_nan(hon_0_0) {
        print("    ❌ ERROR: Output norm contains NaN!")
        exit(1)
    }
    print("    ✓ Output norm is valid")
    print("")

    // Step 18: Final logits
    print("  [Step 18] Compute logits")
    let logits = linear(h_output_norm, output_weight)

    let l_0_0 = get_scalar(logits, 0, 0)
    let l_0_1 = get_scalar(logits, 0, 1)
    let l_0_2 = get_scalar(logits, 0, 2)
    print("    logits[0, 0:3] =", l_0_0, l_0_1, l_0_2)

    if is_nan(l_0_0) {
        print("    ❌ ERROR: Final logits contain NaN!")
        exit(1)
    }
    print("    ✓ Final logits are valid")
    print("")

    print("================================================================================")
    print("✅ All intermediate values are valid - No NaN detected!")
    print("================================================================================")
}

// Helper function to check for NaN
fn is_nan(x: float16) -> bool {
    x != x
}

// Helper function to check for infinity
fn is_inf(x: float16) -> bool {
    x == (1.0 / 0.0) || x == (-1.0 / 0.0)
}

// Helper function for SiLU activation
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

// Helper function to format chat prompt
fn format_chat_prompt(system: string, user: string) -> string {
    let s1 = "<|system|>\n"
    let s2 = system
    let s3 = "</s>\n<|user|>\n"
    let s4 = user
    let s5 = "</s>\n<|assistant|>\n"
    concat_strings([s1, s2, s3, s4, s5])
}

// Helper function to get a scalar value from a 2D tensor
fn get_scalar(tensor: float16[?, ?], i: int, j: int) -> float16 {
    let slice_i = slice(tensor, i, i + 1, 0)
    let slice_j = slice(slice_i, j, j + 1, 1)
    let flat = flatten(slice_j)
    flat[0]
}

// Helper function to get a scalar value from a 3D tensor
fn get_scalar(tensor: float16[?, ?, ?], i: int, j: int, k: int) -> float16 {
    let slice_i = slice(tensor, i, i + 1, 0)
    let slice_j = slice(slice_i, j, j + 1, 1)
    let slice_k = slice(slice_j, k, k + 1, 2)
    let flat = flatten(slice_k)
    flat[0]
}
