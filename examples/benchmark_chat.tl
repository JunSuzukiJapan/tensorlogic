// ============================================================================
// TinyLlama 1.1B Performance Benchmark
// ============================================================================
//
// Purpose: Measure performance of key operations to compare with Candle
//
// Benchmarks:
//   1. Single layer forward pass
//   2. Attention computation
//   3. FFN computation
//   4. Full 22-layer inference
//   5. Total generation time
//
// Usage:
//   cargo build --release
//   timeout 120 ./target/release/tl run examples/benchmark_chat.tl
//
// ============================================================================

// ----------------------------------------------------------------------------
// Helper Functions
// ----------------------------------------------------------------------------

fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

fn swiglu_ffn(
    x: float16[?, ?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let gate = linear(x, W_gate)
    let gate_act = silu(gate)
    let up = linear(x, W_up)
    let intermediate = gate_act * up
    linear(intermediate, W_down)
}

fn tinyllama_gqa_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?],
    W_o: float16[?, ?]
) -> float16[?, ?] {
    let Q_shape_tensor = shape(Q)
    let seq_len = Q_shape_tensor[0]

    let Q_heads = reshape(Q, [seq_len, 32, 64])
    let K_heads = reshape(K, [seq_len, 4, 64])
    let V_heads = reshape(V, [seq_len, 4, 64])

    let Q_rope = rope(Q_heads)
    let K_rope = rope(K_heads)

    let K_with_group = reshape(K_rope, [seq_len, 4, 1, 64])
    let V_with_group = reshape(V_heads, [seq_len, 4, 1, 64])
    let K_broadcast = broadcast_to(K_with_group, [seq_len, 4, 8, 64])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, 4, 8, 64])
    let K_expanded = reshape(K_broadcast, [seq_len, 32, 64])
    let V_expanded = reshape(V_broadcast, [seq_len, 32, 64])

    let scores = einsum("ihd,jhd->ihj", Q_rope, K_expanded)
    let scaled_scores = scores * 0.125

    let seq_len_int = seq_len as int
    let mask_2d = causal_mask(seq_len_int)
    let mask_3d = reshape(mask_2d, [seq_len, 1, seq_len])
    let mask_broadcast = broadcast_to(mask_3d, [seq_len, 32, seq_len])
    let masked_scores = apply_attention_mask(scaled_scores, mask_broadcast)

    let attn_weights = softmax(masked_scores, 2)
    let attn_output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    let attn_reshaped = reshape(attn_output, [seq_len, 2048])
    linear(attn_reshaped, W_o)
}

fn transformer_layer(
    x: float16[?, ?],
    attn_norm: float16[?],
    W_q: float16[?, ?],
    W_k: float16[?, ?],
    W_v: float16[?, ?],
    W_o: float16[?, ?],
    ffn_norm: float16[?],
    W_gate: float16[?, ?],
    W_up: float16[?, ?],
    W_down: float16[?, ?]
) -> float16[?, ?] {
    let x_norm1 = rms_norm(x, attn_norm)
    let Q = linear(x_norm1, W_q)
    let K = linear(x_norm1, W_k)
    let V = linear(x_norm1, W_v)
    let attn_out = tinyllama_gqa_attention(Q, K, V, W_o)
    let x1 = x + attn_out
    let x_norm2 = rms_norm(x1, ffn_norm)
    let ffn_out = swiglu_ffn(x_norm2, W_gate, W_up, W_down)
    x1 + ffn_out
}

// ============================================================================
// Main: Performance Benchmark
// ============================================================================
main {
    print("================================================================================")
    print("🔬 TinyLlama 1.1B Performance Benchmark")
    print("================================================================================")
    print("")

    // ----------------------------------------------------------------------------
    // Setup
    // ----------------------------------------------------------------------------
    print("[Setup] Loading model and tokenizer...")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-q4_0.gguf"
    let model = load_model(model_path)

    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"
    let tokenizer = load_tokenizer(tokenizer_path)

    let emb_weight = get_tensor(model, "token_embd.weight")
    let output_norm_weight = get_tensor(model, "output_norm.weight")
    let output_weight = get_tensor(model, "output.weight")

    print("        ✓ Setup complete")
    print("")

    // Prepare prompt (short)
    let prompt = "<|system|>\nYou are a friendly chatbot.</s>\n<|user|>\nHello!</s>\n<|assistant|>\n"
    let tokens = tokenize(tokenizer, prompt, false)
    let prompt_len = len(tokens)
    print("        Prompt tokens:", prompt_len)
    print("")

    // ----------------------------------------------------------------------------
    // Benchmark 1: Prefill (first forward pass with full sequence)
    // ----------------------------------------------------------------------------
    print("================================================================================")
    print("📊 Benchmark 1: Prefill Phase (", prompt_len, "tokens)")
    print("================================================================================")
    print("")

    let result_emb = []
    let result_x = []

    bench prefill {
        result_emb = embedding(emb_weight, tokens)
        result_x = result_emb

        for layer_idx in range_i(22) {
            let attn_norm = model.blk[layer_idx].attn_norm.weight
            let W_q = model.blk[layer_idx].attn_q.weight
            let W_k = model.blk[layer_idx].attn_k.weight
            let W_v = model.blk[layer_idx].attn_v.weight
            let W_o = model.blk[layer_idx].attn_output.weight
            let ffn_norm = model.blk[layer_idx].ffn_norm.weight
            let W_gate = model.blk[layer_idx].ffn_gate.weight
            let W_up = model.blk[layer_idx].ffn_up.weight
            let W_down = model.blk[layer_idx].ffn_down.weight

            result_x = transformer_layer(result_x, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)
        }
    }

    print("")

    // ----------------------------------------------------------------------------
    // Benchmark 2: Single Layer Forward Pass (decode phase, seq_len=1)
    // ----------------------------------------------------------------------------
    print("================================================================================")
    print("📊 Benchmark 2: Single Layer (decode phase, seq_len=", prompt_len + 1, ")")
    print("================================================================================")
    print("")

    // Generate one token first to get seq_len+1 state
    let gen_tokens = tokens
    let x_full = result_x

    let last_hidden_1d = slice_last(x_full, 0)
    let last_hidden = reshape(last_hidden_1d, [1, 2048])
    let normed = rms_norm(last_hidden, output_norm_weight)
    let logits = linear(normed, output_weight)
    let next_token = temperature_sample(logits, 0.8)
    gen_tokens = append(gen_tokens, next_token)

    // Now benchmark single layer with seq_len+1
    let decode_emb = embedding(emb_weight, gen_tokens)
    let decode_x = decode_emb

    let attn_norm_0 = model.blk[0].attn_norm.weight
    let W_q_0 = model.blk[0].attn_q.weight
    let W_k_0 = model.blk[0].attn_k.weight
    let W_v_0 = model.blk[0].attn_v.weight
    let W_o_0 = model.blk[0].attn_output.weight
    let ffn_norm_0 = model.blk[0].ffn_norm.weight
    let W_gate_0 = model.blk[0].ffn_gate.weight
    let W_up_0 = model.blk[0].ffn_up.weight
    let W_down_0 = model.blk[0].ffn_down.weight

    let result_layer = []

    bench single_layer {
        result_layer = transformer_layer(decode_x, attn_norm_0, W_q_0, W_k_0, W_v_0, W_o_0, ffn_norm_0, W_gate_0, W_up_0, W_down_0)
    }

    print("")

    // ----------------------------------------------------------------------------
    // Benchmark 3: Attention Only (decode phase)
    // ----------------------------------------------------------------------------
    print("================================================================================")
    print("📊 Benchmark 3: Attention Computation (decode phase)")
    print("================================================================================")
    print("")

    let x_norm1 = rms_norm(decode_x, attn_norm_0)
    let Q_bench = linear(x_norm1, W_q_0)
    let K_bench = linear(x_norm1, W_k_0)
    let V_bench = linear(x_norm1, W_v_0)

    let result_attn = []

    bench attention {
        result_attn = tinyllama_gqa_attention(Q_bench, K_bench, V_bench, W_o_0)
    }

    print("")

    // ----------------------------------------------------------------------------
    // Benchmark 4: FFN Only (decode phase)
    // ----------------------------------------------------------------------------
    print("================================================================================")
    print("📊 Benchmark 4: FFN Computation (decode phase)")
    print("================================================================================")
    print("")

    let x1_bench = decode_x + result_attn
    let x_norm2 = rms_norm(x1_bench, ffn_norm_0)

    let result_ffn = []

    bench ffn {
        result_ffn = swiglu_ffn(x_norm2, W_gate_0, W_up_0, W_down_0)
    }

    print("")

    // ----------------------------------------------------------------------------
    // Benchmark 5: Full Decode Pass (all 22 layers)
    // ----------------------------------------------------------------------------
    print("================================================================================")
    print("📊 Benchmark 5: Full Decode Pass (22 layers)")
    print("================================================================================")
    print("")

    let result_decode = []

    bench full_decode {
        result_decode = decode_emb

        for layer_idx in range_i(22) {
            let attn_norm = model.blk[layer_idx].attn_norm.weight
            let W_q = model.blk[layer_idx].attn_q.weight
            let W_k = model.blk[layer_idx].attn_k.weight
            let W_v = model.blk[layer_idx].attn_v.weight
            let W_o = model.blk[layer_idx].attn_output.weight
            let ffn_norm = model.blk[layer_idx].ffn_norm.weight
            let W_gate = model.blk[layer_idx].ffn_gate.weight
            let W_up = model.blk[layer_idx].ffn_up.weight
            let W_down = model.blk[layer_idx].ffn_down.weight

            result_decode = transformer_layer(result_decode, attn_norm, W_q, W_k, W_v, W_o, ffn_norm, W_gate, W_up, W_down)
        }
    }

    print("")

    // ----------------------------------------------------------------------------
    // Benchmark 6: Token Sampling (GPU sync overhead)
    // ----------------------------------------------------------------------------
    print("================================================================================")
    print("📊 Benchmark 6: Token Sampling (includes GPU sync)")
    print("================================================================================")
    print("")

    let last_1d = slice_last(result_decode, 0)
    let last_2d = reshape(last_1d, [1, 2048])
    let normed_final = rms_norm(last_2d, output_norm_weight)
    let logits_final = linear(normed_final, output_weight)

    let sampled_token = 0

    bench sampling {
        sampled_token = temperature_sample(logits_final, 0.8)
    }

    print("")

    // ----------------------------------------------------------------------------
    // Summary
    // ----------------------------------------------------------------------------
    print("================================================================================")
    print("✅ Benchmark Complete")
    print("================================================================================")
    print("")
    print("💡 Key Metrics:")
    print("   - Prefill: First forward pass with full prompt")
    print("   - Single Layer: One transformer layer in decode phase")
    print("   - Attention: QKV projection + GQA + output projection")
    print("   - FFN: SwiGLU feed-forward network")
    print("   - Full Decode: All 22 layers in decode phase")
    print("   - Sampling: CPU sync overhead for token selection")
    print("")
    print("📊 Compare these timings with Candle's llama.cpp or quantized example")
    print("")
}
