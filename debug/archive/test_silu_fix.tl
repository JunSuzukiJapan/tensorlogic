// Test: Verify silu function bug is fixed
//
// Bug: fn silu(x) { x * sigmoid(x) } returned 0
// Fix: Added GPU flush before returning from functions
// This test verifies the fix works

// Original buggy form (should now work)
fn silu_direct(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}

// Workaround form (for comparison)
fn silu_workaround(x: float16[?, ?]) -> float16[?, ?] {
    let sig_x = sigmoid(x)
    let result = x * sig_x
    result
}

main {
    print("================================================================================")
    print("SiLU Function Bug Fix Test")
    print("================================================================================")
    print("")

    let model_path = env("HOME") + "/.llm/models/tinyllama-1.1b-chat-f16.gguf"
    let tokenizer_path = env("HOME") + "/.llm/tokenizers/tinyllama-tokenizer.json"

    print("[1/4] Loading model and tokenizer...")
    let model = load_model(model_path)
    let tokenizer = load_tokenizer(tokenizer_path)
    print("      ✓ Loaded")
    print("")

    print("[2/4] Loading Layer 0 FFN weights...")
    let W_gate_0 = get_tensor(model, "blk.0.ffn_gate.weight")
    let W_up_0 = get_tensor(model, "blk.0.ffn_up.weight")
    let embed_table = get_tensor(model, "token_embd.weight")
    let ffn_norm_0 = get_tensor(model, "blk.0.ffn_norm.weight")
    print("      ✓ Weights loaded")
    print("")

    print("[3/4] Creating test input...")
    let system_prompt = "You are a friendly chatbot."
    let user_message = "Hello!"
    let chat_prompt = "<|system|>\\n" + system_prompt + "</s>\\n<|user|>\\n" + user_message + "</s>\\n<|assistant|>\\n"
    let tokens = tokenize(tokenizer, chat_prompt, false)
    let e = embedding(embed_table, tokens)
    let x_norm = rms_norm(e, ffn_norm_0)
    let gate = linear(x_norm, W_gate_0)
    print("      ✓ Test input prepared")
    print("      gate sum:", sum(gate))
    print("")

    print("[4/4] Testing silu functions...")
    print("")

    print("  Test A: Direct form (original bug)")
    let silu_direct_result = silu_direct(gate)
    let sum_direct = sum(silu_direct_result)
    print("    Result sum:", sum_direct)

    if sum_direct == 0.0 {
        print("    ❌ FAIL: Returns 0 (BUG NOT FIXED)")
    }

    if sum_direct != 0.0 {
        print("    ✅ PASS: Returns non-zero value")
    }

    print("")

    print("  Test B: Workaround form (for comparison)")
    let silu_workaround_result = silu_workaround(gate)
    let sum_workaround = sum(silu_workaround_result)
    print("    Result sum:", sum_workaround)

    if sum_workaround == 0.0 {
        print("    ❌ FAIL: Workaround also returns 0")
    }

    if sum_workaround != 0.0 {
        print("    ✅ PASS: Returns non-zero value")
    }

    print("")

    print("  Comparison:")
    if sum_direct == sum_workaround {
        print("    ✅ PASS: Both forms return the same value")
    }

    if sum_direct != sum_workaround {
        print("    ⚠️  WARNING: Different values - direct:", sum_direct, "workaround:", sum_workaround)
    }

    print("")
    print("================================================================================")
    print("✅ SiLU Function Bug Fix Test Complete")
    print("================================================================================")
}
