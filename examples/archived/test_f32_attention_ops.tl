// Test attention operations with real model sizes

main {
    print("=== Testing f32 Attention Operations (Real Sizes) ===")
    print("")

    // Real sizes from TinyLlama
    // Prompt length: ~20 tokens
    // Hidden dim: 2048
    // Num heads: 32
    // Head dim: 64

    let seq_len = 20.0
    let hidden = 2048.0
    let num_heads = 32.0
    let head_dim = 64.0

    // Test 1: Reshape to heads
    print("[1] Reshaping to heads...")
    let x = f32::ones([seq_len, hidden])
    let x_heads = reshape(x, [seq_len, num_heads, head_dim])
    print("  ✓ Reshape OK, shape:", shape(x_heads))
    print("")

    // Test 2: RoPE
    print("[2] Applying RoPE...")
    let x_rope = rope(x_heads)
    print("  ✓ RoPE OK")
    print("")

    // Test 3: Key expansion (4 heads → 32 heads)
    print("[3] Key expansion...")
    let kv_heads = 4.0
    print("  3a: Creating K tensor [", seq_len, ",", kv_heads, ",", head_dim, "]...")
    let k_heads = f32::ones([seq_len, kv_heads, head_dim])
    print("  3b: Applying RoPE to K...")
    let k_rope = rope(k_heads)
    print("  3c: Reshaping to [", seq_len, ",", kv_heads, ", 1,", head_dim, "]...")
    let k_exp = reshape(k_rope, [seq_len, kv_heads, 1.0, head_dim])
    print("  3d: Broadcasting to [", seq_len, ",", kv_heads, ", 8,", head_dim, "]...")
    let k_broadcast = broadcast_to(k_exp, [seq_len, kv_heads, 8.0, head_dim])
    print("  3e: Reshaping to [", seq_len, ",", num_heads, ",", head_dim, "]...")
    let k_expanded = reshape(k_broadcast, [seq_len, num_heads, head_dim])
    print("  ✓ Key expansion OK")
    print("")

    // Test 4: Attention scores (THIS IS LIKELY THE SLOW PART)
    print("[4] Computing attention scores (einsum)...")
    print("    Q shape: [", seq_len, ",", num_heads, ",", head_dim, "]")
    print("    K shape: [", seq_len, ",", num_heads, ",", head_dim, "]")
    print("    Running einsum(\"ihd,jhd->ihj\")...")
    let scores = einsum("ihd,jhd->ihj", x_rope, k_expanded)
    print("  ✓ Attention scores OK, shape:", shape(scores))
    print("")

    // Test 5: Softmax on scores
    print("[5] Softmax...")
    let attn = softmax(scores, 2)
    print("  ✓ Softmax OK")
    print("")

    // Test 6: Attention output (einsum)
    print("[6] Attention output (einsum)...")
    let v_expanded = k_expanded
    let out = einsum("ihj,jhd->ihd", attn, v_expanded)
    print("  ✓ Attention output OK, shape:", shape(out))
    print("")

    print("=== All attention operations completed successfully ===")
}
