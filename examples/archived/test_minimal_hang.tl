// Minimal test to reproduce hang

main {
    print("=== Minimal Hang Reproduction ===")
    print("")

    let seq_len = 20.0
    let hidden = 2048.0
    let num_heads = 32.0
    let head_dim = 64.0

    // Execute reshape→rope twice
    print("[1] First reshape→rope...")
    let x1 = reshape(f32::ones([seq_len, hidden]), [seq_len, num_heads, head_dim])
    let r1 = rope(x1)
    print("  OK")

    print("[2] Second reshape→rope...")
    let x2 = reshape(f32::ones([seq_len, hidden]), [seq_len, num_heads, head_dim])
    let r2 = rope(x2)
    print("  OK")

    print("[3] Reshape result...")
    let r2_reshaped = reshape(r2, [seq_len, num_heads, 1.0, head_dim])
    print("  OK")

    print("[4] Broadcast (will call to_cpu internally)...")
    let r2_broadcast = broadcast_to(r2_reshaped, [seq_len, num_heads, 8.0, head_dim])
    print("  OK")

    print("=== Completed ===")
}
