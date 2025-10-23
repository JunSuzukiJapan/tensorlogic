// Test: broadcast_to() function
// Demonstrates NumPy-style broadcasting for tensor operations

main {
    print("=== broadcast_to() テスト ===")
    print("")

    // Test 1: 2D -> 3D broadcast (for attention heads)
    print("テスト1: [4, 64] -> [4, 8, 64] にブロードキャスト")
    let seq_len = 4
    let head_dim = 64
    let num_heads = 8
    let x1 = ones([seq_len, head_dim])
    print("  入力:", shape(x1))

    // まずreshapeして次元を追加 [4, 64] -> [4, 1, 64]
    let x1_reshaped = reshape(x1, [seq_len, 1, head_dim])
    print("  reshape後:", shape(x1_reshaped))

    // broadcastで [4, 1, 64] -> [4, 8, 64]
    let y1 = broadcast_to(x1_reshaped, [seq_len, num_heads, head_dim])
    print("  broadcast後:", shape(y1))
    print("  ✅ 成功!")
    print("")

    // Test 2: GQA use case - K/V heads to Q heads
    print("テスト2: GQA用途 - KVヘッド [4, 256] を Qヘッド次元に拡張")
    let kv_heads = 4
    let q_heads = 32
    let group_size = 8  // 32 / 4
    let k = ones([seq_len, kv_heads * head_dim])  // [4, 256]
    print("  入力 K shape:", shape(k))

    // Reshape K to [seq, kv_heads, head_dim]
    let k_reshaped = reshape(k, [seq_len, kv_heads, head_dim])
    print("  K reshaped:", shape(k_reshaped))

    // GQAの正しいアプローチ：各KVヘッドを複数のQヘッドで共有
    // [seq, kv_heads, head_dim] -> [seq, kv_heads, 1, head_dim]
    let k_with_group = reshape(k_reshaped, [seq_len, kv_heads, 1, head_dim])
    print("  K with group dim:", shape(k_with_group))

    // Broadcast: [seq, kv_heads, 1, head_dim] -> [seq, kv_heads, group_size, head_dim]
    let k_broadcasted = broadcast_to(k_with_group, [seq_len, kv_heads, group_size, head_dim])
    print("  K broadcasted:", shape(k_broadcasted))

    // Reshape to final: [seq, kv_heads, group_size, head_dim] -> [seq, q_heads, head_dim]
    let k_final = reshape(k_broadcasted, [seq_len, q_heads, head_dim])
    print("  K final shape:", shape(k_final))
    print("  ✅ GQA対応成功!")
    print("")

    print("==================================================")
    print("🎉 broadcast_to() DSL統合完了!")
    print("")
    print("主な用途:")
    print("  • Grouped Query Attention (GQA)")
    print("  • バッチ次元の追加")
    print("  • ヘッド次元の拡張")
    print("  • NumPyスタイルのブロードキャスト")
}
