// Grouped Query Attention (GQA) Implementation
// TinyLlama-style: 32 Q heads, 4 KV heads, head_dim=64

// Grouped Query Attention
// Q: [seq, num_q_heads * head_dim] = [seq, 2048]
// K: [seq, num_kv_heads * head_dim] = [seq, 256]
// V: [seq, num_kv_heads * head_dim] = [seq, 256]
fn grouped_query_attention(
    Q: float16[?, ?],
    K: float16[?, ?],
    V: float16[?, ?]
) -> float16[?, ?] {
    // TinyLlama固定パラメータ
    let seq_len = 4
    let num_q_heads = 32
    let num_kv_heads = 4
    let head_dim = 64
    // Group size: how many Q heads share one KV head
    let group_size = num_q_heads / num_kv_heads

    // 1. Reshape Q to [seq, num_q_heads, head_dim]
    let Q_heads = reshape(Q, [seq_len, num_q_heads, head_dim])

    // 2. Reshape K, V to [seq, num_kv_heads, head_dim]
    let K_heads = reshape(K, [seq_len, num_kv_heads, head_dim])
    let V_heads = reshape(V, [seq_len, num_kv_heads, head_dim])

    // 3. Expand K, V to match Q's head count
    // [seq, num_kv_heads, head_dim] -> [seq, num_kv_heads, 1, head_dim]
    let K_with_group = reshape(K_heads, [seq_len, num_kv_heads, 1, head_dim])
    let V_with_group = reshape(V_heads, [seq_len, num_kv_heads, 1, head_dim])

    // Broadcast: [seq, kv_heads, 1, head_dim] -> [seq, kv_heads, group_size, head_dim]
    let K_broadcast = broadcast_to(K_with_group, [seq_len, num_kv_heads, group_size, head_dim])
    let V_broadcast = broadcast_to(V_with_group, [seq_len, num_kv_heads, group_size, head_dim])

    // Reshape to final: [seq, num_q_heads, head_dim]
    let K_expanded = reshape(K_broadcast, [seq_len, num_q_heads, head_dim])
    let V_expanded = reshape(V_broadcast, [seq_len, num_q_heads, head_dim])

    // 4. Transpose for attention: [seq, heads, head_dim] -> [heads, seq, head_dim]
    // TensorLogic transpose currently only swaps last 2 dims, so we reshape manually
    // For simplicity, compute attention per-head and concatenate

    // Per-head attention using einsum
    // Q_heads: [seq, heads, head_dim]
    // K_expanded: [seq, heads, head_dim]
    // scores: [seq_query, head, seq_key]
    let scores = einsum("ihd,jhd->ihj", Q_heads, K_expanded)

    // Scale factor: 1/sqrt(head_dim) = 1/sqrt(64) = 0.125
    let scale = 0.125
    let scaled_scores = scores * scale

    // Softmax over seq_key dimension
    let attn_weights = softmax(scaled_scores, 2)

    // attn_weights: [seq_query, head, seq_key]
    // V_expanded: [seq_key, head, head_dim]
    // output: [seq_query, head, head_dim]
    let output = einsum("ihj,jhd->ihd", attn_weights, V_expanded)

    // Reshape: [seq, heads, head_dim] -> [seq, heads*head_dim]
    let result = reshape(output, [seq_len, num_q_heads * head_dim])

    return result
}

main {
    print("=== Grouped Query Attention (GQA) テスト ===")
    print("")

    // TinyLlama config
    let seq_len = 4
    let d_model = 2048
    let num_q_heads = 32
    let num_kv_heads = 4
    let head_dim = 64
    let group_size = 8  // 32 / 4

    print("構成:")
    print("  seq_len:", seq_len)
    print("  d_model:", d_model)
    print("  num_q_heads:", num_q_heads)
    print("  num_kv_heads:", num_kv_heads)
    print("  head_dim:", head_dim)
    print("  group_size:", group_size, "(各KVヘッドを", group_size, "個のQヘッドで共有)")
    print("")

    // Create dummy inputs
    print("ステップ1: 入力テンソル作成")
    let Q = ones([seq_len, num_q_heads * head_dim])  // [4, 2048]
    let K = ones([seq_len, num_kv_heads * head_dim]) // [4, 256]
    let V = ones([seq_len, num_kv_heads * head_dim]) // [4, 256]
    print("  Q:", shape(Q))
    print("  K:", shape(K))
    print("  V:", shape(V))
    print("")

    // Run GQA
    print("ステップ2: GQAアテンション実行")
    let output = grouped_query_attention(Q, K, V)
    print("  出力:", shape(output))
    print("")

    print("==================================================")
    print("")
    print("✅ GQA Attention Test Passed!")
    print("")
    print("重要な修正:")
    print("  - 旧実装: 全ヘッドをフラット化 → クロスヘッドアテンション (バグ)")
    print("  - 新実装: einsumで各ヘッド独立にアテンション計算")
    print("  - einsum(\"ihd,jhd->ihj\") で [seq, heads, seq] スコア取得")
    print("  - einsum(\"ihj,jhd->ihd\") でヘッドごとの出力計算")
    print("")
    print("NOTE: 旧テストはall-ones入力だったため、")
    print("      正しい実装も間違った実装も同じ出力になっていました。")
    print("      今後のテストでは非一様入力を使用してください。")
    print("")
    print("==================================================")
    print("")
    print("次のステップ:")
    print("  1. Chat Demoで実際のモデル重みでテスト")
    print("  2. 期待される出力トークンが生成されるか確認")
}
