# 非決定性の根本原因特定

**日時**: 2025年11月14日
**状態**: 🔴 根本原因特定 - GPU並列実行順序の問題

---

## 実験結果サマリー

### ✅ 決定的（単体テスト合格）
- Buffer pool reuse ✓
- matmul/linear ✓
- RoPE ✓
- rms_norm ✓
- softmax ✓
- reshape ✓

### ❌ 非決定的（フル実行）
- chat_demo_optimized (temperature=0.0)
- Full forward pass (22 layers)
- **Layer 0で既に分岐**

---

## レイヤー分岐ポイント

```
[Run 1]
  Embedding sum: 3.416015625 ✅
  Layer 0 output: 3.47265625
  Layer 1 output: -18.625
  Layer 2 output: -54.3125

[Run 2]
  Embedding sum: 3.416015625 ✅ (同じ)
  Layer 0 output: 2.73046875  ❌ (異なる！)
  Layer 1 output: 1259        ❌
  Layer 2 output: 1252        ❌
```

**重要**: Embeddingは決定的だが、**Layer 0の最初のTransformer層で既に分岐**

---

## 根本原因

### GPU並列実行の順序問題

**症状**:
- 個別GPU kernelは決定的
- 複数kernelの組み合わせで非決定的

**推測メカニズム**:

Layer 0の処理順序：
```
1. x_norm1 = rms_norm(x, attn_norm)     ← GPU kernel 1
2. Q = linear(x_norm1, W_q)              ← GPU kernel 2
3. K = linear(x_norm1, W_k)              ← GPU kernel 3
4. V = linear(x_norm1, W_v)              ← GPU kernel 4
5. [GQA attention with 10+ kernels]      ← GPU kernels 5-15
6. attn_out = linear(...)                ← GPU kernel 16
7. x1 = x + attn_out                     ← GPU kernel 17
8. x_norm2 = rms_norm(x1, ffn_norm)      ← GPU kernel 18
9. [SwiGLU FFN with 5+ kernels]          ← GPU kernels 19-24
10. return x1 + ffn_out                  ← GPU kernel 25
```

**問題**:
- Metal GPUで25+個のkernelが並列実行
- バッファプールから再利用されるバッファ
- **kernel実行順序が非決定的**で、バッファの読み書きタイミングが変わる
- 未完了のkernel結果を次のkernelが読み取る可能性

---

## 証拠

### 1. Buffer Poolの動作
```rust
// buffer_pool.rs:222
// NOTE: DO NOT zero out buffers here!
// Kernels overwrite all buffer contents anyway, so uninitialized is safe.
```

この前提が**複数kernel並列実行では成立しない**

### 2. GPU同期の不足

`linear()`の後に`min(logits)`で強制同期しても非決定的 → 個別操作だけでなく、**操作間の依存関係**で同期が必要

### 3. ゼロLogitsの発生

Run 4で全logitsがゼロ：
```
Logit range: [0, 0]
```

これは`linear()`が完了する前に結果が読まれた証拠

---

## Candleとの比較

### Candleの実装
```rust
// candle-metal-kernels/src/metal/command_buffer.rs
impl Drop for CommandBuffer {
    fn drop(&mut self) {
        self.flush();  // 各CommandBuffer dropで自動flush
    }
}
```

- 各操作後に自動的にflush
- バッファプールなし（毎回新規作成）

### TensorLogicの実装
```rust
// src/device/metal_buffer.rs
pub fn to_vec(&self) -> Vec<T> {
    self.device.wait_until_completed().expect(...);  // 読み取り時のみsync
    // ...
}
```

- 読み取り時のみ同期
- バッファプール使用（パフォーマンス重視）

---

## 解決策の方向性

### Option 1: 各kernel後に明示的sync（遅い）
```rust
pub fn matmul(...) -> TensorResult<Self> {
    // ... GPU kernel実行 ...
    self.device.wait_until_completed()?;  // ← 追加
    Ok(result)
}
```

**問題**: パフォーマンス大幅低下

### Option 2: CommandBuffer単位でflush（Candle方式）
```rust
// 各kernel実行後に自動flush
executor.encode_and_dispatch(...)?;
executor.flush()?;  // ← 追加
```

**利点**: 適切な粒度で同期

### Option 3: バッファプールをゼロクリア
```rust
// buffer_pool.rs: 再利用時にゼロクリア
let buffer = buffers.pop()?;
zero_buffer_async(&buffer)?;  // ← 追加
```

**問題**: パフォーマンス低下、根本解決にならない

### Option 4: 依存関係グラフで明示的管理
- kernelの依存関係を追跡
- 依存kernel完了後にのみ次のkernelを実行

**問題**: 実装が複雑

---

## 推奨アプローチ

**Option 2 (Candle方式) + 部分的Option 1**

1. **基本**: CommandBuffer単位でflush
2. **クリティカル箇所**: 明示的`wait_until_completed()`
   - logits計算後
   - sampling前
   - 大きなバッファ再利用前

これにより決定性を確保しつつ、パフォーマンスも維持

---

## 次のステップ

1. ✅ 根本原因特定完了
2. ⏳ Option 2実装テスト
3. ⏳ パフォーマンス測定
4. ⏳ 決定性検証

---

## 関連ファイル

### 実装
- `src/device/kernel_executor.rs` - GPU kernel実行
- `src/device/metal_buffer.rs` - バッファ管理
- `src/device/buffer_pool.rs` - バッファプール

### テスト
- `tests/test_buffer_pool_determinism.rs` ✅
- `tests/test_rope_determinism.rs` ✅
- `tests/test_gpu_kernel_determinism.rs` ✅
- `examples/debug/test_forward_determinism.tl` ❌
- `examples/debug/debug_layer_divergence.tl` ❌

### ドキュメント
- `claudedocs/non_determinism_analysis.md` - 初期分析
- `claudedocs/root_cause_found.md` - 本ファイル
