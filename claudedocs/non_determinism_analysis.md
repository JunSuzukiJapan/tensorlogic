# TensorLogic非決定性バグ分析報告

**日時**: 2025年11月14日
**状態**: 🔴 重大バグ - greedy decoding（temperature=0.0）が非決定的

---

## 問題の概要

TensorLogicのLLM推論が完全に非決定的です。温度0.0（greedy decoding）で同じプロンプトを実行しても、毎回異なるトークンを生成します。

### 実験結果

| Framework | 決定性 | First Token | Decoded |
|-----------|--------|-------------|---------|
| **Candle** | ✅ 決定的 | 18585 | "Sure" |
| **TensorLogic Run 1** | ❌ | 7989 | "arse" |
| **TensorLogic Run 2** | ❌ | 2083 | "sum" |
| **TensorLogic Run 3** | ❌ | 514 | ? |
| **TensorLogic Run 4** | ❌ | 10421 | ? |

- **同じモデル**: tinyllama-1.1b-chat-q4_0.gguf
- **同じプロンプト**: `<|system|>\nYou are a friendly and helpful AI assistant.</s>\n<|user|>\nHello! Tell me a short fun fact about computers.</s>\n<|assistant|>\n`
- **同じ温度**: 0.0 (greedy)

---

## 調査経過

### 1. GPU同期の問題？ → ❌ 違う

**仮説**: GPU操作が完了する前に結果を読み取っている

**テスト**: `linear()`の直後に`min(logits)`を追加して強制同期

```tl
let logits = linear(normed, output_weight)
let _logit_min = min(logits)  // Force GPU sync
let next_token_id = temperature_sample(logits, temperature)
```

**結果**: まだ非決定的 → GPU同期は問題ではない

### 2. ゼロlogitsの問題

一部の実行で全logitsがゼロになる現象を確認：

```
[SAMPLING DEBUG] Top 10 logits:
  #1: token_id=0 logit=0.000000
  #2: token_id=1 logit=0.000000
  ...all zeros...
```

**原因**: GPU同期不足
**解決**: `min(logits)`による強制同期で解決

### 3. ループの有無で結果が変わる

**発見**: `temperature_sample()`をloopで囲むと結果が変わる

```tl
// Without loop → ALL ZERO logits
let top_token = temperature_sample(logits, 0.0)

// With loop → Valid logits
for i in range_i(1) {
    let top_token = temperature_sample(logits, 0.0)
}
```

これもGPU同期のタイミング問題として解決済み。

---

## 現在の状況

### 解決済み
✅ ゼロlogits問題（GPU同期不足）
✅ ループ依存の動作（同上）

### 未解決 🔴
❌ **計算自体の非決定性** - これが根本問題

---

## 根本原因の候補

### 1. 未初期化メモリ（最も疑わしい）

**BufferPool**が古いデータを含むバッファを再利用：

```rust
// buffer_pool.rs:222-230
// NOTE: DO NOT zero out buffers here!
// Reasons:
// 1. new_uninit_pooled() expects uninitialized buffers (for performance)
// 2. CPU write to GPU memory (write_bytes) causes implicit GPU sync,
//    which hangs when many GPU operations are pending
// 3. Kernels overwrite all buffer contents anyway, so uninitialized is safe.
```

**問題**: もしGPUカーネルが一部の要素しか書き込まなかったら？

### 2. RoPE実装のバグ

**疑わしい点**:
- Position encoding計算が複雑
- 浮動小数点演算の順序
- Metal GPU三角関数の精度

```metal
// shaders/unified.metal:rope_f16
const float theta = float(pos) * freq;
const float cos_theta = cos(theta);
const float sin_theta = sin(theta);
```

### 3. Metal GPUカーネルの並列実行順序

**可能性**: GPUスレッドの実行順序が非決定的で、浮動小数点演算の結果に影響

---

## 次のステップ

### 優先度1: バッファプールゼロクリア
再利用時にバッファを明示的にゼロクリアしてテスト

### 優先度2: レイヤーごと比較
TensorLogicとCandleで各Transformerレイヤーの出力を比較

### 優先度3: RoPEテスト
RoPEカーネル単体で決定性を確認

---

## 関連ファイル

- `examples/chat_demo_optimized.tl` - メインテストスクリプト
- `examples/debug/debug_first_token_logits.tl` - First token詳細デバッグ
- `src/device/buffer_pool.rs:185-256` - BufferPool allocate実装
- `src/ops/rope.rs` - RoPE実装
- `shaders/unified.metal:rope_f16` - RoPE GPU kernel
- `/tmp/candle_output.txt` - Candle baseline結果

---

## 参考情報

### Candleの実装
- GPU同期: `to_cpu()`で自動同期
- バッファ管理: 毎回新規作成（プールなし）
- RoPE: 事前計算したcos/sin tablesを使用

### TensorLogicの現在の実装
- GPU同期: `wait_until_completed()`を明示的に呼ぶ
- バッファ管理: プールで再利用（パフォーマンス向上のため）
- RoPE: オンザフライで計算
