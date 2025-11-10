# 1層テスト結果 - TensorLogic vs Candle比較

## テスト概要

**目的**: 1層のみのtransformerで、TensorLogicとCandleの出力を比較し、GPU同期問題の原因を特定

**テスト日時**: 2025-11-10

## テスト環境

- **モデル**: TinyLlama 1.1B Chat (f16)
- **プロンプト**: "You are a friendly chatbot." / "Hello! How are you?"
- **トークン数**: 38トークン (BOS tokenなし)

## Candle参照値 (HuggingFace Transformers)

| ステップ | Sum値 |
|---------|-------|
| Embedding | 3.660156 |
| Layer 0 output | **4.902344** |
| Final norm | 774.0 |
| Logits (last token) | -34784.0 |
| Top prediction | token_id=26012 ("jd"), logit=5.8203 |

## TensorLogic結果

| ステップ | Sum値 | 状態 |
|---------|-------|------|
| Embedding | 3.673828125 | ✅ ほぼ一致 |
| x_norm (attn) | 82.875 | ✅ 正常 |
| Q | 29.21875 | ✅ 正常 |
| K | -1040 | ✅ 正常 |
| V | 1.6875 | ✅ 正常 |
| attn_out | 4.70703125 | ✅ 正常 |
| hidden_1 (after attn) | 8.3515625 | ✅ 正常 |
| x_norm2 (ffn) | 59.125 | ✅ 正常 |
| gate | -7884 | ✅ 正常 |
| up | 330.5 | ✅ 正常 |
| **silu_gate** | **0** | ❌ **問題箇所！** |
| gated | 0 | ❌ silu_gateが0なので0 |
| ffn_out | 0 | ❌ すべて0 |
| **Layer 0 output** | **8.3515625** | ❌ **FFNが効いていない** |
| Final norm | 790 | △ Candleの774に近いが不正確 |
| Logits | inf | ❌ 無限大 |
| Top prediction | token_id=29414, logit=7.26 | ❌ 全く違う |

## 決定性テスト

3回実行した結果、**完全に決定的に同じ値**を返します：
- silu_gate sum: 0 (3回とも)
- ffn_out sum: 0 (3回とも)
- Layer 0 output sum: 8.3515625 (3回とも)

これは、非決定的なGPU同期問題ではなく、**関数実装の問題**であることを示しています。

## 根本原因

### `silu`関数の問題

```tl
fn silu(x: float16[?, ?]) -> float16[?, ?] {
    x * sigmoid(x)
}
```

**症状**:
- Input: `gate` with sum=-7884 (正常)
- Output: `silu_gate` with sum=0 (異常)

**仮説**:
1. `sigmoid(x)`のGPU実装に問題がある
2. `x * sigmoid(x)`の乗算がGPUで正しく実行されていない
3. GPU同期が欠けており、結果が未初期化（0）になっている

**影響**:
- すべてのFFN（Feed-Forward Network）出力が0になる
- 22層すべてでFFNが機能しない
- Attention出力のみで動作するため、モデルの能力が大幅に低下
- 最終的なlogitsが不正確になり、生成が破綻する

## 次のステップ

1. **`sigmoid`関数の実装を確認**
   - `src/ops/*.rs`で`sigmoid`の実装を探す
   - GPUカーネルの実装を確認

2. **`silu`を直接実装**
   - `sigmoid`を使わず、`silu(x) = x / (1 + exp(-x))`を直接実装
   - またはbuiltin関数として実装

3. **中間値のGPU同期確認**
   - `sigmoid(x)`の出力を読み取って、0になっているか確認
   - 必要に応じて明示的な同期を追加

## 結論

**1層テストで根本原因を特定できました**:
- **`silu`関数（FFN内）が完全に0を返す**
- これがすべての層で発生し、22層デモが失敗する原因
- GPU同期の非決定性ではなく、関数実装の問題

**優先度**: 🔴 **CRITICAL** - これを修正しない限り、すべての層で正しい出力が得られません。
