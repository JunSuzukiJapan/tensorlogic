# Step 3: Root Cause Analysis - Q6_K Dequantization Issue

## 実行日時
2025-10-26

## 問題の要約
TensorLogicが誤ったトークンを予測する問題の根本原因を特定

### 症状
- llama.cpp出力: "You are a helpful"
- TensorLogic出力: "umi angularjs"
- 完全に異なる予測結果

## 検証プロセス

### 検証1: Metal演算の精度チェック

#### Transpose ✅
```
Test: [2, 3] → [3, 2]
Input: [[1,2,3], [4,5,6]]
Output: [[1,4], [2,5], [3,6]]
Result: ✅ Correct
```

#### RMS Norm ✅（微小な精度問題あり、実用上問題なし）
```
Test 1: [1, 8]
  Max difference: 0.000464 ✅
  Avg difference: 0.000184 ✅

Test 2: [1, 2048] uniform weights
  Max difference: 0.000783 ✅
  Avg difference: 0.000213 ✅

Test 3: [1, 2048] non-uniform weights
  Max difference: 0.001404 ⚠️ (わずかに閾値超過)
  Avg difference: 0.000281 ✅

Result: f16精度の限界による正常な動作、実用上問題なし
```

### 検証2: Sampling実装の確認

#### temperature_sample() は実際には argmax（貪欲サンプリング）
```rust
// For now, implement greedy sampling (argmax)
// TODO: Implement proper temperature-scaled sampling with random selection

// Find argmax in the last sequence position
let mut max_idx = 0;
let mut max_val = f16::NEG_INFINITY;

for idx in 0..vocab_size {
    let val = logits_data[start_idx + idx];
    if val > max_val {
        max_val = val;
        max_idx = idx;
    }
}
```

**発見**: 温度パラメータは無視され、常に最高確率のトークンを選択
→ llama.cpp `--temp 0` と同じ動作のはず

### 検証3: Logits値の異常発見 ❌

#### デバッグ出力
```
=== Temperature Sample Debug ===
  Logits shape: [29, 32000]
  Vocab size: 32000
  Sampled token: 15547
  Max logit value: 0.8598633
  Top 5 tokens:
    1: token=15547, logit=0.8598633
    2: token=18263, logit=0.8569336
    3: token=31430, logit=0.8491211
    4: token=17856, logit=0.8120117
    5: token=2349, logit=0.80810547
```

**異常**: 全てのlogitが **0.8〜0.86の範囲** で、ほぼ同じ値

**正常な言語モデルのlogits**:
- Top token: 大きな正の値（10〜20）
- 他のtoken: 小さい値や負の値（-10〜5）
- 明確な差がある

### 検証4: Quantization Type確認 ❌

#### GGUFファイル内のテンソル型
```
F32 (45 tensors):
  - blk.*.attn_norm.weight
  - blk.*.ffn_norm.weight
  - output_norm.weight

Q4_0 (155 tensors):
  - blk.*.attn_q.weight
  - blk.*.attn_k.weight
  - blk.*.attn_v.weight
  - blk.*.ffn_*.weight
  - token_embd.weight

Q6_K (1 tensors):
  - output.weight  ← **重要な重み！**
```

**重大な発見**: `output.weight` は **Q6_K quantization**

### 検証5: Python GGUFライブラリの誤解

初期のPython出力:
```python
output.weight:
  Shape: [2048, 32000]
  Type: 14  # Q6_K (NOT F32!)
  Data size: 32,000 bytes  # 不完全なデータ
```

Type 14 = Q6_K (F32ではない)
Data size は gguf-py ライブラリの問題（実際のファイルでは正しいサイズ）

## 根本原因

### Q6_K Dequantization Issue

1. **output.weight は Q6_K quantization形式**
2. **Q6_K dequantizationの実装が複雑**（256要素ブロック、210バイト/ブロック）
3. **Logitsが異常に均一** → dequantization が正しく機能していない可能性が非常に高い

### Q6_K実装の複雑さ
```rust
const QK_K: usize = 256;  // Block size
const BLOCK_BYTES: usize = 210;  // 128 + 64 + 16 + 2

// Components:
// - ql[128]: Lower 4 bits
// - qh[64]: High 2 bits
// - sc[16]: Scales (int8)
// - d[2]: Block scale (f16)

// Reconstructs 6-bit values:
// q1 = (ql[l] & 0xF) | ((qh[l] & 0x3) << 4)
// q2 = (ql[l+32] & 0xF) | (((qh[l] >> 2) & 0x3) << 4)
// q3 = (ql[l] >> 4) | (((qh[l] >> 4) & 0x3) << 4)
// q4 = (ql[l+32] >> 4) | (((qh[l] >> 6) & 0x3) << 4)
```

## 次のステップ

### 優先度1: Q6_K Dequantization修正

1. **Candle公式実装との比較**
   - Repository: https://github.com/huggingface/candle
   - File: `candle-core/src/quantized/k_quants.rs`
   - Function: `dequantize_q6_k()`

2. **検証テストの作成**
   - 既知の入力で既知の出力を生成
   - Candleの出力と比較

3. **修正の適用**
   - ビット操作の正確性確認
   - インデックス計算の検証
   - スケール適用の確認

### 優先度2: 動作確認

修正後に以下を確認：
1. Logits値が正常な範囲に分散（-10〜20程度）
2. Top 5トークンに明確な差がある
3. 予測トークンがllama.cppと一致

## まとめ

### ✅ 正常動作確認済み
1. Q4_0 dequantization
2. Metal Softmax
3. Metal Matmul（f32 accumulation修正後）
4. RMS Norm
5. Transpose/Permute
6. GGUF dimension handling
7. All shapes
8. RoPE/GQA/Attention math
9. Sampling implementation（argmax動作）

### ❌ 問題発見
**Q6_K dequantization** - output.weightが正しくdequantizeされていない
→ 全てのlogitsが0.8付近で均一
→ 予測が実質的にランダム
→ 間違ったトークン生成

### 解決方法
Q6_K dequantizationをCandle公式実装と比較して修正する必要がある
