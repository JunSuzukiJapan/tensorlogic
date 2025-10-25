# TensorLogic vs llama.cpp 出力比較レポート

## テスト条件

### 共通設定
- **モデル**: tinyllama-1.1b-chat-q4_0.gguf
- **入力プロンプト**: "Hello"
- **システムプロンプト**: "You are a helpful assistant."

## 実行結果

### llama.cpp (参照実装)
```
入力プロンプト:
<|system|>
You are a helpful assistant.
<|user|>
Hello
<|assistant|>

出力: "How are you?"

状態: ✅ 正しい動作
```

### TensorLogic (linear関数使用)

#### テスト1: BOSトークンのみ (test_linear_function.tl)
```
入力: [1.0] (BOS token only)
温度: 0.0 (greedy)
レイヤー: 2層のみ

出力: Token 2579

状態: ✅ 正常動作
```

#### テスト2: フルプロンプト (chat_demo_full_22_layers.tl)
```
入力プロンプト:
<|system|>
You are a helpful assistant.
<|user|>
Hello
<|assistant|>

入力トークン: [1, 529, 29989, 5205, 29989, 29958, 13, 3492, 526, 263, 8444, 20255, 29889, 13, 29966, 29989, 1792, 29989, 29958, 13, 10994, 13, 29966, 29989, 465, 22137, 29989, 29958, 13]
温度: 0.8
レイヤー: 22層全て

出力: Token 0, 0, 0, 0, ...

状態: ❌ 異常動作 (token 0 = <unk> トークン)
```

## 問題の分析

### 成功パターン vs 失敗パターン

| 項目 | test_linear_function.tl | chat_demo_full_22_layers.tl |
|------|------------------------|----------------------------|
| 入力長 | 1トークン（BOS） | 26トークン（フルプロンプト） |
| 温度 | 0.0 (greedy) | 0.8 (sampling) |
| レイヤー数 | 2層 | 22層 |
| RoPE | なし | あり |
| KV cache | なし | なし |
| 結果 | ✅ Token 2579 | ❌ Token 0連続 |

### 問題の可能性

#### 1. RoPE (Rotary Position Embedding) ⚠️
```rust
// chat_demo_full_22_layers.tl
let Q_rope = rope(Q_heads)
let K_rope = rope(K_heads)
```
- RoPEの実装が間違っている可能性
- 複数トークンでのRoPE適用が正しくない可能性

#### 2. 複数トークン処理 ⚠️
- シーケンス長26での処理に問題がある可能性
- Attention maskingが正しくない可能性

#### 3. linear関数の実装 🤔
- BOSのみでは動作するので、関数自体は正しい
- ただし、バッチ次元の処理に問題がある可能性

#### 4. Temperature sampling 🤔
- 温度0.8でのサンプリングに問題がある可能性
- ただし、test_linear_functionでgreedyは成功している

## 検証すべきこと

### 優先度 高
1. **RoPE実装の確認**
   - `/tmp/candle_reference/candle`のRoPE実装と比較
   - 複数トークンでのRoPE動作を確認

2. **シーケンス長の影響**
   - chat_demo_full_22_layersで温度0.0に変更してテスト
   - シーケンス長を変えてテスト（1, 5, 10, 26）

3. **中間値の確認**
   - logitsの値を出力（全て0になっていないか）
   - attentionの出力を確認

### 優先度 中
4. **linear関数のバッチ処理**
   - [batch, seq, features]形状での動作確認
   - transposeがバッチ次元に影響していないか

5. **Temperature samplingの動作**
   - 温度0.0と0.8での比較
   - logitsからの確率分布計算を確認

### 優先度 低
6. **22層全体の伝播**
   - 2層では成功するが22層で失敗する理由
   - 勾配消失・爆発の可能性（推論なので低い）

## 次のステップ

### ステップ1: 簡略版でテスト
```bash
# chat_demo_full_22_layersを修正
# - RoPEを無効化
# - 温度を0.0に設定
# - 2層のみで実行
```

### ステップ2: RoPE実装の確認
```bash
# Candleのrope実装と比較
cat /tmp/candle_reference/candle/candle-nn/src/rotary_emb.rs
# TensorLogicのrope実装を確認
grep -A50 "fn rope" src/ops/*.rs
```

### ステップ3: 中間値デバッグ
```rust
// logitsを出力
print("Logits shape:", shape(logits1))
print("Logits[0] first 10:", ...)
```

## 比較まとめ

| 実装 | 入力 | 出力 | 状態 |
|------|------|------|------|
| llama.cpp | "Hello" | "How are you?" | ✅ 正常 |
| TensorLogic (BOS) | [1] | Token 2579 | ✅ 正常 |
| TensorLogic (フル) | [1, 529, ...] | Token 0連続 | ❌ 異常 |

**結論**:
- linear関数の基本実装は正しい（BOS単独テストで成功）
- 問題は複数トークン処理、RoPE、またはその組み合わせにある可能性が高い
