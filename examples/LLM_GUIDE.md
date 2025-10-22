# Building Language Models with TensorLogic

このガイドでは、TensorLogicを使ってローカルLLM（大規模言語モデル）の基本概念を学ぶ方法を説明します。

## LLMの基本コンセプト

### 1. Token Embeddings（トークン埋め込み）

**コンセプト**: 離散的なトークン（単語や文字）を連続的なベクトルに変換

**TensorLogicでの実装例**:
```tensorlogic
// 語彙サイズ8、埋め込み次元16
tensor emb_0: float16[16] learnable = randn([16]) * 0.1
tensor emb_1: float16[16] learnable = randn([16]) * 0.1
// ... 他のトークン用の埋め込み
```

**参考example**: `examples/gnn_node_classification.tl` のノード埋め込み部分

### 2. Feedforward Neural Network（順伝播ニューラルネットワーク）

**コンセプト**: 入力を変換して高レベルの表現を学習

**TensorLogicでの実装例**:
```tensorlogic
// 隠れ層
tensor W1: float16[16, 32] learnable = randn([16, 32]) * 0.1
tensor b1: float16[32] learnable = zeros([32])

// Forward pass
tensor hidden: float16[32] = relu(matmul(input, W1) + b1)
```

**参考example**: `examples/gnn_node_classification.tl` のGNN層

### 3. Attention Mechanism（注意機構）

**コンセプト**: シーケンスの異なる部分に注目する能力（Transformerの核心）

**数式**:
```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

**TensorLogicでの実装**:
```tensorlogic
// Query, Key, Value projections
tensor Q: float16[seq_len, d_k] = matmul(input, W_q)
tensor K: float16[seq_len, d_k] = matmul(input, W_k)
tensor V: float16[seq_len, d_v] = matmul(input, W_v)

// Attention scores
tensor scores: float16[seq_len, seq_len] = matmul(Q, transpose(K))
tensor attn_weights: float16[seq_len, seq_len] = softmax(scores / sqrt_dk)

// Attention output
tensor output: float16[seq_len, d_v] = matmul(attn_weights, V)
```

**参考example**: `examples/attention.tl` - 完全なattention実装を参照

### 4. Next Token Prediction（次トークン予測）

**コンセプト**: LLMの基本タスク - 与えられたコンテキストから次のトークンを予測

**TensorLogicでの実装**:
```tensorlogic
learn {
    // Input embedding
    tensor input_emb: float16[embedding_dim]
    input_emb = embeddings[token_id]

    // Forward pass
    tensor hidden: float16[hidden_dim] = relu(matmul(input_emb, W1) + b1)
    tensor logits: float16[vocab_size] = matmul(hidden, W_out) + b_out

    // Softmax for probability distribution
    tensor probs: float16[vocab_size] = softmax(logits)

    // Target (one-hot vector for next token)
    tensor target: float16[vocab_size] = [0.0, ..., 1.0, ..., 0.0]

    // Training objective
    minimize mean((probs - target) ** 2.0)
    with adam(lr=0.01)
    epochs 100
}
```

## 実践ガイド：LLM風モデルの構築手順

### ステップ1: 語彙とパラメータの定義

```tensorlogic
main {
    // 語彙: 10トークン（例: 頻出単語）
    // 埋め込み次元: 16
    // 隠れ層次元: 32

    tensor emb_0: float16[16] learnable = randn([16]) * 0.1
    tensor emb_1: float16[16] learnable = randn([16]) * 0.1
    // ... emb_2 から emb_9 まで

    tensor W_hidden: float16[16, 32] learnable = randn([16, 32]) * 0.1
    tensor b_hidden: float16[32] learnable = zeros([32])

    tensor W_output: float16[32, 10] learnable = randn([32, 10]) * 0.1
    tensor b_output: float16[10] learnable = zeros([10])
}
```

### ステップ2: 学習ループの実装

```tensorlogic
learn {
    // パターン1: トークン0 → トークン1
    tensor h1: float16[32] = relu(matmul(emb_0, W_hidden) + b_hidden)
    tensor logits1: float16[10] = matmul(h1, W_output) + b_output
    tensor probs1: float16[10] = softmax(logits1)

    tensor target1: float16[10] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    minimize mean((probs1 - target1) ** 2.0)
    with adam(lr=0.01)
    epochs 50
    report_every 10
}
```

### ステップ3: 推論の実装

```tensorlogic
// テスト: トークン0が与えられたときの次トークン予測
tensor test_h: float16[32] = relu(matmul(emb_0, W_hidden) + b_hidden)
tensor test_logits: float16[10] = matmul(test_h, W_output) + b_output
tensor test_probs: float16[10] = softmax(test_logits)

print("Predicted probabilities:")
print("  Token 0:", [test_probs[0]])
print("  Token 1:", [test_probs[1]], "<-- Expected high")
// ...
```

## スケーリング: ミニモデルから本物のLLMへ

| 要素 | TensorLogicミニ実装 | GPT-3/4クラス |
|------|-------------------|--------------|
| **語彙サイズ** | 10-100トークン | 50,000-100,000トークン |
| **埋め込み次元** | 8-32 | 4,096-12,288 |
| **隠れ層数** | 1-2層 | 96-120+層 |
| **パラメータ数** | ~500-5000 | 1750億-1兆7600億 |
| **コンテキスト長** | 1-10トークン | 4,096-128,000トークン |
| **学習時間** | 秒-分 | 週-月 |
| **学習データ** | 手作りパターン | インターネット規模のテキスト |

## 主要な概念の参考Example

### Attention Mechanism
- **ファイル**: `examples/attention.tl`
- **学べること**:
  - Scaled Dot-Product Attention
  - Query, Key, Value projections
  - Attention weights の計算

### Graph Neural Network (構造が似ている)
- **ファイル**: `examples/gnn_node_classification.tl`
- **学べること**:
  - Embedding の学習
  - メッセージパッシング（AttentionのアナロジーREAD)`
  - 分類タスクの実装

### Batch Normalization
- **ファイル**: `examples/batch_norm_demo.tl`
- **学べること**:
  - 正規化テクニック
  - 学習の安定化

## 実装上の注意点

### 1. TensorLogicの制約

- **固定サイズ**: すべてのテンソルは固定サイズ
- **静的グラフ**: 動的な計算グラフは未サポート
- **シーケンス処理**: RNN/LSTMは現状では手動実装が必要

### 2. 推奨アプローチ

**初心者向け**:
1. `examples/attention.tl` を実行して理解
2. 埋め込みの概念を `gnn_node_classification.tl` で学習
3. 簡単なトークン予測モデルを構築

**中級者向け**:
1. Multi-head attentionの実装
2. Position encodingsの追加
3. 複数層のTransformerブロック

**上級者向け**:
1. ミニTransformerの完全実装
2. テキスト生成のサンプリング戦略
3. Fine-tuningのシミュレーション

## 学習パス

```
1. 基礎
   ↓
   - Tensor操作の理解
   - 行列乗算、ReLU、Softmax

2. 埋め込み
   ↓
   - Token → Vector変換
   - 類似性の学習

3. Attention
   ↓
   - Query/Key/Value の概念
   - Attention weightsの計算

4. Transformer Block
   ↓
   - Self-attention + FFN
   - Residual connections
   - Layer normalization

5. Language Model
   ↓
   - Next token prediction
   - Autoregressive generation
   - Training objectives
```

## 追加リソース

### 論文
- "Attention Is All You Need" (Vaswani et al., 2017)
- "Language Models are Unsupervised Multitask Learners" (GPT-2, Radford et al., 2019)
- "BERT: Pre-training of Deep Bidirectional Transformers" (Devlin et al., 2019)

### オンラインリソース
- [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/) by Jay Alammar
- [The Illustrated GPT-2](http://jalammar.github.io/illustrated-gpt2/) by Jay Alammar
- [Hugging Face Course](https://huggingface.co/course/chapter1/1)

### 書籍
- "Speech and Language Processing" by Dan Jurafsky & James H. Martin
- "Deep Learning" by Ian Goodfellow, Yoshua Bengio, Aaron Courville

## よくある質問

**Q: TensorLogicで本物のLLMを動かせますか？**
A: TensorLogicは教育目的であり、GPT-4のような大規模モデルの実行には適していません。しかし、**同じ原理**を小規模で学ぶには最適です。

**Q: どこから始めればいいですか？**
A: まず `examples/attention.tl` を実行して、Attentionメカニズムを理解することをお勧めします。

**Q: なぜCross-Entropyではなく MSEを使っているのですか？**
A: 簡潔さのためです。実際のLLMはCross-Entropyを使用しますが、MSEでも学習の概念は理解できます。

**Q: 生成されたテキストはどのように取得しますか？**
A: 現在のTensorLogicでは、確率分布を出力として得ます。サンプリング（temperatureベースなど）は手動で実装する必要があります。

## まとめ

TensorLogicを使って、LLMの核心的な概念を理解できます：

✅ **Token Embeddings** - 単語をベクトルに変換
✅ **Neural Networks** - パターンを学習
✅ **Attention** - コンテキストに注目
✅ **Next Token Prediction** - 言語モデリングの基本タスク
✅ **Gradient Descent** - モデルの最適化

これらは、GPT-4のような最先端のLLMでも使われている**全く同じ基礎**です！

---

Happy Learning! 🚀
