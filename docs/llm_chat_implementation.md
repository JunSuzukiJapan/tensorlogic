# Local LLM Chat Implementation in TensorLogic

## 概要

TensorLogicを使用してローカルLLM（TinyLlama 1.1B）によるチャットアプリケーションを実装しています。

## 現在の実装状況

### ✅ 完了した機能

1. **配列リテラル内での変数使用**
   - 構文: `ones([seq_len, d_model])`
   - AST、パーサー、インタプリタ、型チェッカーを更新
   - 例: `examples/test_array_variables.tl`

2. **Transformer アーキテクチャの実装（TensorLogic DSL）**
   - Multi-Head Self-Attention: `examples/transformer_attention.tl`
   - Complete Transformer Block: `examples/transformer_block.tl`
   - LLM Inference Pipeline: `examples/llm_inference.tl`
   - 使用可能な操作: `matmul`, `softmax`, `gelu`, `layer_norm`, `transpose`, `apply_attention_mask`, etc.

3. **モデル管理**
   - TinyLlama 1.1B Chat (Q4_0) モデルのダウンロード完了 (609MB)
   - GGUFフォーマットのモデルローダー実装済み
   - モデルパス: `~/.tensorlogic/models/tinyllama-1.1b-chat-q4_0.gguf`
   - テンソル数: 201個

4. **トークナイザー**
   - HuggingFace tokenizersライブラリを使用
   - `load_tokenizer(path)` - トークナイザーのロード
   - `tokenize(tokenizer, text)` - テキストのトークン化
   - `detokenize(tokenizer, token_ids)` - トークンIDのデコード

5. **サンプリング戦略**
   - Top-k sampling: `top_k(logits, k)` - 上位k個のトークンに限定
   - Top-p (nucleus) sampling: `top_p(logits, p)` - 累積確率pまでのトークンに限定
   - `sample(probs)` - 確率分布からトークンをサンプリング
   - 例: `examples/text_generation_sampling.tl` - 各手法のデモ

6. **自己回帰生成パイプライン**
   - `examples/autoregressive_generation.tl` - 完全な生成ループのデモ
   - トークン単位の生成プロセス（3ステップのデモンストレーション）
   - Top-k → Top-p → Softmax → Sample の統合パイプライン
   - 本番環境のハイパーパラメータ例（チャット、コード生成、創作）
   - KV-Cache、Temperature scaling、特殊トークンの説明

7. **チャットインターフェース**
   - 基本的なチャットデモ: `examples/local_llm_chat.tl`
   - Chat REPL アーキテクチャ: `examples/chat_repl_demo.tl`
   - マルチターン会話のシミュレーション
   - ChatMLフォーマット対応
   - REPLコンポーネント詳細（セッション状態、入力処理、プロンプト構築）
   - 特殊コマンド: /help, /clear, /exit, /config, /temp, /tokens
   - `generate(model, prompt, max_tokens, temperature)` API（プレースホルダー）

### 🚧 実装中・未実装の機能

1. **Transformer Forward Pass（Rustによる実装が必要）**
   - Embedding層からの入力処理
   - 複数レイヤーの順伝播
   - 最終層からlogitsの生成
   - GGUFモデルからの重みのロード

2. **KV-Cache**
   - 自己回帰生成の効率化
   - 以前のトークンのKey/Valueテンソルをキャッシュ
   - メモリ効率的な実装

3. **Temperature Scaling**
   - `logits = logits / temperature` の実装
   - generate()関数への統合

4. **ストリーミング出力**
   - トークン単位のリアルタイム出力
   - ユーザー体験の向上
   - コールバック/非同期アーキテクチャ

5. **インタラクティブREPL（完全実装）**
   - 対話型チャットループの実装
   - チャット履歴の管理とトランケーション
   - 特殊コマンドの処理
   - セッション保存/ロード

## アーキテクチャ

### TensorLogic DSLによるTransformer実装

```tensorlogic
# Multi-Head Self-Attention (examples/transformer_attention.tl)
let seq_len = 8
let d_model = 512
let embeddings = positional_encoding(seq_len, d_model)

let Q = matmul(embeddings, W_q)
let K = matmul(embeddings, W_k)
let V = matmul(embeddings, W_v)

let scores = matmul(Q, transpose(K))
let mask = ones([seq_len, seq_len])
let masked_scores = apply_attention_mask(scores, mask)
let attn_weights = softmax(masked_scores, 1)
let output = matmul(attn_weights, V)
```

### モデルロードと推論

```tensorlogic
# Chat Application (examples/local_llm_chat.tl)
let model = load_model("/path/to/tinyllama-1.1b-chat-q4_0.gguf")
let response = generate(model, "What is the capital of Japan?", 100, 0.7)
```

## 使用可能な操作（48個）

### Tensor Creation (2)
- `zeros([m, n])` - ゼロテンソル
- `ones([m, n])` - 1のテンソル

### Shape Operations (6)
- `reshape(x, [new_shape])` - 形状変更
- `flatten(x)` - 1次元化
- `transpose(x)` - 転置
- `permute(x, [dims])` - 次元の並び替え
- `unsqueeze(x, dim)` - 次元追加
- `squeeze(x, dim)` - 次元削除

### Math Functions (6)
- `exp(x)`, `log(x)`, `sqrt(x)`
- `pow(x, y)`, `sin(x)`, `cos(x)`, `tan(x)`

### Aggregation (4)
- `max(x, dim)`, `min(x, dim)`
- `argmax(x, dim)`, `argmin(x, dim)`

### Activation (2)
- `gelu(x)` - GELU活性化
- `tanh(x)` - Tanh活性化

### Normalization (2)
- `layer_norm(x, shape, eps)` - Layer Normalization
- `batch_norm(x, mean, var, eps)` - Batch Normalization

### Attention/Masking (3)
- `apply_attention_mask(scores, mask)` - Attention mask適用
- `padding_mask(lengths, max_len)` - Padding mask生成
- `combine_masks(mask1, mask2)` - マスクの結合

### Broadcast (1)
- `broadcast_to(x, shape)` - ブロードキャスト

### Fused Operations (4)
- `fused_add_relu(a, b)` - Add + ReLU
- `fused_mul_relu(a, b)` - Mul + ReLU
- `fused_affine(x, scale, bias)` - Affine変換
- `fused_gelu_linear(x, weight, bias)` - GELU + Linear

### その他 (18)
- `matmul(a, b)` - 行列積
- `softmax(x, dim)` - Softmax
- `relu(x)` - ReLU
- `positional_encoding(seq_len, d_model)` - 位置エンコーディング
- など

## 次のステップ

### Phase 1: Rust実装によるTransformer Forward Pass
1. Embedding層の実装
2. Multi-Head Attentionの最適化実装
3. FFNの実装
4. Layer Normalizationの統合

### Phase 2: 推論パイプラインの実装
1. KV-Cacheの実装
2. Top-k/Top-pサンプリング
3. Temperature scalingの実装
4. `generate()` 関数の完全実装

### Phase 3: ユーザー体験の向上
1. ストリーミング出力
2. インタラクティブREPL
3. チャット履歴の管理
4. 複数のプロンプトフォーマット対応

### Phase 4: パフォーマンス最適化
1. Metal GPUの完全活用
2. Batch処理のサポート
3. モデル量子化の最適化
4. メモリ管理の改善

## 参考資料

- TinyLlama: https://github.com/jzhang38/TinyLlama
- GGUF Format: https://github.com/ggerganov/llama.cpp
- Attention is All You Need: https://arxiv.org/abs/1706.03762
- Tensor-Logic Paper: https://arxiv.org/abs/2510.12269

## 実装例の実行方法

```bash
# Transformer アーキテクチャのデモ
./target/debug/tl run examples/transformer_attention.tl
./target/debug/tl run examples/transformer_block.tl
./target/debug/tl run examples/llm_inference.tl

# 配列リテラル内での変数使用のテスト
./target/debug/tl run examples/test_array_variables.tl

# ローカルLLMチャットデモ
./target/debug/tl run examples/local_llm_chat.tl
```
