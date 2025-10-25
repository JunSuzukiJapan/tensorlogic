# TensorLogic Examples

このディレクトリにはTensorLogicの使用例が含まれています。

**最終更新**: 2025-10-25 - 大規模整理実施（130ファイル → トップレベル5ファイルに整理）

## 🚀 クイックスタート

### 基本動作確認
```bash
# Softmax、RMSNorm、Embedding、Matmulの動作確認
./target/release/tl run examples/verify_operations.tl

# SwiGLU活性化関数の確認
./target/release/tl run examples/verify_swiglu.tl
```

### チャット推論
```bash
# 10層モデルでチャット（最新版・稼働中）
./target/release/tl run examples/chat_10layers_kv_rope.tl
```

## 📂 トップレベルファイル（重要・稼働中）

- **[chat_10layers_kv_rope.tl](chat_10layers_kv_rope.tl)** - 最新の10層チャットデモ（KVキャッシュ + RoPE対応）
- **[chat_repl_demo.tl](chat_repl_demo.tl)** - ChatGPT風REPLアーキテクチャドキュメント
- **[verify_operations.tl](verify_operations.tl)** - 基本演算の動作確認（RMSNorm, Softmax, Embedding, Matmul）
- **[verify_swiglu.tl](verify_swiglu.tl)** - SwiGLU活性化関数の動作確認
- **[check_weight_shapes.tl](check_weight_shapes.tl)** - モデル重み形状のチェックツール

## ディレクトリ構成

### 📁 basics/ - 基本機能
言語の基本的な機能とコントロールフローをテストするサンプル

- `env_input.tl` - 環境変数とユーザー入力
- `builtins.tl` - 組み込み関数
- `keywords.tl` - キーワードのテスト
- `function.tl` - 関数定義と呼び出し
- `if.tl` - if文
- `control_flow.tl` - 制御構造
- `break.tl` - break文
- `variable_redefinition.tl` - 変数の再定義
- `variable_update.tl` - 変数の更新
- `array_variables.tl` - 配列変数

### 📁 tensor_ops/ - テンソル演算
テンソル操作と数値計算の機能をテストするサンプル

- `advanced_ops.tl` - 高度なテンソル演算
- `all_20_ops.tl` - 主要な20の演算
- `argmax.tl` - argmax演算
- `broadcast.tl` - ブロードキャスト機能
- `squeeze_unsqueeze.tl` - 次元の追加/削除
- `split_chunk.tl` - テンソルの分割
- `rms_norm.tl` - RMSノーマライゼーション
- `new_builtins.tl` - 新しい組み込み関数

### 📁 llm/ - LLM/Transformer
大規模言語モデルとTransformer関連の機能をテストするサンプル

- `generation.tl` - テキスト生成（詳細版）
- `tokenizer.tl` - トークナイザーの基本機能
- `tokenizer_embedding.tl` - トークナイザーと埋め込みの統合
- `model_tensors.tl` - モデルからのテンソル取得
- `transformer_ops.tl` - Transformer演算
- `transformer_functional.tl` - Transformerの関数型実装
- `sampling.tl` - サンプリング手法
- `softmax_sample.tl` - Softmaxサンプリング

### 📁 features/ - 機能別デモ

#### features/attention/ (9ファイル)
Attention機構とTransformer関連のデモ
- 基本Attention実装
- Multi-head Attention
- Grouped Query Attention (GQA)
- Transformer Block

#### features/gnn/ (6ファイル)
Graph Neural NetworkとKnowledge Graph関連のデモ
- GNNメッセージパッシング
- ノード分類
- Knowledge Graph埋め込み
- 関係予測

#### features/tutorials/ (4ファイル)
TensorLogicの基本チュートリアル
- Linear Regression
- Logistic Regression
- Neural Network
- Logic Programming

### 📁 tests/ - テストファイル (7ファイル)

基本機能のテストファイル
- Buffer管理テスト（buffer_stats_test.tl, simple_buffer_test.tl, concat_test.tl）
- 数学演算テスト（test_rmsnorm_math.tl, test_rope_simple.tl, test_softmax_simple.tl）
- モデルロードテスト（test_model_basic.tl）

### 📁 archived/ - アーカイブ済みファイル (64ファイル)

古いデバッグファイルや実験的なコード。参考用に保持。

#### archived/debug/ (20ファイル)
開発中のデバッグスクリプト
- レイヤー別デバッグ（1層、2層、5層）
- トークン数別テスト（29トークン、30トークン）
- Attention、Embedding、RoPEの個別デバッグ

#### archived/old_chat/ (10ファイル)
古いチャットデモのバージョン
- 2層、5層、22層版
- greedy sampling版
- RoPEなし版

#### archived/kv_tests/ (6ファイル)
KVキャッシュ機能の初期テスト

#### archived/tinyllama_tests/ (4ファイル)
TinyLlamaモデルの推論テスト

#### archived/profiling/ (2ファイル)
パフォーマンスプロファイリング用スクリプト

#### archived/old_demos/ (22ファイル)
その他の古いデモファイル
- 自動回帰生成
- Batch normalization
- Dropout
- サンプリング戦略
- CoreML Neural Engine

### 📁 gnn/ - Graph Neural Networks
グラフニューラルネットワークの機能をテストするサンプル（既存のサブディレクトリ）

- `gnn_comprehensive.tl` - 包括的なGNNテスト
- `gnn_simple.tl` - 簡略版GNNテスト（学習用）

### 📁 integration/ - 統合テスト
複数機能の統合や新機能のテストサンプル

- `unified_syntax.tl` - 統一構文のテスト
- `embedding.tl` - 埋め込み機能
- `new_features.tl` - 新機能のテスト

### 📄 ルートディレクトリ

- `test_gpu_usage.tl` - GPU使用のテスト

## 📝 整理履歴

**整理日**: 2025-10-25

**整理内容**:
- 130個のファイルから90個（69.2%）を整理
- トップレベルを5個の重要ファイルのみに整理
- 機能別・用途別にディレクトリ構造化
- デバッグファイルとテストファイルを分離

**移動したファイル**:
| カテゴリ | 移動先 | ファイル数 |
|---------|--------|-----------|
| Debug scripts | `archived/debug/` | 20 |
| Old chat demos | `archived/old_chat/` | 10 |
| KV cache tests | `archived/kv_tests/` | 6 |
| TinyLlama tests | `archived/tinyllama_tests/` | 4 |
| Profiling scripts | `archived/profiling/` | 2 |
| Old demos | `archived/old_demos/` | 22 |
| Attention/Transformer | `features/attention/` | 9 |
| GNN/KG | `features/gnn/` | 6 |
| Tutorials | `features/tutorials/` | 4 |
| Tests | `tests/` | 7 |

## 実行方法

```bash
# プロジェクトルートから実行
./target/release/tl run examples/verify_operations.tl
./target/release/tl run examples/chat_10layers_kv_rope.tl
./target/debug/tl run examples/gnn/gnn_simple.tl
```

## 統一構文について

TensorLogicは統一構文を採用しています：

- **ファクト**: `Parent(alice, bob)` （`<-` 接頭辞なし）
- **クエリ**: `Parent(alice, X)?` （`?` 接尾辞のみ）
- **ルール**: テンソル方程式として記述 `Ancestor[x, z] = H(Ancestor[x, y] Parent[y, z])`

詳細は `integration/unified_syntax.tl` を参照してください。
