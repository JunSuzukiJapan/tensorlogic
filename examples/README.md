# TensorLogic Examples

このディレクトリには、TensorLogicの機能を示すサンプルコードが含まれています。

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

### 📁 gnn/ - Graph Neural Networks
グラフニューラルネットワークの機能をテストするサンプル

- `gnn_comprehensive.tl` - 包括的なGNNテスト
- `gnn_simple.tl` - 簡略版GNNテスト（学習用）

### 📁 integration/ - 統合テスト
複数機能の統合や新機能のテストサンプル

- `unified_syntax.tl` - 統一構文のテスト
- `embedding.tl` - 埋め込み機能
- `new_features.tl` - 新機能のテスト

### 📄 ルートディレクトリ

- `test_gpu_usage.tl` - GPU使用のテスト

## 実行方法

```bash
# プロジェクトルートから実行
./target/debug/tl run examples/basics/env_input.tl
./target/debug/tl run examples/llm/generation.tl
./target/debug/tl run examples/gnn/gnn_simple.tl
```

## 統一構文について

TensorLogicは統一構文を採用しています：

- **ファクト**: `Parent(alice, bob)` （`<-` 接頭辞なし）
- **クエリ**: `Parent(alice, X)?` （`?` 接尾辞のみ）
- **ルール**: テンソル方程式として記述 `Ancestor[x, z] = H(Ancestor[x, y] Parent[y, z])`

詳細は `integration/unified_syntax.tl` を参照してください。
