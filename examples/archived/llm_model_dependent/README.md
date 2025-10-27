# モデル依存サンプル

このディレクトリのファイルは、特定のモデルとトークナイザーの組み合わせが必要です。

## 必要条件

これらのファイルを実行するには：

1. **モデルとトークナイザーのvocab_sizeが一致**している必要があります
2. 正しい組み合わせの例：
   - `tinyllama-1.1b-chat-q4_0.gguf` (vocab_size=32000)
   - `tinyllama-tokenizer.json` (vocab_size=32000)

## エラー例

モデルとトークナイザーが不一致の場合：
```
error: Token ID 15040 out of range for vocab size 2048
```

## ファイル

- `generation.tl` - テキスト生成デモ（モデル+トークナイザー必須）
- `tokenizer_embedding.tl` - トークナイザーとembedding統合テスト

## 実行方法

正しいモデルとトークナイザーを配置後：
```bash
tl run generation.tl
tl run tokenizer_embedding.tl
```
