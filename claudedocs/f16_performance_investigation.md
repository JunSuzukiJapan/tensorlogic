# F16/F32 Performance Investigation

## Problem
22-layer F16/F32 models hang at "Assistant:" with no token generation.

## Root Cause Identified

### Timeline of Commits
- ✅ **09ccab9** (`apply_rope_k` 前): トークン生成成功
- ❌ **5605e23以降**: `apply_rope_k()` 関数追加後にハング

### Bottleneck: `apply_rope_k()` 関数

```tensorlogic
fn apply_rope_k(K: float16[?, ?], pos: float) -> float16[?, ?] {
    let shp = shape(K)          // ← ボトルネック！
    let s = shp[0]
    let K_h = reshape(K, [s, 4.0, 64.0])
    let K_r = rope(K_h, pos)
    result := reshape(K_r, [s, 256.0])
}
```

### Performance Degradation Path

1. `shape(K)` → 新しいGPUテンソル作成（小さいが頻繁）
2. `reshape()` → 非連続テンソル作成（メタデータのみ変更）
3. `rope()` → `.contiguous()` 呼び出し（reshape後は必ず非連続）
4. `.contiguous()` **CPU実装** → **GPU→CPU→GPU転送** (128KB)
5. 22層 × prefill/decode × 毎トークン → 莫大なオーバーヘッド

### Measurements

| Commit | Description | Result |
|--------|-------------|--------|
| 09ccab9 | `apply_rope_k` なし | ✅ トークン生成（"Assistant: Or"） |
| 5605e23 | `apply_rope_k` 追加 | ❌ ハング（2分でタイムアウト） |
| 68cb9e3 | `.contiguous()` 追加 | ❌ ハング |
| 991b4a3 | GPU sampling実装 | ❌ ハング |
| 338605e | GPU contiguous実装 | （未テスト、根本原因未解決）|

## Solutions

### Option 1: Rust Builtin Implementation (最良)
`apply_rope_k()` をRust builtinとして実装：
- メリット: 最速、`shape()`呼び出し不要
- デメリット: 実装コスト

### Option 2: Shape-Free Function (一時対策)
形状を引数として渡す：
```tensorlogic
fn apply_rope_k_fast(K: float16[?, ?], seq_len: float, pos: float) -> float16[?, ?] {
    let K_h = reshape(K, [seq_len, 4.0, 64.0])
    let K_r = rope(K_h, pos)
    result := reshape(K_r, [seq_len, 256.0])
}
```

### Option 3: GPU Contiguous (部分的改善)
GPU `contiguous()` 実装は完了（commit 338605e）：
- メリット: GPU→CPU→GPU転送を排除
- デメリット: `shape()`呼び出しオーバーヘッドは残る
- 推定改善: 30-50%（根本解決ではない）

## Recommendation

1. **短期**: Option 2（`shape()`削除）で動作確認
2. **中期**: GPU `contiguous()` 有効化でさらに改善
3. **長期**: Option 1（Rust builtin）で最適化

## Implementation Status

- ✅ GPU `contiguous()` 実装完了（commit 338605e、現在無効化）
- ⏳ `shape()` 削除版 `apply_rope_k()` 未実装
- ⏳ Rust builtin `apply_rope_k()` 未実装

