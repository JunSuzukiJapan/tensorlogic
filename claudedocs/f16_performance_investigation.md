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

- ✅ GPU `contiguous()` 実装完了（commit 338605e、e6ff6d7で有効化）
- ✅ `shape()` 削除版 `apply_rope_k()` 実装完了（commit e6ff6d7）
- ❌ **改善効果なし**: 依然としてハング状態
- ⏳ Rust builtin `apply_rope_k()` 未実装（推奨）

## 追加調査結果（commit e6ff6d7）

### 実装した最適化
1. `apply_rope_k(K, seq_len, pos)`: seq_lenをパラメータ化
2. Prefill: `shape(x)`を1回だけ呼び出し、全22層に渡す
3. Decode: seq_len=1.0を固定で渡す（新K常に1トークン）
4. GPU `contiguous()` 有効化

### 効果
- `shape()` 呼び出し: 44回/token → 22回/token（50%削減）
- GPU contiguous: CPU→GPU転送を排除

### 問題: 依然ハング
デバッグ出力（`TL_DEBUG_ROPE=1 TL_DEBUG_CONTIGUOUS=1`）でROPE/CONTIGUOUSメッセージが一切出力されないことから、**最初の`rope()`呼び出し前**でハングしていると判明。

### 根本原因の再分析
`reshape()` が非連続テンソルを作成することが問題：

```tensorlogic
fn apply_rope_k(K: float16[?, ?], seq_len: float, pos: float) -> float16[?, ?] {
    let K_h = reshape(K, [seq_len, 4.0, 64.0])  // ← 非連続テンソル作成
    let K_r = rope(K_h, pos)                     // ← .contiguous()トリガー
    result := reshape(K_r, [seq_len, 256.0])
}
```

GPUで`.contiguous()`を実装しても、`reshape()`によって作成される非連続性が根本的なボトルネックとなっている。

### 最終推奨解決策
**Rust builtin `apply_rope_k()` 実装**が唯一の解決策：
- メリット: reshape中間テンソル不要、GPU最適化パス
- デメリット: 実装コスト
- 見積もり: TensorLogicの`rope()`と同様の実装パターン

