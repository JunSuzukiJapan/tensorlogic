# Candle Alignment Plan - Phase 3 (REVISED)

## Overview

This plan aligns TensorLogic's chat demo implementation with Candle's architecture. Based on analysis in Phase 2, the following differences must be addressed:

| Aspect | Candle | TensorLogic (Current) | Status |
|--------|--------|----------------------|---------|
| Prefill | `model.forward(all_tokens, 0)` | Manual 22-layer loop | ⚠️ Needs unified loop |
| Decode | `model.forward([token], pos)` | Manual 22-layer loop | ⚠️ Needs unified loop |
| Position | CPU `index_pos` parameter | CPU `current_pos` variable | ✅ Already aligned |
| KV Cache | Auto-managed in Model | Manual concat() × 44/token | ⚠️ Needs abstraction |
| Command Buffer | Unified device.command_buffer() | ✅ Working correctly | ✅ Already aligned |

## Analysis Results

After investigating commit c631522 and current state:
- ✅ Command buffer architecture is correct (device.command_buffer() exists)
- ✅ EncoderProvider trait properly implemented
- ✅ CPU position tracking already in place
- ❌ Prefill/decode have duplicate 22-layer loops (lines ~120-260 and ~335-521)
- ❌ Manual KV cache management with 44 concat operations per token

## Critical Constraints

⚠️ **NO RUNTIME TESTING** until all checkboxes complete
✅ **Build checks OK** (cargo build, cargo test --no-run)
🎯 **Complete alignment first**, then test once

## Phase 3.1: Command Buffer Architecture Fix

### Goal: Restore unified command buffer management (Candle-style)

- [ ] **3.1.1** Analyze EncoderProvider changes in commit c631522
  - Read src/device/mod.rs for EncoderProvider trait
  - Read src/device/commands.rs for Commands implementation
  - Identify how buffer management broke

- [ ] **3.1.2** Restore device.command_buffer() method (if missing)
  - Ensure Device has direct command_buffer() access
  - Match Candle's metal/device.rs interface
  - Build check: `cargo build`

- [ ] **3.1.3** Fix encoder acquisition in ops
  - Update all ops to use unified buffer access
  - Remove any ad-hoc buffer pool workarounds
  - Build check: `cargo build`

- [ ] **3.1.4** Verify buffer lifecycle matches Candle
  - Check buffer commit/wait logic
  - Ensure single buffer per thread
  - Build check: `cargo build`

## Phase 3.2: Model Abstraction Layer

### Goal: Replace manual 22-layer loops with model.forward() (Candle-style)

- [ ] **3.2.1** Create Model struct for LLaMA
  - New file: src/model/llama.rs
  - Struct fields: layers, embeddings, norm, lm_head
  - Match Candle's candle-transformers/src/models/llama.rs structure
  - Build check: `cargo build`

- [ ] **3.2.2** Move KV cache to Model struct
  - Add `cache: Cache` field to Model
  - Cache manages all 22 layers' KV tensors
  - Remove manual concat() from chat demo
  - Build check: `cargo build`

- [ ] **3.2.3** Implement model.forward() method
  ```rust
  pub fn forward(&self, x: &Tensor, index_pos: usize, cache: &mut Cache) -> Result<Tensor>
  ```
  - Take input tokens tensor
  - Loop through layers internally
  - Update cache internally
  - Return logits tensor
  - Build check: `cargo build`

- [ ] **3.2.4** Implement Cache struct
  - Store K and V tensors for all layers
  - Handle prefill (create) vs decode (concat) logic
  - Match Candle's Cache implementation
  - Build check: `cargo build`

## Phase 3.3: Prefill Simplification

### Goal: Replace manual prefill loop with single model.forward() call

- [ ] **3.3.1** Update prefill in chat_full_22layers_f16.tl
  - Replace lines ~120-260 (22-layer manual loop)
  - Single call: `let logits = model.forward(input_tokens, 0, &mut cache)`
  - index_pos = 0 for prefill
  - Build check: `cargo build`

- [ ] **3.3.2** Remove prefill-specific KV initialization
  - Cache.forward() handles prefill automatically
  - No manual KV tensor creation
  - Build check: `cargo build`

- [ ] **3.3.3** Simplify prefill sampling
  - Keep temperature_sample call
  - Remove unnecessary shape() operations
  - Build check: `cargo build`

## Phase 3.4: Decode Simplification

### Goal: Replace manual decode loop with model.forward() per token

- [ ] **3.4.1** Update decode loop in chat_full_22layers_f16.tl
  - Replace lines ~335-521 (22-layer manual loop)
  - Single call per token: `let logits = model.forward([token], current_pos, &mut cache)`
  - Keep position increment: `current_pos = current_pos + 1`
  - Build check: `cargo build`

- [ ] **3.4.2** Remove decode-specific KV concat operations
  - Cache.forward() handles concat automatically
  - No manual KV0-KV21 concat calls
  - Build check: `cargo build`

- [ ] **3.4.3** Keep decode sampling unchanged
  - temperature_sample already correct
  - No shape() calls (already removed in Phase 2)
  - Build check: `cargo build`

## Phase 3.5: Integration and Cleanup

### Goal: Final alignment verification and cleanup

- [ ] **3.5.1** Remove all manual layer variables
  - Delete K0-K21, V0-V21 variables
  - Delete hidden0-hidden21 intermediate tensors
  - Cache manages all state
  - Build check: `cargo build`

- [ ] **3.5.2** Verify no explicit sync calls
  - No sync_and_read() calls (Candle doesn't use)
  - No flush_gpu() calls (Candle doesn't use)
  - Sync happens implicitly via command buffer
  - Build check: `cargo build`

- [ ] **3.5.3** Match Candle's initialization pattern
  - Load model weights into Model struct
  - Initialize empty Cache
  - Match Candle's main.rs setup flow
  - Build check: `cargo build`

- [ ] **3.5.4** Final compilation check
  - `cargo build --release`
  - `cargo test --no-run`
  - All tests compile (not run yet)

## Phase 3.6: First Runtime Test

### Goal: Validate complete alignment

- [ ] **3.6.1** Run prefill test
  - `timeout 60 ./target/release/tl run examples/chat_full_22layers_f16.tl`
  - Should process "User: Hi" prompt
  - Should generate first token

- [ ] **3.6.2** Run generation test
  - Should generate multiple tokens
  - Should not hang
  - Should match Candle's performance

- [ ] **3.6.3** Document results
  - Update candle_migration_phase2_shape_elimination.md
  - Record sync count per token
  - Record performance metrics

## Comparison Checklist

After completion, verify these match Candle:

- [ ] Prefill: Single `model.forward(tokens, 0)` call
- [ ] Decode: Single `model.forward([token], pos)` call per token
- [ ] Position: CPU `index_pos` parameter (not GPU shape())
- [ ] KV Cache: Auto-managed in Cache struct
- [ ] Command Buffer: Unified device interface
- [ ] Sync Points: Only implicit (no explicit calls)
- [ ] Performance: ~1 GPU sync per token

## Success Criteria

✅ **All checkboxes complete**
✅ **Cargo build succeeds**
✅ **Cargo test compiles**
✅ **Runtime test generates tokens**
✅ **No explicit sync calls**
✅ **Architecture matches Candle**
