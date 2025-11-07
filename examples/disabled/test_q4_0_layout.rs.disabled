/// Test Q4_0 dequantization layout
///
/// Verify that Q4_0 dequantization matches the expected layout.
///
/// Q4_0 format:
/// - Block size: 32 values
/// - Structure: 2 bytes f16 scale + 16 bytes 4-bit values (32 values)
/// - Each byte contains 2 4-bit values
///
/// Critical question: What is the layout?
/// Layout A (Interleaved): [low0, high0, low1, high1, ..., low15, high15]
/// Layout B (Grouped):     [low0-15, high0-15]

use half::f16;

fn dequantize_q4_0_layout_a(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18; // 2 + 16

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = vec![0.0f32; num_elements];

    for block_idx in 0..num_blocks {
        let block_offset = block_idx * BLOCK_BYTES;
        if block_offset + BLOCK_BYTES > data.len() {
            break;
        }

        // Read scale (f16)
        let scale_bytes = [data[block_offset], data[block_offset + 1]];
        let scale = f16::from_le_bytes(scale_bytes).to_f32();

        // Layout A: Interleaved [low0, high0, low1, high1, ...]
        let values_offset = block_offset + 2;
        let base_idx = block_idx * BLOCK_SIZE;

        for i in 0..16 {
            if base_idx + i * 2 >= num_elements {
                break;
            }

            let byte = data[values_offset + i];

            // Low nibble → even index
            let x0 = ((byte & 0x0F) as i8 - 8) as f32;
            result[base_idx + i * 2] = x0 * scale;

            // High nibble → odd index
            if base_idx + i * 2 + 1 < num_elements {
                let x1 = ((byte >> 4) as i8 - 8) as f32;
                result[base_idx + i * 2 + 1] = x1 * scale;
            }
        }
    }

    result
}

fn dequantize_q4_0_layout_b(data: &[u8], num_elements: usize) -> Vec<f32> {
    const BLOCK_SIZE: usize = 32;
    const BLOCK_BYTES: usize = 18; // 2 + 16

    let num_blocks = (num_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let mut result = vec![0.0f32; num_elements];

    for block_idx in 0..num_blocks {
        let block_offset = block_idx * BLOCK_BYTES;
        if block_offset + BLOCK_BYTES > data.len() {
            break;
        }

        // Read scale (f16)
        let scale_bytes = [data[block_offset], data[block_offset + 1]];
        let scale = f16::from_le_bytes(scale_bytes).to_f32();

        // Layout B: Grouped [low0-15, high0-15]
        let values_offset = block_offset + 2;
        let base_idx = block_idx * BLOCK_SIZE;

        for j in 0..16 {
            if base_idx + j >= num_elements {
                break;
            }

            let byte = data[values_offset + j];

            // Low nibble → first half
            let x0 = ((byte & 0x0F) as i8 - 8) as f32;
            result[base_idx + j] = x0 * scale;

            // High nibble → second half
            let second_idx = base_idx + j + 16;
            if second_idx < num_elements {
                let x1 = ((byte >> 4) as i8 - 8) as f32;
                result[second_idx] = x1 * scale;
            }
        }
    }

    result
}

fn main() {
    println!("=== Q4_0 Layout Verification ===\n");

    // Create test data: 1 block of Q4_0
    // Scale = 1.0, values = [0, 1, 2, ..., 31]
    let mut data = vec![0u8; 18];

    // Set scale to 1.0
    let scale = f16::from_f32(1.0);
    data[0..2].copy_from_slice(&scale.to_le_bytes());

    // Set 4-bit values representing [0, 1, 2, ..., 31] as signed
    // In 4-bit signed: value = nibble - 8
    // So for value 0: nibble = 8 (0x8)
    //    for value 1: nibble = 9 (0x9)
    //    for value 7: nibble = 15 (0xF)
    //    for value -8: nibble = 0 (0x0)

    // We want output [0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, ..., 7] (cycling)
    // Let's use a simpler pattern: bytes where we can distinguish layouts

    // Pattern: Byte i contains nibbles (i, i+1)
    // Layout A would give: [0, 1, 2, 3, 4, 5, ...]
    // Layout B would give: [0, 2, 4, 6, ..., 1, 3, 5, 7, ...]

    for i in 0..16 {
        let low_nibble = i as u8;
        let high_nibble = (i + 1) as u8;
        data[2 + i] = (high_nibble << 4) | low_nibble;
    }

    println!("Test data:");
    println!("  Scale: 1.0");
    println!("  16 bytes, each byte i contains nibbles (i, i+1)");
    println!("  Byte 0: nibbles (0, 1)");
    println!("  Byte 1: nibbles (2, 3)");
    println!("  ...");
    println!("  Byte 15: nibbles (14, 15)");
    println!();

    // Dequantize with both layouts
    let result_a = dequantize_q4_0_layout_a(&data, 32);
    let result_b = dequantize_q4_0_layout_b(&data, 32);

    println!("Layout A (Interleaved): [low0, high0, low1, high1, ...]");
    print!("  Result: [");
    for (i, val) in result_a.iter().take(16).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.0}", val);
    }
    println!(", ...]");
    println!();

    println!("Layout B (Grouped): [low0-15, high0-15]");
    print!("  Result: [");
    for (i, val) in result_b.iter().take(16).enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.0}", val);
    }
    println!(", ...]");
    println!();

    println!("Expected pattern:");
    println!("  If correct layout:");
    println!("    - Layout A: [-8, -7, -6, -5, ..., 7] (interleaved nibbles)");
    println!("    - Layout B: [-8, -6, -4, -2, ..., -7, -5, -3, -1, ...]");
    println!();

    println!("Analysis:");
    println!("  Layout A first 8 values: {:?}", &result_a[0..8]);
    println!("  Layout B first 8 values: {:?}", &result_b[0..8]);
    println!("  Layout B indices [16-23]:  {:?}", &result_b[16..24]);
    println!();

    // Now test with actual GGUF file to see which layout matches
    println!("=== Verification with Actual Model ===");
    println!("To determine the correct layout, we need to:");
    println!("1. Load a known Q4_0 tensor from the GGUF file");
    println!("2. Dequantize with both layouts");
    println!("3. Compare results with llama.cpp or candle reference");
    println!();
    println!("For now, checking TensorLogic's current implementation:");
    println!("  TensorLogic uses Layout B (Grouped)");
    println!("  Python reference uses Layout A (Interleaved)");
    println!();
    println!("⚠️  This difference could cause numerical discrepancies!");
}
