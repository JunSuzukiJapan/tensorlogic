use half::f16;

fn main() {
    println!("=== f16 Accumulation Test ===\n");

    // Test 1: TinyLlama normalized_size = 2048
    let normalized_size = 2048;
    println!("Test 1: Accumulating 2048 values");

    // Simulate typical RMSNorm input (normalized values around -1 to 1)
    let values: Vec<f16> = (0..normalized_size)
        .map(|i| f16::from_f32((i as f32 / normalized_size as f32) * 2.0 - 1.0))
        .collect();

    // Method 1: f16 accumulation
    let mut sq_sum_f16 = f16::ZERO;
    for &x in &values {
        sq_sum_f16 += x * x;
    }
    let mean_sq_f16 = sq_sum_f16 / f16::from_f32(normalized_size as f32);

    // Method 2: f32 accumulation
    let sq_sum_f32: f32 = values.iter().map(|&x| {
        let v = x.to_f32();
        v * v
    }).sum();
    let mean_sq_f32 = sq_sum_f32 / normalized_size as f32;

    println!("  f16 accumulation: sq_sum={} mean_sq={}",
             sq_sum_f16, mean_sq_f16);
    println!("  f32 accumulation: sq_sum={} mean_sq={}",
             sq_sum_f32, mean_sq_f32);
    println!("  Difference: {:.6}", (mean_sq_f16.to_f32() - mean_sq_f32).abs());
    println!("  Relative error: {:.2}%\n",
             ((mean_sq_f16.to_f32() - mean_sq_f32) / mean_sq_f32 * 100.0).abs());

    // Test 2: Overflow check
    println!("Test 2: Overflow check with larger values");
    let large_values: Vec<f16> = vec![f16::from_f32(10.0); 2048];

    let mut sq_sum_large_f16 = f16::ZERO;
    for &x in &large_values {
        let sq = x * x;
        if sq.is_infinite() {
            println!("  OVERFLOW: x={} → x²={}", x, sq);
            break;
        }
        sq_sum_large_f16 += sq;
        if sq_sum_large_f16.is_infinite() {
            println!("  OVERFLOW: Accumulated sum became Inf after {} additions", large_values.len());
            break;
        }
    }

    let sq_sum_large_f32: f32 = large_values.iter().map(|&x| {
        let v = x.to_f32();
        v * v
    }).sum();

    println!("  f16 result: {} (is_infinite={})", sq_sum_large_f16, sq_sum_large_f16.is_infinite());
    println!("  f32 result: {}", sq_sum_large_f32);

    // Test 3: Precision loss demonstration
    println!("\nTest 3: Precision loss in f16");
    let base = f16::from_f32(2000.0);
    let small = f16::from_f32(0.5);
    let result = base + small;
    println!("  2000.0 + 0.5 = {} (expected 2000.5)", result);
    println!("  Lost {} due to precision limit", (2000.5 - result.to_f32()).abs());

    // Test 4: What's the maximum value we can accumulate?
    println!("\nTest 4: f16 accumulation limit");
    println!("  f16::MAX = {}", f16::MAX);
    println!("  For normalized_size=2048:");
    println!("  Max safe value per element: sqrt(f16::MAX / 2048) = {}",
             (f16::MAX.to_f32() / 2048.0).sqrt());
    println!("  If x_i > 5.68, sq_sum will overflow!");
}
