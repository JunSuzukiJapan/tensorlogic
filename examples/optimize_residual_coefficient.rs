// Optimize residual scaling coefficient
// Phase 1: Test 0.0 to 1.0 in 0.1 increments
// Phase 2: Refine best range in 0.01 increments

use tensorlogic::device::MetalDevice;
use tensorlogic::tensor::Tensor;

fn tensor_stats(tensor: &Tensor) -> (f32, f32, f32, f32) {
    let values = tensor.to_vec();
    let mut min_val = f32::MAX;
    let mut max_val = f32::MIN;
    let mut sum = 0.0f32;
    let mut sum_sq = 0.0f32;

    for val in &values {
        let f = val.to_f32();
        if f < min_val { min_val = f; }
        if f > max_val { max_val = f; }
        sum += f;
        sum_sq += f * f;
    }

    let count = values.len() as f32;
    let mean = sum / count;
    let variance = (sum_sq / count) - (mean * mean);
    let std = variance.sqrt();

    (min_val, max_val, mean, std)
}

fn test_coefficient(device: &MetalDevice, coeff: f32, hidden_dim: usize) -> (f32, f32) {
    // Create input
    let input_data: Vec<f32> = (0..hidden_dim)
        .map(|i| ((i as f32 * 0.01) % 2.0 - 1.0) * 0.1)
        .collect();
    let input_data_f16: Vec<half::f16> = input_data.iter()
        .map(|&x| half::f16::from_f32(x))
        .collect();
    let input = Tensor::from_vec_metal(device, input_data_f16.clone(), vec![1, hidden_dim]).unwrap();

    // Create weights
    let norm_weight_data: Vec<half::f16> = vec![half::f16::from_f32(1.0); hidden_dim];
    let norm_weight = Tensor::from_vec_metal(device, norm_weight_data, vec![hidden_dim]).unwrap();

    let weight_data: Vec<f32> = (0..(hidden_dim * hidden_dim))
        .map(|i| ((i as f32 * 0.001) % 1.0 - 0.5) * 0.01)
        .collect();
    let weight_data_f16: Vec<half::f16> = weight_data.iter()
        .map(|&x| half::f16::from_f32(x))
        .collect();
    let weight = Tensor::from_vec_metal(device, weight_data_f16, vec![hidden_dim, hidden_dim]).unwrap();

    let mut x = input.clone();
    let (_, _, _, initial_std) = tensor_stats(&x);

    // Simulate 22 layers
    for _ in 0..22 {
        let normed = x.rms_norm(vec![hidden_dim], &norm_weight, 1e-5).unwrap();
        let proj = normed.matmul(&weight).unwrap();

        // Apply residual scaling
        let proj_data = proj.to_vec();
        let proj_scaled_data: Vec<half::f16> = proj_data.iter()
            .map(|&v| half::f16::from_f32(v.to_f32() * coeff))
            .collect();
        let proj_scaled = Tensor::from_vec_metal(device, proj_scaled_data, vec![1, hidden_dim]).unwrap();

        x = x.add(&proj_scaled).unwrap();
    }

    let (_, _, _, final_std) = tensor_stats(&x);
    let std_growth = final_std / initial_std;

    (final_std, std_growth)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Residual Scaling Coefficient Optimization ===\n");
    println!("Target: Find coefficient where std growth = 1.3x (±0.2 acceptable)\n");

    let device = MetalDevice::new()?;
    let hidden_dim = 2048;
    let target_growth = 1.3;

    // Phase 1: Coarse search (0.0 to 0.5, step 0.1)
    println!("=== PHASE 1: Coarse Search (0.0 to 0.5, step 0.1) ===\n");
    println!("{:<10} {:<15} {:<15} {:<10}", "Coeff", "Final Std", "Std Growth", "Rating");
    println!("{}", "-".repeat(55));

    let mut phase1_results: Vec<(f32, f32, f32)> = Vec::new();

    for i in 0..=50 {
        let coeff = i as f32 / 100.0;
        let (final_std, std_growth) = test_coefficient(&device, coeff, hidden_dim);

        // Rating: closer to target_growth (1.3) is better
        let deviation = (std_growth - target_growth).abs();
        let rating = (1.0 - deviation / target_growth).max(0.0) * 100.0;

        phase1_results.push((coeff, std_growth, rating));

        // Only print every 10th result in Phase 1
        if i % 10 == 0 {
            println!("{:<10.2} {:<15.6} {:<15.2} {:<10.1}",
                     coeff, final_std, std_growth, rating);
        }
    }

    println!("\n{}", "=".repeat(55));

    // Find best coefficient from Phase 1
    let mut best_coeff = 0.0;
    let mut best_rating = 0.0;

    for (coeff, growth, rating) in &phase1_results {
        if *rating > best_rating {
            best_rating = *rating;
            best_coeff = *coeff;
        }
    }

    println!("\nPhase 1 Best: coeff={:.2}, rating={:.1}", best_coeff, best_rating);

    // Determine Phase 2 range (±0.05 around best)
    let phase2_start = (best_coeff - 0.05).max(0.0);
    let phase2_end = (best_coeff + 0.05).min(0.5);

    println!("\n=== PHASE 2: Fine Search ({:.2} to {:.2}, step 0.01) ===\n",
             phase2_start, phase2_end);
    println!("{:<10} {:<15} {:<15} {:<10}", "Coeff", "Final Std", "Std Growth", "Rating");
    println!("{}", "-".repeat(55));

    let mut phase2_results: Vec<(f32, f32, f32)> = Vec::new();
    let num_steps = ((phase2_end - phase2_start) / 0.01) as usize + 1;

    for i in 0..num_steps {
        let coeff = phase2_start + (i as f32 * 0.01);
        let coeff = (coeff * 100.0).round() / 100.0;  // Round to 2 decimals

        if coeff > phase2_end {
            break;
        }

        let (final_std, std_growth) = test_coefficient(&device, coeff, hidden_dim);
        let deviation = (std_growth - target_growth).abs();
        let rating = (1.0 - deviation / target_growth).max(0.0) * 100.0;

        phase2_results.push((coeff, std_growth, rating));

        println!("{:<10.2} {:<15.6} {:<15.2} {:<10.1}",
                 coeff, final_std, std_growth, rating);
    }

    println!("\n{}", "=".repeat(55));

    // Find optimal coefficient
    let mut optimal_coeff = 0.0;
    let mut optimal_rating = 0.0;
    let mut optimal_growth = 0.0;

    for (coeff, growth, rating) in &phase2_results {
        if *rating > optimal_rating {
            optimal_rating = *rating;
            optimal_coeff = *coeff;
            optimal_growth = *growth;
        }
    }

    println!("\n=== OPTIMIZATION COMPLETE ===\n");
    println!("Optimal Coefficient: {:.3}", optimal_coeff);
    println!("Std Growth: {:.2}x", optimal_growth);
    println!("Best Rating: {:.1}", optimal_rating);
    println!();
    println!("Interpretation:");
    println!("  - Target growth: {:.1}x (ideal for TinyLlama 22 layers)", target_growth);
    println!("  - Actual growth: {:.2}x", optimal_growth);
    println!("  - Deviation: {:.2}x", (optimal_growth - target_growth).abs());
    println!();
    println!("Recommendation:");
    println!("  Use residual_scale = {:.3} in TensorLogic scripts", optimal_coeff);
    println!();
    println!("For reference:");
    println!("  - Theoretical value (1/sqrt(44)): 0.151");
    println!("  - No scaling (original): 1.0");
    println!("  - Your optimal value: {:.3}", optimal_coeff);

    Ok(())
}
