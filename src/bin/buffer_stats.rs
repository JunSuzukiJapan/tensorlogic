/// Buffer pool statistics utility
/// Shows memory usage and buffer reuse statistics

use std::env;
use half::f16;
use tensorlogic::device::MetalDevice;
use tensorlogic::error::TensorResult;
use tensorlogic::tensor::{Tensor, TensorCreation};

fn main() -> TensorResult<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: buffer_stats <test_type>");
        eprintln!("  test_type: simple | layers");
        std::process::exit(1);
    }

    let test_type = &args[1];
    let device = MetalDevice::new()?;

    println!("=== Buffer Pool Statistics Test ===\n");

    match test_type.as_str() {
        "simple" => test_simple(&device)?,
        "layers" => test_layers(&device)?,
        _ => {
            eprintln!("Unknown test type: {}", test_type);
            std::process::exit(1);
        }
    }

    Ok(())
}

fn test_simple(device: &MetalDevice) -> TensorResult<()> {
    println!("[Test: Simple Buffer Allocation]\n");

    // Initial stats
    print_stats(device, "Initial");

    // Allocate some tensors
    {
        let _t1 = Tensor::<f16>::zeros(device, vec![100, 100])?;
        print_stats(device, "After t1 allocation");

        let _t2 = Tensor::<f16>::zeros(device, vec![100, 100])?;
        print_stats(device, "After t2 allocation");

        let _t3 = Tensor::<f16>::zeros(device, vec![200, 200])?;
        print_stats(device, "After t3 allocation");
    }

    // After tensors go out of scope
    print_stats(device, "After scope exit");

    Ok(())
}

fn test_layers(device: &MetalDevice) -> TensorResult<()> {
    println!("[Test: Sequential Layer Processing]\n");

    print_stats(device, "Initial");

    // Simulate 5 layers
    for i in 0..5 {
        {
            // Allocate tensors for this layer
            let _input = Tensor::<f16>::zeros(device, vec![29, 2048])?;
            let _q = Tensor::<f16>::zeros(device, vec![29, 2048])?;
            let _k = Tensor::<f16>::zeros(device, vec![29, 256])?;
            let _v = Tensor::<f16>::zeros(device, vec![29, 256])?;
            let _attn_scores = Tensor::<f16>::zeros(device, vec![29, 32, 29])?;
            let _attn_weights = Tensor::<f16>::zeros(device, vec![29, 32, 29])?;
            let _attn_output = Tensor::<f16>::zeros(device, vec![29, 32, 64])?;
            let _ffn_gate = Tensor::<f16>::zeros(device, vec![29, 5632])?;
            let _ffn_up = Tensor::<f16>::zeros(device, vec![29, 5632])?;
            let _output = Tensor::<f16>::zeros(device, vec![29, 2048])?;

            print_stats(device, &format!("Layer {} - inside scope", i));
        }

        print_stats(device, &format!("Layer {} - after scope", i));
    }

    Ok(())
}

fn print_stats(device: &MetalDevice, label: &str) {
    let stats = device.buffer_pool_stats();

    println!("--- {} ---", label);
    println!("  Pooled buffers: {}", stats.total_pooled);
    println!("  Size classes: {}", stats.size_classes);
    println!("  Total memory: {} bytes ({:.2} MB)",
        stats.total_memory,
        stats.total_memory as f64 / 1024.0 / 1024.0
    );
    println!("  Allocations: {}", stats.allocation_count);
    println!("  Reuses: {}", stats.reuse_count);
    println!("  Reuse rate: {:.1}%\n",
        if stats.allocation_count > 0 {
            stats.reuse_count as f64 / (stats.allocation_count + stats.reuse_count) as f64 * 100.0
        } else {
            0.0
        }
    );
}
