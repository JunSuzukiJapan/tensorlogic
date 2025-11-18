//! GPU Memory Check CLI Tool
//!
//! Simple command-line tool to check GPU memory usage and detect leaks.
//!
//! Usage:
//!   gpu_memory_check              - Show current GPU memory usage once
//!   gpu_memory_check --watch      - Continuously monitor GPU memory (Ctrl+C to stop)
//!   gpu_memory_check --interval 5 - Monitor with custom interval in seconds

use std::env;
use std::io::Write;
use std::thread;
use std::time::Duration;

fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = KB * 1024.0;
    const GB: f64 = MB * 1024.0;

    if bytes == 0 {
        return "0 B".to_string();
    }

    let bytes_f64 = bytes as f64;
    if bytes_f64 >= GB {
        format!("{:.2} GB", bytes_f64 / GB)
    } else if bytes_f64 >= MB {
        format!("{:.2} MB", bytes_f64 / MB)
    } else if bytes_f64 >= KB {
        format!("{:.2} KB", bytes_f64 / KB)
    } else {
        format!("{} B", bytes)
    }
}

fn show_memory_info(show_header: bool, show_details: bool) {
    let stats = tensorlogic::device::MetalBuffer::<half::f16>::pool_stats();

    if show_header {
        println!("╔═══════════════════════════════════════════════════╗");
        println!("║         GPU Memory Usage (Metal Framework)       ║");
        println!("╚═══════════════════════════════════════════════════╝");
    }

    println!("  Allocated Memory: {}", format_bytes(stats.total_memory));

    if show_details {
        println!("  Buffer Pool Stats:");
        println!("    - Total Pooled Buffers: {}", stats.total_pooled);
        println!("    - Size Classes: {}", stats.size_classes);
        println!("    - Reuse Count: {}", stats.reuse_count);
        println!("    - Allocation Count: {}", stats.allocation_count);
        println!("    - Eviction Count: {}", stats.eviction_count);
        if stats.allocation_count > 0 {
            let reuse_rate = (stats.reuse_count as f64 / (stats.reuse_count + stats.allocation_count) as f64) * 100.0;
            println!("    - Reuse Rate: {:.1}%", reuse_rate);
        }
    }

    // Show warning if memory is allocated
    if stats.total_memory > 0 {
        let mb = stats.total_memory as f64 / (1024.0 * 1024.0);
        if mb > 1000.0 {
            println!("  ⚠️  WARNING: High GPU memory usage detected!");
        } else if mb > 100.0 {
            println!("  ⚠️  Moderate GPU memory usage");
        } else if mb > 1.0 {
            println!("  ℹ️  Low GPU memory usage");
        } else {
            println!("  ✅ Minimal GPU memory usage");
        }
    } else {
        println!("  ✅ No GPU memory allocated");
    }

    if show_header {
        println!();
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();

    // Parse command line arguments
    let watch_mode = args.iter().any(|arg| arg == "--watch" || arg == "-w");
    let show_details = args.iter().any(|arg| arg == "--details" || arg == "-d");
    let mut interval_secs = 2u64;

    if let Some(pos) = args.iter().position(|arg| arg == "--interval" || arg == "-i") {
        if let Some(interval_str) = args.get(pos + 1) {
            if let Ok(interval) = interval_str.parse::<u64>() {
                interval_secs = interval;
            } else {
                eprintln!("Error: Invalid interval value");
                std::process::exit(1);
            }
        }
    }

    // Show help
    if args.iter().any(|arg| arg == "--help" || arg == "-h") {
        println!("GPU Memory Check - Monitor GPU memory usage");
        println!();
        println!("USAGE:");
        println!("  gpu_memory_check [OPTIONS]");
        println!();
        println!("OPTIONS:");
        println!("  -h, --help              Show this help message");
        println!("  -w, --watch             Continuously monitor GPU memory");
        println!("  -d, --details           Show detailed buffer pool statistics");
        println!("  -i, --interval <secs>   Update interval in seconds (default: 2)");
        println!();
        println!("EXAMPLES:");
        println!("  gpu_memory_check              Check GPU memory once");
        println!("  gpu_memory_check --details    Show detailed statistics");
        println!("  gpu_memory_check --watch      Monitor continuously");
        println!("  gpu_memory_check -w -i 5      Monitor with 5 second interval");
        return;
    }

    if watch_mode {
        println!("Monitoring GPU memory usage (press Ctrl+C to stop)");
        println!("Update interval: {} seconds", interval_secs);
        println!();

        let mut iteration = 0;
        loop {
            if iteration == 0 {
                show_memory_info(true, show_details);
            } else {
                // Show compact format for continuous monitoring
                let stats = tensorlogic::device::MetalBuffer::<half::f16>::pool_stats();
                print!("\r[Update #{}] GPU Memory: {} (Buffers: {}) ",
                    iteration,
                    format_bytes(stats.total_memory),
                    stats.total_pooled
                );
                std::io::stdout().flush().unwrap();
            }

            thread::sleep(Duration::from_secs(interval_secs));
            iteration += 1;
        }
    } else {
        // Single check mode
        show_memory_info(true, show_details);
    }
}
