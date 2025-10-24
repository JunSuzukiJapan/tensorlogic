#!/bin/bash

# KV Cache Performance Benchmark Script
# Measures performance with different token counts and layer configurations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=========================================="
echo "KV Cache Performance Benchmark"
echo "=========================================="
echo ""

# Check if binary exists
if [ ! -f "./target/release/tl" ]; then
    echo "Error: Binary not found. Please run 'cargo build --release' first."
    exit 1
fi

# Results directory
RESULTS_DIR="$PROJECT_ROOT/benchmark_results"
mkdir -p "$RESULTS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_FILE="$RESULTS_DIR/benchmark_$TIMESTAMP.txt"

echo "Results will be saved to: $RESULTS_FILE"
echo ""

# Function to run benchmark
run_benchmark() {
    local test_name=$1
    local test_file=$2
    local timeout_sec=$3

    echo "================================================"
    echo "Test: $test_name"
    echo "File: $test_file"
    echo "Timeout: ${timeout_sec}s"
    echo "================================================"

    # Record start time
    local start_time=$(date +%s)

    # Run test with timeout
    if timeout ${timeout_sec}s ./target/release/tl run "$test_file" > /dev/null 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        local status="SUCCESS"
        echo "✅ $test_name: ${duration}s"
        echo "$test_name,$status,$duration" >> "$RESULTS_FILE"
    else
        local exit_code=$?
        if [ $exit_code -eq 124 ]; then
            local status="TIMEOUT"
            echo "⏱️  $test_name: TIMEOUT (>${timeout_sec}s)"
        else
            local status="FAILED"
            echo "❌ $test_name: FAILED (exit code: $exit_code)"
        fi
        echo "$test_name,$status,>$timeout_sec" >> "$RESULTS_FILE"
    fi
    echo ""
}

# Write header to results file
echo "Test Name,Status,Duration (seconds)" > "$RESULTS_FILE"
echo "Benchmark started at: $(date)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Run benchmarks
echo ""
echo "Starting benchmarks..."
echo ""

# Basic tests
run_benchmark "2-Layer 2-Token KV Cache" "examples/kv_cache_demo.tl" 60
run_benchmark "2-Layer 3-Token Autoregressive" "examples/kv_cache_3_tokens.tl" 120

# Baseline (no KV cache)
run_benchmark "5-Layer Baseline (no cache)" "examples/chat_demo_short_context.tl" 60
run_benchmark "22-Layer Baseline (no cache, 5 tokens)" "examples/chat_demo_full_22_layers.tl" 300

# Print summary
echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Results summary:"
cat "$RESULTS_FILE"
echo ""
echo "Full results saved to: $RESULTS_FILE"
