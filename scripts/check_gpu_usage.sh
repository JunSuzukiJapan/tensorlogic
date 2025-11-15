#!/bin/bash
# GPU Usage Monitor for TensorLogic
# Measures GPU usage before and after running tl script

echo "=========================================="
echo "GPU Usage Monitor - TensorLogic"
echo "=========================================="
echo ""

# Function to get GPU info
get_gpu_info() {
    local label=$1
    echo "[$label]"
    echo "----------------------------------------"

    # macOS GPU info via ioreg
    echo "GPU Active Residency:"
    ioreg -r -d 1 -w 0 -c "IOAccelerator" 2>/dev/null | grep -i "PerformanceStatistics" | head -1

    # Get Metal device utilization (if available)
    echo ""
    echo "Metal Processes:"
    ps aux | grep -E "MTL|Metal" | grep -v grep | awk '{printf "  PID: %-6s CPU: %5s%% MEM: %5s%%  %s\n", $2, $3, $4, $11}'

    # Memory pressure
    echo ""
    echo "Memory Pressure:"
    memory_pressure | grep "System-wide memory free percentage"

    # VM stats
    echo ""
    echo "VM Stats (Pages):"
    vm_stat | grep -E "free|active|inactive|wired|speculative" | head -5

    echo "----------------------------------------"
    echo ""
}

# Check before running
get_gpu_info "BEFORE TL EXECUTION"

# Run the tl script if provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 <tl_script_path>"
    echo "Example: $0 examples/chat_demo_short.tl"
    exit 1
fi

TL_SCRIPT=$1
echo "Running: ./target/release/tl run $TL_SCRIPT"
echo ""

# Execute with timeout
timeout 60 ./target/release/tl run "$TL_SCRIPT" 2>&1 | tail -50

echo ""
echo ""

# Check after running
get_gpu_info "AFTER TL EXECUTION"

# Check for lingering processes
echo "[LINGERING PROCESSES CHECK]"
echo "----------------------------------------"
echo "TensorLogic processes:"
pgrep -fl tl | grep -v "localization\|spotlight\|check_gpu"

if [ $? -eq 0 ]; then
    echo ""
    echo "⚠️  WARNING: TensorLogic processes still running!"
    echo "   These processes may be holding GPU resources"
else
    echo "✅ No lingering tl processes"
fi
echo ""

echo "=========================================="
echo "GPU Usage Check Complete"
echo "=========================================="
