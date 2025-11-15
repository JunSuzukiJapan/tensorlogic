#!/bin/bash
# Continuous GPU Memory Monitor
# Samples GPU memory every 5 seconds during tl script execution

TL_SCRIPT=$1
if [ -z "$TL_SCRIPT" ]; then
    echo "Usage: $0 <tl_script_path>"
    exit 1
fi

LOG_FILE="/tmp/gpu_monitor.log"
> "$LOG_FILE"  # Clear log

echo "==========================================="
echo "Continuous GPU Memory Monitor"
echo "==========================================="
echo ""

# Function to get GPU memory
get_gpu_memory() {
    ioreg -r -d 1 -w 0 -c "IOAccelerator" 2>/dev/null | \
        grep "In use system memory\"=" | \
        sed 's/.*"In use system memory"=\([0-9]*\).*/\1/'
}

# Get baseline
BASELINE=$(get_gpu_memory)
echo "Baseline GPU memory: $(echo "scale=2; $BASELINE / 1048576" | bc) MB"
echo ""

# Start monitoring in background
(
    while true; do
        CURRENT=$(get_gpu_memory)
        TIMESTAMP=$(date '+%H:%M:%S')
        DIFF=$((CURRENT - BASELINE))
        CURRENT_MB=$(echo "scale=2; $CURRENT / 1048576" | bc)
        DIFF_MB=$(echo "scale=2; $DIFF / 1048576" | bc)
        echo "[$TIMESTAMP] GPU Memory: ${CURRENT_MB} MB (${DIFF_MB:+}${DIFF_MB} MB from baseline)" | tee -a "$LOG_FILE"
        sleep 5
    done
) &
MONITOR_PID=$!

# Run tl script with timeout
echo "Starting tl script: $TL_SCRIPT"
echo ""
timeout 120 ./target/release/tl run "$TL_SCRIPT"
EXIT_CODE=$?

# Stop monitoring
kill $MONITOR_PID 2>/dev/null
wait $MONITOR_PID 2>/dev/null

echo ""
echo "==========================================="
echo "Execution finished with exit code: $EXIT_CODE"
echo "==========================================="

# Final measurement
FINAL=$(get_gpu_memory)
FINAL_MB=$(echo "scale=2; $FINAL / 1048576" | bc)
DIFF=$((FINAL - BASELINE))
DIFF_MB=$(echo "scale=2; $DIFF / 1048576" | bc)

echo ""
echo "Final GPU memory: ${FINAL_MB} MB"
echo "Change from baseline: ${DIFF_MB:+}${DIFF_MB} MB"
echo ""
echo "Full log saved to: $LOG_FILE"
