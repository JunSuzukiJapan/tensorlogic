#!/bin/bash
# GPU Memory Leak Process Diagnosis Script
#
# Usage: ./scripts/diagnose_memory_leak.sh
#

echo "==================================="
echo "GPU Memory Leak Process Diagnosis"
echo "==================================="
echo ""

echo "1. All Processes (Top 20 by Memory):"
echo "-------------------------------------"
ps aux | awk '{printf "%-8s %6s %6s %6s %s\n", $2, $3, $4, $5, $11}' | head -1
ps aux | awk '{print $2, $3, $4, $5, $11}' | tail -n +2 | sort -k3 -rn | head -20 | \
  awk '{printf "%-8s %6s %6s %6s %s\n", $1, $2, $3, $4, $5}'
echo ""

echo "2. TensorLogic Related Processes:"
echo "-------------------------------------"
ps aux | grep -E "tl|tensorlogic" | grep -v grep | grep -v diagnose
echo ""

echo "3. Metal/GPU Related Processes:"
echo "-------------------------------------"
ps aux | grep -E "Metal|GPU|MTL" | grep -v grep
echo ""

echo "4. VSCode GPU Processes:"
echo "-------------------------------------"
ps aux | grep "Code Helper (GPU)" | grep -v grep
echo ""

echo "5. System GPU Memory Info (if available):"
echo "-------------------------------------"
if command -v system_profiler &> /dev/null; then
    system_profiler SPDisplaysDataType | grep -A 5 "VRAM"
else
    echo "system_profiler not available"
fi
echo ""

echo "==================================="
echo "Diagnosis Complete"
echo "==================================="
