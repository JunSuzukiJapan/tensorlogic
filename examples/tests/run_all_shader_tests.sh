#!/bin/bash

# Comprehensive shader test runner
# Runs all Metal shader operation tests sequentially

echo "======================================"
echo "  TensorLogic Shader Test Suite"
echo "======================================"
echo ""

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

TESTS=(
    "test_arithmetic_ops.tl"
    "test_activation_ops.tl"
    "test_math_ops.tl"
    "test_reduction_ops.tl"
    "test_shape_ops.tl"
    "test_broadcast_ops.tl"
    "test_matmul_kernel.tl"
)

PASSED=0
FAILED=0

for test in "${TESTS[@]}"; do
    echo "Running: $test"
    echo "--------------------------------------"

    if timeout 60 ./target/release/tl run examples/tests/$test; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        ((FAILED++))
    fi
    echo ""
    echo ""
done

echo "======================================"
echo "  Test Suite Summary"
echo "======================================"
echo "Total tests:  $((PASSED + FAILED))"
echo -e "${GREEN}Passed:       $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed:       $FAILED${NC}"
else
    echo "Failed:       $FAILED"
fi
echo "======================================"

if [ $FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}All shader tests passed! 🎉${NC}"
    exit 0
else
    echo ""
    echo -e "${RED}Some tests failed. Please review the output above.${NC}"
    exit 1
fi
