#!/bin/bash
# Run all GraphAgent examples

set -e

cd "$(dirname "$0")/../../.."
source venv/bin/activate

echo "========================================"
echo "Running All GraphAgent Examples"
echo "========================================"
echo ""

examples=(
    "01_basic"
    "02_conditional_routing"
    "03_cyclic_execution"
    "15_enhanced_routing"
    "04_checkpointing"
    "05_interrupts_basic"
    "06_interrupts_reasoning"
    "07_callbacks"
    "08_rewind"
    "09_parallel_wait_all"
    "10_parallel_wait_any"
    "11_parallel_wait_n"
    "12_parallel_checkpointing"
    "13_parallel_interrupts"
    "14_parallel_rewind"
)

for example in "${examples[@]}"; do
    echo "----------------------------------------"
    echo "Running: $example"
    echo "----------------------------------------"
    python -m "contributing.samples.graph_examples.${example}.agent" 2>&1 | grep -v "UserWarning" || true
    echo ""
done

echo "========================================"
echo "✅ All Examples Complete!"
echo "========================================"
