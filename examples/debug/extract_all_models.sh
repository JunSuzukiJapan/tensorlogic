#!/bin/bash
# Extract reference values for all available models
# Creates a reference database for future comparisons

set -e

MODELS_DIR="$HOME/.llm/models"
OUTPUT_DIR="candle_reference_data"

mkdir -p "$OUTPUT_DIR"

echo "=================================="
echo "Multi-Model Reference Extractor"
echo "=================================="
echo ""

# Function to extract reference for a single model
extract_model() {
    local model_file="$1"
    local model_name=$(basename "$model_file" .gguf)
    local output_file="$OUTPUT_DIR/${model_name}_reference.md"

    echo "Processing: $model_name"
    echo "  Input:  $model_file"
    echo "  Output: $output_file"

    # Run extractor with model path as argument
    if [ -f "./target/release/extract_candle_values" ]; then
        ./target/release/extract_candle_values "$model_file" > "$output_file" 2>&1
        echo "  ✅ Complete"
    else
        echo "  ⚠️  Extractor not built - run: cargo build --release --bin extract_candle_values"
    fi
    echo ""
}

# Find all GGUF models
if [ -d "$MODELS_DIR" ]; then
    echo "Searching for GGUF models in: $MODELS_DIR"
    echo ""

    # Common model patterns
    MODELS=(
        "tinyllama-1.1b-chat-q4_0.gguf"
        "tinyllama-1.1b-chat-q8_0.gguf"
        "tinyllama-1.1b-chat-f16.gguf"
        "llama-2-7b-chat.Q4_0.gguf"
        "llama-2-7b-chat.Q8_0.gguf"
        "mistral-7b-instruct-v0.2.Q4_0.gguf"
        "phi-2.Q4_0.gguf"
    )

    for model in "${MODELS[@]}"; do
        model_path="$MODELS_DIR/$model"
        if [ -f "$model_path" ]; then
            extract_model "$model_path"
        else
            echo "⏭️  Skipping $model (not found)"
        fi
    done
else
    echo "⚠️  Models directory not found: $MODELS_DIR"
    echo "Please set MODELS_DIR to your model directory"
    exit 1
fi

echo "=================================="
echo "✅ Reference extraction complete"
echo "=================================="
echo ""
echo "Reference data saved in: $OUTPUT_DIR/"
ls -lh "$OUTPUT_DIR/" 2>/dev/null || echo "(No files generated)"
