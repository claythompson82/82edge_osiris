#!/bin/bash

# Model details
MODEL_NAME="microsoft/phi-3-mini-4k-instruct"
# IMPORTANT: The AWQ_FILENAME is a placeholder.
# You may need to verify the exact filename for the AWQ ONNX model on Hugging Face.
# Look for .onnx files, prioritizing those with "AWQ" and "4bit" (or similar, e.g., int4) in their names.
# It might be located in a subdirectory like 'onnx_int4_awq', 'onnx/cpu_and_mobile/cpu-int4-awq-block-128', etc.
# For example, if the file is 'onnx/cpu_and_mobile/cpu-int4-awq-block-128/model.onnx',
# then AWQ_FILENAME should be 'onnx/cpu_and_mobile/cpu-int4-awq-block-128/model.onnx'.
AWQ_FILENAME="phi-3-mini-4k-instruct-AWQ-4bit.onnx"
GGUF_FALLBACK_FILENAME="phi-3-mini-4k-instruct.Q8_0.gguf"
LOCAL_DIR="models/llm_micro"

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

echo "Attempting to download AWQ ONNX model: $MODEL_NAME ($AWQ_FILENAME)"
huggingface-cli download "$MODEL_NAME" "$AWQ_FILENAME" \
  --local-dir "$LOCAL_DIR" \
  --repo-type model \
  --local-dir-use-symlinks False

# Check if AWQ download was successful
if [ $? -eq 0 ]; then
  echo "AWQ ONNX model ($AWQ_FILENAME) downloaded successfully to $LOCAL_DIR."
else
  echo "Failed to download AWQ ONNX model ($AWQ_FILENAME)."
  echo "Attempting to download GGUF fallback model: $MODEL_NAME ($GGUF_FALLBACK_FILENAME)"
  
  huggingface-cli download "$MODEL_NAME" "$GGUF_FALLBACK_FILENAME" \
    --local-dir "$LOCAL_DIR" \
    --repo-type model \
    --local-dir-use-symlinks False

  # Check if GGUF download was successful
  if [ $? -eq 0 ]; then
    echo "GGUF fallback model ($GGUF_FALLBACK_FILENAME) downloaded successfully to $LOCAL_DIR."
  else
    echo "Failed to download GGUF fallback model ($GGUF_FALLBACK_FILENAME)."
    echo "Both AWQ ONNX and GGUF downloads failed."
    exit 1 # Exit with an error code if both downloads fail
  fi
fi

echo "Script finished."
exit 0
