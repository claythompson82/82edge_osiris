#!/bin/bash

# Model details
MODEL_NAME="microsoft/Phi-3-mini-4k-instruct-onnx"
ONNX_SUBDIR="cuda/cuda-int4-rtn-block-32"
ONNX_TEMP_DIR_NAME="temp_onnx_files"
EXPECTED_ONNX_FILENAME="phi3-mini-4k-instruct-cuda-int4-rtn-block-32.onnx"
EXPECTED_ONNX_DATA_FILENAME="phi3-mini-4k-instruct-cuda-int4-rtn-block-32.onnx.data"
FINAL_ONNX_FILENAME="phi3.onnx"
FINAL_ONNX_DATA_FILENAME="phi3.onnx.data"
GGUF_FALLBACK_MODEL_NAME="microsoft/phi-3-mini-4k-instruct" # GGUF is in the base model repo
GGUF_FALLBACK_FILENAME="phi-3-mini-4k-instruct.Q8_0.gguf"
LOCAL_DIR="models/llm_micro"

# Create the local directory if it doesn't exist
mkdir -p "$LOCAL_DIR"

# Define the temporary directory for ONNX files
ONNX_TEMP_DIR="$LOCAL_DIR/$ONNX_TEMP_DIR_NAME"

echo "Attempting to download ONNX model: $MODEL_NAME (sub-directory: $ONNX_SUBDIR)"
# Create a temporary directory for the download
mkdir -p "$ONNX_TEMP_DIR"

huggingface-cli download "$MODEL_NAME" "$ONNX_SUBDIR" \
  --include "*" \
  --exclude "" \
  --repo-type model \
  --local-dir "$ONNX_TEMP_DIR" \
  --local-dir-use-symlinks False

# Check if ONNX download was successful
if [ $? -eq 0 ]; then
  echo "ONNX model files from $ONNX_SUBDIR downloaded successfully to $ONNX_TEMP_DIR."

  # Move files from the subdirectory (if they are in a nested folder, adjust path)
  # Assuming the files are directly in $ONNX_TEMP_DIR/$ONNX_SUBDIR after download
  SOURCE_FILES_PATH="$ONNX_TEMP_DIR/$ONNX_SUBDIR"
  
  # Check if the source files path exists
  if [ ! -d "$SOURCE_FILES_PATH" ]; then
    echo "Error: Expected source directory $SOURCE_FILES_PATH not found after download."
    # Attempt to list contents of ONNX_TEMP_DIR for debugging
    echo "Contents of $ONNX_TEMP_DIR:"
    ls -R "$ONNX_TEMP_DIR"
    rm -rf "$ONNX_TEMP_DIR" # Clean up temp directory
    exit 1 # Exit with an error code
  fi

  echo "Moving files from $SOURCE_FILES_PATH to $LOCAL_DIR..."
  mv "$SOURCE_FILES_PATH"/* "$LOCAL_DIR/"
  if [ $? -ne 0 ]; then
    echo "Error moving files from $SOURCE_FILES_PATH to $LOCAL_DIR."
    rm -rf "$ONNX_TEMP_DIR" # Clean up temp directory
    exit 1
  fi

  # Rename the ONNX model file
  if [ -f "$LOCAL_DIR/$EXPECTED_ONNX_FILENAME" ]; then
    echo "Renaming $LOCAL_DIR/$EXPECTED_ONNX_FILENAME to $LOCAL_DIR/$FINAL_ONNX_FILENAME..."
    mv "$LOCAL_DIR/$EXPECTED_ONNX_FILENAME" "$LOCAL_DIR/$FINAL_ONNX_FILENAME"
    if [ $? -ne 0 ]; then
      echo "Error renaming $EXPECTED_ONNX_FILENAME to $FINAL_ONNX_FILENAME."
      rm -rf "$ONNX_TEMP_DIR" # Clean up temp directory
      exit 1
    fi
  else
    echo "Error: Expected ONNX file $LOCAL_DIR/$EXPECTED_ONNX_FILENAME not found for renaming."
    rm -rf "$ONNX_TEMP_DIR" # Clean up temp directory
    exit 1
  fi

  # Rename the ONNX data file if it exists
  if [ -f "$LOCAL_DIR/$EXPECTED_ONNX_DATA_FILENAME" ]; then
    echo "Renaming $LOCAL_DIR/$EXPECTED_ONNX_DATA_FILENAME to $LOCAL_DIR/$FINAL_ONNX_DATA_FILENAME..."
    mv "$LOCAL_DIR/$EXPECTED_ONNX_DATA_FILENAME" "$LOCAL_DIR/$FINAL_ONNX_DATA_FILENAME"
    if [ $? -ne 0 ]; then
      echo "Error renaming $EXPECTED_ONNX_DATA_FILENAME to $FINAL_ONNX_DATA_FILENAME."
      # No need to exit if this fails, as .data file might not always exist or be critical for all setups
    fi
  else
    echo "Info: ONNX data file $LOCAL_DIR/$EXPECTED_ONNX_DATA_FILENAME not found. This might be okay."
  fi
  
  echo "ONNX model processed and renamed successfully."
  
  # Clean up the temporary directory
  echo "Cleaning up temporary directory $ONNX_TEMP_DIR..."
  rm -rf "$ONNX_TEMP_DIR"
  if [ $? -ne 0 ]; then
      echo "Warning: Failed to clean up temporary directory $ONNX_TEMP_DIR."
  fi

else
  echo "Failed to download ONNX model from $MODEL_NAME (sub-directory: $ONNX_SUBDIR)."
  # Clean up the temporary directory in case of download failure
  if [ -d "$ONNX_TEMP_DIR" ]; then
    echo "Cleaning up temporary directory $ONNX_TEMP_DIR..."
    rm -rf "$ONNX_TEMP_DIR"
  fi
  
  echo "Attempting to download GGUF fallback model: $GGUF_FALLBACK_MODEL_NAME ($GGUF_FALLBACK_FILENAME)"
  
  huggingface-cli download "$GGUF_FALLBACK_MODEL_NAME" "$GGUF_FALLBACK_FILENAME" \
    --local-dir "$LOCAL_DIR" \
    --repo-type model \
    --local-dir-use-symlinks False

  # Check if GGUF download was successful
  if [ $? -eq 0 ]; then
    echo "GGUF fallback model ($GGUF_FALLBACK_FILENAME) downloaded successfully to $LOCAL_DIR."
  else
    echo "Failed to download GGUF fallback model ($GGUF_FALLBACK_FILENAME)."
    echo "Both ONNX and GGUF downloads failed."
    exit 1 # Exit with an error code if both downloads fail
  fi
fi

echo "Script finished."
exit 0
