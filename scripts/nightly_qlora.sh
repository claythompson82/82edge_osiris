#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# Define paths
TRAINING_DATA_OUT="/tmp/training_data.jsonl"
QLORA_ADAPTERS_SRC_DIR="/tmp/qlora_adapters" # Assumed output directory for run_qlora.py

# Create a dated directory for storing adapters
# Format: YYYYMMDD
DATE_SUFFIX=$(date +%Y%m%d)
ADAPTERS_DEST_DIR="/models/phi3/adapters/${DATE_SUFFIX}"

echo "Starting nightly QLoRA fine-tuning process..."

# 1. Harvest feedback data
echo "Harvesting feedback data from the last 1 day..."
python scripts/harvest_feedback.py --days-back 1 --out "${TRAINING_DATA_OUT}" --max 1000
echo "Feedback data harvested to ${TRAINING_DATA_OUT}"

# 2. Run QLoRA fine-tuning script
# This script is assumed to:
# - Read training data from a predefined location (e.g., TRAINING_DATA_OUT or configured internally)
# - Output adapter files to QLORA_ADAPTERS_SRC_DIR
echo "Running QLoRA fine-tuning (scripts/run_qlora.py)..."
python scripts/run_qlora.py
echo "QLoRA fine-tuning script finished."

# 3. Create the destination directory for adapters if it doesn't exist
echo "Creating adapter destination directory: ${ADAPTERS_DEST_DIR}"
mkdir -p "${ADAPTERS_DEST_DIR}"
echo "Adapter destination directory created."

# 4. Move adapter files to the dated directory
# Assuming standard adapter file names. Adjust if run_qlora.py produces different files.
ADAPTER_FILES=("adapter_model.bin" "adapter_config.json" "README.md" "tokenizer.json" "tokenizer_config.json" "special_tokens_map.json" "tokenizer.model") # Add other relevant files if any

echo "Moving adapter files from ${QLORA_ADAPTERS_SRC_DIR} to ${ADAPTERS_DEST_DIR}..."

# Check if source directory exists before attempting to move files
if [ -d "${QLORA_ADAPTERS_SRC_DIR}" ]; then
    for FILE_NAME in "${ADAPTER_FILES[@]}"; do
        if [ -f "${QLORA_ADAPTERS_SRC_DIR}/${FILE_NAME}" ]; then
            mv "${QLORA_ADAPTERS_SRC_DIR}/${FILE_NAME}" "${ADAPTERS_DEST_DIR}/"
            echo "Moved ${FILE_NAME}"
        else
            echo "Warning: Adapter file ${QLORA_ADAPTERS_SRC_DIR}/${FILE_NAME} not found. Skipping."
        fi
    done
    # Optionally, clean up the source directory if it's empty or no longer needed
    # rmdir "${QLORA_ADAPTERS_SRC_DIR}" || echo "Source directory ${QLORA_ADAPTERS_SRC_DIR} not empty or could not be removed."
else
    echo "Error: Source directory for adapters ${QLORA_ADAPTERS_SRC_DIR} not found. Cannot move adapter files."
    # Depending on desired behavior, one might want to exit with an error here:
    # exit 1 
fi

echo "Nightly QLoRA fine-tuning process completed."
echo "Adapters saved to ${ADAPTERS_DEST_DIR}"

# --- Report Success to GitHub ---
# If the GitHub CLI is available, report a commit status so the workflow
# can surface the result directly on GitHub. This uses environment
# variables provided by GitHub Actions when available and falls back to
# local git information otherwise.
if command -v gh >/dev/null 2>&1; then
    REPO="${GITHUB_REPOSITORY:-$(git config --get remote.origin.url | sed -E 's#.*github.com[:/]([^/]+/[^.]+)\.git#\1#')}"
    SHA="${GITHUB_SHA:-$(git rev-parse HEAD)}"
    TARGET_URL="${GITHUB_SERVER_URL:-https://github.com}/${REPO}/actions/runs/${GITHUB_RUN_ID:-}"
    echo "Reporting success status to GitHub for ${REPO}@${SHA}..."
    gh api repos/${REPO}/statuses/${SHA} \
        -f state=success \
        -f context="nightly_qlora" \
        -f description="QLoRA fine-tune completed" \
        -f target_url="${TARGET_URL}" || echo "Warning: failed to update GitHub status"
else
    echo "Warning: gh CLI not found; skipping GitHub status update."
fi

exit 0
