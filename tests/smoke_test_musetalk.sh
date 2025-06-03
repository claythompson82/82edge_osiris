#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Create a unique temporary directory
TMP_DIR=$(mktemp -d -t musetalk_smoke_test.XXXXXX)

# Function to clean up the temporary directory on script exit
cleanup() {
  echo "Cleaning up temporary directory: $TMP_DIR"
  rm -rf "$TMP_DIR"
}

# Register the cleanup function to be called on EXIT signal
trap cleanup EXIT

# Check if AVATAR environment variable is set and points to an existing file
if [ -z "$AVATAR" ]; then
  echo "Error: AVATAR environment variable is not set."
  exit 1
elif [ ! -f "$AVATAR" ]; then
  echo "Error: AVATAR file '$AVATAR' does not exist."
  exit 1
else
  echo "AVATAR variable is set to '$AVATAR' and file exists."
fi

# Define the destination path for the copied avatar
DEST_AVATAR_PATH="$TMP_DIR/copied_avatar.png"

# Perform the copy operation
echo "Copying '$AVATAR' to '$DEST_AVATAR_PATH'..."
cp "$AVATAR" "$DEST_AVATAR_PATH"

# Verify that the copied file exists in the temporary directory
if [ -f "$DEST_AVATAR_PATH" ]; then
  echo "Successfully copied avatar to '$DEST_AVATAR_PATH'."
else
  echo "Error: Failed to copy avatar to '$DEST_AVATAR_PATH'."
  exit 1
fi

echo "MuseTalk smoke test completed successfully."

exit 0
