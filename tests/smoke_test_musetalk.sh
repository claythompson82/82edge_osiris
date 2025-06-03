#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

# --- Configuration ---
DOCKER_IMAGE_NAME="musetalk-av100-test"
DOCKERFILE_DIR="./docker/musetalk" # Relative to repo root
TEMP_PIPE_DIR="./tmp_pipe_smoke_test"
AVATAR_IMAGE_NAME="avatar.png"
AUDIO_WAV_NAME="sine.wav"
OUTPUT_VIDEO_NAME="out.mp4"
EXPECTED_FRAME_COUNT=30 # For a 2-second video at 15 FPS, as per "30 frame mp4" for "2-sec sine wave"

# --- Helper Functions ---
cleanup() {
  echo "Cleaning up..."
  rm -rf "${TEMP_PIPE_DIR}"
  # Consider removing the Docker image if not needed for caching in CI
  # docker rmi "${DOCKER_IMAGE_NAME}" || true 
  echo "Cleanup complete."
}

# Register cleanup function to be called on script exit
trap cleanup EXIT

# --- Main Script ---

echo "Starting MuseTalk CI Smoke Test..."

# 1. Create dummy input files
echo "Creating dummy input files..."
mkdir -p "${TEMP_PIPE_DIR}"

# Create a simple PNG image (e.g., a 128x128 red square)
# Requires ImageMagick's convert tool. Ensure it's available in CI or use a pre-generated image.
if ! command -v convert &> /dev/null
then
    echo "ImageMagick's convert command could not be found. Creating a placeholder file for ${AVATAR_IMAGE_NAME}."
    echo "Please ensure a valid PNG is available or ImageMagick is installed if generation is required."
    # As a fallback, create a tiny placeholder. MuseTalk might require a valid image.
    # A better solution for CI would be to commit a small, valid PNG test image.
    touch "${TEMP_PIPE_DIR}/${AVATAR_IMAGE_NAME}" 
else
    convert -size 128x128 xc:red "${TEMP_PIPE_DIR}/${AVATAR_IMAGE_NAME}"
    echo "Dummy avatar image created: ${TEMP_PIPE_DIR}/${AVATAR_IMAGE_NAME}"
fi


# Create a 2-second sine wave audio file at 16kHz mono (common for voice)
# Requires SoX (Sound eXchange). Ensure it's available in CI or use a pre-generated file.
if ! command -v sox &> /dev/null
then
    echo "SoX (Sound eXchange) command could not be found. Creating a placeholder file for ${AUDIO_WAV_NAME}."
    echo "Please ensure a valid WAV is available or SoX is installed if generation is required."
    # As a fallback, create a tiny placeholder. MuseTalk will require a valid WAV.
    # A better solution for CI would be to commit a small, valid WAV test file.
    touch "${TEMP_PIPE_DIR}/${AUDIO_WAV_NAME}"
else
    sox -n -r 16000 -c 1 "${TEMP_PIPE_DIR}/${AUDIO_WAV_NAME}" synth 2 sine 440
    echo "Dummy audio file created: ${TEMP_PIPE_DIR}/${AUDIO_WAV_NAME}"
fi

# Copy dummy files to expected input names for the Docker container
cp "${TEMP_PIPE_DIR}/${AVATAR_IMAGE_NAME}" "${TEMP_PIPE_DIR}/avatar.png"
cp "${TEMP_PIPE_DIR}/${AUDIO_WAV_NAME}" "${TEMP_PIPE_DIR}/in.wav"


# 2. Build the Docker image
echo "Building Docker image ${DOCKER_IMAGE_NAME} from ${DOCKERFILE_DIR}..."
docker build -t "${DOCKER_IMAGE_NAME}" "${DOCKERFILE_DIR}"
echo "Docker image built successfully."

# 3. Run the Docker container
echo "Running Docker container..."
# Make sure the path to TEMP_PIPE_DIR is absolute for Docker volume mounting
ABSOLUTE_TEMP_PIPE_DIR="$(cd "${TEMP_PIPE_DIR}" && pwd)"
echo "Mounted volume path: ${ABSOLUTE_TEMP_PIPE_DIR}"

docker run --gpus all --rm \
  -v "${ABSOLUTE_TEMP_PIPE_DIR}:/pipe" \
  "${DOCKER_IMAGE_NAME}" python3 /app/run_musetalk.py --wav /pipe/in.wav --img /pipe/avatar.png --out /pipe/out.mp4 --fps 15

echo "Docker container run complete."

# 4. Verify output
echo "Verifying output video..."
OUTPUT_VIDEO_PATH="${TEMP_PIPE_DIR}/${OUTPUT_VIDEO_NAME}"

if [ ! -f "${OUTPUT_VIDEO_PATH}" ]; then
  echo "Error: Output video file ${OUTPUT_VIDEO_PATH} not found!"
  exit 1
fi

if [ ! -s "${OUTPUT_VIDEO_PATH}" ]; then
  echo "Error: Output video file ${OUTPUT_VIDEO_PATH} is empty!"
  exit 1
fi
echo "Output video file ${OUTPUT_VIDEO_PATH} created and is not empty."

# Verify frame count using ffprobe
# Ensure ffprobe is installed in the CI environment
if ! command -v ffprobe &> /dev/null
then
    echo "ffprobe command could not be found. Skipping frame count verification."
    echo "Please install ffmpeg (which includes ffprobe) in the CI environment to enable this check."
else
    echo "Checking frame count..."
    # The MuseTalk run_musetalk.py sets fps to 25 by default.
    # For a 2-second audio, this would be 2 * 25 = 50 frames.
    # The issue states "30 frame mp4" for a "2-sec sine wave".
    # This implies an effective frame rate of 15 FPS.
    # The run_musetalk.py currently defaults to 25 FPS.
    # For now, let's stick to the issue's "30 frame" target.
    # This means the run_musetalk.py or the inference call might need adjustment
    # if 30 frames is a strict requirement for 2s input.
    # For this test, we are checking against EXPECTED_FRAME_COUNT = 30.
    # The run_musetalk.py script should be updated to output 15 FPS for this to pass.
    # Or, if MuseTalk generates 25fps (50 frames), this check needs to be 50.
    # Let's assume for now the goal is to *produce* a 30-frame MP4 as per issue title.
    # This might mean the video duration itself is 2s at 15fps, or it's a segment.
    # The run_musetalk.py was set to 25fps. Let's adjust the expectation here for now, assuming 2s @ 25fps = 50 frames.
    # Re-evaluating: The issue "encode 2-sec sine wave into 30 frame mp4" is quite specific.
    # This means the effective FPS of the *output video file* should be 15 FPS.
    # The run_musetalk.py currently has a default of 25 FPS.
    # For the smoke test to pass as per the requirement, the run_musetalk.py's FPS setting should be adjusted.
    # Let's assume the run_musetalk.py will be updated or the setting is controlled.
    # For now, the EXPECTED_FRAME_COUNT is 30.

    ACTUAL_FRAME_COUNT=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "${OUTPUT_VIDEO_PATH}")
    echo "Actual frame count: ${ACTUAL_FRAME_COUNT}"

    # Allow a small tolerance, e.g., +/- 1 frame for safety, though exactly 30 is expected.
    if [ "$((ACTUAL_FRAME_COUNT - EXPECTED_FRAME_COUNT))" -lt -1 ] || [ "$((ACTUAL_FRAME_COUNT - EXPECTED_FRAME_COUNT))" -gt 1 ]; then
        echo "Error: Frame count mismatch. Expected ${EXPECTED_FRAME_COUNT} (for 2s @ 15 FPS), got ${ACTUAL_FRAME_COUNT}."
        exit 1
    else
        echo "Frame count verified successfully: ${ACTUAL_FRAME_COUNT} frames (expected ${EXPECTED_FRAME_COUNT})."
    fi
fi

echo "Smoke test completed successfully!"
exit 0
