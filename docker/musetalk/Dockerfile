# Base image
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# Set DEBIAN_FRONTEND to noninteractive to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Create and activate a virtual environment
RUN python3.10 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"

# Upgrade pip
RUN pip install --upgrade pip

# Clone the MuseTalk repository
RUN git clone https://github.com/TMElyralab/MuseTalk.git musetalk_code

# Install PyTorch with CUDA 12.1 compatibility
# According to https://pytorch.org/get-started/previous-versions/
# For PyTorch 2.2.2 and CUDA 12.1
RUN pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121

# Install MuseTalk dependencies
RUN pip install -r /app/musetalk_code/requirements.txt

# Install MMLab packages using mim
RUN pip install --no-cache-dir -U openmim
RUN mim install mmengine
RUN mim install "mmcv==2.0.1" -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html
RUN mim install "mmdet==3.1.0"
RUN mim install "mmpose==1.1.0"

# Download MuseTalk model weights
# Ensure the script is executable and then run it
RUN cd /app/musetalk_code && chmod +x download_weights.sh && ./download_weights.sh && cd /app

# Create /pipe directory for input/output
RUN mkdir /pipe

# Copy the run_musetalk.py script (will be created in a later step)
# This line assumes run_musetalk.py will be in the same directory as the Dockerfile when building
COPY run_musetalk.py /app/run_musetalk.py

# Set PYTHONPATH to include the musetalk_code directory
ENV PYTHONPATH="/app/musetalk_code:${PYTHONPATH}"

# Define the command to run MuseTalk
CMD ["python3", "/app/run_musetalk.py", "--wav", "/pipe/in.wav", "--img", "/pipe/avatar.png", "--out", "/pipe/out.mp4"]
