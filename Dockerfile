# Use a suitable base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ARG HF_TOKEN

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, and git
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt /tmp/requirements.txt
COPY constraints.txt /tmp/constraints.txt

# Install Python dependencies from requirements.txt and constraints.txt
RUN pip install --upgrade -r /tmp/requirements.txt -c /tmp/constraints.txt

# Clone the Hermes-3 8B GPTQ model files
# Using a specific revision known to work with auto-gptq 0.7.1
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

# Download Chatterbox model
RUN mkdir -p /models/tts/chatterbox && \
    huggingface-cli download ResembleAI/Chatterbox --local-dir /models/tts/chatterbox --local-dir-use-symlinks False

# Set up the working directory
WORKDIR /app

# Create a simple server using FastAPI
COPY ./server.py /app/server.py

# Expose the necessary port for the API
EXPOSE 8000

# Copy VRAM Watchdog script into container and make it executable
COPY ./scripts/vram_watchdog.sh /usr/local/bin/vram_watchdog.sh
RUN chmod +x /usr/local/bin/vram_watchdog.sh

# Command to run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
