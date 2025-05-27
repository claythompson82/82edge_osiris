# Use a suitable base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
ARG HF_TOKEN

# Set environment variables to prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python, pip, and git
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install necessary dependencies for running a GPTQ model
# Pinning versions for compatibility, ensure these are suitable for Hermes-3 8B
RUN pip3 install --no-cache-dir \
    auto-gptq==0.7.1 \
    transformers==4.38.2 \
    optimum==1.19.0 \
    torch==2.1.0 \
    fastapi==0.109.2 \
    uvicorn==0.27.1 \
    sentencepiece==0.1.99 \
    accelerate==0.27.2

# Clone the Hermes-3 8B GPTQ model files
# Using a specific revision known to work with auto-gptq 0.7.1
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

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
