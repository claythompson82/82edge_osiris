# Use a suitable base image
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

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
RUN git clone https://huggingface.co/TheBloke/Hermes-Trismegistus-III-8B-GPTQ /app/hermes-model --branch gptq-8bit-128g-actorder_True --depth 1

# Set up the working directory
WORKDIR /app

# Create a simple server using FastAPI
COPY ./server.py /app/server.py

# Expose the necessary port for the API
EXPOSE 8000

# Command to run the server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
