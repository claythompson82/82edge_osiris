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

# Set up the working directory
WORKDIR /app

# Copy application code
COPY common /app/common
COPY osiris_policy /app/osiris_policy
COPY advisor /app/advisor # Added as osiris_policy/orchestrator.py imports from advisor
COPY llm_sidecar /app/llm_sidecar # Added as osiris_policy/orchestrator.py imports from llm_sidecar for EventBus and db

# Command to run the orchestrator
CMD ["python", "-m", "osiris_policy.orchestrator"]
