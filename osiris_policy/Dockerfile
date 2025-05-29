# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY ./requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy necessary directories
COPY ./osiris_policy ./osiris_policy
COPY ./llm_sidecar ./llm_sidecar
COPY ./advisor ./advisor

# Command to run the orchestrator
CMD ["python", "-u", "osiris_policy/orchestrator.py"]
