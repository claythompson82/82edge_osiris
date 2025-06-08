# ---- Builder Stage ----
FROM python:3.10-slim as builder
WORKDIR /app

# Set build arguments for PyTorch configuration
ARG PYTORCH_TARGET=cu121
ENV PYTORCH_INDEX_URL=https://download.pytorch.org/whl/${PYTORCH_TARGET}

# Install pip-tools and compile requirements inside the builder
RUN pip install --no-cache-dir pip-tools
COPY requirements.in .
RUN pip-compile requirements.in -o requirements.txt --extra-index-url ${PYTORCH_INDEX_URL} --resolver=backtracking

# Create a virtual environment and install dependencies
ENV VENV_DIR=/opt/venv
RUN python3 -m venv ${VENV_DIR}
ENV PATH="${VENV_DIR}/bin:$PATH"
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url ${PYTORCH_INDEX_URL}

# ---- Runtime Stage ----
FROM python:3.10-slim as runtime
WORKDIR /app

# Copy virtual environment and application code from builder stage
COPY --from=builder /opt/venv /opt/venv
COPY ./src/ /app/src/

# Set environment variables correctly
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONPATH="/app/src"

# Expose port if necessary (e.g., for the server)
EXPOSE 8000

# Set a default command or entrypoint
# CMD ["python", "/app/src/osiris/main.py"]
