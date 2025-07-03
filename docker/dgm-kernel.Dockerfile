FROM python:3.12-slim

ARG REQUIREMENTS=requirements-dgm-runtime.txt

WORKDIR /app

# Install runtime dependencies via build-arg to leverage cache
COPY ${REQUIREMENTS} /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Copy source code
COPY ./src /app/src

ENV PYTHONPATH=/app/src

# Default command runs the kernel once
CMD ["python", "-m", "dgm_kernel", "--once"]
