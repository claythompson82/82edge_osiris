FROM python:3.12-slim

WORKDIR /app

# Install minimal runtime dependencies
RUN pip install --no-cache-dir redis==6.2.0

# Copy source code
COPY ./src /app/src

ENV PYTHONPATH=/app/src

CMD ["python", "-m", "dgm_kernel"]
