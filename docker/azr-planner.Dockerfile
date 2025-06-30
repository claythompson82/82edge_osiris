FROM python:3.12-slim

WORKDIR /app

# Install build tools if necessary, and then install dependencies
# For now, assuming direct pip install is enough
COPY requirements-azr-runtime.txt .
RUN pip install --no-cache-dir -r requirements-azr-runtime.txt

COPY ./src /app/src

ENV PYTHONPATH=/app/src

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "osiris.server:app", "--host", "0.0.0.0", "--port", "8000"]
