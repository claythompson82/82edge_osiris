# Base image
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Then copy rest of code
COPY . .

EXPOSE 8080

CMD ["uvicorn", "services.azr_planner.main:app", "--host", "0.0.0.0", "--port", "8080"]
