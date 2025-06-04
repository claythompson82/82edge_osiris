#!/bin/bash
set -e

DATE=$(date +%Y%m%d)

echo "Pulling test adapter image..."
docker pull ghcr.io/osiris/test-adapter:latest

CID=$(docker create ghcr.io/osiris/test-adapter:latest)
mkdir -p ./test-adapter

echo "Copying adapter from container..."
docker cp "${CID}:/adapter" ./test-adapter

docker rm "$CID"

SIDECAR_ID=$(docker compose -f docker/compose.yaml ps -q llm-sidecar)

echo "Copying adapter into llm-sidecar container..."
docker cp ./test-adapter "${SIDECAR_ID}:/app/models/phi3/adapters/${DATE}"

curl -s -X POST http://localhost:8000/adapters/swap
sleep 5

EXPECTED=$(echo "$DATE" | sed 's/\(....\)\(..\)\(..\)/\1-\2-\3/')
HEALTH=$(curl -s http://localhost:8000/health)

echo "$HEALTH"
ADAPTER_DATE=$(echo "$HEALTH" | jq -r '.phi3_adapter_date')

if [ "$ADAPTER_DATE" != "$EXPECTED" ]; then
  echo "Adapter date mismatch: expected $EXPECTED got $ADAPTER_DATE"
  exit 1
fi

echo "Adapter swap smoke test passed."
