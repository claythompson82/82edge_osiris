# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04 AS builder
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y python3 python3-pip git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install Python dependencies first to leverage Docker cache
COPY constraints.txt requirements.txt ./
RUN pip3 install --no-cache-dir --prefix=/install -r requirements.txt -c constraints.txt

# Copy application source
COPY llm_sidecar ./llm_sidecar
COPY common ./common
COPY scripts/fetch_phi3.sh ./scripts/

RUN chmod +x ./scripts/fetch_phi3.sh && ./scripts/fetch_phi3.sh && \
    if [ -f ./models/llm_micro/phi-3-mini-4k-instruct-AWQ-4bit.onnx ]; then \
        mv ./models/llm_micro/phi-3-mini-4k-instruct-AWQ-4bit.onnx ./models/llm_micro/phi3.onnx; \
    fi || true

RUN git clone https://huggingface.co/TheBloke/Hermes-Trismegistus-III-8B-GPTQ hermes-model --branch gptq-8bit-128g-actorder_True --depth 1


FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages and application code
COPY --from=builder /install/ /usr/local/
COPY --from=builder /build/llm_sidecar ./llm_sidecar
COPY --from=builder /build/common ./common
COPY --from=builder /build/hermes-model ./hermes-model
COPY --from=builder /build/models ./models
COPY docker/server.py ./server.py

EXPOSE 8000
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
