# LLM Sidecar with Hermes-3 8B GPTQ & Phi-3-mini

[![E2E Orchestrator Smoke Test](https://github.com/OWNER/REPO/actions/workflows/e2e-orchestrator.yaml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/e2e-orchestrator.yaml)

## Project Overview

This project provides a Dockerised side-car service for running two local Large-Language-Models:

* **Hermes-Trismegistus-III-8B-GPTQ** – the main “conscious-component” model.
* **Phi-3-mini-4k-instruct (int8 ONNX)** – a lightweight JSON-planning head that is continuously improved by a nightly QLoRA / DPO feedback loop.

The container exposes a small REST API (FastAPI) and ships with a GPU-VRAM watchdog so your 4070 Super never OOM-crashes.

---

## Architecture v3

**(Placeholder for new architecture diagram - `docs/arch_v3.png` needs to be created/updated manually)**

A detailed description of the v3 architecture, including components and interaction flows, can be found here:
[Link to diagram description](docs/arch_v3_description.txt) (Note: This file also needs to be updated for v3)

---

## Prerequisites

| Requirement                 | Purpose                                                                     |
| --------------------------- | --------------------------------------------------------------------------- |
| **Docker + Docker Compose** | Container orchestration                                                     |
| **NVIDIA driver w/ CUDA**   | GPU inference inside the container                                          |
| **≈ 10 GB free VRAM**       | 6-7 GB for Hermes-3 GPTQ, ≈ 2 GB for Phi-3 int8, plus head-room for tensors |

---

## Quick-start (Hermes-3 side-car)

```bash
# 1. clone & cd
git clone https://github.com/your/fork.git # Replace with your fork/repo URL
cd your-repo-name # Replace with your directory name

# 2. copy env file and tweak if you like
cp .env.template .env      # default Hermes ctx = 6144

# 3. spin it up
docker compose -f docker/compose.yaml up -d llm-sidecar redis
```
*Note: `redis` service is added to the compose command if you intend to use features relying on it, like event bus or TTS streaming.*

FastAPI is now live on **[http://localhost:8000](http://localhost:8000)**.

---

## Quick-start (Full Simulator & Orchestrator Loop)

This mode runs the market data simulator, the policy orchestrator, the LLM sidecar, and Redis. The orchestrator listens to ticks from the simulator and triggers proposal workflows.

**Prerequisites:**
* Ensure you have a market data CSV file (e.g., `data/spy_1h.csv`). You might need to create this directory and file.
* A `docker-compose.yaml` (or a dedicated one like `docker-compose.full.yaml`) that defines services for `sim`, `orchestrator`, `llm-sidecar`, and `redis`.

**Example `docker-compose.full.yaml` (illustrative - adapt as needed):**
```yaml
version: '3.8'
services:
  redis:
    image: redis:alpine
    ports:
      - "6379:6379"
    # command: redis-server --save "" --appendonly no # Optional: disable persistence for dev

  llm-sidecar:
    build:
      context: .
      dockerfile: Dockerfile # Assuming your main Dockerfile for llm-sidecar
    env_file: .env
    ports:
      - "8000:8000"
    volumes:
      - ./models:/models # If downloading models outside image
      - ./phi3_feedback_log.jsonl:/app/phi3_feedback_log.jsonl
      - ./phi3_feedback_data.jsonl:/app/phi3_feedback_data.jsonl
    deploy: # Requires Docker Swarm mode or `docker compose --compatibility`
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    depends_on:
      - redis

  sim:
    build:
      context: .
      dockerfile: Dockerfile.sim # Create a separate Dockerfile for the simulator
    # Example Dockerfile.sim:
    # FROM python:3.9-slim
    # WORKDIR /app
    # COPY requirements.txt .
    # RUN pip install -r requirements.txt
    # COPY sim /app/sim
    # COPY llm_sidecar /app/llm_sidecar # For EventBus dependency
    # ENTRYPOINT ["python", "-m", "sim.engine"]
    volumes:
      - ./data:/app/data # Mount your CSV data directory
    command: >
      python -m sim.engine 
      /app/data/spy_1h.csv 
      --redis_url redis://redis:6379/0 
      --speed 10x 
      --from_date 2023-10-01
    depends_on:
      - redis
    # network_mode: host # Or ensure they are on the same docker network

  orchestrator:
    build:
      context: .
      dockerfile: Dockerfile.orchestrator # Create a separate Dockerfile for the orchestrator
    # Example Dockerfile.orchestrator:
    # FROM python:3.9-slim
    # WORKDIR /app
    # COPY requirements.txt .
    # RUN pip install -r requirements.txt
    # COPY osiris_policy /app/osiris_policy
    # COPY llm_sidecar /app/llm_sidecar # For EventBus and DB dependencies
    # ENTRYPOINT ["python", "-m", "osiris_policy.orchestrator"]
    env_file: .env # If it needs any env vars like API keys (not currently the case)
    command: >
      python -m osiris_policy.orchestrator 
      --redis_url redis://redis:6379/0 
      --market_channel market.ticks 
      --ticks_per_proposal 5
    depends_on:
      - redis
      - llm-sidecar # Orchestrator calls the sidecar's API
    # network_mode: host

  # vram-watchdog: (if used with the full loop)
  #   build:
  #     context: .
  #     dockerfile: Dockerfile.watchdog
  #   depends_on:
  #     - llm-sidecar
  #   environment:
  #     - CONTAINER_NAME_TO_RESTART=llm-sidecar 
  #   volumes:
  #     - /var/run/docker.sock:/var/run/docker.sock
  #   privileged: true # Required for Docker socket access
```

**To run (assuming you have the compose file, e.g., `docker-compose.full.yaml`):**
```bash
# 1. Ensure Dockerfiles for sim and orchestrator exist if not using a single large image.
# 2. Prepare your data/spy_1h.csv (or other data file).
# 3. docker compose -f docker-compose.full.yaml up -d sim orchestrator llm-sidecar redis
# (Remove -d to see logs, or use `docker compose -f ... logs -f <service_name>`)
```
This setup will start the simulator feeding data into Redis, and the orchestrator processing these ticks to generate and evaluate trade proposals using the LLM sidecar.

---

## VRAM Watchdog

`vram_watchdog.sh` runs in a second container (`vram-watchdog`) and:

* polls `nvidia-smi` every 60 s;
* if usage > **10.5 GB** for 3 consecutive reads it restarts `llm-sidecar`.

---

## API Endpoints

*All endpoints live in `server.py`.*

### **POST `/generate/`** (Unified)

Generate with **either** Hermes or Phi-3.

| Field        | Type   | Default    | Notes                  |
| ------------ | ------ | ---------- | ---------------------- |
| `prompt`     | string | –          | Required               |
| `max_length` | int    | 256        | Optional               |
| `model_id`   | string | `"hermes"` | `"hermes"` or `"phi3"` |

*If `model_id` is omitted Hermes is used.*
*Phi-3 responses respect a strict JSON schema and, once the nightly QLoRA/DPO pipeline is running, will embed the latest fine-tuned weights.*

#### Response (Hermes success)

```json
{
  "generated_text": "The model's response..."
}
```

#### Response (Phi-3 success — example)

```json
{
  "ticker": "XYZ",
  "action": "adjust",
  "side": "LONG",
  "new_stop_pct": 5.0,
  "new_target_pct": 10.0,
  "confidence": 0.85,
  "rationale": "Price action suggests upward trend."
}
```

---

### **Model-specific shortcuts**

* **POST `/generate/hermes/`** – vanilla Hermes text.
* **POST `/generate/phi3/`** – structured JSON via Phi-3 (same schema).

---

### **POST `/propose_trade_adjustments/`**

1. Phi-3 writes a JSON trade proposal.
2. Hermes critiques it in plain English.
3. Both objects are logged for the nightly feedback loop.

---

### **POST `/feedback/phi3/`**

Submit corrections or ratings for a previous Phi-3 proposal.

```json
{
  "transaction_id": "uuid-from-log",
  "feedback_type": "correction",
  "feedback_content": { "comment": "Stop too tight." },
  "corrected_proposal": { /* full corrected JSON … */ }
}
```

---

### **GET `/health`**

Returns:

```json
{
  "status": "ok",                // "partial_error" or "error"
  "hermes_loaded": true,
  "phi3_loaded": true,
  "phi3_model_file_exists": true,
  "device": "cuda"
}
```

---

## Voice Output (Text-to-Speech)

The system can generate voice output for certain events, primarily for Hermes assessments.

### 1. Direct API Call (`/speak`)

You can directly request TTS synthesis for any text:

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world, this is a test.", "exaggeration": 0.5}' \
  http://localhost:8000/speak \
  --output speech_output.wav
```
This will save the WAV audio data to `speech_output.wav`.
Optional fields in the JSON payload:
- `exaggeration` (float, default 0.5): Controls the expressiveness.
- `ref_wav_b64` (string, optional): Base64 encoded WAV data to be used as a voice reference (voice cloning).

### 2. Redis Channel (`audio.bytes`)

When TTS is triggered by an internal process (like the orchestrator after a Hermes assessment), the raw WAV audio data (base64 encoded) is published to the Redis channel named `audio.bytes`.
You can subscribe to this channel using any Redis client to receive the audio data programmatically.

### 3. Web Audio Console

A simple web-based audio console is available to listen to the audio streamed from the `audio.bytes` Redis channel in real-time.

*   **URL**: `static/audio_console.html`
    *   If you are running the LLM sidecar locally, you can typically access it at [http://localhost:8000/static/audio_console.html](http://localhost:8000/static/audio_console.html) (assuming the `static` directory is served by the FastAPI app, which might require adding `StaticFiles` mount to `server.py`).
    *   Alternatively, open the `static/audio_console.html` file directly in your browser from your local file system if the FastAPI server isn't configured to serve static files from that path.

The console uses Server-Sent Events (SSE) to connect to the `/stream/audio` endpoint on the LLM sidecar, which streams the audio data from the Redis channel.

---

## Phi-3 Feedback Loop (Nightly)

1. **Logging** – every `/propose_trade_adjustments/` call appends to `/app/phi3_feedback_log.jsonl`.
2. **User feedback** – collected via `/feedback/phi3/`, stored in `/app/phi3_feedback_data.jsonl`.
3. **QLoRA / DPO fine-tune** – a nightly job trains on the feedback set and produces a new int8 ONNX; the next morning `llm-sidecar` hot-loads it.
4. **Prompt-augmentation** – latest high-quality corrections are few-shot-prepended to Phi-3 prompts during the day.

---

## CI / CD

- **`.github/workflows/ci.yaml`**: Builds the main image, boots the container, and hits `/health` on every push & PR to `main`.
- **`.github/workflows/ci-audio-smoke.yaml`**: Performs a smoke test on the TTS audio generation by calling `/speak` and verifying the WAV output.
- **`.github/workflows/e2e-orchestrator.yaml`**: (Assumed from badge) Runs an end-to-end test of the orchestrator loop.

### Chaos Testing

To ensure system resilience and validate graceful restart capabilities, a chaos testing mode and an automated CI workflow have been implemented.

**`CHAOS_MODE` Environment Variable:**

Setting the environment variable `CHAOS_MODE=1` activates the chaos script (`scripts/chaos_restarts.py`). When enabled, this script will periodically and randomly restart the `llm-sidecar` and `orchestrator` Docker services. This helps simulate unstable conditions and verify that the system can recover and continue processing without data loss.

**Chaos Smoke Test CI Workflow:**

The `chaos-smoke.yaml` workflow (located in `.github/workflows/`) automates chaos testing. This CI job performs the following:
1. Sets `CHAOS_MODE=1`.
2. Starts all services using Docker Compose (including Redis, LLM Sidecar, and Orchestrator).
3. Runs the `chaos_restarts.py` script in the background to randomly restart services.
4. Simulates market tick data being sent to the system using `scripts/publish_ticks.py`.
5. After a period of chaotic activity and data simulation, it verifies that critical data (specifically, "advice" entries in LanceDB) has been successfully processed and logged using `scripts/verify_advice.py`.
6. Cleans up all services by stopping and removing Docker containers and volumes.

This workflow helps to continuously ensure the robustness of the system against unexpected service interruptions.

---

Patience Profits!
