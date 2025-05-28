# LLM Sidecar with Hermes-3 8B GPTQ & Phi-3-mini

[![E2E Orchestrator Smoke Test](https://github.com/OWNER/REPO/actions/workflows/e2e-orchestrator.yaml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/e2e-orchestrator.yaml)

## Project Overview

This project provides a Dockerised side-car service for running two local Large-Language-Models:

* **Hermes-Trismegistus-III-8B-GPTQ** – the main “conscious-component” model.
* **Phi-3-mini-4k-instruct (int8 ONNX)** – a lightweight JSON-planning head that is continuously improved by a nightly QLoRA / DPO feedback loop.

The container exposes a small REST API (FastAPI) and ships with a GPU-VRAM watchdog so your 4070 Super never OOM-crashes.

---

## Architecture v2

(Placeholder for new architecture diagram - `docs/arch_v2.png` will be here)

A detailed description of the v2 architecture, including components and interaction flows, can be found here:
[Link to diagram description](docs/arch_v2_description.txt)

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
git clone https://github.com/your/fork.git
cd 82edge_osiris

# 2. copy env file and tweak if you like
cp .env.template .env      # default Hermes ctx = 6144

# 3. spin it up
docker compose -f docker/compose.yaml up -d llm-sidecar
```

FastAPI is now live on **[http://localhost:8000](http://localhost:8000)**.

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

## Phi-3 Feedback Loop (Nightly)

1. **Logging** – every `/propose_trade_adjustments/` call appends to `/app/phi3_feedback_log.jsonl`.
2. **User feedback** – collected via `/feedback/phi3/`, stored in `/app/phi3_feedback_data.jsonl`.
3. **QLoRA / DPO fine-tune** – a nightly job trains on the feedback set and produces a new int8 ONNX; the next morning `llm-sidecar` hot-loads it.
4. **Prompt-augmentation** – latest high-quality corrections are few-shot-prepended to Phi-3 prompts during the day.

---

## CI

`.github/workflows/ci.yaml` builds the image, boots the container, and hits `/health` on every push & PR to `main`.

Patience Profits!
