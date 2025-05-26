# LLM Roster

This document provides an overview of the Large Language Models (LLMs) used in this project, their configurations, and expected performance characteristics.

## Microsoft Phi-3 Mini (4k Instruct) - Micro LLM

- **Hugging Face ID:** `microsoft/phi-3-mini-4k-instruct`
- **Internal Path (Primary):** `/app/models/llm_micro/phi3.onnx`
- **Internal Path (Fallback):** `/app/models/llm_micro/phi-3-mini-4k-instruct.Q8_0.gguf`
- **Quantization (Primary):** INT4 RTN (via ONNX Runtime)
- **Quantization (Fallback):** Q8_0 GGUF
- **Expected Throughput:** ~150 tokens/second (target for INT4 RTN on RTX 4070 for 50 new tokens)
- **Context Window:** 4096 tokens
- **Primary Use Case:** Low-latency tasks requiring structured JSON output (e.g., initial trade adjustment proposals).
- **Key Features:**
    - JSON schema enforcement using `outlines`.
    - Target latency: <= 8 ms for batch-1 generation (50 new tokens) on RTX 4070.
    - **ONNX Source:** `microsoft/Phi-3-mini-4k-instruct-onnx/cuda/cuda-int4-rtn-block-32/`
- **Configuration ENV:** `MICRO_LLM_MODEL_PATH` (defaults to `/app/models/llm_micro/phi3.onnx`)

---
*Note: Performance metrics like throughput and latency are hardware-dependent and may vary.*
