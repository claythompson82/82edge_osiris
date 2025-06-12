import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from typing import Optional
from datetime import datetime
from peft import AutoPeftModel

# Globals for loaded models/tokenizers
hermes_model = None
hermes_tokenizer = None
phi3_model = None
phi3_tokenizer = None
phi3_adapter_date = None

# Paths
MICRO_LLM_MODEL_PARENT_DIR = "/app/models/llm_micro"
MICRO_LLM_MODEL_PATH = os.getenv(
    "MICRO_LLM_MODEL_PATH",
    os.path.join(MICRO_LLM_MODEL_PARENT_DIR, "phi3.onnx"),
)

# One-shot override of the ONNX model
_phi3_override = os.getenv("PHI3_MODEL_PATH")
if _phi3_override and os.path.exists(_phi3_override):
    MICRO_LLM_MODEL_PATH = _phi3_override
    MICRO_LLM_MODEL_PARENT_DIR = os.path.dirname(_phi3_override)
    print(f"[Side-car] PHI3_MODEL_PATH override → {MICRO_LLM_MODEL_PATH}")

PHI3_TOKENIZER_PATH = "microsoft/phi-3-mini-4k-instruct"
HERMES_MODEL_PATH = "/app/hermes-model"
_hermes_override = os.getenv("HERMES_MODEL_PATH")
if _hermes_override and os.path.isdir(_hermes_override):
    HERMES_MODEL_PATH = _hermes_override
    print(f"[Side-car] HERMES_MODEL_PATH override → {HERMES_MODEL_PATH}")


def load_hermes_model():
    """Load the Hermes GPTQ model & tokenizer (once)."""
    global hermes_model, hermes_tokenizer
    if hermes_model is None or hermes_tokenizer is None:
        hermes_tokenizer = AutoTokenizer.from_pretrained(
            HERMES_MODEL_PATH, use_fast=True, local_files_only=True
        )
        hermes_model = AutoModelForCausalLM.from_pretrained(
            HERMES_MODEL_PATH,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True,
        )


def load_phi3_model():
    """
    1) Always load the base Phi-3 tokenizer + ONNX model.
    2) Then, if there's any date-formatted subdir under `…/adapters/`, wrap with
       the newest PEFT adapter.
    """
    global phi3_model, phi3_tokenizer, phi3_adapter_date

    # --- 1) Base load (unconditional) ---
    phi3_tokenizer = AutoTokenizer.from_pretrained(
        PHI3_TOKENIZER_PATH, use_fast=True, local_files_only=True
    )
    phi3_model = ORTModelForCausalLM.from_pretrained(
        MICRO_LLM_MODEL_PATH,
        provider="CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider",
        use_io_binding=torch.cuda.is_available(),
    )

    # --- 2) Adapter logic ---
    adapter_base = os.path.join(MICRO_LLM_MODEL_PARENT_DIR, "adapters")
    latest_adapter = get_latest_adapter_dir(adapter_base)
    if latest_adapter:
        phi3_model = AutoPeftModel.from_pretrained(phi3_model, latest_adapter)
        # Parse YYYYMMDD from folder name
        try:
            dt = datetime.strptime(os.path.basename(latest_adapter), "%Y%m%d")
            phi3_adapter_date = dt.strftime("%Y-%m-%d")
        except ValueError:
            phi3_adapter_date = None
    else:
        phi3_adapter_date = None


def get_hermes_model_and_tokenizer():
    """Return (model, tokenizer) for Hermes."""
    return hermes_model, hermes_tokenizer


def get_phi3_model_and_tokenizer():
    """Return (model, tokenizer) for Phi-3."""
    return phi3_model, phi3_tokenizer


def get_latest_adapter_dir(base_path: str) -> Optional[str]:
    """
    Scan `base_path` for subdirectories named YYYYMMDD and return the newest one,
    or None if there are no date-formatted folders.
    """
    if not os.path.isdir(base_path):
        return None

    latest_date = None
    latest_dir = None

    for d in os.listdir(base_path):
        candidate = os.path.join(base_path, d)
        if not os.path.isdir(candidate):
            continue
        try:
            dt = datetime.strptime(d, "%Y%m%d").date()
        except ValueError:
            continue
        if latest_date is None or dt > latest_date:
            latest_date = dt
            latest_dir = candidate

    return latest_dir
