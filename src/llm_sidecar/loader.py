import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from typing import Optional
from datetime import datetime
from peft import AutoPeftModel

# Globals
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
    global hermes_model, hermes_tokenizer
    if hermes_model and hermes_tokenizer:
        return
    hermes_tokenizer = AutoTokenizer.from_pretrained(HERMES_MODEL_PATH, use_fast=True)
    hermes_model = AutoModelForCausalLM.from_pretrained(
        HERMES_MODEL_PATH, device_map="auto", trust_remote_code=True
    )

def load_phi3_model():
    """
    1) Always load the base Phi-3 tokenizer & ONNX model.
    2) Then, if an adapter directory exists, wrap with that adapter.
    """
    global phi3_model, phi3_tokenizer, phi3_adapter_date
    if phi3_model and phi3_tokenizer:
        return

    # 1) Base load
    phi3_tokenizer = AutoTokenizer.from_pretrained(PHI3_TOKENIZER_PATH, use_fast=True)
    phi3_model = ORTModelForCausalLM.from_pretrained(
        MICRO_LLM_MODEL_PATH
    )

    # 2) Adapter logic
    adapter_base = os.path.join(MICRO_LLM_MODEL_PARENT_DIR, "adapters")
    if os.path.isdir(adapter_base):
        # find latest date-formatted subfolder
        valid = []
        for d in os.listdir(adapter_base):
            path = os.path.join(adapter_base, d)
            cfg = os.path.join(path, "adapter_config.json")
            if os.path.isdir(path) and os.path.exists(cfg):
                try:
                    date = datetime.strptime(d, "%Y%m%d")
                    valid.append((date, path))
                except ValueError:
                    pass
        if valid:
            _, latest_path = sorted(valid)[-1]
            # apply adapter
            phi3_model = AutoPeftModel.from_pretrained(phi3_model, latest_path)
            phi3_adapter_date = datetime.strptime(
                os.path.basename(latest_path), "%Y%m%d"
            ).strftime("%Y-%m-%d")
        else:
            phi3_adapter_date = None
    else:
        phi3_adapter_date = None

def get_hermes_model_and_tokenizer():
    return hermes_model, hermes_tokenizer

def get_phi3_model_and_tokenizer():
    return phi3_model, phi3_tokenizer
