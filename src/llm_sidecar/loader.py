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
    # Load once
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
    global phi3_model, phi3_tokenizer, phi3_adapter_date

    # 1) Base load (always)
    phi3_tokenizer = AutoTokenizer.from_pretrained(
        PHI3_TOKENIZER_PATH,
        use_fast=True,
        local_files_only=True,
    )
    phi3_model = ORTModelForCausalLM.from_pretrained(
        MICRO_LLM_MODEL_PATH,
        provider=("CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider"),
        use_io_binding=torch.cuda.is_available(),
    )

    # 2) Adapter logic
    adapter_base = os.path.join(MICRO_LLM_MODEL_PARENT_DIR, "adapters")
    if os.path.isdir(adapter_base):
        valid = []
        for d in os.listdir(adapter_base):
            p = os.path.join(adapter_base, d)
            cfg = os.path.join(p, "adapter_config.json")
            if os.path.isdir(p) and os.path.exists(cfg):
                try:
                    dt = datetime.strptime(d, "%Y%m%d")
                    valid.append((dt, p))
                except ValueError:
                    pass
        if valid:
            latest_path = sorted(valid)[-1][1]
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


def get_latest_adapter_dir(base_path: str) -> Optional[str]:
    if not os.path.isdir(base_path):
        return None
    latest_dir = None
    latest_date = None
    for d in os.listdir(base_path):
        p = os.path.join(base_path, d)
        cfg = os.path.join(p, "adapter_config.json")
        if os.path.isdir(p) and os.path.exists(cfg):
            try:
                dt = datetime.strptime(d, "%Y%m%d")
                if latest_date is None or dt.date() > latest_date:
                    latest_date = dt.date()
                    latest_dir = p
            except ValueError:
                pass
    return latest_dir
