import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from optimum.onnxruntime import ORTModelForCausalLM
from typing import Optional
from datetime import datetime
from peft import AutoPeftModel

# Global variables for models and tokenizers
hermes_model = None
hermes_tokenizer = None
phi3_model = None
phi3_tokenizer = None
phi3_adapter_date = None

# Model and Tokenizer Paths
# The MICRO_LLM_MODEL_PATH should point to the directory containing the ONNX model file (e.g., model.onnx)
# or directly to the .onnx file if ORTModelForCausalLM handles it.
# The fetch_phi3.sh script renames the downloaded ONNX model to phi3.onnx in /app/models/llm_micro/
MICRO_LLM_MODEL_PARENT_DIR = (
    "/app/models/llm_micro"  # Directory where phi3.onnx is located
)
MICRO_LLM_MODEL_PATH = os.getenv(
    "MICRO_LLM_MODEL_PATH", os.path.join(MICRO_LLM_MODEL_PARENT_DIR, "phi3.onnx")
)
# ------------------------------------------------------------------
# Allow a one-shot override via PHI3_MODEL_PATH ...
_phi3_override = os.getenv("PHI3_MODEL_PATH")
if _phi3_override and os.path.exists(_phi3_override):
    MICRO_LLM_MODEL_PATH = _phi3_override
    MICRO_LLM_MODEL_PARENT_DIR = os.path.dirname(MICRO_LLM_MODEL_PATH)
    print(f"[Side-car] Using PHI3_MODEL_PATH override: {MICRO_LLM_MODEL_PATH}")
# ------------------------------------------------------------------


PHI3_TOKENIZER_PATH = "microsoft/phi-3-mini-4k-instruct"
HERMES_MODEL_PATH = "/app/hermes-model"  # This is a directory
# ------------------------------------------------------------------
# Allow an override via HERMES_MODEL_PATH env-var so users can
# point to any local folder without editing the code again.
_hermes_override = os.getenv("HERMES_MODEL_PATH")
if _hermes_override and os.path.isdir(_hermes_override):
    HERMES_MODEL_PATH = _hermes_override
    print(f"[Side-car] Using HERMES_MODEL_PATH override: {HERMES_MODEL_PATH}")
# ------------------------------------------------------------------


def load_hermes_model():
    """Load the Hermes GPTQ model and tokenizer."""
    global hermes_model, hermes_tokenizer
    if hermes_model is not None and hermes_tokenizer is not None:
        print("Hermes model and tokenizer already loaded.")
        return

    print(f"Loading Hermes model from: {HERMES_MODEL_PATH}")
    try:
        hermes_tokenizer = AutoTokenizer.from_pretrained(
            HERMES_MODEL_PATH, use_fast=True
        )
        hermes_model = AutoModelForCausalLM.from_pretrained(
            HERMES_MODEL_PATH,
            device_map="auto",  # Automatically select device (CPU/GPU)
            trust_remote_code=True,  # Required for some custom model architectures
        )
        # No explicit .to(device) needed here due to device_map="auto"
        print("Hermes model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading Hermes model or tokenizer: {e}")
        hermes_model = None
        hermes_tokenizer = None


def load_phi3_model():
    """Load the Phi-3 ONNX model and tokenizer."""
    global phi3_model, phi3_tokenizer
    if phi3_model is not None and phi3_tokenizer is not None:
        print("Phi-3 model and tokenizer already loaded.")
        return

    print(f"Loading Phi-3 tokenizer from: {PHI3_TOKENIZER_PATH}")
    print(f"Loading Phi-3 ONNX model from: {MICRO_LLM_MODEL_PATH}")

    # Determine device for ONNX model (provider will handle actual placement on CUDA device)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(
        f"Phi-3 ONNX model will use provider: {'CUDAExecutionProvider' if device == 'cuda' else 'CPUExecutionProvider'}"
    )

    try:
        phi3_tokenizer = AutoTokenizer.from_pretrained(
            PHI3_TOKENIZER_PATH, use_fast=True
        )

        # For ORTModelForCausalLM, from_pretrained expects the directory containing the onnx model and config files.
        # If MICRO_LLM_MODEL_PATH is a direct file path, we should use its directory.
        model_dir_to_load = MICRO_LLM_MODEL_PARENT_DIR

        # Check if the specific ONNX file exists, as a sanity check from the script.
        if not os.path.exists(MICRO_LLM_MODEL_PATH):
        	print(f"Error: ONNX model file not found at {MICRO_LLM_MODEL_PATH}")
                return None, None  # abort early

        if not os.path.isdir(model_dir_to_load):
        	print(f"Warning: Parent dir '{model_dir_to_load}' not found; loading ONNX file directly.")
        	model_dir_to_load = MICRO_LLM_MODEL_PATH

        phi3_model = ORTModelForCausalLM.from_pretrained(
            model_dir_to_load,  # Expects directory containing model.onnx and config.json etc.
            # file_name="phi3.onnx", # Explicitly specify if model_dir_to_load is a directory and ONNX file has a specific name
            provider=(
                "CUDAExecutionProvider" if device == "cuda" else "CPUExecutionProvider"
            ),
            use_io_binding=True if device == "cuda" else False,  # IO binding for GPU
        )
        # Even if we never load an adapter, we always return base model & tokenizer
	return phi3_model, phi3_tokenizer

	# No explicit .to(device) for ORTModelForCausalLM, provider handles it.
        # Inputs to generate will need .to(device)

        # Load PEFT adapter if available
        global phi3_adapter_date  # Ensure we are updating the global variable
        adapter_base_path = "/app/models/phi3/adapters/"
        latest_adapter_path = get_latest_adapter_dir(adapter_base_path)

        if latest_adapter_path:
            print(f"Loading PEFT adapter for Phi-3 from: {latest_adapter_path}")
            try:
                # Assuming phi3_model is the base model loaded by ORTModelForCausalLM
                # and it's compatible with AutoPeftModel.from_pretrained's expected model argument.
                # This might need adjustment if ORTModelForCausalLM's output isn't directly usable.
                # For now, we proceed assuming it is.
                phi3_model = AutoPeftModel.from_pretrained(phi3_model, latest_adapter_path)

                # Extract date string from path and format it
                adapter_dir_name = os.path.basename(
                    latest_adapter_path
                )  # e.g., "20231022"
                try:
                    date_obj = datetime.strptime(adapter_dir_name, "%Y%m%d")
                    phi3_adapter_date = date_obj.strftime("%Y-%m-%d")
                    print(
                        f"Successfully loaded PEFT adapter {latest_adapter_path} with date {phi3_adapter_date}"
                    )
                except ValueError as ve:
                    print(
                        f"Warning: Could not parse date from adapter directory name {adapter_dir_name}: {ve}. Adapter date will not be set."
                    )
                    phi3_adapter_date = None  # Reset if parsing fails

            except Exception as peft_e:
                print(
                    f"Error loading PEFT adapter from {latest_adapter_path}: {peft_e}"
                )
                print("Proceeding with the base Phi-3 model without adapter.")
                phi3_adapter_date = None  # Ensure it's None if adapter loading fails
        else:
            print("No PEFT adapter found for Phi-3, using base model.")
            phi3_adapter_date = None  # Explicitly set to None

        print("Phi-3 ONNX model and tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading Phi-3 ONNX model or tokenizer: {e}")
        phi3_model = None
        phi3_tokenizer = None


def get_hermes_model_and_tokenizer():
    """Return the loaded Hermes model and tokenizer."""
    return hermes_model, hermes_tokenizer


def get_phi3_model_and_tokenizer():
    """Return the loaded Phi-3 model and tokenizer."""
    return phi3_model, phi3_tokenizer


if __name__ == "__main__":
    # For basic testing
    print("Attempting to load Hermes model...")
    load_hermes_model()
    if hermes_model and hermes_tokenizer:
        print("Hermes loaded.")
    else:
        print("Hermes failed to load.")

    print("\nAttempting to load Phi-3 ONNX model...")
    load_phi3_model()
    if phi3_model and phi3_tokenizer:
        print("Phi-3 loaded.")
    else:
        print("Phi-3 failed to load.")


def get_latest_adapter_dir(base_path: str) -> Optional[str]:
    """
    Scan a base path for date-formatted subdirectories (YYYYMMDD)
    containing an 'adapter_config.json' file and return the path
    to the most recent valid adapter directory.

    Args:
        base_path: The base directory to scan (e.g., /models/phi3/adapters/).

    Returns:
        The full path to the latest valid adapter directory, or None if not found.
    """
    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        print(f"Adapter base path {base_path} does not exist or is not a directory.")
        return None

    latest_dir = None
    latest_date = None

    try:
        for dirname in os.listdir(base_path):
            dirpath = os.path.join(base_path, dirname)
            if not os.path.isdir(dirpath):
                continue

            # Check if dirname matches YYYYMMDD format
            try:
                dir_date = datetime.strptime(dirname, "%Y%m%d").date()
            except ValueError:
                # Not a valid date-formatted directory name
                continue

            # Check for adapter_config.json
            adapter_config_path = os.path.join(dirpath, "adapter_config.json")
            if not os.path.exists(adapter_config_path):
                # Missing adapter_config.json
                continue

            # If this is the first valid directory or newer than current latest
            if latest_date is None or dir_date > latest_date:
                latest_date = dir_date
                latest_dir = dirpath

    except OSError as e:
        print(f"Error accessing adapter base path {base_path}: {e}")
        return None

    if latest_dir:
        print(f"Found latest adapter directory: {latest_dir}")
    else:
        print(f"No valid adapter directories found in {base_path}")

    return latest_dir
