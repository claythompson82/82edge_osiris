import time
import os
import torch
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
import statistics

# Constants for the benchmark
# These can be overridden by environment variables for flexibility, e.g., in CI/CD pipelines.

# MODEL_PATH: Path to the ONNX model file.
# This path is typically valid inside the Docker container where the model is downloaded by scripts/fetch_phi3.sh
# and potentially renamed to phi3.onnx in the /app/models/llm_micro directory.
MODEL_PATH = os.getenv("MICRO_LLM_MODEL_PATH", "/app/models/llm_micro/phi3.onnx")

# TOKENIZER_PATH: Path or Hugging Face identifier for the tokenizer.
TOKENIZER_PATH = os.getenv("PHI3_TOKENIZER_PATH", "microsoft/phi-3-mini-4k-instruct")

# LATENCY_TARGET_MS: The target average latency in milliseconds for the Phi-3 ONNX model.
# This target (<= 8.0ms) is specified for an NVIDIA RTX 4070 GPU for generating MAX_NEW_TOKENS.
# Adjust this target based on your specific hardware and performance expectations.
LATENCY_TARGET_MS = float(os.getenv("LATENCY_TARGET_MS", "8.0"))

# NUM_WARMUP_RUNS: Number of initial generation runs to perform before benchmarking.
# These runs help to warm up the GPU and ensure that any JIT compilation or initial setup overhead
# does not affect the benchmark results.
NUM_WARMUP_RUNS = int(os.getenv("NUM_WARMUP_RUNS", "5"))

# NUM_BENCHMARK_RUNS: Number of generation runs to perform for the actual benchmark.
# A higher number of runs provides more stable and reliable latency metrics.
NUM_BENCHMARK_RUNS = int(os.getenv("NUM_BENCHMARK_RUNS", "20"))

# SAMPLE_PROMPT: A sample prompt used for the generation task during the benchmark.
SAMPLE_PROMPT = os.getenv(
    "SAMPLE_PROMPT",
    "Translate the following English sentence to French: 'Hello, how are you?'",
)

# MAX_NEW_TOKENS: The number of new tokens to generate in each benchmark run.
# This directly impacts the generation latency.
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "50"))


def run_benchmark():
    """
    Runs a latency benchmark for the Phi-3 ONNX model.
    Measures the average and P95 latency for generating a fixed number of new tokens.
    The test asserts that the average latency is within the LATENCY_TARGET_MS.
    """
    print(f"Starting Phi-3 ONNX model latency benchmark...")
    print(f"Model Path: {MODEL_PATH}")
    print(f"Tokenizer Path: {TOKENIZER_PATH}")
    print(f"Latency Target: {LATENCY_TARGET_MS} ms (for RTX 4070 or similar)")
    print(f"Warm-up Runs: {NUM_WARMUP_RUNS}")
    print(f"Benchmark Runs: {NUM_BENCHMARK_RUNS}")
    print(f"Max New Tokens: {MAX_NEW_TOKENS}")
    print(f'Sample Prompt: "{SAMPLE_PROMPT}"')

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}.")
        print("Please ensure the model is downloaded and the path is correct.")
        print(
            "This script is typically run in an environment (like a Docker container post-build)"
        )
        print("where the model defined by MODEL_PATH is available.")
        return False  # Indicate failure

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if (
        device == "cpu" and LATENCY_TARGET_MS < 100
    ):  # Arbitrary threshold to warn if target is aggressive for CPU
        print(
            "WARNING: Running on CPU. The default latency target is set for GPU (e.g., RTX 4070)."
        )
        print(
            "Benchmark results on CPU will likely be significantly higher and may not meet the target."
        )

    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        print("Tokenizer loaded.")

        print("Loading ONNX model...")
        # ORTModelForCausalLM.from_pretrained expects the directory containing the .onnx file and potentially other config files.
        model_dir = os.path.dirname(MODEL_PATH)
        if (
            not model_dir
        ):  # Handle case where MODEL_PATH might be just "model.onnx" (less likely for full path)
            model_dir = "."

        model_load_kwargs = {"provider": "CPUExecutionProvider"}
        if device == "cuda":
            model_load_kwargs["provider"] = "CUDAExecutionProvider"
            # use_io_binding is generally recommended for GPU for better performance
            model_load_kwargs["use_io_binding"] = True

        model = ORTModelForCausalLM.from_pretrained(model_dir, **model_load_kwargs)
        # Model is already on the correct device due to the provider setting in from_pretrained.
        # Explicit .to(device) is not typically needed for ORTModelForCausalLM after specifying provider.
        print("ONNX model loaded.")

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        import traceback

        traceback.print_exc()
        return False

    print("Tokenizing sample prompt...")
    try:
        inputs = tokenizer(SAMPLE_PROMPT, return_tensors="pt").to(device)
        input_length = inputs.input_ids.shape[1]
    except Exception as e:
        print(f"Error tokenizing prompt or moving to device: {e}")
        return False

    print(f"Input prompt token length: {input_length}")

    # Warm-up runs
    print(f"Performing {NUM_WARMUP_RUNS} warm-up runs...")
    try:
        for i in range(NUM_WARMUP_RUNS):
            print(f"Warm-up run {i+1}/{NUM_WARMUP_RUNS}...")
            _ = model.generate(
                inputs.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                min_length=input_length + 1,
            )
        print("Warm-up runs completed.")
    except Exception as e:
        print(f"Error during warm-up runs: {e}")
        return False

    # Benchmark runs
    print(f"Performing {NUM_BENCHMARK_RUNS} benchmark runs...")
    latencies = []
    try:
        for i in range(NUM_BENCHMARK_RUNS):
            print(f"Benchmark run {i+1}/{NUM_BENCHMARK_RUNS}...")
            start_time = time.perf_counter()
            # Using default generation parameters (greedy search) for latency measurement.
            # For more complex scenarios, parameters like do_sample, num_beams could be set.
            _ = model.generate(
                inputs.input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                min_length=input_length + 1,
            )
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000)  # Convert to milliseconds
        print("Benchmark runs completed.")
    except Exception as e:
        print(f"Error during benchmark runs: {e}")
        return False

    if not latencies:
        print("Error: No benchmark runs were completed successfully.")
        return False

    average_latency_ms = statistics.mean(latencies)
    median_latency_ms = statistics.median(latencies)
    stdev_latency_ms = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    # Calculate P95 latency
    # statistics.quantiles requires Python 3.8+. For n=100, it gives percentiles.
    # P95 is the 95th percentile, so index 94 for 0-indexed list of 100 quantiles.
    # If using older Python, numpy.percentile(latencies, 95) could be an alternative if numpy is available.
    try:
        p95_latency_ms = statistics.quantiles(latencies, n=100)[94]
    except (
        AttributeError
    ):  # statistics.quantiles might not be available in older Python versions
        print(
            "Warning: statistics.quantiles not available (requires Python 3.8+). P95 latency not calculated."
        )
        p95_latency_ms = -1  # Placeholder if not calculable

    print(f"\n--- Benchmark Results ---")
    print(f"Number of Generated Tokens per Run: {MAX_NEW_TOKENS}")
    print(f"Average Latency: {average_latency_ms:.2f} ms")
    print(f"Median Latency: {median_latency_ms:.2f} ms")
    if p95_latency_ms != -1:
        print(f"P95 Latency: {p95_latency_ms:.2f} ms")
    print(f"Standard Deviation: {stdev_latency_ms:.2f} ms")
    print(f"Min Latency: {min(latencies):.2f} ms")
    print(f"Max Latency: {max(latencies):.2f} ms")
    print(
        "\nNote: This benchmark measures the end-to-end generation latency for the specified model"
    )
    print(
        "and parameters on the current hardware. The target latency (<= 8.0ms) is specified"
    )
    print("for an NVIDIA RTX 4070 GPU or equivalent, generating 50 new tokens.")
    print(
        "Results may vary significantly based on hardware, software versions, and model specifics."
    )

    # Assertion against the latency target
    # The primary target is average latency for "batch-1" performance.
    if average_latency_ms <= LATENCY_TARGET_MS:
        print(
            f"\nSUCCESS: Average latency {average_latency_ms:.2f}ms is within the target of {LATENCY_TARGET_MS:.2f}ms."
        )
        return True
    else:
        print(
            f"\nFAILURE: Average latency {average_latency_ms:.2f}ms exceeds target of {LATENCY_TARGET_MS:.2f}ms."
        )
        return False


if __name__ == "__main__":
    print("Executing Phi-3 ONNX Model Latency Benchmark Script...")
    success = run_benchmark()

    if success:
        print("Benchmark completed successfully and met the performance target.")
        exit(0)  # Exit with zero code on success
    else:
        print("Benchmark failed or did not meet the performance target.")
        exit(1)  # Exit with non-zero code on failure
