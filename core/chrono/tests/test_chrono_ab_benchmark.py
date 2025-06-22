import json
import time
import torch
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from core.chrono.tcn_layer import TCNChronoLayer
from core.chrono.tide_layer import TiDEChronoLayer
from core.chrono.tests.utils import compute_mse

# ——————————————————————————————————————————————————————————————————————————
TEST_DIR  = Path(__file__).parent
DATA_PATH = TEST_DIR / "data" / "spy_2024-05-17.parquet"
LOG_PATH  = TEST_DIR / "chrono_benchmark_results.json"
# ——————————————————————————————————————————————————————————————————————————

def load_and_engineer(ticks_path):
    """
    Returns a list of np.arrays [n_features] after:
    - computing log-returns on price
    - rolling Z-score normalization over a 16-tick window for each feature
    """
    df = pd.read_parquet(ticks_path)
    # raw features assumed: ['feature1','feature2','price']
    arr = df[['feature1','feature2','price']].values
    # compute log-returns of price column
    prices = arr[:, 2]
    logret = np.zeros_like(prices)
    logret[1:] = np.log(prices[1:] / prices[:-1])
    # replace price column with log-return
    arr[:, 2] = logret
    # rolling z-score (window=16) per column
    window = 16
    engineered = []
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        window_data = arr[start:i+1]
        mu = window_data.mean(axis=0)
        sigma = window_data.std(axis=0) if window_data.shape[0] > 1 else 1.0
        engineered.append((arr[i] - mu) / (sigma + 1e-6))
    return engineered

@pytest.fixture(scope="module")
def ticks_list():
    return load_and_engineer(DATA_PATH)

def benchmark_layer(layer_cls, ticks, history_size, device="cpu"):
    """
    Warmup + timed one-step-ahead forecasting:
    - history tensor shape [history_size, n_features]
    - next price predicted by layer.step
    """
    n_features = len(ticks[0])
    # instantiate model
    model = layer_cls(in_channels=n_features, history_size=history_size)
    model = model.to(device).eval()

    # initialize history buffer
    history = torch.zeros(history_size, n_features, device=device)
    # preload first history_size ticks
    for i in range(history_size):
        history[i] = torch.tensor(ticks[i], device=device)

    latencies, preds, trues = [], [], []
    # iterate through ticks
    for i in range(history_size, len(ticks) - 1):
        true_next = ticks[i + 1][-1]  # price/log-ret feature
        # time prediction
        start = time.time()
        pred_tensor = model.step(history)
        latencies.append((time.time() - start) * 1000)
        preds.append(pred_tensor.item())
        trues.append(true_next)
        # roll and append current tick
        history = torch.roll(history, shifts=-1, dims=0)
        history[-1] = torch.tensor(ticks[i], device=device)

    avg_latency = sum(latencies) / len(latencies)
    mse = compute_mse(preds, trues)
    return avg_latency, mse

def log_result(model_name, device, hist_size, avg_latency, mse):
    entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": model_name,
        "history_size": hist_size,
        "device": device,
        "avg_latency_ms": avg_latency,
        "mse": mse,
    }
    records = []
    if LOG_PATH.exists():
        records = json.loads(LOG_PATH.read_text())
    records.append(entry)
    LOG_PATH.write_text(json.dumps(records, indent=2))

# Parametrize over three history sizes
@pytest.mark.parametrize("history_size", [16, 32, 64])
def test_tcn_forecast(ticks_list, history_size):
    lat, mse = benchmark_layer(TCNChronoLayer, ticks_list, history_size, device="cpu")
    print(f"TCN | hist={history_size} | CPU: {lat:.2f} ms | MSE: {mse:.5f}")
    log_result("TCNChronoLayer", "cpu", history_size, lat, mse)
    assert lat <= 50, f"TCN {history_size} >50ms: {lat:.2f} ms"

@pytest.mark.parametrize("history_size", [16, 32, 64])
def test_tide_forecast(ticks_list, history_size):
    lat, mse = benchmark_layer(TiDEChronoLayer, ticks_list, history_size, device="cpu")
    print(f"TiDE | hist={history_size} | CPU: {lat:.2f} ms | MSE: {mse:.5f}")
    log_result("TiDEChronoLayer", "cpu", history_size, lat, mse)
    assert lat <= 50, f"TiDE {history_size} >50ms: {lat:.2f} ms"
