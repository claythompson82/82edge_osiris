import json
import time
import torch
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from core.chrono.tcn_layer import TCNChronoLayer
from core.chrono.tests.utils import compute_mse

# Paths
TEST_DIR  = Path(__file__).parent
DATA_PATH = TEST_DIR / "data" / "spy_2024-05-17.parquet"
LOG_PATH  = TEST_DIR / "chrono_benchmark_results.json"

def load_and_engineer(ticks_path):
    df = pd.read_parquet(ticks_path)
    arr = df[['feature1','feature2','price']].values
    prices = arr[:, 2]
    logret = np.zeros_like(prices)
    logret[1:] = np.log(prices[1:] / prices[:-1])
    arr[:, 2] = logret
    window = 16
    engineered = []
    for i in range(len(arr)):
        start = max(0, i - window + 1)
        win = arr[start:i+1]
        mu, sigma = win.mean(0), (win.std(0) if win.shape[0]>1 else 1.0)
        engineered.append((arr[i] - mu) / (sigma + 1e-6))
    return engineered

@pytest.fixture(scope="module")
def ticks_list():
    return load_and_engineer(DATA_PATH)

def benchmark_layer(layer_cls, ticks, history_size, device="cpu"):
    n_feat = len(ticks[0])
    model = layer_cls(in_channels=n_feat, history_size=history_size).to(device).eval()

    history = torch.zeros(history_size, n_feat, device=device)
    for i in range(history_size):
        history[i] = torch.tensor(ticks[i], device=device)

    lat, preds, trues = [], [], []
    for i in range(history_size, len(ticks)-1):
        true_next = ticks[i+1][-1]
        start = time.time()
        pred = model.step(history)
        lat.append((time.time()-start)*1000)
        preds.append(pred.item())
        trues.append(true_next)
        history = torch.roll(history, -1, 0)
        history[-1] = torch.tensor(ticks[i], device=device)

    return sum(lat)/len(lat), compute_mse(preds, trues)

def log_result(model, hist, dev, lat, mse):
    rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "model": model, "history_size": hist, "device": dev,
           "avg_latency_ms": lat, "mse": mse}
    arr = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
    arr.append(rec)
    LOG_PATH.write_text(json.dumps(arr, indent=2))

@pytest.mark.parametrize("history_size", [16,32,64])
def test_tcn_forecast(history_size, ticks_list):
    lat, mse = benchmark_layer(TCNChronoLayer, ticks_list, history_size, "cpu")
    print(f"TCN | hist={history_size} | CPU: {lat:.2f} ms | MSE: {mse:.5f}")
    log_result("TCNChronoLayer", history_size, "cpu", lat, mse)
    assert lat <= 50, f"TCN {history_size} latency >50ms: {lat:.2f}"
