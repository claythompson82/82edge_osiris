# core/chrono/tests/test_chrono_ab_benchmark.py

import json, time, torch, pytest, numpy as np, pandas as pd
from pathlib import Path
from typing import List

# ensure src/ and repo root on PYTHONPATH
import sys
HERE = Path(__file__).parent
ROOT = HERE.parent.parent.parent
SRC = ROOT / "src"
for p in (SRC, ROOT):
    sys.path.insert(0, str(p))

from core.chrono.api import StepOutput
from core.chrono.tcn_layer import TCNChronoLayer
from core.chrono.tide_layer import TiDEChronoLayer
from core.chrono.tests.utils import compute_mse

TEST_DIR = HERE
DATA_PATH = TEST_DIR / "data" / "spy_2024-05-17.parquet"
LOG_PATH  = TEST_DIR / "chrono_benchmark_results.json"

def load_and_engineer_features(path: Path) -> List[np.ndarray]:
    df = pd.read_parquet(path)
    arr = df[['feature1','feature2','price']].values
    prices = arr[:, -1]
    lr = np.zeros_like(prices)
    mask = prices[:-1]>0
    lr[1:][mask] = np.log(prices[1:][mask]/prices[:-1][mask])
    arr[:, -1] = lr
    rolled = pd.DataFrame(arr).rolling(20, min_periods=1)
    means = rolled.mean().fillna(0).values
    stds  = rolled.std().fillna(1).values
    return [(row - m)/(s+1e-6) for row,m,s in zip(arr,means,stds)]

@pytest.fixture(scope="module")
def ticks_list():
    return load_and_engineer_features(DATA_PATH)

def _unwrap(step_out):
    if isinstance(step_out, StepOutput):
        return step_out
    # legacy dict
    return StepOutput(prediction=step_out['prediction'],
                      embedding=step_out.get('embedding'))

def benchmark_layer_cpu(layer_cls, ticks, history_size):
    n = len(ticks[0])
    model = layer_cls(input_dim=n, history_size=history_size).cpu().eval()
    hist = torch.stack([torch.tensor(t, dtype=torch.float32) for t in ticks[:history_size]])
    # warm-up
    for i in range(history_size, min(history_size+50, len(ticks)-1)):
        _ = _unwrap(model.step(hist))
        hist = torch.roll(hist, -1, 0)
        hist[-1] = torch.tensor(ticks[i], dtype=torch.float32)
    # timed
    lats, preds, trues = [], [], []
    for i in range(history_size, len(ticks)-1):
        true = ticks[i+1][-1]
        start = time.perf_counter()
        out = _unwrap(model.step(hist))
        lats.append((time.perf_counter()-start)*1000)
        preds.append(out.prediction.item())
        trues.append(true)
        hist = torch.roll(hist, -1, 0)
        hist[-1] = torch.tensor(ticks[i], dtype=torch.float32)
    return sum(lats)/len(lats), compute_mse(preds, trues)

def benchmark_layer_gpu(layer_cls, ticks, history_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    n = len(ticks[0])
    model = layer_cls(input_dim=n, history_size=history_size).to(device).eval()
    hist = torch.stack([torch.tensor(t, dtype=torch.float32, device=device)
                        for t in ticks[:history_size]])
    # warm-up
    for i in range(history_size, min(history_size+50, len(ticks)-1)):
        with torch.no_grad():
            _ = _unwrap(model.step(hist))
        hist = torch.roll(hist, -1, 0)
        hist[-1] = torch.tensor(ticks[i], dtype=torch.float32, device=device)
    # timed
    lats, preds, trues = [], [], []
    torch.cuda.synchronize(device)
    for i in range(history_size, len(ticks)-1):
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt   = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        out = _unwrap(model.step(hist))
        end_evt.record()
        torch.cuda.synchronize(device)
        lats.append(start_evt.elapsed_time(end_evt))
        preds.append(out.prediction.item())
        trues.append(ticks[i+1][-1])
        hist = torch.roll(hist, -1, 0)
        hist[-1] = torch.tensor(ticks[i], dtype=torch.float32, device=device)
    return (sum(lats)/len(lats)) if lats else 0, compute_mse(preds, trues)

def log_result(name, hs, dev, lat, mse):
    rec = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": name, "history_size": hs,
        "device": dev, "avg_latency_ms": lat, "mse": mse
    }
    allr = json.loads(LOG_PATH.read_text()) if LOG_PATH.exists() else []
    allr.append(rec)
    LOG_PATH.write_text(json.dumps(allr, indent=2))

@pytest.mark.parametrize("history_size", [16, 32, 64])
def test_micro_layer_models(history_size, ticks_list):
    tcn_lat, tcn_mse = benchmark_layer_cpu(TCNChronoLayer, ticks_list, history_size)
    log_result("TCNChronoLayer", history_size, "cpu", tcn_lat, tcn_mse)
    if torch.cuda.is_available():
        gl, gm = benchmark_layer_gpu(TCNChronoLayer, ticks_list, history_size)
        log_result("TCNChronoLayer", history_size, "cuda", gl, gm)

    tide_lat, tide_mse = benchmark_layer_cpu(TiDEChronoLayer, ticks_list, history_size)
    log_result("TiDEChronoLayer", history_size, "cpu", tide_lat, tide_mse)
    if torch.cuda.is_available():
        gl, gm = benchmark_layer_gpu(TiDEChronoLayer, ticks_list, history_size)
        log_result("TiDEChronoLayer", history_size, "cuda", gl, gm)

    assert min(tcn_lat, tide_lat) <= 50, f"Neither model ≤50 ms for history={history_size}"
