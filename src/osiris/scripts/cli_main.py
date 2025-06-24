# osiris/scripts/cli_main.py

from __future__ import annotations
import sys, time
import torch
from torch.multiprocessing import Process, Queue, set_start_method

from core.chrono.api import StepOutput
from core.chrono.factory import create_chrono_layers

cfg = {
    "chrono_stack": {
        "micro_enabled": True,
        "micro_model": "tcn",       # or "tide"
        "meso_enabled": True,
        "input_dim": 3,
        "history_size": 64,
        "tide_hidden_dims": [64, 32],
        "tide_output_dim": 32,
    }
}

def meso_layer_process(layer, in_q: Queue, out_q: Queue):
    print("[Meso] starting")
    while True:
        data = in_q.get()
        if data is None:
            break
        try:
            step_out: StepOutput = layer.step(data)
            if step_out:
                out_q.put(step_out)
        except Exception as e:
            print(f"[Meso] error: {e}")
            time.sleep(1)
    print("[Meso] exiting")

def cli_main(argv: list[str] | None = None) -> None:
    print("--- Osiris Agent Initializing ---")
    layers = create_chrono_layers(cfg)

    # spawn meso process
    meso_proc = None
    to_meso = None
    from_meso = None
    if "meso" in layers:
        to_meso = Queue()
        from_meso = Queue()
        meso_proc = Process(target=meso_layer_process,
                            args=(layers["meso"], to_meso, from_meso))
        meso_proc.start()
        print("[Main] meso process started")

    cfg["chrono_stack"]["layers"] = layers

    print("\n--- Simulating Agent Loop ---")
    micro = layers.get("micro")
    if micro and meso_proc:
        for tick in range(5):
            print(f"\n[Main] Tick {tick}")
            hist = torch.randn(cfg["chrono_stack"]["history_size"],
                               cfg["chrono_stack"]["input_dim"])
            out: StepOutput = micro.step(hist)
            print(f"[Main] Micro→prediction {out.prediction.item():.4f}")
            if out.embedding is not None:
                to_meso.put(out.embedding)
            try:
                meso_out: StepOutput = from_meso.get(timeout=0.1)
                print(f"[Main] Meso→prediction {meso_out.prediction.item():.4f}")
            except:
                pass
            time.sleep(1)

    # shutdown
    if meso_proc:
        print("\n--- Shutting down ---")
        to_meso.put(None)
        meso_proc.join(timeout=5)
        if meso_proc.is_alive():
            meso_proc.terminate()
        print("Shutdown complete")

if __name__ == "__main__":
    try:
        set_start_method("spawn")
    except RuntimeError:
        pass
    cli_main(sys.argv[1:])
