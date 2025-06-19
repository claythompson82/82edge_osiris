#!/usr/bin/env python3
"""
Yield live-mic audio in 1-second numpy chunks (16 kHz, mono).
"""
import asyncio, sounddevice as sd, numpy as np

SAMPLE_RATE = 16_000
CHUNK_SEC   = 1.0
CHUNK_SIZE  = int(SAMPLE_RATE * CHUNK_SEC)

async def mic_chunks():
    """Asynchronous generator yielding (CHUNK_SIZE,) float32 numpy arrays."""
    loop  = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def _callback(indata, frames, time, status):        # sounddevice callback
        if status:
            print("⚠️ mic status:", status)
        loop.call_soon_threadsafe(queue.put_nowait, indata[:, 0].copy())

    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=1,
                        dtype="float32",
                        blocksize=CHUNK_SIZE,
                        callback=_callback):
        try:
            while True:
                chunk = await queue.get()               # (CHUNK_SIZE,) np.float32
                yield chunk
        finally:                                        # runs on cancellation/CTRL-C
            print("🎤 Mic stopped.")
