#!/usr/bin/env python3
"""
Record 5 seconds from the mic into /tmp/test.wav (16 kHz, mono, PCM-16).
CTRL-C to stop early.
"""
import asyncio, torch, torchaudio, pathlib
from mic import mic_chunks

SAMPLE_RATE = 16_000
OUT_PATH    = pathlib.Path("/tmp/test.wav")
SECONDS     = 5                         # total duration

# ---- tiny helper because builtin enumerate() is *sync* only ----
async def aenumerate(aiter, start=0):
    idx = start
    async for item in aiter:
        yield idx, item
        idx += 1
# ----------------------------------------------------------------

async def record_to_wav():
    print("🎙️  Mic is hot — start talking!")
    chunks = []

    async for idx, np_chunk in aenumerate(mic_chunks()):
        tensor = torch.from_numpy(np_chunk).unsqueeze(0)   # (1, samples)
        chunks.append(tensor)
        print(f"📦 Recorded chunk {idx+1}/{SECONDS}")
        if idx + 1 >= SECONDS:
            break

    if not chunks:
        print("⚠️ Nothing captured.")
        return

    audio = torch.cat(chunks, dim=1)                      # (1, total_samples)
    torchaudio.save(OUT_PATH.as_posix(), audio, SAMPLE_RATE)
    print(f"✅ Saved {OUT_PATH}")

def main():
    try:
        asyncio.run(record_to_wav())
    except KeyboardInterrupt:
        # graceful bailout if user hits CTRL-C mid-record
        pass

if __name__ == "__main__":
    main()
