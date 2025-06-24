import threading
import queue
import numpy as np
import sounddevice as sd
import whisper
import torch
import webrtcvad

# ── CONFIG ───────────────────────────────────────────────────────────
SAMPLE_RATE       = 16000    # must be 16000 for WebRTC VAD
BLOCK_DURATION_MS = 30       # 30 ms per frame for VAD
BLOCK_SIZE        = int(SAMPLE_RATE * BLOCK_DURATION_MS / 1000)
IN_DEV            = 1        # your mic index
GAIN              = 3.0      # mic gain (1.0–5.0)
MODEL_NAME        = "large"  # tiny|base|small|medium|large
BEAM_SIZE         = 10
VAD_AGGRESSIVENESS= 2        # 0-3, higher = more sensitive to speech

# ── SETUP ────────────────────────────────────────────────────────────
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Whisper {MODEL_NAME} on {device}…")
model = whisper.load_model(MODEL_NAME, device=device).to(device)
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

audio_q = queue.Queue()

# ── AUDIO CALLBACK ───────────────────────────────────────────────────
def audio_callback(indata, frames, time, status):
    if status:
        print("⚠️", status)
    # mono audio → gain → clip
    mono      = indata[:,0].astype(np.int32)
    amplified = np.clip(mono * GAIN, -32768, 32767).astype(np.int16)
    # push raw PCM16 to queue
    audio_q.put(amplified.tobytes())

# ── TRANSCRIPTION WORKER ─────────────────────────────────────────────
def transcribe_worker():
    buffer_bytes = b""
    in_speech    = False
    speech_buffer = []

    while True:
        frame = audio_q.get()
        if frame is None:
            break
        buffer_bytes += frame

        # While we have at least one block of bytes
        while len(buffer_bytes) >= BLOCK_SIZE * 2:  # 2 bytes per sample
            chunk = buffer_bytes[: BLOCK_SIZE * 2]
            buffer_bytes = buffer_bytes[BLOCK_SIZE * 2:]
            is_speech = vad.is_speech(chunk, SAMPLE_RATE)

            if is_speech:
                in_speech = True
                speech_buffer.append(chunk)
            elif in_speech:
                # Heard end of utterance
                utterance = b"".join(speech_buffer)
                speech_buffer = []
                in_speech = False

                # Convert bytes → float32 array
                pcm16 = np.frombuffer(utterance, dtype=np.int16)
                audio = pcm16.astype(np.float32) / 32768.0

                # ASR on GPU
                audio_tensor = torch.from_numpy(audio).to(device)
                res = model.transcribe(
                    audio=audio_tensor,
                    fp16=True,
                    language="en",
                    beam_size=BEAM_SIZE
                )
                text = res["text"].strip()
                if text:
                    print(text)

    # flush any remaining speech
    if speech_buffer:
        utterance = b"".join(speech_buffer)
        pcm16 = np.frombuffer(utterance, dtype=np.int16)
        audio = pcm16.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio).to(device)
        res = model.transcribe(
            audio=audio_tensor,
            fp16=True,
            language="en",
            beam_size=BEAM_SIZE
        )
        text = res["text"].strip()
        if text:
            print(text)

# ── MAIN ──────────────────────────────────────────────────────────────
def main():
    worker = threading.Thread(target=transcribe_worker, daemon=True)
    worker.start()

    print("🗣️ Live ASR (VAD-based utterances) — press Enter to stop")
    with sd.InputStream(device=IN_DEV,
                        samplerate=SAMPLE_RATE,
                        blocksize=BLOCK_SIZE,
                        dtype="int16",
                        channels=1,
                        callback=audio_callback):
        input()

    audio_q.put(None)
    worker.join()

if __name__ == "__main__":
    main()
