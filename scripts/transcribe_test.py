# scripts/transcribe_test.py
import whisper
import time

print("🔍 Loading Whisper model...")
model = whisper.load_model("base")

print("🎧 Transcribing audio...")
start = time.time()
result = model.transcribe("/tmp/test.wav")
end = time.time()

print("📝 Transcription:", result["text"])
print(f"⏱️ Completed in {end - start:.2f} seconds.")
