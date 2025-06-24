# scripts/transcribe_test.py
import whisper

model = whisper.load_model("base")  # or "small" if you want more speed
result = model.transcribe("/tmp/test.wav")

print("📝 Transcription:", result["text"])
