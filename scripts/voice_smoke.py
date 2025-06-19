"""
Run with:  python scripts/voice_smoke.py
"""
import asyncio, subprocess
import whisper, sounddevice as sd, torchaudio
from chatterbox.tts import ChatterboxTTS

async def main():
    # 1. silent chunk ➜ Whisper (should print '') ------------------------
    subprocess.run(
        ["ffmpeg","-f","lavfi","-i","anullsrc=r=16000:cl=mono","-t","1",
         "-acodec","pcm_s16le","/tmp/silence.wav","-y"],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    empty = whisper.load_model("base","cpu").transcribe(
        "/tmp/silence.wav", no_speech_threshold=0.05)["text"]
    print("Silence transcript:", repr(empty))

    # 2. TTS round-trip ---------------------------------------------------
    tts = ChatterboxTTS.from_pretrained(device="cpu")
    wav = tts.generate("Hello world! 🎉")
    sd.play(wav.squeeze(), tts.sr, device=1)
    sd.wait()

    print("TTS OK")

if __name__ == "__main__":
    asyncio.run(main())
