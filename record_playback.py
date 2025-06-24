import sounddevice as sd

SAMPLE_RATE = 16000      # 16 kHz matches Whisper’s expectations
DURATION    = 3         # record 3 seconds
CHANNELS    = 1         # mono

print(f"🎤 Recording {DURATION} seconds of audio... Speak now!")
audio = sd.rec(int(SAMPLE_RATE * DURATION),
               samplerate=SAMPLE_RATE,
               channels=CHANNELS,
               dtype='int16')
sd.wait()  # block until recording is done

print("▶️  Recording complete. Playing back now...")
sd.play(audio, samplerate=SAMPLE_RATE)
sd.wait()  # block until playback is done

print("✅ Playback finished.")
