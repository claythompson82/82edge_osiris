import numpy as np
import sounddevice as sd

# Parameters for a 440 Hz sine wave
DURATION = 2.0       # seconds
FS       = 44100     # sample rate
FREQ     = 440.0     # A4

# Generate the waveform
t = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)
tone = 0.5 * np.sin(2 * np.pi * FREQ * t)

print(f"▶️ Playing a {FREQ} Hz test tone for {DURATION} s…")
sd.play(tone, FS)
sd.wait()  # wait until done
print("✅ Done.")
