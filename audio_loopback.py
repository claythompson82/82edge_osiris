import sounddevice as sd

# Match Whisper’s 16 kHz sample rate
SAMPLE_RATE = 16000
# About 1024 samples per block ≈64 ms latency
BLOCK_SIZE = 1024

# Use device 0 for both mic (input) and speakers (output)
DEVICE_INDEX = 1  

def callback(indata, outdata, frames, time, status):
    if status:
        print(f"⚠️ Status: {status}")
    outdata[:] = indata  # loop mic → speaker

print(f"🔊 Loopback on device #{DEVICE_INDEX}. Speak now!")
with sd.Stream(device=(DEVICE_INDEX, DEVICE_INDEX),
               samplerate=SAMPLE_RATE,
               blocksize=BLOCK_SIZE,
               dtype='int16',
               channels=1,
               callback=callback):
    input("Press Enter to stop the loopback…")
