import pyaudio

CHUNK = 4096
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 48000  # Match the device's sample rate

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                input=True, frames_per_buffer=CHUNK)

print("Recording...")
try:
    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        print(f"Captured {len(data)} bytes")
except KeyboardInterrupt:
    print("Stopped.")
stream.stop_stream()
stream.close()
p.terminate()