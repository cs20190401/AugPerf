import threading
import numpy as np
import sounddevice as sd
from pydub import AudioSegment
from pythonosc import dispatcher, osc_server

sample_rate = 44100
duration = 0.1
play_song = True

# Create a click sound
def create_click_sound(sr, duration, freq):
	t = np.linspace(0, duration, int(sr * duration), False)
	click = 0.5 * np.sin(2 * np.pi * freq * t)
	return click

# Play a sound
def play_sound(sound, sr):
	sd.play(sound, sr, blocksize=64, latency="low")
	sd.wait()

# Play a click sound
def play_click(address, beat_type):
	if beat_type == 1:
		play_sound(beat_click, sample_rate)
	elif beat_type == 2: 
		play_sound(downbeat_click, sample_rate)

# Play audio data with sounddevice given a file path
def play_audio_stream(file_path):
	audio = AudioSegment.from_file(file_path, format="mp3")
	audio = audio.set_frame_rate(sample_rate)
	samples = np.array(audio.get_array_of_samples())
	samples = samples.astype(np.float32) / 32768.0 # Normalize to -1.0 to 1.0

	# Reduce the volume
	volume = 0.6
	samples *= volume

	channels = audio.channels
	if channels == 2:
		samples = samples.reshape(-1, 2) # Stereo
	else:
		samples = samples.reshape(-1, 1) # Mono

	current_frame = 0

	def callback(outdata, frames, time, status):
		nonlocal current_frame
		if status:
			print(status)
		chunksize = min(len(samples) - current_frame, frames)
		outdata[:chunksize] = samples[current_frame:current_frame + chunksize]
		if chunksize < frames:
			outdata[chunksize:] = 0
			raise sd.CallbackStop()
		current_frame += chunksize

	background_stream = sd.OutputStream(callback=callback, samplerate=sample_rate, channels=channels, latency="low")
	background_stream.start()

# Start audio playback
def start_audio(address, file_path):
	audio_thread = threading.Thread(target=play_audio_stream, args=(file_path,), daemon=True)
	audio_thread.start()
	print("Audio playback started.")
	# TODO: fix "output underflow" -> due to too much CPU time used?

# Start a persistent stream for click playback
click_buffer = np.zeros(4410)

def play_click_stream(address, beat_type):
	global click_buffer
	if beat_type == 1:
		click_buffer[:len(beat_click)] = beat_click
	elif beat_type == 2:
		click_buffer[:len(downbeat_click)] = downbeat_click

def click_stream_callback(outdata, frames, time, status):
	if status:
		print(status)
	outdata[:, 0] = click_buffer[:frames]
	click_buffer.fill(0) # Clear buffer

click_stream = sd.OutputStream(callback=click_stream_callback, samplerate=sample_rate, channels=1, latency="low")
click_stream.start()

beat_click = create_click_sound(sample_rate, duration, 1000)
downbeat_click = create_click_sound(sample_rate, duration, 1500)

dispatcher = dispatcher.Dispatcher()
dispatcher.map("/click", play_click_stream)
if play_song:
	dispatcher.map("/audio", start_audio)

# Setup OSC server
ip = "127.0.0.1" # localhost
port = 8000
server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)

try:
	print(f"OSC server on ({ip}:{port})")
	server.serve_forever()
except KeyboardInterrupt:
    print("Server stopped.")