#import sys
# Remove user site-packages from sys.path
#sys.path = [p for p in sys.path if not p.startswith('/Users/wonseonjae/.local')]

import pyaudio
import queue
import threading
import numpy as np
import time
import librosa
from scipy.signal import lfilter
from preprocess import (
    preprocess_reference_audio,
    get_reference_beat_label,
    convert_beat_label,
    get_tempo,
    get_fx_label,
    convert_fx_label,
)
from kalman_class import KalmanFilter
from pythonosc.udp_client import SimpleUDPClient
from BeatNet.BeatNet import BeatNet
from matchmaker.features.audio import MFCCProcessor, MelSpectrogramProcessor, ChromagramProcessor
from matchmaker.dp import OnlineTimeWarpingArzt

# Audio stream settings
CHUNK = 4096  # Samples per frame
#FORMAT = pyaudio.paInt16  # 16-bit audio
FORMAT = pyaudio.paFloat32  # 32-bit float audio
CHANNELS = 1  # Mono
RATE = 44100  # Sample rate

# Queues for communication between threads
audio_queue_beatnet = queue.Queue()
audio_queue_dtw = queue.Queue()
beat_signal_queue = queue.Queue()
dtw_frame_queue = queue.Queue()

# OSC client for sending messages
OSC_IP = "127.0.0.1"
OSC_PORT = 8000
osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)

# Paths to input files
reference_audio_path = "/Users/wonseonjae/rep/resources/dataset/mini_band_harmonix/record/tracks/0004_abc.mp3"
beat_label_path = "/Users/wonseonjae/rep/resources/dataset/mini_band_harmonix/record/beats/0004_abc.txt"
fx_label_path = "/Users/wonseonjae/rep/resources/dataset/mini_band_harmonix/live/fx_list/0004_abc.txt"

# Preprocessing step
print("Preprocessing reference data...")
reference_features = preprocess_reference_audio(reference_audio_path, feature_type="chroma", sample_rate=RATE)
beat_labels = get_reference_beat_label(beat_label_path)
beat_frames = convert_beat_label(beat_labels, sample_rate=RATE, hop_length=512)
tempo = get_tempo(beat_labels)
fx_list = get_fx_label(fx_label_path)
converted_fx = convert_fx_label(fx_list, beat_labels, delay_threshold=0.1, sample_rate=RATE, hop_length=512)
print("Preprocessing complete.")

# Function to send OSC messages
def send_message(delay, fx_id):
    time.sleep(delay)
    #osc_client.send_message("/fx", fx_id)
    print(f"Sent OSC message: FX ID = {fx_id}, Delay = {delay:.2f}s")

# Audio stream function
def audio_stream():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Audio stream started.")
    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            #print("Read audio chunk")
            audio_queue_beatnet.put(data)
            audio_queue_dtw.put(data)
            #print(f"Put chunk: beatnet_q={audio_queue_beatnet.qsize()}, dtw_q={audio_queue_dtw.qsize()}")
        except OSError as e:
            print(f"Audio input overflowed: {e}")

# BeatNet thread
def run_beatnet():
    print("Running BeatNet in stream mode...")
    estimator = BeatNet(1, mode="stream", inference_model="PF", plot=[], thread=False, audio_source=audio_queue_beatnet)
    try:
        while True:
            estimator.process()
            #print(f"BeatNet queue size: {audio_queue_beatnet.qsize()}")
            if estimator.beat:  # If BeatNet detects a beat
                print(f"[Thread] BeatNet.beat: {estimator.beat}")
                beat_signal_queue.put(True)
    except KeyboardInterrupt:
        print("BeatNet stopped.")

# Online DTW thread
def run_online_dtw():
    print("Running online DTW...")
    dtw = OnlineTimeWarpingArzt(reference_features=reference_features, distance_func="Euclidean", frame_rate=RATE / CHUNK)
    processor = ChromagramProcessor(sample_rate=RATE)
    while True:
        audio_data = audio_queue_dtw.get()
        #print(f"DTW queue size: {audio_queue_dtw.qsize()}")
        if audio_data is None:
            break
        np_audio = np.frombuffer(audio_data, dtype=np.float32)
        input_features = processor(np_audio)
        #print(f"DTW audio mean: {np.mean(np_audio)}, max: {np.max(np_audio)}")
        #print(f"DTW features shape: {input_features.shape}, mean: {np.mean(input_features)}, std: {np.std(input_features)}")
        #print(f"DTW features shape: {reference_features.shape}, mean: {np.mean(reference_features)}, std: {np.std(reference_features)}")
        #if input_features.ndim == 2:
            #input_features = input_features[-1]
        if input_features.ndim == 1:
            input_features = input_features[np.newaxis, :]
        if np.std(input_features) != 0:
            input_features = (input_features - np.mean(input_features)) / np.std(input_features)
        else:
            input_features = input_features - np.mean(input_features)  # Only mean normalization
        matching_frame = dtw.step(input_features)
        print(f"dtw.step returned: {matching_frame}")
        if matching_frame is not None:
            print(f"DTW matching frame: {matching_frame}")
            dtw_frame_queue.put(matching_frame)

# Kalman Filter thread
def run_kalman_filter(tempo, beat_frames, converted_fx):
    print("Running Kalman Filter...")
    kf = KalmanFilter()  # Initialize without additional arguments
    kf.x_esti = np.array([[0], [tempo]])  # Set the initial state manually
    last_beat_time = None
    recent_beats = []
    while True:
        matching_frame = dtw_frame_queue.get()
        is_beat = not beat_signal_queue.empty()
        if is_beat:
            beat_signal_queue.get()  # Consume the beat signal
            current_time = time.time()
            if last_beat_time is not None:
                interval = current_time - last_beat_time
                if abs(interval - (60 / tempo)) <= 0.15 * (60 / tempo):  # Valid beat
                    last_beat_time = current_time
                    recent_beats.append(interval)
                    if len(recent_beats) > 5:
                        recent_beats.pop(0)
                        tempo = 60 / np.mean(recent_beats)  # Update tempo using LPF
                    print(f"Valid beat detected. Updated tempo: {tempo:.2f} BPM")
                    for beat_frame in beat_frames:
                        if abs(beat_frame - matching_frame) <= librosa.time_to_frames(0.2, sr=RATE, hop_length=512):
                            kf.update(np.array([[beat_frame], [tempo]]))
                            break
            else:
                last_beat_time = current_time
        else:
            kf.update(np.array([[matching_frame], [tempo]]))  # Predict using the current frame and tempo
        current_frame = int(kf.get_state()[0, 0])
        print(f"Current frame: {current_frame}")
        for frame, delay, fx_id in converted_fx: # sort converted_fx in preprocess.py, keep track of next idx.
            if frame == current_frame:
                send_message(delay, fx_id)

# Start threads
audio_thread = threading.Thread(target=audio_stream, daemon=True)
beatnet_thread = threading.Thread(target=run_beatnet, daemon=True)
dtw_thread = threading.Thread(target=run_online_dtw, daemon=True)
kalman_thread = threading.Thread(
    target=run_kalman_filter,
    args=(tempo, beat_frames, converted_fx),
    daemon=True
)

audio_thread.start()
#beatnet_thread.start()
dtw_thread.start()
#kalman_thread.start()

audio_thread.join()
#beatnet_thread.join()
dtw_thread.join()
#kalman_thread.join()