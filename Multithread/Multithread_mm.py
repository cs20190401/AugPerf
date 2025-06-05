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
from matchmaker import Matchmaker
import soundfile as sf
import traceback
from matchmaker.features.audio import MFCCProcessor, MelSpectrogramProcessor, ChromagramProcessor
from matchmaker.dp import OnlineTimeWarpingArzt

# Audio stream settings
CHUNK = 441  # Samples per frame
#FORMAT = pyaudio.paInt16  # 16-bit audio
FORMAT = pyaudio.paFloat32  # 32-bit float audio
CHANNELS = 1  # Mono
#CHANNELS = 2  # Stereo
RATE = 44100  # Sample rate
BEATNET_SR = 22050

FEATURE_TYPE = "chroma"  # Feature type for Matchmaker, can be "mel", "mfcc", or "chroma"

audio_stream_start_time = None
BeatTracking_start_time = None

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
reference_audio_path = "/Users/wonseonjae/rep/AugPerf/resources/dataset/mini_band_harmonix/record/tracks/0004_abc.mp3"
beat_label_path = "/Users/wonseonjae/rep/AugPerf/resources/dataset/mini_band_harmonix/record/beats/0004_abc.txt"
fx_label_path = "/Users/wonseonjae/rep/AugPerf/resources/dataset/mini_band_harmonix/live/fx_list/0004_abc.txt"

performance_audio_path = "/Users/wonseonjae/rep/AugPerf/resources/dataset/mini_band_harmonix/live/tracks_441/0004_abc.mp3"

# Preprocessing step
print("Preprocessing reference data...")
reference_features = preprocess_reference_audio(reference_audio_path, feature_type=FEATURE_TYPE, sample_rate=BEATNET_SR)
#reference_features = preprocess_reference_audio(reference_audio_path, feature_type=FEATURE_TYPE, sample_rate=RATE, mono=False)
beat_labels = get_reference_beat_label(beat_label_path)
beat_frames = convert_beat_label(beat_labels, sample_rate=BEATNET_SR, hop_length=CHUNK)
tempo = get_tempo(beat_labels)
fx_list = get_fx_label(fx_label_path)
converted_fx = convert_fx_label(fx_list, beat_labels, delay_threshold=0.1, sample_rate=BEATNET_SR, hop_length=512)
print("Preprocessing complete.")

# Function to send OSC messages
def send_message(delay, fx_id):
    time.sleep(delay)
    #osc_client.send_message("/fx", fx_id)
    print(f"Sent OSC message: FX ID = {fx_id}, Delay = {delay:.2f}s")

def faux_audio_stream(performance_audio_path):
    global audio_stream_start_time

    # Load the audio file (stereo)
    #audio, sr = sf.read(performance_audio_path, dtype='float32', always_2d=True)  # shape: (n_samples, 2)


    audio, sr = sf.read(performance_audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert to mono if needed

    '''
    if sr != RATE:
        import librosa
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=RATE, axis=1).T  # keep stereo shape (n_samples, 2)
        sr = RATE
    '''
    if sr != BEATNET_SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=BEATNET_SR)
        sr = BEATNET_SR

    #open for playback
    p = pyaudio.PyAudio()
    stream_out = p.open(format=FORMAT, channels=CHANNELS, rate=BEATNET_SR, output=True)
    audio_stream_start_time = time.time()  # Record the start time here
    print(f"Faux audio stream started at {audio_stream_start_time:.6f} (epoch seconds)")

    idx = 0
    total_samples = audio.shape[0]
    #print(f"Producer queue id: {id(audio_queue_dtw)}")
    while idx < total_samples:
        #chunk = audio[idx:idx+CHUNK, :]
        chunk = audio[idx:idx+CHUNK]
        if chunk.shape[0] < CHUNK:
            # Pad the last chunk
            #chunk = np.pad(chunk, ((0, CHUNK - chunk.shape[0]), (0, 0)))
            chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
        #print(f"chunk min: {chunk.min()}, max: {chunk.max()}, mean: {chunk.mean()}")
        # Interleave stereo channels for PyAudio
        #chunk_interleaved = chunk.astype(np.float32).flatten()
        # Put chunk into both queues
        #audio_queue_beatnet.put(chunk_interleaved.tobytes())
        #audio_queue_dtw.put(chunk_interleaved.tobytes())
        audio_queue_beatnet.put(chunk.astype(np.float32).tobytes())
        #audio_queue_dtw.put(chunk.astype(np.float32).tobytes())
        chroma = ChromagramProcessor(sample_rate=BEATNET_SR, hop_length=CHUNK//2)(chunk.astype(np.float32))
        audio_queue_dtw.put((chroma, idx / BEATNET_SR))
        #audio_queue_dtw.put((chunk.astype(np.float32), idx / BEATNET_SR))  # Pass the timestamp as the second argument
        #print(f"MM queue size in MT: {audio_queue_dtw.qsize()}")
        # Play audio through speakers
        #stream_out.write(chunk_interleaved.tobytes())
        # Play audio through speakers
        stream_out.write(chunk.astype(np.float32).tobytes())
        idx += CHUNK
        #time.sleep(CHUNK / RATE)  # Real-time pacing - mute for audio playback

    stream_out.stop_stream()
    stream_out.close()
    p.terminate()
    print("Faux audio stream finished.")

# Audio stream function
def audio_stream():
    global audio_stream_start_time
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    audio_stream_start_time = time.time()  # Record the start time here
    print(f"Audio stream started at {audio_stream_start_time:.6f} (epoch seconds)")
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
    global BeatTracking_start_time
    l = 1
    estimator = BeatNet(1, mode="stream", inference_model="PF", plot=[], thread=False, audio_source=audio_queue_beatnet)
    BeatTracking_start_time = time.time()  # Record the start time here
    print(f"BeatNet started at {BeatTracking_start_time:.6f} (epoch seconds)")
    try:
        while True:
            output = estimator.process()
            if len(output)==l:
                print("BeatNet.process() output:", output[-1])
                print(f"Beat Time: {time.time()-audio_stream_start_time:.6f} (epoch seconds)")
                l+=1
            if len(output): beat_signal_queue.put(output[-1])  # Put the last output into the beat signal queue
            #print(f"BeatNet queue size: {audio_queue_beatnet.qsize()}")

    except KeyboardInterrupt:
        print("BeatNet stopped.")


# Online DTW thread
def run_online_dtw():
    print("Running online DTW with Matchmaker...")
    # Initialize Matchmaker for live alignment
    mm = Matchmaker(
        reference_audio=reference_audio_path,
        performance_file=None,  # Live input, not file
        input_type="audio",
        feature_type="chroma",  # or "mel", as appropriate
        method="arzt",
        sample_rate=BEATNET_SR,
        frame_rate=BEATNET_SR // CHUNK,
        queue=audio_queue_dtw,  # Use the live audio queue
    )

    print("Starting mm.run() generator...")
    n_frames = reference_features.shape[0]
    audio, sr = sf.read(reference_audio_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert to mono if needed

    length_seconds = len(audio) / sr
    # Run the alignment process in a loop
    try:
        for matching_frame in mm.run(verbose=False, wait=True):
            print(f"DTW (Matchmaker) matching frame: {matching_frame} | {matching_frame*length_seconds/n_frames} s | Time since start: {time.time() - audio_stream_start_time:.6f} s")
            dtw_frame_queue.put(matching_frame)
    except Exception as e:
        print(f"DTW thread exception: {e}")
        traceback.print_exc()
        

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
        #output = beat_output_queue.get()
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
#audio_thread = threading.Thread(target=audio_stream, daemon=True)
audio_thread = threading.Thread(target=faux_audio_stream, args=(performance_audio_path,), daemon=True)
beatnet_thread = threading.Thread(target=run_beatnet, daemon=True)
dtw_thread = threading.Thread(target=run_online_dtw, daemon=True)
kalman_thread = threading.Thread(
    target=run_kalman_filter,
    args=(tempo, beat_frames, converted_fx),
    daemon=True
)

audio_thread.start()
beatnet_thread.start()
dtw_thread.start()
#kalman_thread.start()

audio_thread.join()
beatnet_thread.join()
dtw_thread.join()
#kalman_thread.join()