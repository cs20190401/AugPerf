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
from matchmaker.utils.misc import plot_and_save_score_following_result
from pathlib import Path
#from matchmaker.dp import OnlineTimeWarpingArzt

# Audio stream settings
CHUNK = 441  # Samples per frame
#FORMAT = pyaudio.paInt16  # 16-bit audio
FORMAT = pyaudio.paFloat32  # 32-bit float audio
CHANNELS = 1  # Mono
#CHANNELS = 2  # Stereo
STREAM_RATE = 44100  # Sample rate
BEATNET_SR = 22050 # 441/22050 = 20ms resolution

BEAT_TIME_THRESHOLD = 0.1  # Threshold for beat time detection
TEMPO_UPDATE_ALPHA = 0.7  # Low-pass filter alpha for tempo updates

FEATURE_TYPE = "chroma"  # Feature type for Matchmaker, can be "mel", "mfcc", or "chroma"

audio_stream_start_time = None
BeatTracking_start_time = None

# Queues for communication between threads
audio_queue_beatnet = queue.Queue()
audio_queue_dtw = queue.Queue()
beat_signal_queue = queue.Queue()
dtw_frame_queue = queue.Queue()

online_dtw_beats = []
beatnet_beats = []
kalman_beats = []
gt_beats = []

# OSC client for sending messages
OSC_IP = "127.0.0.1"
OSC_PORT = 8000
osc_client = SimpleUDPClient(OSC_IP, OSC_PORT)

# Paths to input files
resource_path = "./AugPerf/resources/dataset/mini_band_harmonix"
file_name = "0004_abc"
mode = "record"  # or "live"
reference_audio_path = resource_path+"/record/tracks/"+file_name+".mp3"
beat_label_path = resource_path+"/record/beats/"+file_name+".txt"
fx_label_path = resource_path+"/live/fx_list/"+file_name+".txt"

performance_audio_path = resource_path+"/"+mode+"/tracks/"+file_name+".mp3"

folder_path = Path("./results/"+file_name+"_"+mode)

# Preprocessing step
print("Preprocessing reference data...")
reference_features = preprocess_reference_audio(reference_audio_path, feature_type=FEATURE_TYPE, sample_rate=BEATNET_SR)
beat_labels = get_reference_beat_label(beat_label_path)
beat_frames = convert_beat_label(beat_labels, sample_rate=BEATNET_SR, hop_length=CHUNK)
tempo = get_tempo(beat_labels)
fx_list = get_fx_label(fx_label_path)
converted_fx = convert_fx_label(fx_list, beat_labels, delay_threshold=0.1, sample_rate=BEATNET_SR, hop_length=CHUNK)
print("Preprocessing complete.")

# Function to send OSC messages
def send_message(delay, fx_id):
    time.sleep(delay)
    #osc_client.send_message("/fx", fx_id)
    print(f"Sent OSC message: FX ID = {fx_id}, Delay = {delay:.2f}s")

def bpm_to_frames_per_update(bpm, chunk, sample_rate):
    beats_per_second = bpm / 60.0
    seconds_per_update = chunk / sample_rate
    return beats_per_second * chunk  # frames per update

def faux_audio_stream(performance_audio_path):
    global audio_stream_start_time

    audio, sr = sf.read(performance_audio_path, dtype='float32')
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert to mono if needed

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
    gt_beat_idx = 0
    sample_idx = 1
    total_samples = audio.shape[0]
    while idx < total_samples:
        chunk = audio[idx:idx+CHUNK]
        if chunk.shape[0] < CHUNK:
            # Pad the last chunk
            chunk = np.pad(chunk, (0, CHUNK - len(chunk)))
        # Put chunk into both queues
        audio_queue_beatnet.put((chunk.astype(np.float32).tobytes(), idx))
        chroma = ChromagramProcessor(sample_rate=BEATNET_SR, hop_length=CHUNK//2)(chunk.astype(np.float32))
        audio_queue_dtw.put((chroma, idx))
        # Play audio through speakers
        stream_out.write(chunk.astype(np.float32).tobytes())
        while gt_beat_idx < len(beat_frames) and beat_frames[gt_beat_idx] < sample_idx:
            timestamp = time.time() - audio_stream_start_time
            gt_beats.append(timestamp)
            gt_beat_idx += 1
        sample_idx += 1
        idx += CHUNK
        #time.sleep(CHUNK / RATE)  # Real-time pacing - mute for audio playback

    stream_out.stop_stream()
    stream_out.close()
    p.terminate()
    folder_path.mkdir(parents=True, exist_ok=True)

    with open(folder_path / "gt_beats.txt", "w") as f:
        for t in gt_beats:
            f.write(f"{t:.6f}\n")

    with open(folder_path / "online_dtw_beats.txt", "w") as f:
        for t in online_dtw_beats:
            f.write(f"{t:.6f}\n")

    with open(folder_path / "beatnet_beats.txt", "w") as f:
        for t in beatnet_beats:
            f.write(f"{t:.6f}\n")

    with open(folder_path / "kalman_beats.txt", "w") as f:
        for t in kalman_beats:
            f.write(f"{t:.6f}\n")

    print("Faux audio stream finished.")

# Audio stream function
def audio_stream():
    global audio_stream_start_time
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=STREAM_RATE, input=True, frames_per_buffer=CHUNK)
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
    BeatTracking_start_time = 0
    try:
        while True:
            output = estimator.process()
            if len(output)==l:
                current_time = time.time()
                l+=1
                beat_signal_queue.put(output[-1])
                if BeatTracking_start_time == 0:
                    BeatTracking_start_time = current_time - output[0][0]*(CHUNK/441)
                # Log BeatNet beat time
                timestamp = current_time - BeatTracking_start_time
                beatnet_beats.append(timestamp)
                #beatnet_beats.append(output[-1][0]*2)
                #print(output[-1][0]*2, current_time - audio_stream_start_time, current_time - BeatTracking_start_time, audio_queue_beatnet.qsize())

    except KeyboardInterrupt:
        print("BeatNet stopped.")


# Online DTW thread
def run_online_dtw():
    #print("Running online DTW with Matchmaker...")
    # Initialize Matchmaker for live alignment
    mm = Matchmaker(
        reference_audio=reference_audio_path,
        performance_file=None,  # Live input, not file
        input_type="audio",
        feature_type=FEATURE_TYPE,  # or "mel", as appropriate
        method="arzt",
        sample_rate=BEATNET_SR,
        frame_rate=BEATNET_SR // CHUNK,
        queue=audio_queue_dtw,  # Use the live audio queue
    )

    print("Starting mm.run() generator...")
    n_frames = reference_features.shape[0]
    audio, sr = librosa.load(reference_audio_path, sr=BEATNET_SR, mono=True)  # Load reference audio for length calculation
    if audio.ndim > 1:
        audio = audio.mean(axis=1)  # Convert to mono if needed

    length_seconds = len(audio) / sr
    last_beat_idx = 0
    start_time = time.time()
    print(f"Online DTW started at {start_time:.6f} (epoch seconds)")
    # Run the alignment process in a loop
    try:
        #track queue draw time interval (출력보다 queue.get이 빠른지 확인)
        for matching_frame in mm.run(verbose=True, wait=True):
            dtw_frame_queue.put(matching_frame)
            # Efficiently log beat times using ordered beat_frames
            while last_beat_idx < len(beat_frames) and beat_frames[last_beat_idx] <= matching_frame:
                last_beat_idx += 1
                timestamp = time.time() - audio_stream_start_time
                online_dtw_beats.append(timestamp)
    except Exception as e:
        print(f"DTW thread exception: {e}")
        traceback.print_exc()
    print(mm.score_follower.warping_path)

    # --- Visualization ---
    # Prepare arguments for plotting
    wp = mm.score_follower.warping_path
    ref_features = mm.score_follower.reference_features
    input_features = mm.score_follower.input_features
    save_dir = Path("./dtw_results")
    save_dir.mkdir(exist_ok=True)
    score_annots = np.array(beat_labels)  # or your reference beat times in seconds
    perf_annots = np.array([])  # If you have ground-truth performance beats, otherwise leave empty or None
    print("Plotting and saving score following result...")
    np.savez("./dtw_results/dtw_data_online_dtw.npz",
             wp=wp,
             ref_features=ref_features,
             input_features=input_features,
             score_annots=score_annots,
             perf_annots=perf_annots,
             beat_labels=beat_labels)
    print("DTW data saved to ./dtw_results/dtw_data_online_dtw.npz")

# Kalman Filter thread
def run_kalman_filter(tempo, beat_frames, converted_fx):
    print("Running Kalman Filter...")
    kf = KalmanFilter()
    # Initial state: [position, speed]
    kf.x_esti = np.array([[0], [tempo]])  # position (frame), speed (BPM)
    last_beat_time = None
    recent_beats = []
    last_fx_check_idx = 0  # To track which fx has been triggered
    last_beat_idx = 0
    start_time = time.time()
    print(f"Kalman Filter started at {start_time:.6f} (epoch seconds)")

    while True:
        # Get the latest DTW matching frame (integer)
        matching_frame = dtw_frame_queue.get()
        if matching_frame is None:
            break  # End thread
        # Try to get a beat signal (list, first element is beat time in seconds)
        is_beat = not beat_signal_queue.empty()
        if is_beat:
            beat_info = beat_signal_queue.get()
            if isinstance(beat_info, list) or isinstance(beat_info, np.ndarray):
                beat_time = beat_info[0]*(CHUNK/441)
            else:
                beat_time = beat_info  # fallback if not a list

            # Find the closest beat label to this beat_time
            closest_beat_idx = np.argmin(np.abs(np.array(beat_labels) - beat_time))
            closest_beat_time = beat_labels[closest_beat_idx]
            closest_beat_frame = beat_frames[closest_beat_idx]

            current_time = time.time()
            # Check beat validity
            if last_beat_time is not None:
                interval = beat_time - last_beat_time
                expected_interval = 60.0 / tempo
                n = round(interval / expected_interval)
                if abs(interval - n * expected_interval) <= (BEAT_TIME_THRESHOLD * expected_interval):
                    # Valid beat
                    last_beat_time = beat_time
                    recent_beats.append(interval)
                    if len(recent_beats) > 5:
                        recent_beats.pop(0)
                        # Update tempo with low-pass filter
                        if np.mean(recent_beats) > 0:
                            new_tempo = (60 / np.mean(recent_beats)) * (1 - TEMPO_UPDATE_ALPHA) + tempo * TEMPO_UPDATE_ALPHA
                            #print(f"[BEAT] Valid beat detected at {beat_time:.2f}s. Updated tempo: {tempo:.2f} -> {new_tempo:.2f} BPM")
                            tempo = new_tempo
                        #else: print(f"[BEAT] Valid beat detected at {beat_time:.2f}s, but mean interval is zero. Tempo unchanged.")
                    # Update Kalman Filter with true beat frame and new tempo
                    kf.update(np.array([[closest_beat_frame], [(tempo)]]))
                    #print(f"[BEAT] Kalman updated with frame {closest_beat_frame}, tempo {tempo:.2f}")
                else:
                    #print(f"[BEAT] Invalid beat. Interval: {interval:.2f}s, Expected: {expected_interval*n:.2f}s")
                    # Still update with predicted position
                    kf.update(np.array([[matching_frame], [tempo]]))
            else:
                if beat_time > 2.0:  # Ignore beats too close to start
                    last_beat_time = beat_time
                    #print(f"[BEAT] First beat at {beat_time:.2f}s")
                    kf.update(np.array([[closest_beat_frame], [tempo]]))
                    #print(f"[BEAT] Kalman initialized with frame {closest_beat_frame}, tempo {tempo:.2f}")
        else:
            # No beat: predict using current matching frame and tempo
            kf.update(np.array([[matching_frame], [tempo]]))
            #print(f"[NO BEAT] Kalman predicted with frame {matching_frame}, tempo {tempo:.2f}")

        # Get current Kalman state
        current_frame = int(kf.get_state()[0, 0])
        
        # Efficiently log beat times using ordered beat_frames
        while last_beat_idx < len(beat_frames) and beat_frames[last_beat_idx] <= current_frame:
            last_beat_idx += 1
            timestamp = time.time() - audio_stream_start_time
            kalman_beats.append(timestamp)
        #if current_frame!=matching_frame: print(f"[KF STATE] True matching frame: {current_frame} | DTW matching frame: {matching_frame} | Tempo: {tempo:.2f} BPM")

        # FX triggering: check if current_frame passes any new fx frame
        while last_fx_check_idx < len(converted_fx) and converted_fx[last_fx_check_idx][0] <= current_frame:
            fx_frame, delay, fx_id = converted_fx[last_fx_check_idx]
            send_message(delay, fx_id)
            last_fx_check_idx += 1

# Start threads
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
kalman_thread.start()

audio_thread.join()
beatnet_thread.join()
dtw_thread.join()
kalman_thread.join()