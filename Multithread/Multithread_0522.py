import pyaudio
import queue
import threading
import numpy as np
import time
from scipy.io.wavfile import write
from pydub import AudioSegment
import librosa

from BeatNet.BeatNet import BeatNet
from matchmaker.features.audio import MFCCProcessor, MelSpectrogramProcessor, ChromagramProcessor  # Import feature processor
from matchmaker.dp import OnlineTimeWarpingArzt  # Import online DTW

from kalman_class import KalmanFilter  # Import Kalman filter class
from preprocess import (
    preprocess_reference_audio,
    get_reference_beat_label,
    convert_beat_label,
    get_tempo,
    get_fx_label,
    convert_fx_label,
)

# Audio stream settings
CHUNK = 9600  # Samples per frame
FORMAT = pyaudio.paInt16  # 16-bit audio
CHANNELS = 1  # Mono
RATE = 44100  # Sample rate

# Queue for processing audio
audio_queue = queue.Queue()
result_queue = queue.Queue()


# Path to the reference audio file
reference_audio_path = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/resources/dataset/mini_band_harmonix/record/tracks/0004_abc.mp3"
beat_label_path = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/resources/dataset/mini_band_harmonix/record/beats/0004_abc.txt"
fx_label_path = "/path/to/fx_labels.txt"


def preprocess(reference_audio_path, feature_type="mfcc", sample_rate=RATE):
    # Parameters
    hop_length = 512
    delay_threshold = 0.1  # 100ms

    # Preprocessing
    reference_features = preprocess_reference_audio(reference_audio_path, feature_type=feature_type, sample_rate=sample_rate)
    beat_labels = get_reference_beat_label(beat_label_path)
    beat_frames = convert_beat_label(beat_labels, sample_rate = RATE, hop_length = hop_length)
    tempo = get_tempo(beat_labels)
    fx_list = get_fx_label(fx_label_path)
    converted_fx = convert_fx_label(fx_list, beat_labels, delay_threshold = delay_threshold, sample_rate = RATE, hop_length = hop_length)

    # Use the preprocessed data
    print(f"Reference features shape: {reference_features.shape}")
    print(f"Beat frames: {beat_frames}")
    print(f"Tempo: {tempo} BPM")
    print(f"Converted FX labels: {converted_fx}")

# Function for capturing audio
def audio_stream():
    p = pyaudio.PyAudio()

    # Get the default input device index
    try:
        default_input_device = p.get_default_input_device_info()
        print("Default Input Device:", default_input_device)
        input_device_index = default_input_device['index']
        #global CHANNELS  # Declare RATE and CHANNELS as global
        #global RATE  # Declare RATE and CHANNELS as global
        #RATE = int(default_input_device['defaultSampleRate'])
        #CHANNELS = min(CHANNELS, default_input_device['maxInputChannels'])
    except IOError:
        print("No default input device found. Please check your system settings.")
        return

    # Open the audio stream with the default input device
    try:
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK,
                        input_device_index=input_device_index)
    except Exception as e:
        print(f"Error opening audio stream: {e}")
        return

    print("Streaming audio...")

    while True:
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_queue.put(data)  # Send data to processing threads
        except IOError as e:
            print(f"Audio stream error: {e}")

# Worker 1 - Analysis Program A: Run BeatNet in 'stream' mode
def analysis_program_a():
    print("Analysis Program A: Running BeatNet in 'stream' mode...")
    
    # Initialize BeatNet in 'stream' mode
    estimator = BeatNet(1, mode='stream', inference_model='PF', plot=[], thread=False)
    
    try:
        # Start processing audio in 'stream' mode
        estimator.process()
    except KeyboardInterrupt:
        print("Analysis Program A: Stopped by user.")
    except Exception as e:
        print(f"Analysis Program A: Error occurred - {e}")

# Worker 2 - Analysis Program B: Perform online DTW
def analysis_program_b():
    print("Analysis Program B: Performing online DTW...")
    
    # Preprocess the reference audio to extract features
    reference_features = preprocess(reference_audio_path, feature_type="mfcc", sample_rate=RATE)
    print(f"Reference features shape: {reference_features.shape}")
    
    # Initialize the online DTW algorithm
    dtw = OnlineTimeWarpingArzt(reference_features=reference_features, distance_func="Euclidean", frame_rate=RATE / CHUNK)
    
    # Process the input audio stream
    while True:
        audio_data = audio_queue.get()
        if audio_data is None:  # Exit when None is received
            break
        
        # Convert the audio data to a NumPy array
        np_audio = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0  # Normalize to [-1, 1]
        
        # Extract features from the input audio frame
        processor = MFCCProcessor(sample_rate=RATE)
        input_features = processor(np_audio)  # This is likely a 2D array
        
        # Ensure input_features is a 1D array (select the last frame or average across frames)
        if input_features.ndim == 2:
            input_features = input_features[-1]  # Select the last frame
        
        # Normalize input features
        input_features = (input_features - np.mean(input_features)) / np.std(input_features)
        print(f"Input features shape: {input_features.shape}")
        print(f"Input features: {input_features}")
        
        # Perform online DTW
        matching_frame = dtw.step(input_features)
        
        # Handle cases where no match is found
        if matching_frame is None:
            print("No matching frame found for the current input.")
        else:
            print(f"Matching frame in reference audio: {matching_frame}")

# Result Aggregation (Optional, not used in this case)
def result_handler():
    while True:
        result = result_queue.get()
        if result is None:
            break
        print(f"Processed Result: {result}")

preprocess(reference_audio_path="/path/to/reference_audio.mp3")  # Replace with actual path

# Start Threads
audio_thread = threading.Thread(target=audio_stream, daemon=True)
worker_a = threading.Thread(target=analysis_program_a, daemon=True)
worker_b = threading.Thread(target=analysis_program_b, daemon=True)

audio_thread.start()
worker_a.start()
worker_b.start()

# Keep the main thread alive
audio_thread.join()
worker_a.join()
worker_b.join()