import librosa
import numpy as np
import os

def preprocess_reference_audio(reference_audio_path, feature_type="mel", sample_rate=48000, mono=True):
    """
    Load the reference audio and extract its audio features.
    """
    print(f"Preprocessing reference audio: {reference_audio_path}")
    
    # Load the reference audio file
    reference_audio, _ = librosa.load(reference_audio_path, sr=sample_rate, mono=mono)
    
    # Extract features based on the specified feature type
    if feature_type == "mfcc":
        processor = librosa.feature.mfcc
        reference_features = processor(y=reference_audio, sr=sample_rate, n_mfcc=13)
    elif feature_type == "mel":
        processor = librosa.feature.melspectrogram
        reference_features = processor(y=reference_audio, sr=sample_rate)
    elif feature_type == "chroma":
        processor = librosa.feature.chroma_stft
        reference_features = processor(y=reference_audio, sr=sample_rate)
    else:
        raise ValueError(f"Unsupported feature type: {feature_type}")
    
    reference_features = reference_features.T  # Transpose to match expected shape
    print(f"Extracted {feature_type} features from reference audio.")
    return reference_features


def get_reference_beat_label(beat_label_path):
    """
    Read the beat label of the reference audio from a text file.
    """
    print(f"Reading beat labels from: {beat_label_path}")
    beat_labels = []
    with open(beat_label_path, "r") as f:
        for line in f:
            parts = line.strip().split("\t")  # Split the line by tab
            timestamp = float(parts[0])  # Take the first value as the timestamp
            beat_labels.append(timestamp)
    print(f"Loaded {len(beat_labels)} beat labels.")
    return np.array(beat_labels)


def convert_beat_label(beat_labels, sample_rate, hop_length):
    """
    Convert beat label timestamps to frame numbers.
    """
    print("Converting beat labels to frame numbers...")
    beat_frames = librosa.time_to_frames(beat_labels, sr=sample_rate, hop_length=hop_length)
    print(f"Converted {len(beat_frames)} beat labels to frame numbers.")
    return beat_frames


def get_tempo(beat_labels):
    """
    Measure the tempo of the reference audio using the beat labels.
    """
    print("Calculating tempo from beat labels...")
    if len(beat_labels) < 2:
        raise ValueError("Not enough beat labels to calculate tempo.")
    inter_beat_intervals = np.diff(beat_labels)
    tempo = 60.0 / np.mean(inter_beat_intervals)  # Convert interval to BPM
    print(f"Calculated tempo: {tempo:.2f} BPM")
    return tempo


def get_fx_label(fx_label_path):
    """
    Read the FX_list from a text file.
    """
    print(f"Reading FX labels from: {fx_label_path}")
    fx_list = []
    with open(fx_label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            timestamp = float(parts[0])
            fx_id = int(parts[1])
            fx_list.append((timestamp, fx_id))
    print(f"Loaded {len(fx_list)} FX labels.")
    return fx_list


def convert_fx_label(fx_list, beat_labels, delay_threshold, sample_rate, hop_length):
    """
    Convert FX labels to (frame#, delay) using the beat labels.
    """
    print("Converting FX labels to frame numbers and delays...")
    converted_fx = []
    for timestamp, fx_id in fx_list:
        # Find the last beat before (timestamp - delay_threshold)
        target_time = timestamp - delay_threshold
        previous_beats = beat_labels[beat_labels <= target_time]
        if len(previous_beats) == 0:
            print(f"No beat found before timestamp {timestamp} for FX ID {fx_id}. Skipping.")
            continue
        last_beat = previous_beats[-1]
        frame_number = librosa.time_to_frames([last_beat], sr=sample_rate, hop_length=hop_length)[0]
        delay = timestamp - last_beat
        converted_fx.append((frame_number, delay, fx_id))
    print(f"Converted {len(converted_fx)} FX labels.")
    return converted_fx