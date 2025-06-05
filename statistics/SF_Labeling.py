from matchmaker import Matchmaker
import numpy as np
import csv
import os

import matplotlib.pyplot as plt
import pandas as pd
import librosa
from matchmaker.features.audio import SAMPLE_RATE

def plot_warping_path(warping_path, reference_features, performance_features=None, 
                      score_ann=None, perf_ann=None, frame_rate=100, show=True):
    """
    Visualize the DTW warping path and feature matrices.
    
    Parameters:
        warping_path: list or np.ndarray, warping path indices (should be shape (2, N) or list of frame times)
        reference_features: np.ndarray, shape (n_frames, n_features)
        performance_features: np.ndarray, shape (n_frames, n_features), optional
        score_ann: str, path to reference annotation file (optional)
        perf_ann: str, path to performance annotation file (optional)
        frame_rate: int, frame rate for x/y axis ticks
        show: bool, whether to show the plot
    """
    # Prepare warping path indices
    if isinstance(warping_path, list):
        warping_path = np.array(warping_path)
    if warping_path.ndim == 1:
        # If warping_path is a list of times, plot as diagonal
        ref_indices = np.arange(len(warping_path))
        perf_indices = np.arange(len(warping_path))
    else:
        ref_indices, perf_indices = warping_path

    '''
    plt.figure(figsize=(14, 10))

    # Plot reference features
    plt.subplot(211)
    plt.title("Reference Features")
    plt.imshow(reference_features.T, aspect="auto", origin="lower")
    plt.ylabel("Feature Index")
    plt.colorbar(label="Feature Value")
    plt.xticks([])
    plt.grid(False)

    # Plot performance features if provided
    if performance_features is not None:
        plt.subplot(212)
        plt.title("Performance Features")
        plt.imshow(performance_features.T, aspect="auto", origin="lower")
        plt.ylabel("Feature Index")
        plt.xlabel("Frame")
        plt.colorbar(label="Feature Value")
        plt.grid(False)
    else:
        plt.subplot(212)
        plt.title("DTW Warping Path")
        plt.plot(ref_indices, perf_indices, ".", color="purple", alpha=0.7, markersize=3, label="Warping Path")
        plt.xlabel("Reference Frame")
        plt.ylabel("Performance Frame")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    plt.show()
    '''

    # Optionally, plot accumulated distance matrix with warping path and ground-truth labels
    if score_ann is not None and perf_ann is not None:
        # Load annotations
        ref_annots = pd.read_csv(score_ann, delimiter="\t", header=None)[0]
        perf_annots = pd.read_csv(perf_ann, delimiter="\t", header=None)[0]
        # Plot distance matrix and warping path
        if isinstance(warping_path, np.ndarray) and warping_path.ndim == 2:
            n_perf = warping_path.shape[1]
        else:
            n_perf = len(warping_path)
        dist = np.zeros((reference_features.shape[0], n_perf))
        plt.figure(figsize=(10, 10))
        plt.imshow(dist, aspect="auto", origin="lower", interpolation="nearest")
        plt.title("Accumulated distance matrix with warping path & ground-truth labels")
        plt.xlabel("Performance Features in Time (s)")
        plt.ylabel("Reference Features in Time (s)")
        # Plot warping path
        plt.plot(perf_indices, ref_indices, ".", color="purple", alpha=0.5, markersize=3, label="Warping Path")
        # Plot ground-truth labels
        for i, (ref, perf) in enumerate(zip(ref_annots, perf_annots)):
            plt.plot(perf * frame_rate, ref * frame_rate, "x", color="r", alpha=1, markersize=5, label="Ground Truth" if i == 0 else "")
        plt.legend()
        plt.show()

def align_and_transfer_beats(reference_audio, performance_audio, reference_beat_label, output_beat_label):
    """
    Align reference and performance audio using matchmaker and transfer beat labels.

    Parameters:
    - reference_audio (str): Path to the reference audio file.
    - performance_audio (str): Path to the performance audio file.
    - reference_beat_label (str): Path to the reference beat label file (e.g., 0004_abc.txt).
    - output_beat_label (str): Path to save the transferred beat label for the performance audio.
    """
    # Initialize Matchmaker for alignment
    mm = Matchmaker(
        reference_audio=reference_audio,  # Reference audio as the "score"
        performance_file=performance_audio,  # Performance audio as the "input"
        input_type="audio",  # Input type is audio
        feature_type="mel",  # Use MFCC features for alignment
        method="arzt",  # Use the Arzt online DTW method
        sample_rate=48000,  # Sample rate of the audio
        frame_rate=100,  # Frame rate for processing
    )

    # Run the alignment process
    #print("Running alignment...")
    warping_path = []
    try:
        for frame in mm.run(verbose=False, wait=False):
            warping_path.append(frame)
    except Exception as e:
        print(f"Alignment stopped early due to: {e}")
        print(f"Collected {len(warping_path)} frames before interruption.")

    print(f"Generated warping path with {len(warping_path)} frames")

    # Load reference beat labels
    print("Loading reference beat labels...")
    reference_beats = []
    with open(reference_beat_label, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            timestamp, beat_position = float(row[0]), int(row[1])
            reference_beats.append((timestamp, beat_position))
    reference_beats = np.array(reference_beats)

    # Transfer beat labels to performance audio
    print("Transferring beat labels...")
    performance_beats = []
    for ref_timestamp, beat_position in reference_beats:
        # Find the closest frame in the warping path
        closest_frame = np.argmin(np.abs(np.array(warping_path) - ref_timestamp))
        performance_timestamp = warping_path[closest_frame]
        performance_beats.append((performance_timestamp, beat_position))

    # Save the transferred beat labels
    print(f"Saving transferred beat labels to {output_beat_label}...")
    with open(output_beat_label, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for timestamp, beat_position in performance_beats:
            writer.writerow([f"{timestamp:.9f}", beat_position])

    print("Beat label transfer complete.")

    audio, _ = librosa.load(performance_audio, sr=SAMPLE_RATE, mono=True)
    performance_features = mm.processor(audio)

    plot_warping_path(
        warping_path, 
        reference_features = mm.reference_features,
        performance_features = performance_features,
        score_ann=reference_beat_label_path, 
        perf_ann=output_beat_label_path, 
        frame_rate=100
    )

# Example usage
dataset_path = "/Users/wonseonjae/rep/resources/dataset/mini_band_harmonix/trimmed"
reference_audio_path = os.path.join(dataset_path, "record/tracks/0004_abc.mp3")
performance_audio_path = os.path.join(dataset_path, "live/tracks_cut/0004_abc.mp3")
reference_beat_label_path = os.path.join(dataset_path, "record/beats/0004_abc.txt")
output_beat_label_path = os.path.join(dataset_path, "live/beats_mm_mel_arzt/0004_abc.txt")

align_and_transfer_beats(
    reference_audio=reference_audio_path,
    performance_audio=performance_audio_path,
    reference_beat_label=reference_beat_label_path,
    output_beat_label=output_beat_label_path,
)

