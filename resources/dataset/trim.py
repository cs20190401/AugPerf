from pydub import AudioSegment
import numpy as np
import librosa
import csv
import os

def save_audio_as_mp3(audio_data, sample_rate, output_path):
    """
    Save audio data as an MP3 file using pydub.

    Parameters:
    - audio_data (numpy.ndarray): The audio data to save.
    - sample_rate (int): The sample rate of the audio.
    - output_path (str): The path to save the MP3 file.
    """
    audio_segment = AudioSegment(
        audio_data.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_data.dtype.itemsize,
        channels=1 if len(audio_data.shape) == 1 else audio_data.shape[0]
    )
    audio_segment.export(output_path, format="mp3")
    print(f"Saved MP3 audio to {output_path}")

def cut_audio_and_beat_labels(reference_audio, performance_audio, reference_beat_label, output_reference_audio, output_performance_audio, output_beat_label, duration=20):
    # Load and cut the reference audio
    print(f"Cutting the first {duration} seconds of the reference audio...")
    ref_audio, ref_sr = librosa.load(reference_audio, sr=None)
    ref_audio_cut = ref_audio[:int(ref_sr * duration)]
    save_audio_as_mp3(ref_audio_cut, ref_sr, output_reference_audio)

    # Load and cut the performance audio
    print(f"Cutting the first {duration} seconds of the performance audio...")
    perf_audio, perf_sr = librosa.load(performance_audio, sr=None)
    perf_audio_cut = perf_audio[:int(perf_sr * duration)]
    save_audio_as_mp3(perf_audio_cut, perf_sr, output_performance_audio)

    # Filter the beat labels
    print(f"Filtering beat labels under {duration} seconds...")
    filtered_beats = []
    with open(reference_beat_label, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            timestamp, beat_position = float(row[0]), int(row[1])
            if timestamp <= duration:
                filtered_beats.append((timestamp, beat_position))

    # Save the filtered beat labels
    with open(output_beat_label, "w") as f:
        writer = csv.writer(f, delimiter="\t")
        for timestamp, beat_position in filtered_beats:
            writer.writerow([timestamp, beat_position])
    print(f"Saved filtered beat labels to {output_beat_label}")

# Example usage
dataset_path = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/resources/dataset/mini_band_harmonix"
reference_audio_path = os.path.join(dataset_path, "record/tracks/0004_abc.mp3")
performance_audio_path = os.path.join(dataset_path, "live/tracks_cut/0004_abc.mp3")
reference_beat_label_path = os.path.join(dataset_path, "record/beats/0004_abc.txt")

output_reference_audio_path = os.path.join(dataset_path, "trimmed/record/tracks/0004_abc_cut.mp3")
output_performance_audio_path = os.path.join(dataset_path, "trimmed/live/tracks/0004_abc_cut.mp3")
output_beat_label_path = os.path.join(dataset_path, "trimmed/record/beats/0004_abc_cut.txt")

cut_audio_and_beat_labels(
    reference_audio=reference_audio_path,
    performance_audio=performance_audio_path,
    reference_beat_label=reference_beat_label_path,
    output_reference_audio=output_reference_audio_path,
    output_performance_audio=output_performance_audio_path,
    output_beat_label=output_beat_label_path,
    duration=20
)