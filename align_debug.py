import os
import synctoolbox
import numpy as np
import pandas as pd
import librosa
from synctoolbox.dtw.mrmsdtw import sync_via_mrmsdtw
from synctoolbox.feature.chroma import pitch_to_chroma, quantize_chroma
from synctoolbox.feature.pitch import audio_to_pitch_features
from synctoolbox.feature.utils import estimate_tuning
from synctoolbox.feature.dlnco import pitch_onset_features_to_DLNCO
from synctoolbox.feature.novelty import spectral_flux
from synctoolbox.feature.pitch_onset import audio_to_pitch_onset_features

# Define paths
reference_audio_dir = 'resources/dataset/mini_band_harmonix/record/tracks'
new_audio_dir = 'resources/dataset/mini_band_harmonix/live/tracks_cut'
reference_beats_dir = 'resources/dataset/mini_band_harmonix/record/beats'
output_beats_dir = 'resources/dataset/mini_band_harmonix/live/beats'

# Ensure output directory exists
os.makedirs(output_beats_dir, exist_ok=True)

# Function to read beat timestamps
def read_beats(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    beats = [tuple(map(float, line.strip().split())) for line in lines]
    return beats

# Function to write beat timestamps
def write_beats(file_path, beats):
    with open(file_path, 'w') as f:
        for beat in beats:
            f.write(f"{beat[0]:.6f}\t{int(beat[1])}\n")

# Function to get chroma features
def get_chroma_features(audio, tuning_offset, Fs=22050, feature_rate=50):
    f_pitch = audio_to_pitch_features(f_audio=audio, Fs=Fs, tuning_offset=tuning_offset, feature_rate=feature_rate)
    f_chroma = pitch_to_chroma(f_pitch=f_pitch)
    f_chroma_quantized = quantize_chroma(f_chroma=f_chroma)
    return f_chroma_quantized

# Function to get DLNCO features
def get_dlnco_features(audio, tuning_offset, feature_sequence_length, Fs=22050, feature_rate=50):
    f_pitch_onset = audio_to_pitch_onset_features(f_audio=audio, Fs=Fs, tuning_offset=tuning_offset)
    f_dlnco = pitch_onset_features_to_DLNCO(f_peaks=f_pitch_onset, feature_rate=feature_rate, feature_sequence_length=feature_sequence_length)
    return f_dlnco

# Function to get spectral flux features
def get_spectral_flux_features(audio, feature_sequence_length, gamma=10.0, Fs=22050, feature_rate=50):
    f_novelty = spectral_flux(audio, Fs=Fs, feature_rate=feature_rate, gamma=gamma)
    if f_novelty.size < feature_sequence_length:
        diff = feature_sequence_length - f_novelty.size
        pad = int(diff / 2)
        f_novelty = np.concatenate((np.zeros(pad), f_novelty, np.zeros(pad)))
    return f_novelty.reshape(1, -1)

# Set the feature type here
feature_type = 'chroma'  # Change this to 'chroma_dlnco' or 'chroma_sf' as needed

# Select one pair of files to process
ref_file = '0321_wonderwall.mp3'  # Change this to the actual file name
new_file = '0321_wonderwall.mp3'  # Change this to the actual file name

ref_audio_path = os.path.join(reference_audio_dir, ref_file)
new_audio_path = os.path.join(new_audio_dir, new_file)
ref_beats_path = os.path.join(reference_beats_dir, ref_file.replace('.mp3', '.txt'))
output_beats_path = os.path.join(output_beats_dir, new_file.replace('.mp3', '.txt'))

# Read reference beats
ref_beats = read_beats(ref_beats_path)

# Load audio files
ref_audio, sr_ref = librosa.load(ref_audio_path, sr=22050)
new_audio, sr_new = librosa.load(new_audio_path, sr=22050)

# Estimate tuning
tuning_offset_ref = estimate_tuning(ref_audio, sr_ref)
tuning_offset_new = estimate_tuning(new_audio, sr_new)

# Generate features based on the selected feature type
if feature_type == 'chroma':
    f_chroma_ref = get_chroma_features(ref_audio, tuning_offset_ref)
    f_chroma_new = get_chroma_features(new_audio, tuning_offset_new)
    wp = sync_via_mrmsdtw(f_chroma1=f_chroma_ref, f_chroma2=f_chroma_new, input_feature_rate=50, step_weights=np.array([1.5, 1.5, 2.0]), threshold_rec=10**6)
elif feature_type == 'chroma_dlnco':
    f_chroma_ref = get_chroma_features(ref_audio, tuning_offset_ref)
    f_chroma_new = get_chroma_features(new_audio, tuning_offset_new)
    f_dlnco_ref = get_dlnco_features(ref_audio, tuning_offset_ref, f_chroma_ref.shape[1])
    f_dlnco_new = get_dlnco_features(new_audio, tuning_offset_new, f_chroma_new.shape[1])
    wp = sync_via_mrmsdtw(f_chroma1=f_chroma_ref, f_onset1=f_dlnco_ref, f_chroma2=f_chroma_new, f_onset2=f_dlnco_new, input_feature_rate=50, step_weights=np.array([1.5, 1.5, 2.0]), threshold_rec=10**6)
elif feature_type == 'chroma_sf':
    f_chroma_ref = get_chroma_features(ref_audio, tuning_offset_ref)
    f_chroma_new = get_chroma_features(new_audio, tuning_offset_new)
    f_sf_ref = get_spectral_flux_features(ref_audio, f_chroma_ref.shape[1])
    f_sf_new = get_spectral_flux_features(new_audio, f_chroma_new.shape[1])
    wp = sync_via_mrmsdtw(f_chroma1=f_chroma_ref, f_onset1=f_sf_ref, f_chroma2=f_chroma_new, f_onset2=f_sf_new, input_feature_rate=50, step_weights=np.array([1.5, 1.5, 2.0]), threshold_rec=10**6)

# Print the warping path
print("Warping Path (wp):")
print(wp)

print(ref_beats)

# Map reference beats to new audio
wp = np.array(wp)
new_beats = []
for ref_time, beat in ref_beats:
    new_time = np.interp(ref_time, wp[:, 0] / 50, wp[:, 1] / 50)
    new_beats.append((new_time, beat))

# Print the new beats
print("New Beats:")
for beat in new_beats:
    print(f"{beat[0]:.6f}\t{int(beat[1])}")

# Write new beats to file
write_beats(output_beats_path, new_beats)