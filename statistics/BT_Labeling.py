#Make beat labels by Beat Tracking

import os
import numpy as np
from BeatNet.BeatNet import BeatNet
from tqdm import tqdm
import contextlib

# Define paths
harmonix_folder = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/resources/dataset/mini_band_harmonix/live"
tracks_folder = os.path.join(harmonix_folder, "tracks_cut")
output_folder = os.path.join(harmonix_folder, "beats_BT_Online")  # Change this to your desired output directory
batch_size = 10  # Number of songs to process per run

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Function to process a track with BeatNet and save results
def process_and_save_track(track_path, mode, output_folder):
    estimator = BeatNet(1, mode=mode, inference_model='PF', plot=[], thread=False)
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            output = estimator.process(track_path)
    beat_times = output[:, 0]  # Extract beat times
    output_file = os.path.join(output_folder, f"{os.path.basename(track_path).replace('.mp3', f'_{mode}.txt')}")
    np.savetxt(output_file, beat_times, fmt='%.6f')

# Get list of already processed files
processed_files = set(f[:4] for f in os.listdir(output_folder) if f.endswith("_online.txt") or f.endswith("_offline.txt"))

# Process each track in the Harmonix dataset
track_files = [f for f in os.listdir(tracks_folder) if f.endswith(".mp3") and f[:4] not in processed_files][:batch_size]
for track_file in tqdm(track_files, desc="Processing tracks"):
    track_path = os.path.join(tracks_folder, track_file)
    try:
        process_and_save_track(track_path, 'online', output_folder)
        #process_and_save_track(track_path, 'offline', output_folder)
    except Exception as e:
        print(f"Error processing {track_file}: {e}")

print(f"Beat results saved to {output_folder}")