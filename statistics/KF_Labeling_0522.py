import numpy as np
import os

def load_label(filepath):
    """Load beat labels from a file."""
    beats = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 1:
                beats.append(float(parts[0]))  # Use the first number (beat position in seconds)
    return np.array(beats)

def kalman_filter(dtw_beats, bt_beats, process_variance=0.01, dtw_variance=0.1, bt_variance=0.05, tolerance_ms=200):
    """
    Combine DTW and BeatTracker beat labels using a Kalman Filter.
    Args:
        dtw_beats: Beat positions from DTW.
        bt_beats: Beat positions from BeatTracker.
        process_variance: Variance of the process noise.
        dtw_variance: Variance of DTW observations.
        bt_variance: Variance of BeatTracker observations.
        tolerance_ms: Tolerance window in milliseconds for matching beats.
    Returns:
        Combined beat labels as a NumPy array.
    """
    # Initialize Kalman Filter variables
    x = 0  # Initial state (beat position)
    P = 1  # Initial state covariance
    Q = process_variance  # Process noise covariance
    R_dtw = dtw_variance  # DTW observation noise covariance
    R_bt = bt_variance  # BeatTracker observation noise covariance
    tolerance = tolerance_ms / 1000.0  # Convert tolerance to seconds

    combined_beats = []
    j = 0  # Pointer for bt_beats

    for i in range(len(dtw_beats)):
        # Predict step
        x_pred = x  # Predicted state (no motion model, so x stays the same)
        P_pred = P + Q  # Predicted covariance

        # Find the closest match in bt_beats within the tolerance window
        while j < len(bt_beats) and bt_beats[j] < dtw_beats[i] - tolerance:
            j += 1

        if j < len(bt_beats) and abs(bt_beats[j] - dtw_beats[i]) <= tolerance:
            # Use the matched BeatTracker beat
            z = bt_beats[j]
            R = R_bt
            j += 1
        else:
            # No match found, use the DTW beat
            z = dtw_beats[i]
            R = R_dtw

        # Update step
        K = P_pred / (P_pred + R)  # Kalman gain
        x = x_pred + K * (z - x_pred)  # Updated state
        P = (1 - K) * P_pred  # Updated covariance

        combined_beats.append(x)

    return np.array(combined_beats)

if __name__ == "__main__":
    # File paths for DTW and BeatTracker beat labels
    beat_path = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/resources/dataset/mini_band_harmonix/live"
    dtw_file = os.path.join(beat_path, "beats_stb_dlnco/0004_abc.txt")
    bt_file = os.path.join(beat_path, "beats_BT_Online/0036_breakingthegirl.txt")

    # Load beat labels
    dtw_beats = load_label(dtw_file)
    bt_beats = load_label(bt_file)

    # Combine beat labels using Kalman Filter
    combined_beats = kalman_filter(dtw_beats, bt_beats)

    # Save the combined beat labels to a file
    output_file = os.path.join(beat_path, "beats_KF/0036_breakingthegirl.txt")  # Replace with actual path
    with open(output_file, 'w') as f:
        for beat in combined_beats:
            f.write(f"{beat:.6f}\n")

    print(f"Combined beat labels saved to {output_file}")