import numpy as np
import os
import librosa

def load_label(filepath):
    """Load beat labels from a file."""
    beats = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 1:
                beats.append(float(parts[0]))  # Use the first number (beat position in seconds)
    return np.array(beats)

def kalman_filter_oltw_bt(warping_path, bt_beats, process_variance=0.01, oltw_variance=0.1, bt_variance=0.05, tolerance_ms=200):
    """
    Combine OLTW warping path and BeatNet beat labels using a Kalman Filter.
    Args:
        warping_path: Warping path from OLTW (list of frame timestamps).
        bt_beats: Beat positions from BeatNet.
        process_variance: Variance of the process noise.
        oltw_variance: Variance of OLTW observations.
        bt_variance: Variance of BeatNet observations.
        tolerance_ms: Tolerance window in milliseconds for matching beats.
    Returns:
        Combined beat labels as a NumPy array.
    """
    # Initialize Kalman Filter variables
    x = 0  # Initial state (beat position)
    P = 1  # Initial state covariance
    Q = process_variance  # Process noise covariance
    R_oltw = oltw_variance  # OLTW observation noise covariance
    R_bt = bt_variance  # BeatNet observation noise covariance
    tolerance = tolerance_ms / 1000.0  # Convert tolerance to seconds

    combined_beats = []
    j = 0  # Pointer for bt_beats

    for i in range(len(warping_path)):
        # Predict step
        x_pred = x  # Predicted state (no motion model, so x stays the same)
        P_pred = P + Q  # Predicted covariance

        # Find the closest match in bt_beats within the tolerance window
        while j < len(bt_beats) and bt_beats[j] < warping_path[i] - tolerance:
            j += 1

        if j < len(bt_beats) and abs(bt_beats[j] - warping_path[i]) <= tolerance:
            # Use the matched BeatNet beat
            z = bt_beats[j]
            R = R_bt
            j += 1
        else:
            # No match found, use the OLTW warping path
            z = warping_path[i]
            R = R_oltw

        # Update step
        K = P_pred / (P_pred + R)  # Kalman gain
        x = x_pred + K * (z - x_pred)  # Updated state
        P = (1 - K) * P_pred  # Updated covariance

        combined_beats.append(x)

    return np.array(combined_beats)

if __name__ == "__main__":
    # File paths for OLTW warping path and BeatNet beat labels
    beat_path = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/resources/dataset/mini_band_harmonix/live"
    oltw_file = os.path.join(beat_path, "beats_stb_dlnco/0179_moveslikejagger.txt")  # Replace with actual OLTW warping path file
    bt_file = os.path.join(beat_path, "beats_BT_Online/0179_moveslikejagger.txt")

    # Load OLTW warping path and BeatNet beat labels
    warping_path = load_label(oltw_file)
    bt_beats = load_label(bt_file)

    # Combine beat labels using Kalman Filter
    combined_beats = kalman_filter_oltw_bt(warping_path, bt_beats)

    # Save the combined beat labels to a file
    output_file = os.path.join(beat_path, "beats_KF/0179_moveslikejagger.txt")  # Replace with actual path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        for beat in combined_beats:
            f.write(f"{beat:.6f}\n")

    print(f"Combined beat labels saved to {output_file}")