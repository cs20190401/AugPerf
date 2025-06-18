import numpy as np
import os
import csv
from madmom.evaluation import beats
from itertools import product

def load_label(filepath):
    predictions = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 1:
                predictions.append(float(parts[0]))
    return np.array(predictions)

def evaluate_beats(ground_truth, predictions, window=70):
    evaluation = beats.BeatEvaluation(ground_truth, predictions, fmeasure_window=window/1000.0)
    return evaluation

def find_best_latency(ground_truth, predictions, window=70, search_window_ms=200):
    # If either is empty, fall back to default range
    if len(ground_truth) == 0 or len(predictions) == 0:
        search_range_ms = np.arange(-500, 501, 1)
    else:
        '''
        # Center the search range around the difference of first beats
        first_diff = predictions[0] - ground_truth[0]
        center_ms = int(first_diff * 1000)
        search_range_ms = np.arange(center_ms - search_window_ms, center_ms + search_window_ms + 1, 1)
        '''
        search_range_ms = np.arange(0, (predictions[0] - ground_truth[0])*1000, 1)
    best_f1 = -1
    best_latency = 0
    best_eval = None
    for latency_ms in search_range_ms:
        latency = latency_ms / 1000.0
        shifted = predictions - latency
        eval_result = evaluate_beats(ground_truth, shifted, window)
        if eval_result.fmeasure > best_f1:
            best_f1 = eval_result.fmeasure
            best_latency = latency_ms
            best_eval = eval_result
    return best_latency, best_eval

if __name__ == "__main__":
    # Separate lists of file names and modes
    file_names = [
        "0004_abc",
        "0036_breakingthegirl",
        "0119_gunpowderandlead",
        "0179_moveslikejagger",
        # Add more file names as needed
    ]
    modes = [
        "live",
        "record",
        # Add more modes as needed
    ]

    # Define evaluation windows in ms
    windows = [20, 30, 50, 70, 100, 200]

    # Output CSV for all results
    output_csv = "./results/evaluation_latency_results_all.csv"

    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header with columns ordered by window
        header = ["File", "Mode", "Prediction File", "Best Latency (ms)"]
        for w in windows:
            header += [f"F-measure ({w}ms)", f"Mean Error ({w}ms)", f"Std Error ({w}ms)"]
        writer.writerow(header)

        # Try all combinations of file_name and mode
        for file_name, mode in product(file_names, modes):
            gt_file = f"./AugPerf/resources/dataset/mini_band_harmonix/{mode}/beats/{file_name}.txt"
            pred_folder = f"./results/{file_name}_{mode}/"

            if not os.path.exists(gt_file):
                print(f"Ground truth file not found: {gt_file}")
                continue
            if not os.path.isdir(pred_folder):
                print(f"Prediction folder not found: {pred_folder}")
                continue

            ground_truth = load_label(gt_file)

            for pred_file in sorted(os.listdir(pred_folder)):
                if not pred_file.endswith('.txt'):
                    continue
                pred_path = os.path.join(pred_folder, pred_file)
                if not os.path.isfile(pred_path):
                    continue
                try:
                    predictions = load_label(pred_path)
                except UnicodeDecodeError:
                    print(f"Skipping non-text or corrupted file: {pred_file}")
                    continue
                # Find best latency in a range around the first beat difference
                best_latency, _ = find_best_latency(ground_truth, predictions, window=70, search_window_ms=500)
                shifted = predictions - (best_latency / 1000.0)

                fmeasures = []
                mean_errors = []
                std_errors = []
                for w in windows:
                    eval_result = evaluate_beats(ground_truth, shifted, window=w)
                    fmeasures.append(f"{eval_result.fmeasure:.4f}")
                    mean_errors.append(f"{eval_result.mean_error:.6f}")
                    std_errors.append(f"{eval_result.std_error:.6f}")

                # Write row with columns ordered by window
                row = [file_name, mode, pred_file, best_latency]
                for i in range(len(windows)):
                    row += [fmeasures[i], mean_errors[i], std_errors[i]]
                writer.writerow(row)
                print(f"{file_name} ({mode}) - {pred_file}: Best latency={best_latency} ms, F1@70ms={fmeasures[3]}")

    print(f"\nResults saved to {output_csv}")