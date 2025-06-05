import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from madmom.evaluation import beats
import os
import csv


def load_label(filepath):
    """Load beat labels from a file, handling both single and double number formats."""
    predictions = []
    with open(filepath, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 1:  # Handle both single and double number formats
                predictions.append(float(parts[0]))  # Always take the first number (beat position in seconds)
    return np.array(predictions)

def evaluate_beats(ground_truth, predictions, window=70):
    """Evaluate the accuracy of predicted beats against the ground truth."""
    evaluation = beats.BeatEvaluation(ground_truth, predictions, fmeasure_window=window/1000.0)
    return evaluation

if __name__ == "__main__":
    # Load ground truth and predictions
    beat_path = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/resources/dataset/mini_band_harmonix/live"
    ground_truth_dir = os.path.join(beat_path, "beats_GND")
    predictions_dir = os.path.join(beat_path, "beats_KF")
    output_csv = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/statistics/evaluation_results_KF.csv"

    # Define tolerances in milliseconds
    tolerances = [20, 30, 50, 100, 200, 500]

    # Prepare to store results
    all_results = {tol: {"fmeasure": [], "mean_error": [], "std_error": []} for tol in tolerances}

    # Open CSV file for writing results
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write header
        writer.writerow(
            ["File"]
            + [f"F-measure ({tol}ms)" for tol in tolerances]
            + [f"Mean Error ({tol}ms)" for tol in tolerances]
            + [f"Std Error ({tol}ms)" for tol in tolerances]
        )

        # Iterate through ground truth files
        for gt_file in sorted(os.listdir(ground_truth_dir)):
            gt_path = os.path.join(ground_truth_dir, gt_file)
            pred_path = os.path.join(predictions_dir, gt_file)  # Match by filename

            if os.path.exists(pred_path):  # Check if matching prediction file exists
                # Load ground truth and predictions
                ground_truth = load_label(gt_path)
                predictions = load_label(pred_path)

                # Evaluate for all tolerances
                fmeasures = []
                mean_errors = []
                std_errors = []
                for tol in tolerances:
                    result = evaluate_beats(ground_truth, predictions, tol)
                    fmeasures.append(result.fmeasure)
                    mean_errors.append(result.mean_error)
                    std_errors.append(result.std_error)

                    # Store results for averaging later
                    all_results[tol]["fmeasure"].append(result.fmeasure)
                    all_results[tol]["mean_error"].append(result.mean_error)
                    all_results[tol]["std_error"].append(result.std_error)

                # Write results for this file to CSV
                writer.writerow([gt_file] + fmeasures + mean_errors + std_errors)

    # Calculate and print average metrics for all tolerances
    print("\nAverage Metrics for the Total Dataset:")
    for tol in tolerances:
        avg_fmeasure = np.mean(all_results[tol]["fmeasure"])
        avg_mean_error = np.mean(all_results[tol]["mean_error"])
        avg_std_error = np.mean(all_results[tol]["std_error"])
        print(f"Tolerance {tol}ms:")
        print(f"  Average F-measure: {avg_fmeasure:.4f}")
        print(f"  Average Mean Error: {avg_mean_error:.4f} seconds")
        print(f"  Average Std Error: {avg_std_error:.4f} seconds")