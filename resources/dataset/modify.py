def normalize_beats(input_file, output_file, min_diff=0.28, max_diff=0.40):
    """
    Normalize beat labels to ensure all beats are between 0.28 and 0.40 seconds apart.
    Args:
        input_file (str): Path to the input beat label file.
        output_file (str): Path to save the normalized beat label file.
        min_diff (float): Minimum allowed difference between consecutive beats.
        max_diff (float): Maximum allowed difference between consecutive beats.
    """
    # Load the beat labels from the file
    with open(input_file, 'r') as f:
        beats = [float(line.strip()) for line in f if line.strip()]

    normalized_beats = [beats[0]]  # Start with the first beat

    for i in range(len(beats) - 1):
        current_beat = beats[i]
        next_beat = beats[i + 1]
        diff = next_beat - current_beat

        # If the difference is within the window, keep the next beat
        if min_diff <= diff <= max_diff:
            normalized_beats.append(next_beat)
        else:
            # Divide the difference and add intermediate beats
            num_divisions = int(diff // max_diff) + 1  # Calculate the number of divisions needed
            step = diff / num_divisions  # Calculate the step size

            # Add intermediate beats
            for j in range(1, num_divisions):
                normalized_beats.append(current_beat + j * step)

            # Add the next beat
            normalized_beats.append(next_beat)

    # Save the normalized beats to the output file
    with open(output_file, 'w') as f:
        for beat in normalized_beats:
            f.write(f"{beat:.6f}\n")

    print(f"Normalized beats saved to {output_file}")


if __name__ == "__main__":
    # Input and output file paths
    input_file = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/resources/dataset/mini_band_harmonix/live/beats_offline/0036_breakingthegirl.txt"
    output_file = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/resources/dataset/mini_band_harmonix/live/beats_GND/0036_breakingthegirl.txt"

    # Normalize the beats
    normalize_beats(input_file, output_file)