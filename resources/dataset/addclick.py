import os
from pydub import AudioSegment
from pydub.generators import Sine

# Define paths directly
INPUT_PATH = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/resources/dataset/mini_band_harmonix/live/tracks_cut/0119_gunpowderandlead.mp3"
OUTPUT_PATH = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/resources/dataset/mini_band_harmonix/live/tracks_cut_click_offline"
BEAT_LABEL_PATH = "/Users/wonseonjae/Desktop/MacBook_Pro/KAIST/2025_1_URP/rep/resources/dataset/mini_band_harmonix/live/beats_GND"

# Ensure the output directory exists
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Generate a click sound
click_sound = Sine(1000).to_audio_segment(duration=10)  # 10ms click sound

def process_audio_file(audio_path):
    """Process a single audio file and its corresponding beat label file."""
    audio_name = os.path.splitext(os.path.basename(audio_path))[0]
    label_path = os.path.join(BEAT_LABEL_PATH, f"{audio_name}.txt")
    output_path = os.path.join(OUTPUT_PATH, f"{audio_name}.mp3")

    # Check if the corresponding label file exists
    if not os.path.exists(label_path):
        print(f"Warning: No label file found for {audio_path}. Skipping.")
        return

    # Load the input audio
    input_audio = AudioSegment.from_file(audio_path)

    # Create an empty audio segment for the metronome track
    metronome_track = AudioSegment.silent(duration=len(input_audio))

    # Read the beat label file
    with open(label_path, "r") as f:
        beat_positions = [float(line.strip()) for line in f.readlines()]

    # Add click sounds to the metronome track based on the beat positions
    for beat_time in beat_positions:
        frame_position = int(beat_time * 1000)  # Convert seconds to milliseconds
        metronome_track = metronome_track.overlay(click_sound, position=frame_position)

    # Adjust the volume of the input track and the metronome track
    input_audio = input_audio - 2  # Decrease volume (dB)
    metronome_track = metronome_track + 2  # Increase volume (dB)

    # Mix the metronome track with the input audio
    mixed_audio = input_audio.overlay(metronome_track)

    # Save the mixed audio to the output directory
    mixed_audio.export(output_path, format="mp3")
    print(f"Processed and saved: {output_path}")

def process_folder(input_folder):
    """Process all audio files in the given folder."""
    for audio_file in os.listdir(input_folder):
        if audio_file.endswith(".mp3"):  # Process only .mp3 files
            audio_path = os.path.join(input_folder, audio_file)
            process_audio_file(audio_path)

def main():
    """Main function to handle both folders and individual files."""
    if os.path.isdir(INPUT_PATH):
        process_folder(INPUT_PATH)
    elif os.path.isfile(INPUT_PATH) and INPUT_PATH.endswith(".mp3"):
        process_audio_file(INPUT_PATH)
    else:
        print(f"Invalid path or unsupported file type: {INPUT_PATH}")

if __name__ == "__main__":
    main()