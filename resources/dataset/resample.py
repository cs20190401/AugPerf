import librosa
import soundfile as sf
import os
from pydub import AudioSegment

audio_folder = "./AugPerf/resources/dataset/mini_band_harmonix/trimmed/record/tracks/"   # Folder containing input audio files
output_folder = "./AugPerf/resources/dataset/mini_band_harmonix/trimmed/record/tracks_441" # Folder to save resampled files

os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(audio_folder):
    if filename.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
        input_path = os.path.join(audio_folder, filename)
        print(f"Processing {input_path}...")
        audio, sr = librosa.load(input_path, sr=48000, mono=False)
        audio_44100 = librosa.resample(audio, orig_sr=sr, target_sr=44100, axis=1)
        temp_wav = os.path.join(output_folder, os.path.splitext(filename)[0] + "_44100_temp.wav")
        sf.write(temp_wav, audio_44100.T, 44100)
        # Convert WAV to MP3 using pydub
        output_mp3 = os.path.join(output_folder, os.path.splitext(filename)[0] + ".mp3")
        sound = AudioSegment.from_wav(temp_wav)
        sound.export(output_mp3, format="mp3")
        os.remove(temp_wav)
        print(f"Saved to {output_mp3}")