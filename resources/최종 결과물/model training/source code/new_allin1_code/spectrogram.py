import numpy as np
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm
from multiprocessing import Pool
from madmom.audio.signal import FramedSignalProcessor, Signal
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor
from .config import Config
from pydub import AudioSegment
import os
import subprocess
import torchaudio
from madmom.audio.filters import LogarithmicFilterbank
import torch


def extract_spectrograms(demix_paths: List[Path], spec_dir: Path, multiprocess: bool = True):
  todos = []
  spec_paths = []
  os.makedirs(spec_dir, exist_ok=True)

  for src in demix_paths:
    dst = spec_dir / f'{src.name}.npy'
    spec_paths.append(dst)
    if dst.is_file():
      continue
    todos.append((src, dst))

  existing = len(spec_paths) - len(todos)
  print(f'=> Found {existing} spectrograms already extracted, {len(todos)} to extract.')

  if todos:
    # Define a pre-processing chain, which is copied from madmom.
    frames = FramedSignalProcessor(
      frame_size=2048,
      fps=int(44100 / 441)
    )
    stft = ShortTimeFourierTransformProcessor()  # caching FFT window
    filt = FilteredSpectrogramProcessor(
      num_bands=12,
      fmin=30,
      fmax=17000,
      norm_filters=True
    )
    spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
    processor = SequentialProcessor([frames, stft, filt, spec])

    # Process all tracks using multiprocessing.
    if multiprocess:
      pool = Pool()
      map_fn = pool.imap
    else:
      pool = None
      map_fn = map

    iterator = map_fn(_extract_spectrogram, [
      (src, dst, processor)
      for src, dst in todos
    ])
    for _ in tqdm(iterator, total=len(todos), desc='Extracting spectrograms'):
      pass

    if pool:
      pool.close()
      pool.join()

  return spec_paths


def _extract_spectrogram(args: Tuple[Path, Path, SequentialProcessor]):
  src, dst, processor = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  sig_bass = Signal(src / 'bass.wav', num_channels=1)
  sig_drums = Signal(src / 'drums.wav', num_channels=1)
  sig_other = Signal(src / 'other.wav', num_channels=1)
  sig_vocals = Signal(src / 'vocals.wav', num_channels=1)

  spec_bass = processor(sig_bass)
  spec_drums = processor(sig_drums)
  spec_others = processor(sig_other)
  spec_vocals = processor(sig_vocals)

  spec = np.stack([spec_bass, spec_drums, spec_others, spec_vocals])  # instruments, frames, bins

  np.save(str(dst), spec)


def no_demixed_extract_spectrogram(paths: List[Path], spec_dir: Path, cfg:Config, multiprocess: bool = True):
  todos = []
  spec_paths = []

  for path in paths:
    dst = spec_dir / f'{path.stem}.npy'
    spec_paths.append(dst)
    if dst.is_file():
      continue
    todos.append((path, dst))
  
  existing = len(spec_paths) - len(todos)
  print(f'=> Found {existing} spectrograms already extracted, {len(todos)} to extract.')

  if todos:
    # Define a pre-processing chain, which is copied from madmom.
    frames = FramedSignalProcessor(
      frame_size=cfg.window_size,
      fps=int(cfg.sample_rate / cfg.hop_size)
    )
    stft = ShortTimeFourierTransformProcessor()  # caching FFT window
    filt = FilteredSpectrogramProcessor(
      num_bands=cfg.num_bands,
      fmin=cfg.fmin,
      fmax=cfg.fmax,
      norm_filters=True
    )
    spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
    processor = SequentialProcessor([frames, stft, filt, spec])

    # Process all tracks using multiprocessing.
    if multiprocess:
      pool = Pool()
      map_fn = pool.imap
    else:
      pool = None
      map_fn = map

    iterator = map_fn(_no_demixed_extract_spectrogram, [
      (src, dst, processor, cfg)
      for src, dst in todos
    ])
    for _ in tqdm(iterator, total=len(todos), desc='Extracting spectrograms'):
      pass

    if pool:
      pool.close()
      pool.join()

  return spec_paths


def _no_demixed_extract_spectrogram(args: Tuple[Path, Path, SequentialProcessor, Config]):
  src, dst, processor, cfg = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  try:
    if src.suffix == ".mp3":
      sound = AudioSegment.from_mp3(str(src))
      temp_name = f"{src.stem}.wav"
      sound.export(temp_name, format="wav")
    elif src.suffix == ".wav":
      temp_name = str(src)

    sig = Signal(temp_name, sample_rate=cfg.sample_rate if cfg.sample_rate else 22050, num_channels=1)

    if cfg.data.name == "gtzan":
      target_length = 30 * cfg.sample_rate
      sig = np.pad(sig, (0, target_length-len(sig)), mode='constant') if len(sig) < target_length else sig[:target_length]

    spec = processor(sig)
    if src.suffix != ".wav":
      os.remove(temp_name)

    np.save(str(dst), spec)
  except Exception as e:
    # pass
    print(f"Spec Something wrong: {src}, {e}")


class TORCH_SPECT():
  def __init__(self, cfg:Config, sample_rate=22050, win_length=2048, hop_size=441, n_bands=12, fmin=30, fmax=11025, device='cuda'):
    self.device = device
    self.stft = torchaudio.transforms.Spectrogram(n_fft=win_length, hop_length=hop_size, normalized=True, power=1).to(device)
    bin_frequencies = np.linspace(0, sample_rate / 2, win_length // 2 + 1)[:-1]
    self.filterbank = LogarithmicFilterbank(bin_frequencies, num_bands=n_bands, fmin=fmin, fmax=fmax, fref=440.0, norm_filters=True, unique_filters=True)
    self.filterbank = torch.from_numpy(self.filterbank).to(device)
    self.cfg = cfg
    if self.cfg.data.bpfed:
      self.bpfed = self.cfg.data.bpfed
      self.sources = self.cfg.data.sources
      bpf_dir_path = self.cfg.bpf_band_dir
      self.subbands = dict()
      for source in self.sources:
        band_freq = np.load(os.path.join(bpf_dir_path, f"total_{source}_freq.npy"))
        band_energ = np.load(os.path.join(bpf_dir_path, f"total_{source}_energy.npy"))
        freqs = np.linspace(0, self.cfg.sample_rate//2, self.cfg.window_size//2+1)
        average_energy = torch.tensor(np.interp(freqs, band_freq, band_energ)[:-1], dtype=torch.float32).to('cuda')
        average_energy = (average_energy / torch.max(average_energy))
        self.subbands[source] = average_energy

  def process_audio(self, audio):
    spectrogram = self.stft(audio)[:-1,:-1].T
    spec_stack = []
    if self.bpfed:
      for source in self.sources:
        source_spec = spectrogram * self.subbands[source]
        spec_stack.append(torch.matmul(source_spec, self.filterbank))
      spectrogram = torch.stack(spec_stack)
    else:
      spectrogram = torch.matmul(spectrogram, self.filterbank)
    spectrogram = torch.log10(spectrogram * 1.0 + 1.0)
    return spectrogram


def no_demixed_torch_spectrogram(paths: List[Path], spec_dir: Path, cfg:Config):
  todos = []
  spec_paths = []

  for path in paths:
    dst = spec_dir / f'{path.stem}.npy'
    spec_paths.append(dst)
    if dst.is_file():
      continue
    todos.append((path, dst))
  
  existing = len(spec_paths) - len(todos)
  print(f'=> Found {existing} spectrograms already extracted, {len(todos)} to extract.')

  if todos:
    processor = TORCH_SPECT(cfg=cfg,
                            sample_rate=cfg.sample_rate,
                            win_length=cfg.window_size,
                            hop_size=cfg.hop_size,
                            n_bands=cfg.num_bands,
                            fmin=cfg.fmin,
                            fmax=cfg.fmax,
                            device='cuda:0' if torch.cuda.is_available() else 'cpu'
                          )
    for src, dst in tqdm(todos, desc='Extracting spectrograms'):
      _no_demixed_torch_spectrogram((src,dst,processor,cfg))

  return spec_paths


def _no_demixed_torch_spectrogram(args: Tuple[Path, Path, TORCH_SPECT, Config]):
  src, dst, processor, cfg = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  try:
    if src.suffix == ".mp3":
      sound = AudioSegment.from_mp3(str(src))
      temp_name = f"{src.stem}.wav"
      sound.export(temp_name, format="wav")
    elif src.suffix == ".wav":
      temp_name = str(src)

    audio = AudioSegment.from_file(temp_name, format='wav')
    audio = audio.set_frame_rate(cfg.sample_rate).set_channels(1).set_sample_width(2)
    raw_data = np.array(audio.get_array_of_samples()).astype(np.float32) / np.iinfo(np.int16).max

    if cfg.data.name == "gtzan":
      target_length = 30 * cfg.sample_rate
      raw_data = np.pad(raw_data, (0, target_length-len(raw_data)), mode='constant') if len(raw_data) < target_length else raw_data[:target_length]

    raw_data = torch.from_numpy(raw_data).to(processor.device)
    spec = processor.process_audio(raw_data)
    if src.suffix != ".wav":
      os.remove(temp_name)

    np.save(str(dst), spec.cpu().numpy())
  except Exception as e:
    # pass
    print(f"Spec Something wrong: {src}, {e}")


def bpf_torch_spectrogram(paths: List[Path], spec_dir: Path, cfg:Config):
  todos = []
  spec_paths = []

  sources_name = '_'.join(sorted(cfg.data.sources))

  for path in paths:
    dst = spec_dir / f'{path.stem}_{sources_name}.npy'
    spec_paths.append(dst)
    if dst.is_file():
      continue
    todos.append((path, dst))
  
  existing = len(spec_paths) - len(todos)
  print(f'=> Found {existing} spectrograms already extracted, {len(todos)} to extract.')

  if todos:
    processor = TORCH_SPECT(cfg=cfg,
                            sample_rate=cfg.sample_rate,
                            win_length=cfg.window_size,
                            hop_size=cfg.hop_size,
                            n_bands=cfg.num_bands,
                            fmin=cfg.fmin,
                            fmax=cfg.fmax,
                            device='cuda:0' if torch.cuda.is_available() else 'cpu'
                          )
    for src, dst in tqdm(todos, desc='Extracting spectrograms'):
      _bpf_torch_spectrogram((src,dst,processor,cfg))

  return spec_paths


def _bpf_torch_spectrogram(args: Tuple[Path, Path, TORCH_SPECT, Config]):
  src, dst, processor, cfg = args

  dst.parent.mkdir(parents=True, exist_ok=True)

  try:
    spec_list = []
    # for source in cfg.data.sources:
    audio = AudioSegment.from_file(str(src / f"{source}.wav"), format='wav')
    audio = audio.set_frame_rate(cfg.sample_rate).set_channels(1).set_sample_width(2)
    raw_data = np.array(audio.get_array_of_samples()).astype(np.float32) / np.iinfo(np.int16).max

    if cfg.data.name == "gtzan":
      target_length = 30 * cfg.sample_rate
      raw_data = np.pad(raw_data, (0, target_length-len(raw_data)), mode='constant') if len(raw_data) < target_length else raw_data[:target_length]

    raw_data = torch.from_numpy(raw_data).to(processor.device)
    spec = processor.process_audio(raw_data)
    spec_list.append(spec)
    
    spec = torch.stack(spec_list)
    np.save(str(dst), spec.cpu().numpy())
  except Exception as e:
    # pass
    print(f"Spec Something wrong: {src}, {e}")