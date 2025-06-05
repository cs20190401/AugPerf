import numpy as np
import pandas as pd
import math

from pathlib import Path
from typing import Literal, Union
from numpy.typing import NDArray
from ..datasetbase import DatasetBase
from ...utils import widen_temporal_events
from ...eventconverters import GTZANConverter
from .....config import Config
from tqdm import tqdm



class GTZANDataset(DatasetBase):
  def __init__(
    self,
    cfg: Config,
  ):
    super().__init__(cfg, 'unseen')

    beat_padding = None
    if cfg.model in ['nobufferallin1']:
      beat_padding = cfg.fps * cfg.buffer_length
    
    fold = cfg.fold
    # Use rglob to search for track_ids in all subdirectories
    track_ids = sorted(t.stem for t in Path(cfg.data.path_track_dir).rglob('*.wav'))
    print(cfg.data.path_track_dir)

    df = pd.read_csv(cfg.data.path_metadata, converters={'File': str})
    df['id'] = df['filename']
    df["BPM"] = pd.to_numeric(df["tempo"], errors='coerce', downcast="float")
    df = df.set_index('id')
    
    self._track_ids = track_ids

    if cfg.data.bpfed:
      self.feature_dir = Path(cfg.data.path_bpf_feature_dir)
      self.num_channels = len(cfg.data.sources)
    elif cfg.data.demixed:
      self.feature_dir = Path(cfg.data.path_feature_dir)
      self.num_channels = cfg.data.num_instruments
    else:
      self.feature_dir = Path(cfg.data.path_no_demixed_feature_dir)
      self.num_channels = 1

    self.df = df

    self.buffer_size = int(cfg.buffer_length * cfg.sample_rate / cfg.hop_size) # 250 frames
    self.specs = dict()

    self.buffer_id = [] # track id
    self.buffer_jd = []
    self.buffer_true_beats = []
    self.buffer_true_downbeats = []
    self.buffer_true_bpm = []
    self.buffer_bpm_int = []
    self.buffer_widen_beats = []
    self.buffer_widen_downbeats = []
    self.buffer_widen_bpm = []
    self.mask = []

    self.xbeat = False
    if cfg.focal_loss or cfg.dice_loss:
      self.xbeat = True
      self.buffer_true_nobeats = []
      self.buffer_widen_nobeats = []

    self.latency = None
    if cfg.latency_t > 0:
      self.latency = int(cfg.latency_t * cfg.sample_rate / cfg.hop_size)

    max_T = 30 * self.sample_rate // self.hop
    max_T = int(math.ceil(max_T/self.cfg.batch_size)*self.cfg.batch_size)

    remove_i = []

    for i in tqdm(range(len(self.track_ids))):
      track_id = self.track_ids[i]
      try:
        spec_full = self.load_features(track_id)
      except:
        remove_i.append(i)
        continue
      row = self.df.loc[track_id+".wav"]
      num_bpm_units = 300
      true_bpm_int = round(row['BPM'])
      true_bpm = np.zeros(num_bpm_units, dtype='float32')
      true_bpm[true_bpm_int] = 1.0
      widen_true_bpm = widen_temporal_events(true_bpm, num_neighbors=2)

      T, F = spec_full.shape[-2:]
      spec_full = spec_full.reshape(self.num_channels,T,F)
      self.specs[track_id] = spec_full
      
      try:
        st = self.create_converter(i, track_id, T, None, None)
      except:
        remove_i.append(i)
        continue

      true_beat = st.beat.of_frames(encode=True)
      true_downbeat = st.downbeat.of_frames(encode=True)
      true_nobeat = 1. - true_beat
      if self.xbeat:
        true_beat = np.clip(true_beat - true_downbeat, a_min=0, a_max=None)

      widen_true_beat = widen_temporal_events(true_beat, num_neighbors=1)
      widen_true_downbeat = widen_temporal_events(true_downbeat, num_neighbors=1)
      if self.xbeat:
        widen_true_nobeat = widen_temporal_events(true_nobeat, num_neighbors=1)
      
      for j in range(T-self.latency if self.latency is not None else T):
        self.buffer_widen_bpm.append(widen_true_bpm)
        self.buffer_true_bpm.append(true_bpm)
        self.buffer_id.append(track_id)
        self.buffer_bpm_int.append(true_bpm_int)
        self.mask.append(1)
        self.buffer_jd.append(j)
        j_latency = j if self.latency is None else j+self.latency
        if beat_padding is not None:
          self.buffer_widen_beats.append(widen_true_beat[j_latency:j_latency+beat_padding])
          self.buffer_true_beats.append(true_beat[j_latency:j_latency+beat_padding])
          self.buffer_widen_downbeats.append(widen_true_downbeat[j_latency:j_latency+beat_padding])
          self.buffer_true_downbeats.append(true_downbeat[j_latency:j_latency+beat_padding])
          if self.xbeat:
            self.buffer_true_nobeats.append(true_nobeat[j_latency:j_latency+beat_padding])
            self.buffer_widen_nobeats.append(widen_true_nobeat[j_latency:j_latency+beat_padding])
        else:
          self.buffer_widen_beats.append(widen_true_beat[j_latency])
          self.buffer_true_beats.append(true_beat[j_latency])
          self.buffer_widen_downbeats.append(widen_true_downbeat[j_latency])
          self.buffer_true_downbeats.append(true_downbeat[j_latency])
          if self.xbeat:
            self.buffer_true_nobeats.append(true_nobeat[j_latency])
            self.buffer_widen_nobeats.append(widen_true_nobeat[j_latency])

      for j in range(max_T-T if self.latency is None else max_T-(T-self.latency)):
        self.buffer_widen_bpm.append(np.zeros(widen_true_bpm.shape,dtype='float32'))
        self.buffer_true_bpm.append(np.zeros(true_bpm.shape,dtype='float32'))
        self.buffer_id.append(track_id)
        self.buffer_bpm_int.append(0)
        self.mask.append(0)
        self.buffer_jd.append(j+T)
        if beat_padding is not None:
          self.buffer_widen_beats.append(np.zeros(beat_padding,dtype='float32'))
          self.buffer_true_beats.append(np.zeros(beat_padding,dtype='float32'))
          self.buffer_widen_downbeats.append(np.zeros(beat_padding,dtype='float32'))
          self.buffer_true_downbeats.append(np.zeros(beat_padding,dtype='float32'))
          if self.xbeat:
            self.buffer_widen_nobeats.append(np.ones(beat_padding,dtype='float32'))
            self.buffer_true_nobeats.append(np.ones(beat_padding,dtype='float32'))
        else:
          self.buffer_widen_beats.append(0)
          self.buffer_true_beats.append(0)
          self.buffer_widen_downbeats.append(0)
          self.buffer_true_downbeats.append(0)
          if self.xbeat:
            self.buffer_widen_nobeats.append(1)
            self.buffer_true_nobeats.append(1)
    
    for i in sorted(remove_i, reverse=True):
      _ = self._track_ids.pop(i)
    print(len(self.track_ids))

  def create_converter(
    self,
    index: int,
    track_id: str,
    num_frames: int,
    start: float,
    end: Union[float, None],
  ) -> GTZANConverter:
    return GTZANConverter(
      track_id=track_id,
      total_frames=num_frames,
      sr=self.sample_rate,
      hop=self.hop,
      start=start,
      end=end,
      base_dir=self.cfg.data.path_base_dir,
    )
  
  def __getitem__(self, idx):
    if hasattr(self, 'buffers'):
      buffer = self.buffers[idx]
    else:
      spec_full = self.specs[self.buffer_id[idx]]
      T, F = spec_full.shape[-2:]
      j = self.buffer_jd[idx]
      if j < T:
        if j >= self.buffer_size-1:
          buffer = spec_full[:,j-self.buffer_size+1:j+1,:]
        else:
          buffer = np.concatenate((np.zeros((self.num_channels, self.buffer_size-j-1, F), dtype=spec_full.dtype),spec_full[:,:j+1,:].reshape(self.num_channels,j+1,F)), axis=1)
      else:
        buffer = np.zeros((self.num_channels,self.buffer_size,F),dtype=spec_full.dtype)

    if self.xbeat:
      if self.latency is not None:
        return dict(
          track_key = self.buffer_id[idx],
          spec = buffer,
          mag = self.buffer_mag[idx],
          mask = self.mask[idx],
          true_beat = self.buffer_true_beats[idx],
          true_downbeat = self.buffer_true_downbeats[idx],
          true_nobeat = self.buffer_true_nobeats[idx],
          widen_true_beat = self.buffer_widen_beats[idx],
          widen_true_downbeat = self.buffer_widen_downbeats[idx],
          widen_true_nobeat = self.buffer_widen_nobeats[idx],
          true_bpm = self.buffer_true_bpm[idx],
          widen_true_bpm = self.buffer_widen_bpm[idx],
          true_bpm_int = self.buffer_bpm_int[idx],
        )
      else:
        return dict(
          track_key = self.buffer_id[idx],
          spec = buffer,
          mask = self.mask[idx],
          true_beat = self.buffer_true_beats[idx],
          true_downbeat = self.buffer_true_downbeats[idx],
          true_nobeat = self.buffer_true_nobeats[idx],
          widen_true_beat = self.buffer_widen_beats[idx],
          widen_true_downbeat = self.buffer_widen_downbeats[idx],
          widen_true_nobeat = self.buffer_widen_nobeats[idx],
          true_bpm = self.buffer_true_bpm[idx],
          widen_true_bpm = self.buffer_widen_bpm[idx],
          true_bpm_int = self.buffer_bpm_int[idx],
        )
    else:
      if self.latency is not None:
        return dict(
          track_key = self.buffer_id[idx],
          spec = buffer,
          mag = self.buffer_mag[idx],
          mask = self.mask[idx],
          true_beat = self.buffer_true_beats[idx],
          true_downbeat = self.buffer_true_downbeats[idx],
          widen_true_beat = self.buffer_widen_beats[idx],
          widen_true_downbeat = self.buffer_widen_downbeats[idx],
          true_bpm = self.buffer_true_bpm[idx],
          widen_true_bpm = self.buffer_widen_bpm[idx],
          true_bpm_int = self.buffer_bpm_int[idx],
        )
      else:
        return dict(
          track_key = self.buffer_id[idx],
          spec = buffer,
          mask = self.mask[idx],
          true_beat = self.buffer_true_beats[idx],
          true_downbeat = self.buffer_true_downbeats[idx],
          widen_true_beat = self.buffer_widen_beats[idx],
          widen_true_downbeat = self.buffer_widen_downbeats[idx],
          true_bpm = self.buffer_true_bpm[idx],
          widen_true_bpm = self.buffer_widen_bpm[idx],
          true_bpm_int = self.buffer_bpm_int[idx],
        )