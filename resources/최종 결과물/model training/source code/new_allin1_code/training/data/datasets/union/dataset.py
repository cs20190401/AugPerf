import numpy as np
import pandas as pd
import math

from pathlib import Path
from typing import Literal, Union
from numpy.typing import NDArray
from ..datasetbase import DatasetBase
from ...utils import widen_temporal_events
from ...eventconverters import UnionConverter
from .....config import Config
from tqdm import tqdm


class UnionDataset(DatasetBase):
  def __init__(
    self,
    cfg: Config,
    split: Literal['train', 'val', 'test'],
    smc: bool = False,
  ):
    super().__init__(cfg, split)

    beat_padding = None
    if cfg.model in ['nobufferallin1']:
      beat_padding = cfg.fps * cfg.buffer_length
    
    self.smc = smc
    fold = cfg.fold
    if fold < cfg.total_folds or fold >= 0:
      test_fold = fold
      val_fold = (fold + 1) % cfg.total_folds

    track_ids = []
    if split != "test":
      if smc:
        name_list = ["SMC"]
      else:
        name_list = [name for name in cfg.data.names if name != "SMC"]
    else:
      name_list = ["GTZAN"]

    # name_list = cfg.data.names if split != "test" else [cfg.data.test_name]
    for name in name_list:
      print(f"Initializing {name} Dataset...")
      # collecting track ids
      if name == "harmonix":
        temp_track_ids = sorted(t.stem for t in Path(cfg.data.path_base_dir + name + cfg.data.path_track_dir).rglob('*.mp3'))
      else:
        temp_track_ids = sorted(t.stem for t in Path(cfg.data.path_base_dir + name + cfg.data.path_track_dir).rglob('*.wav'))

      
      folds = np.arange(len(temp_track_ids)) % cfg.total_folds
      if fold >= cfg.total_folds or split == 'test':
        pass
      elif split == 'train':
        temp_track_ids = [tid for tid, fold in zip(temp_track_ids, folds) if fold not in [test_fold, val_fold]]
      elif split == 'val':
        temp_track_ids = [tid for tid, fold in zip(temp_track_ids, folds) if fold == val_fold]
      else:
        raise ValueError(f'Unknown dataset split: {split}')

      # collecting tempo information
      if name == "harmonix":
        df = pd.read_csv(cfg.data.path_base_dir+name+cfg.data.path_metadata)
        df['id'] = df['File'].str.split('_').str[0]
        df = df.set_index('id')
        self.df = df
      elif name == "GTZAN":
        df = pd.read_csv(cfg.data.path_base_dir+name+cfg.data.path_metadata, converters={'File': str})
        df['id'] = df['filename']
        df["BPM"] = pd.to_numeric(df["tempo"], errors='coerce', downcast="float")
        df = df.set_index('id')
        self.df = df

      if cfg.data.bpfed:
        self.feature_dir = (Path(cfg.data.path_bpf_feature_dir))
      elif cfg.data.demixed:
        self.feature_dir = Path(cfg.data.path_feature_dir)
      else:
        self.feature_dir = Path(cfg.data.path_base_dir + name + cfg.data.path_no_demixed_feature_dir)

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
      # self.buffer_smc = [] # 0:false, 1:true

      self.xbeat = False
      if cfg.focal_loss or cfg.dice_loss:
        self.xbeat = True
        self.buffer_true_nobeats = []
        self.buffer_widen_nobeats = []

      num_channels = cfg.data.num_instruments if cfg.data.demixed or cfg.data.bpfed else 1
      self.num_channels = num_channels

      self.latency = None
      if cfg.latency_t > 0:
        self.latency = int(cfg.latency_t * cfg.sample_rate / cfg.hop_size)

      remove_i = []

      if split in ['test', "unseen"]:
        max_T = 30 * self.sample_rate // self.hop
        max_T = int(math.ceil(max_T/self.cfg.batch_size)*self.cfg.batch_size)

      for i in tqdm(range(len(temp_track_ids))):
        track_id = temp_track_ids[i]
        try:
          spec_full = self.load_features(track_id)
        except Exception as e:
          print("Spec error:", track_id, e)
          remove_i.append(i)
          continue

        T, F = spec_full.shape[-2:]

        try:
          st = self.create_converter(i, track_id, T, None, None, name)
        except Exception as e:
          # print(f"Converter: {e}")
          remove_i.append(i)
          continue

        num_bpm_units = 300
        if name in ['harmonix', 'GTZAN']:
          row = self.df.loc[track_id.split('_')[0]]
          true_bpm_int = row['BPM']
        else:
          # In cases of datasets except the Harmonix and the GTZAN, calculate true BPM using beat annotations
          beat_times = st.df_beat['time'].values
          true_bpm_int = int(60 / np.mean(np.diff(beat_times)))

        true_bpm = np.zeros(num_bpm_units, dtype='float32')
        true_bpm[true_bpm_int] = 1.0
        widen_true_bpm = widen_temporal_events(true_bpm, num_neighbors=2)
        
        spec_full = spec_full.reshape(num_channels,T,F)
        self.specs[track_id] = spec_full

        true_beat = st.beat.of_frames(encode=True)
        if beat_padding is not None:
          true_beat = np.append(np.zeros(beat_padding-1,dtype='float32'),true_beat)
        if name != "SMC":
          true_downbeat = st.downbeat.of_frames(encode=True)
          if beat_padding is not None:
            true_downbeat = np.append(np.zeros(beat_padding-1,dtype='float32'),true_downbeat)
        if self.xbeat:
          if name != "SMC":
            true_beat = np.clip(true_beat - true_downbeat, a_min=0, a_max=None)
          true_nobeat = 1. - true_beat

        widen_true_beat = widen_temporal_events(true_beat, num_neighbors=1)
        if name != "SMC":
          widen_true_downbeat = widen_temporal_events(true_downbeat, num_neighbors=1)
        if self.xbeat:
          widen_true_nobeat = widen_temporal_events(true_nobeat, num_neighbors=1)

        for j in range(T-self.latency if self.latency is not None else T):
          # self.buffer_smc.append(1 if name=="SMC" else 0)
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
            if name!="SMC":
              self.buffer_widen_downbeats.append(widen_true_downbeat[j_latency:j_latency+beat_padding])
              self.buffer_true_downbeats.append(true_downbeat[j_latency:j_latency+beat_padding])
            if self.xbeat:
              self.buffer_true_nobeats.append(true_nobeat[j_latency:j_latency+beat_padding])
              self.buffer_widen_nobeats.append(widen_true_nobeat[j_latency:j_latency+beat_padding])
          else:
            self.buffer_widen_beats.append(widen_true_beat[j_latency])
            self.buffer_true_beats.append(true_beat[j_latency])
            if name!="SMC":
              self.buffer_widen_downbeats.append(widen_true_downbeat[j_latency])
              self.buffer_true_downbeats.append(true_downbeat[j_latency])
            if self.xbeat:
              self.buffer_true_nobeats.append(true_nobeat[j_latency])
              self.buffer_widen_nobeats.append(widen_true_nobeat[j_latency])
        if split in ["test", "unseen"]:
          for j in range(max_T-T if self.latency is None else max_T-(T-self.latency)):
            # self.buffer_smc.append(1 if name=="SMC" else 0)
            self.buffer_widen_bpm.append(np.zeros(widen_true_bpm.shape,dtype='float32'))
            self.buffer_true_bpm.append(np.zeros(true_bpm.shape,dtype='float32'))
            self.buffer_id.append(track_id)
            self.buffer_bpm_int.append(0)
            self.mask.append(0)
            self.buffer_jd.append(j+T)
            if beat_padding is not None:
              self.buffer_widen_beats.append(np.zeros(beat_padding,dtype='float32'))
              self.buffer_true_beats.append(np.zeros(beat_padding,dtype='float32'))
              if name!="SMC":
                self.buffer_widen_downbeats.append(np.zeros(beat_padding,dtype='float32'))
                self.buffer_true_downbeats.append(np.zeros(beat_padding,dtype='float32'))
              if self.xbeat:
                self.buffer_widen_nobeats.append(np.ones(beat_padding,dtype='float32'))
                self.buffer_true_nobeats.append(np.ones(beat_padding,dtype='float32'))
            else:
              self.buffer_widen_beats.append(0)
              self.buffer_true_beats.append(0)
              if name!="SMC":
                self.buffer_widen_downbeats.append(0)
                self.buffer_true_downbeats.append(0)
              if self.xbeat:
                self.buffer_widen_nobeats.append(1)
                self.buffer_true_nobeats.append(1)

      for i in sorted(remove_i, reverse=True):
        _ = temp_track_ids.pop(i)
      track_ids.extend(temp_track_ids)
      print(len(track_ids))

    self._track_ids = track_ids

  def create_converter(
    self,
    index: int,
    track_id: str,
    num_frames: int,
    start: float,
    end: Union[float, None],
    name: str,
  ) -> UnionConverter:
    return UnionConverter(
      track_id=track_id,
      total_frames=num_frames,
      sr=self.sample_rate,
      hop=self.hop,
      start=start,
      end=end,
      base_dir=self.cfg.data.path_base_dir,
      name=name,
    )
  
  def __getitem__(self, idx):
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
    # smc_bool = self.buffer_smc[idx]
    if self.smc:
      if self.xbeat:
        return dict(
          track_key = self.buffer_id[idx],
          spec = buffer,
          mask = self.mask[idx],
          true_beat = self.buffer_true_beats[idx],
          true_nobeat = self.buffer_true_nobeats[idx],
          widen_true_beat = self.buffer_widen_beats[idx],
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
          widen_true_beat = self.buffer_widen_beats[idx],
          true_bpm = self.buffer_true_bpm[idx],
          widen_true_bpm = self.buffer_widen_bpm[idx],
          true_bpm_int = self.buffer_bpm_int[idx],
        )
    else:
      if self.xbeat:
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
