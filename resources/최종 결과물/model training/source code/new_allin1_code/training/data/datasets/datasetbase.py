import numpy as np

from abc import ABC, abstractmethod
from typing import Literal, List, Union, Tuple
from numpy.typing import NDArray
from torch.utils.data import Dataset
from ..utils import widen_temporal_events
from ..eventconverters import DatasetConverter
from ....config import Config


class DatasetBase(Dataset, ABC):

  def __init__(
    self,
    cfg: Config,
    split: Literal['train', 'val', 'test', 'unseen'],
  ):
    if split not in ['train', 'val', 'test', 'unseen']:
      raise ValueError(f'Unknown dataset split: {split}')

    self.cfg = cfg
    self.split = split
    self.sample_rate = cfg.sample_rate
    self.hop = cfg.hop_size
    self.segment_size = cfg.segment_size if split == 'train' else None


  def load_features(self, track_id: str) -> NDArray:
    # if self.cfg.data.bpfed:
    #   sources_name = '_'.join(sorted(self.cfg.data.sources))
    #   return np.load(self.feature_dir / f'{track_id}_{sources_name}.npy')
    # else:
    #   return np.load(self.feature_dir / f'{track_id}.npy')
    return np.load(self.feature_dir / f'{track_id}.npy')
  
  def load_magnitudes(self, track_id: str) -> Tuple[Tuple[float,float],NDArray]:
    with open(self.feature_dir / f'{track_id}.csv', 'r') as f:
      lines = f.readlines()
      global_min, global_max = [float(x) for x in lines[0].split()]
      data = np.array([[float(x) for x in line.split()] for line in lines[1:]], dtype=np.float32)
      return ((global_min,global_max),data)

  @property
  def track_ids(self) -> List[str]:
    return self._track_ids
  
  @property
  def numsongs(self):
    return len(self._track_ids)

  @abstractmethod
  def create_converter(
    self,
    index: int,
    track_id: str,
    num_frames: int,
    start: float,
    end: Union[float, None],
  ) -> DatasetConverter:
    pass

  def __len__(self):
    return len(self.buffer_id)