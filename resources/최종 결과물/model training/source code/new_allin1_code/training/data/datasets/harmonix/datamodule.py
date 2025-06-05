from lightning import LightningDataModule
from torch.utils.data import DataLoader
from .dataset import HarmonixDataset
from ..collate import collate_fn
from .....config import Config

import torch
import os

class HarmonixDataModule(LightningDataModule):
  dataset_train: HarmonixDataset
  dataset_val: HarmonixDataset
  dataset_test: HarmonixDataset
  
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
  
  def setup(self, stage: str):
    if stage == 'fit':
      self.dataset_train = HarmonixDataset(self.cfg, split='train')
    
    if stage in ['fit', 'validate']:
      if self.cfg.sanity_check:
        self.dataset_val = self.dataset_train
      else:
        self.dataset_val = HarmonixDataset(self.cfg, split='val')
    
    if stage in ['test', 'predict']:
      if self.cfg.sanity_check:
        self.dataset_test = self.dataset_train
      else:
        self.dataset_test = HarmonixDataset(self.cfg, split='test')
  
  def train_dataloader(self):
    return DataLoader(
      self.dataset_train,
      batch_size=self.cfg.batch_size,
      shuffle=True,
      num_workers=15,
      pin_memory=True,
      worker_init_fn=worker_init_fn,
      prefetch_factor=2,
    )
  
  def val_dataloader(self):
    return DataLoader(
      self.dataset_val,
      batch_size=self.cfg.batch_size*8,
      shuffle=False,
      num_workers=2,
      worker_init_fn=worker_init_fn,
    )
  
  def test_dataloader(self):
    return DataLoader(
      self.dataset_test,
      batch_size=self.cfg.batch_size,
      shuffle=False,
      num_workers=15,
      worker_init_fn=worker_init_fn,
    )
  
  def predict_dataloader(self):
    return self.test_dataloader()
  
def worker_init_fn(worker_id):
    os.sched_setaffinity(0, range(os.cpu_count()))