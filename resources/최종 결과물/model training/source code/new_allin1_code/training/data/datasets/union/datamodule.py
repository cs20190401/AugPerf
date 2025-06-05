from lightning import LightningDataModule
from torch.utils.data import DataLoader
from .dataset import UnionDataset
from ..collate import collate_fn
from .....config import Config

import os

class UnionDataModule(LightningDataModule):
  dataset_train_w_downbeat: UnionDataset
  dataset_train_wo_downbeat: UnionDataset
  dataset_val_w_downbeat: UnionDataset
  dataset_val_wo_downbeat: UnionDataset
  dataset_test: UnionDataset
  
  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg
  
  def setup(self, stage: str):
    if stage == 'fit':
      print("Train Setup...")
      self.dataset_train_w_downbeat = UnionDataset(self.cfg, split='train')
      self.dataset_train_wo_downbeat = UnionDataset(self.cfg, split='train', smc=True)
    
    if stage in ['fit', 'validate']:
      if self.cfg.sanity_check:
        self.dataset_val_w_downbeat = self.dataset_train_w_downbeat
        self.dataset_val_wo_downbeat = self.dataset_train_wo_downbeat
      else:
        self.dataset_val_w_downbeat = UnionDataset(self.cfg, split='val')
        self.dataset_val_wo_downbeat = UnionDataset(self.cfg, split='val', smc=True)
    
    if stage in ['test', 'predict']:
      if self.cfg.sanity_check:
        self.dataset_test = self.dataset_train_w_downbeat
      else:
        self.dataset_test = UnionDataset(self.cfg, split='test')
  
  def train_dataloader(self):
    return [DataLoader(
      self.dataset_train_w_downbeat,
      batch_size=self.cfg.batch_size,
      shuffle=True,
      num_workers=15,
      pin_memory=True,
      worker_init_fn=worker_init_fn,
      prefetch_factor=2,
    ), DataLoader(
      self.dataset_train_wo_downbeat,
      batch_size=self.cfg.batch_size,
      shuffle=True,
      num_workers=15,
      pin_memory=True,
      worker_init_fn=worker_init_fn,
      prefetch_factor=2,
    )]
  
  def val_dataloader(self):
    return [DataLoader(
      self.dataset_val_w_downbeat,
      batch_size=self.cfg.batch_size*8,
      shuffle=False,
      num_workers=2,
      worker_init_fn=worker_init_fn,
    ), DataLoader(
      self.dataset_val_wo_downbeat,
      batch_size=self.cfg.batch_size*8,
      shuffle=False,
      num_workers=2,
      worker_init_fn=worker_init_fn,
    )]
  
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