from lightning import LightningDataModule
from .dataset import GTZANDataset
from .....config import Config

from ..harmonix.datamodule import HarmonixDataModule


class GTZANDataModule(HarmonixDataModule):
  dataset_train: GTZANDataset
  dataset_val: GTZANDataset
  dataset_test: GTZANDataset
  
  def __init__(self, cfg: Config):
    super().__init__(cfg)
  
  def setup(self, stage: str):
    if stage == 'fit':
      self.dataset_train = GTZANDataset(self.cfg, split='train')
    
    if stage in ['fit', 'validate']:
      if self.cfg.sanity_check:
        self.dataset_val = self.dataset_train
      else:
        self.dataset_val = GTZANDataset(self.cfg, split='val')
    
    if stage in ['test', 'predict']:
      if self.cfg.sanity_check:
        self.dataset_test = self.dataset_train
      else:
        self.dataset_test = GTZANDataset(self.cfg)