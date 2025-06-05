import torch
import os
from typing import Optional
from omegaconf import OmegaConf
from huggingface_hub import hf_hub_download
from .allinone import AllInOne
from .ensemble import Ensemble
from ..typings import PathLike
from ..training.evaluate import load_wandb_run

NAME_TO_FILE = {
  'harmonix-fold0': 'harmonix-fold0-0vra4ys2.pth',
  'harmonix-fold1': 'harmonix-fold1-3ozjhtsj.pth',
  'harmonix-fold2': 'harmonix-fold2-gmgo0nsy.pth',
  'harmonix-fold3': 'harmonix-fold3-i92b7m8p.pth',
  'harmonix-fold4': 'harmonix-fold4-1bql5qo0.pth',
  'harmonix-fold5': 'harmonix-fold5-x4z5zeef.pth',
  'harmonix-fold6': 'harmonix-fold6-x7t226rq.pth',
  'harmonix-fold7': 'harmonix-fold7-qwwskhg6.pth',
}

ENSEMBLE_MODELS = {
  'harmonix-all': [
    'harmonix-fold0',
    'harmonix-fold1',
    'harmonix-fold2',
    'harmonix-fold3',
    'harmonix-fold4',
    'harmonix-fold5',
    'harmonix-fold6',
    'harmonix-fold7',
  ],
  "own-all": [
    'own-fold0',
    'own-fold1',
    'own-fold2',
    'own-fold3',
    'own-fold4',
    'own-fold5',
    'own-fold6',
    'own-fold7',
  ],
}

MY_VERSION = {
  # 'xxxxxxxx',  # Your W&B run ID here.
  "own-fold0": "v0bodnjq", # fold=0
  "own-fold1": "calgmen1", # fold=1
  "own-fold2": "k1fxdtem", # fold=2
  "own-fold3": "xe7hzmez", # fold=3
  "own-fold4": "91scca0a", # fold=4
  "own-fold5": "ng9rdy5k", # fold=5
  "own-fold6": "ortoefkq", # fold=6
  "own-fold7": "ttjn8dvs", # fold=7
}

TEMPO = {
  "tempo-fold0": "pbe8lfgh", # fold=0
  # "tempo-fold1": "calgmen1", # fold=1
  # "tempo-fold2": "k1fxdtem", # fold=2
  # "tempo-fold3": "xe7hzmez", # fold=3
  # "tempo-fold4": "91scca0a", # fold=4
  # "tempo-fold5": "ng9rdy5k", # fold=5
  # "tempo-fold6": "ortoefkq", # fold=6
  # "tempo-fold7": "ttjn8dvs", # fold=7
  "beat-loss-min": "mlkyqcgt",
  "bpm-loss-min" : "d4qk1zas",
  "dbeat-loss-min":"lvuqi90t",
  "dice-focal-min":"at2s5lpx",
  "all-dataset": "7945zej7",
}


def load_pretrained_model(
  model_name: Optional[str] = None,
  cache_dir: Optional[PathLike] = None,
  device=None,
):
  if device is None:
    if torch.cuda.device_count():
      device = 'cuda'
    else:
      device = 'cpu'

  if model_name in ENSEMBLE_MODELS:
    return load_ensemble_model(model_name, cache_dir, device)

  if "own" in model_name and model_name in ['own-fold0','own-fold1','own-fold2','own-fold3','own-fold4','own-fold5','own-fold6','own-fold7','own-all']:
    if model_name in list(MY_VERSION.keys()):
      model_name = MY_VERSION[model_name]
      trainer, _, _ = load_wandb_run(run_id=model_name, run_dir='eval/')
      model = trainer.model.to(device)
      model.eval()
      return model
    elif model_name == "own-all":
      return load_ensemble_model(model_name, cache_dir, device)
  elif model_name in TEMPO:
    model_name = TEMPO[model_name]
    trainer, _, _ = load_wandb_run(run_id=model_name, run_dir='eval/')
    model = trainer.model.to(device)
    model.eval()
    return model
  else:
    model_name = model_name or list(NAME_TO_FILE.keys())[0]
    assert model_name in NAME_TO_FILE, f'Unknown model name: {model_name} (expected one of {list(NAME_TO_FILE.keys())})'

    filename = NAME_TO_FILE[model_name]
    checkpoint_path = hf_hub_download(repo_id='taejunkim/allinone', filename=filename, cache_dir=cache_dir)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = OmegaConf.create(checkpoint['config'])

    model = AllInOne(config).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    return model


def load_ensemble_model(
  model_name: Optional[str] = None,
  cache_dir: Optional[PathLike] = None,
  device=None,
):
  models = []
  for model_name in ENSEMBLE_MODELS[model_name]:
    model = load_pretrained_model(model_name, cache_dir, device)
    models.append(model)

  ensemble = Ensemble(models).to(device)
  ensemble.eval()

  return ensemble
