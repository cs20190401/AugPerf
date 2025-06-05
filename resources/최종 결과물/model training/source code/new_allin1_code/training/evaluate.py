import numpy as np
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import os
import wandb
import librosa
import torch
import json

from pprint import pprint
from typing import List, Tuple, Dict, Mapping
from tqdm import tqdm
from lightning import Trainer
from madmom.evaluation.beats import BeatEvaluation
from sklearn.metrics import f1_score, accuracy_score

from .data import HarmonixDataModule, GTZANDataModule
from .helpers import makeup_wandb_config
from .trainer import AllInOneTrainer
from ..utils import mkpath
from ..config import Config, GTZANConfig, HarmonixConfig
from ..typings import AllInOneOutput, AllInOnePrediction
from ..postprocessing import postprocess_metrical_structure, Postprocessor
from .preprocess import main as preprocess

DEBUG = False
RUN_ID = [
  # 'xxxxxxxx',  # Your W&B run ID here.
]

# SWEEP_ID = ["f6waasoh"]
OUTDIR = 'eval/'


def main():
  global RUN_ID

  for run_id in tqdm(RUN_ID):
    print(f'=> Running evaluation of {run_id}...')
    evaluate(run_id=run_id)


def evaluate(
  run_id=None,
  model: AllInOneTrainer = None,
  trainer: Trainer = None,
  dataset: str = "gtzan",
  save_dir: str = '/home/jongsoo/beat-tracking/result',
  project_name: str = 'hyundai_final'
):
  print('=> Evaluating...')
  if run_id:
    model, cfg, run = load_wandb_run(run_id, run_dir=OUTDIR, project_name=project_name)
    cfg.debug = DEBUG
  else:
    assert model is not None, 'Either run_id or model should be provided'
    assert trainer is not None, 'Trainer should be provided if model is provided'
    cfg = model.cfg
    run = wandb.run

  print('=> Creating data module...')
  if dataset == 'gtzan':
    model_cfg = cfg
    cfg = Config
    cfg.latency_t = model_cfg.latency_t
    cfg.focal_loss = model_cfg.focal_loss
    cfg.dice_loss = model_cfg.dice_loss
    cfg.loss_weight_dice_downbeat = model_cfg.loss_weight_dice_downbeat
    cfg.loss_weight_dice_beat = model_cfg.loss_weight_dice_beat
    cfg.loss_weight_focal = model_cfg.loss_weight_focal
    cfg.model = model_cfg.model
    cfg.causal = model_cfg.causal
    dm = GTZANDataModule(cfg)
    cfg.data = GTZANConfig
    if model_cfg.data.bpfed==True:
      cfg.data.bpfed = True
      cfg.data.sources= model_cfg.data.sources
    else:
      cfg.data.bpfed = False
    preprocess(cfg)
  elif dataset == 'harmonix':
    dm = HarmonixDataModule(cfg)
    cfg.data = HarmonixConfig
  else:
    raise ValueError(f'Unknown dataset: {cfg.data.name}')

  if trainer is None:
    trainer = Trainer(
      accelerator='cpu' if cfg.debug else 'auto',
      devices=1,
    )

  save_dir = os.path.join(save_dir,run_id)

  postprocessor_list = dict()
  th_beat_list = np.linspace(0.0, 0.4, 21).tolist()
  th_downbeat_list = np.linspace(0.0, 0.4, 21).tolist()
  for tb in th_beat_list:
    if tb == 0.0:
      for tdb in th_downbeat_list:
        postprocessor_list[f"{int(tb*100)}_{int(tdb*100)}"] = Postprocessor(threshold_beat=tb, threshold_downbeat=tdb)
    else:
      postprocessor_list[f"{int(tb*100)}_{int(0*100)}"] = Postprocessor(threshold_beat=tb, threshold_downbeat=0)
  
  postprocessor_list['dbn'] = None
    
  print(f'=> Evaluating with thresholds: {cfg.threshold_beat}, {cfg.threshold_downbeat}')
  predict_outputs = trainer.predict(model, datamodule=dm)
  num_tracks = dm.dataset_test.numsongs

  for k, postprocessor in postprocessor_list.items():
    scores, genre_scores = compute_postprocessed_scores(predict_outputs, cfg, num_tracks, postprocessor, prefix='test/')

    print('=> Postprocessed scores on test set:')
    pprint(scores)
    pprint(genre_scores)
    if not cfg.debug:
      run.summary.update(scores)
    os.makedirs(os.path.join(save_dir,k), exist_ok=True)
    with open(f'{save_dir}/{k}/score_{run_id}.json', 'w') as f : 
      json.dump(scores, f, indent=4)
    with open(f'{save_dir}/{k}/genre_score_{run_id}.json', 'w') as f : 
      json.dump(genre_scores, f, indent=4)


def compute_postprocessed_scores(
  predict_outputs: List[Tuple[Dict, AllInOneOutput, AllInOnePrediction]],
  cfg: Config,
  num_tracks: int,
  postprocessor,
  prefix: str = '',
):
  all_scores: List[Mapping[str, float]] = []
  # print(predict_outputs)
  current_track_key = predict_outputs[0][0][0]
  true_beats = []
  true_downbeats = []
  true_bpms = []
  mags = []
  logits_beats = []
  logits_downbeats = []
  logits_nobeats = []
  pred_bpms = []
  pred_mags = []
  
  genres = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]
  genre_scores = {genre:[] for genre in genres}

  with tqdm(total=num_tracks, desc="TRACKS") as pbar:
    for count, (track_key, bmask, true_beat, true_downbeat, true_bpm_int, mag, logits_beat, logits_downbeat, logits_nobeat, pred) in enumerate(predict_outputs):
      assert len(list(set(track_key))) == 1
      mask = bmask == 1
      true_beats.append(true_beat[mask])
      true_downbeats.append(true_downbeat[mask])
      true_bpms.append(true_bpm_int[mask])
      mags.append(mag[mask] if mag is not None else None)
      logits_beats.append(logits_beat[mask])
      logits_downbeats.append(logits_downbeat[mask])
      logits_nobeats.append(logits_nobeat[mask] if logits_nobeat is not None else None)
      pred_bpms.append(pred.pred_tempos[mask.cpu().numpy()])
      pred_mags.append(pred.pred_mags[mask] if pred.pred_mags is not None else None)

      if count == len(predict_outputs)-1 or predict_outputs[count+1][0][0] != current_track_key:
        genre = current_track_key.split('.')[0]
        true_beats = torch.cat(true_beats)
        true_downbeats = torch.cat(true_downbeats)
        true_bpms = torch.cat(true_bpms).cpu().numpy()
        mags = torch.cat(mags).cpu().numpy() if mags[0] is not None else None
        logits_beats = torch.cat(logits_beats, dim=0)
        logits_downbeats = torch.cat(logits_downbeats, dim=0)
        logits_nobeats = torch.cat(logits_nobeats, dim=0) if logits_nobeats[0] is not None else None
        pred_bpms = np.concatenate(pred_bpms)
        pred_mags = np.concatenate(pred_mags) if pred_mags[0] is not None else None

        if postprocessor is None:
          metrical_structure = postprocess_metrical_structure((logits_beats, logits_downbeats, logits_nobeats), cfg)
        else:
          metrical_structure = postprocessor.postprocess((logits_beats, logits_downbeats, logits_nobeats))
        pred_beats = metrical_structure['beats']
        pred_downbeats = metrical_structure['downbeats']
        
        true_beats = torch.nonzero(true_beats == 1).squeeze().cpu().numpy()
        true_downbeats = torch.nonzero(true_downbeats == 1).squeeze().cpu().numpy()
        true_beats = librosa.frames_to_time(np.array(true_beats), sr=cfg.sample_rate, hop_length=cfg.hop_size)
        true_downbeats = librosa.frames_to_time(np.array(true_downbeats), sr=cfg.sample_rate, hop_length=cfg.hop_size)
        eval_beat = BeatEvaluation(pred_beats,true_beats)
        eval_downbeat = BeatEvaluation(pred_downbeats,true_downbeats)
        bpm_f1 = f1_score(true_bpms,pred_bpms,average='macro')
        bpm_accuracy = accuracy_score(true_bpms, pred_bpms)
        if pred_mags is not None:
          mag_rmse = float(np.sqrt(np.mean((pred_mags - mags)**2)))
          scores = {
            'beat/f1': eval_beat.fmeasure,
            'beat/precision': eval_beat.precision,
            'beat/recall': eval_beat.recall,
            'beat/cmlt': eval_beat.cmlt,
            'beat/amlt': eval_beat.amlt,
            'downbeat/f1': eval_downbeat.fmeasure,
            'downbeat/precision': eval_downbeat.precision,
            'downbeat/recall': eval_downbeat.recall,
            'downbeat/cmlt': eval_downbeat.cmlt,
            'downbeat/amlt': eval_downbeat.amlt,
            'bpm/f1': bpm_f1,
            'bpm/acc': bpm_accuracy,
            'mag/rmse': mag_rmse,
          }
        else:
          scores = {
            'beat/f1': eval_beat.fmeasure,
            'beat/precision': eval_beat.precision,
            'beat/recall': eval_beat.recall,
            'beat/cmlt': eval_beat.cmlt,
            'beat/amlt': eval_beat.amlt,
            'downbeat/f1': eval_downbeat.fmeasure,
            'downbeat/precision': eval_downbeat.precision,
            'downbeat/recall': eval_downbeat.recall,
            'downbeat/cmlt': eval_downbeat.cmlt,
            'downbeat/amlt': eval_downbeat.amlt,
            'bpm/f1': bpm_f1,
            'bpm/acc': bpm_accuracy,
          }
        genre_scores[genre].append(scores)
        all_scores.append(scores)

        true_beats = []
        true_downbeats = []
        true_bpms = []
        mags = []
        logits_beats = []
        logits_downbeats = []
        logits_nobeats = []
        pred_bpms = []
        pred_mags = []
        if count != len(predict_outputs)-1:
          current_track_key = predict_outputs[count+1][0][0]
        pbar.update(1)

  avg_scores = {
    f'{prefix}{k}': np.mean([scores[k] for scores in all_scores])
    for k in all_scores[0].keys()
  }
  avg_genre_scores = dict()
  for genre in genres:
    avg_genre_scores[genre] = {
      f'{prefix}{k}': np.mean([scores[k] for scores in genre_scores[genre]])
      for k in genre_scores[genre][0].keys()
    }

  return avg_scores, avg_genre_scores


def compute_postprocessed_scores_step(
  predict_output: Tuple[Dict, AllInOneOutput, AllInOnePrediction],
  cfg: Config,
) -> Mapping[str, float]:
  inputs, outputs, preds = predict_output

  pred_metrical = postprocess_metrical_structure(outputs, cfg)

  eval_beat = BeatEvaluation(pred_metrical['beats'], inputs['true_beat_times'][0])
  eval_downbeat = BeatEvaluation(pred_metrical['downbeats'], inputs['true_downbeat_times'][0])

  scores = {
    'beat/f1': eval_beat.fmeasure,
    'beat/precision': eval_beat.precision,
    'beat/recall': eval_beat.recall,
    'beat/cmlt': eval_beat.cmlt,
    'beat/amlt': eval_beat.amlt,
    'downbeat/f1': eval_downbeat.fmeasure,
    'downbeat/precision': eval_downbeat.precision,
    'downbeat/recall': eval_downbeat.recall,
    'downbeat/cmlt': eval_downbeat.cmlt,
    'downbeat/amlt': eval_downbeat.amlt,
  }

  return scores


def load_wandb_run(
  run_id: str,
  run_dir: str = './wandb',
  project_name: str = 'hyundai_final',
) -> Tuple[AllInOneTrainer, Config, wandb.apis.public.Run]:
  api = wandb.Api()
  run = api.run(f'{project_name}/{run_id}')
  artifact = api.artifact(f'{project_name}/model-{run_id}:latest', type='model')
  artifact_dir = artifact.download()
  checkpoint_path = mkpath(artifact_dir) / 'model.ckpt'
  outdir = mkpath(run_dir) / run_id
  outdir.mkdir(parents=True, exist_ok=True)
  cfg = makeup_wandb_config(run.config)
  model = AllInOneTrainer.load_from_checkpoint(
    checkpoint_path,
    map_location='cpu',
    cfg=cfg,
  )
  return model, cfg, run


if __name__ == '__main__':
  main()
