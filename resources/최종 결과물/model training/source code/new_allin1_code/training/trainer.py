import warnings
import librosa
import numpy as np
import torch.nn.functional as F
import torch
import gc

from typing import Dict, Union
from lightning import LightningModule
from numpy.typing import NDArray
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from timm.optim.optim_factory import create_optimizer_v2 as create_optimizer
from timm.scheduler import create_scheduler
from timm.scheduler.scheduler import Scheduler

from madmom.audio.signal import FramedSignalProcessor, Signal
from madmom.audio.stft import ShortTimeFourierTransformProcessor
from madmom.processors import SequentialProcessor
from madmom.audio.spectrogram import FilteredSpectrogramProcessor, LogarithmicSpectrogramProcessor

from ..models import AllInOne, AllInOneTempo, AllInOneTempoDepthSep, OldAllInOneTempo, RealTimeAllInOne, AllInOneTempo_with_Head
from ..typings import AllInOneOutput, AllInOnePrediction
from ..config import Config
from .lossfunc import FocalLoss, DiceLoss
from ..postprocessing import postprocess_metrical_structure_realtime

# For ignoring following warnings from madmom.evaluation
warnings.filterwarnings('ignore', category=RuntimeWarning, message='Mean of empty slice')
warnings.filterwarnings('ignore', category=UserWarning, message='Not enough beat annotations')
warnings.filterwarnings('ignore', category=UserWarning, message='The epoch parameter')
warnings.filterwarnings('ignore', category=UserWarning, message='no annotated tempo strengths given')


class AllInOneTrainer(LightningModule):
  scheduler: Scheduler

  def __init__(self, cfg: Config):
    super().__init__()
    self.cfg = cfg

    if cfg.model == 'allinone':
      self.model = AllInOne(cfg)
    elif cfg.model == 'allinonetempo':
      if hasattr(cfg, "causal"):
        self.model = AllInOneTempo(cfg)
      else:
        self.model = OldAllInOneTempo(cfg)
    elif cfg.model == 'depthsepcausal':
      self.model = AllInOneTempoDepthSep(cfg)
    elif cfg.model == 'realtimeallinone':
      self.model = RealTimeAllInOne(cfg)
    elif cfg.model == 'nobufferallin1':
      self.model = AllInOneTempo_with_Head(cfg)
    else:
      raise NotImplementedError(f'Unknown model: {cfg.model}')

    self.lr = cfg.lr
    self.temp_losses_scores = dict()

    self.stream_window = np.zeros(cfg.buffer_length*cfg.sample_rate, dtype=np.float32)
    
    frames = FramedSignalProcessor(frame_size=cfg.window_size, fps=int(cfg.sample_rate / cfg.hop_size))
    stft = ShortTimeFourierTransformProcessor()  # caching FFT window
    filt = FilteredSpectrogramProcessor(num_bands=cfg.num_bands, fmin=cfg.fmin, fmax=cfg.fmax, norm_filters=True)
    spec = LogarithmicSpectrogramProcessor(mul=1, add=1)
    self.processor = SequentialProcessor([frames, stft, filt, spec])
    
    self.window_T = int(cfg.buffer_length * cfg.sample_rate) // cfg.hop_size
    act_buffer_size = self.window_T * 2
    self.buffer = np.zeros((act_buffer_size,2), dtype=np.float32)
    self.temp_beat = 0.

  def forward(self, x):
    return self.model(x)
  
  def process(self, x, frame_no:int, device='cpu'): # for inference
    self.stream_window = np.append(self.stream_window[self.cfg.hop_size:],x)
    sig = Signal(self.stream_window, sample_rate=self.cfg.sample_rate, num_channels=1)
    z = self.processor(sig)
    T, F = z.shape
    z = torch.from_numpy(z).float().to(device).reshape(1,1,T,F)
    logits: AllInOneOutput = self.model(z)
    result = postprocess_metrical_structure_realtime(
              logits_beat=logits.logits_beat,
              logits_downbeat=logits.logits_downbeat,
              logits_nobeat=logits.logits_nobeat if hasattr(logits, "logits_nobeat") else None,
              logits_tempo=logits.logits_tempo,
              frame_no=frame_no, temp_beat=self.temp_beat,
              cfg=self.cfg, buffer=self.buffer, frame_T=self.window_T,
            )
    beat_type, tempo, self.temp_beat, self.buffer, beat_times, beat_position, activations = result
    prob_beat = activations[0]
    prob_downbeat = activations[1]
    return beat_type, tempo, beat_times, beat_position, prob_beat, prob_downbeat


  def configure_optimizers(self):
    optimizer = create_optimizer(
      self,
      opt=self.cfg.optimizer,
      lr=self.cfg.lr,
      weight_decay=self.cfg.weight_decay,
    )
    if self.cfg.sched is not None:
      self.scheduler, _ = create_scheduler(self.cfg, optimizer)

    return {
      'optimizer': optimizer,
    }

  def on_train_epoch_end(self) -> None:
    if self.cfg.sanity_check:
      return

    if self.cfg.sched == 'plateau':
      if (self.current_epoch + 1) % self.cfg.validation_interval_epochs == 0:
        optimizer = self.trainer.optimizers[0]
        old_lr = optimizer.param_groups[0]['lr']

        metric = self.trainer.callback_metrics[self.cfg.eval_metric]
        self.scheduler.step(epoch=self.current_epoch + 1, metric=metric)

        new_lr = optimizer.param_groups[0]['lr']
        if new_lr < old_lr:
          print(f'=> The LR is decayed from {old_lr} to {new_lr}. '
                f'Loading the best model: {self.cfg.eval_metric}={self.trainer.checkpoint_callback.best_model_score}')
          # self.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path, cfg=self.cfg)
          self.__class__.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path, cfg=self.cfg)
      elif self.current_epoch + 1 <= self.cfg.warmup_epochs:
        self.scheduler.step(epoch=self.current_epoch + 1)
    else:
      self.scheduler.step(epoch=self.current_epoch + 1)

  def on_validation_epoch_end(self):
    outputs = list(self.temp_losses_scores.values())

    m_losses = []
    m_scores = []

    for batch_size, losses, scores in outputs:
      m_losses.append(losses)
      m_scores.append(scores)

    print()
    print(m_losses)
    print()
    
    loss_keys = []
    for losses in m_losses:
      loss_keys.extend(list(losses.keys()))
    loss_keys = list(set(loss_keys))

    score_keys = []
    for scores in m_scores:
      score_keys.extend(list(scores.keys()))
    score_keys = list(set(score_keys))

    losses, scores = {}, {}

    for loss_key in loss_keys:
      losses[loss_key] = torch.stack([ml[loss_key] for ml in m_losses if loss_key in ml.keys()]).mean()

    for score_key in score_keys:
      scores[score_key] = np.array([ms[score_key] for ms in m_scores if score_key in ms.keys()]).mean()
    
    self.log_dict(losses, sync_dist=True, batch_size=batch_size)
    self.log_dict(scores, sync_dist=True, batch_size=batch_size)

  def training_step(self, batch, batch_idx):
    if isinstance(batch,list):
      batches = batch
    else:
      batches = [batch]

    m_loss = []
    total_losses = []
    outputs_list = []

    for batch in batches:
      # print(batch_idx, list(batch.keys()))
      batch_size = batch['spec'].shape[0]
      outputs: AllInOneOutput = self(batch['spec'])
      outputs_list.append(outputs)
      losses = self.compute_losses(outputs, batch, prefix='train/')
      total_losses.append(losses)
      loss = losses.pop('train/loss')
      m_loss.append(loss)
    
    loss = torch.stack(m_loss).mean()
    self.log('train/loss', loss, prog_bar=True, batch_size=batch_size)
    
    loss_keys = []
    for losses in total_losses:
      loss_keys.extend(list(losses.keys()))
    loss_keys = list(set(loss_keys))
    losses = dict()
    for loss_key in loss_keys:
      losses[loss_key] = torch.stack([tl[loss_key] for tl in total_losses if loss_key in tl.keys()]).mean()
    self.log_dict(losses, batch_size=batch_size)

    if (self.current_epoch + 1) % self.cfg.validation_interval_epochs == 0 or self.cfg.debug:
      m_scores = []
      for i, outputs in enumerate(outputs_list):
        predictions = self.compute_predictions(outputs, mask=None)
        scores = self.compute_metrics(predictions, batches[i], prefix='train/')
        m_scores.append(scores)
      
      score_keys = []
      for scores in m_scores:
        score_keys.extend(list(scores.keys()))
      score_keys = list(set(score_keys))

      scores = dict()
      for score_key in score_keys:
        scores[score_key] = np.array([ms[score_key] for ms in m_scores if score_key in ms.keys()]).mean()

      self.log_dict(scores, sync_dist=True, on_epoch=True, batch_size=batch_size)

      if self.cfg.sanity_check:
        print('\n')
        for k, v in {**losses, **scores}.items():
          print(k, v.item())
        print('\n')
    
    return loss

  def evaluation_step(self, batch, batch_idx, prefix=None, dataloader_idx=0):
    batch_size = batch['spec'].shape[0]
    outputs: AllInOneOutput = self(batch['spec'])
    losses = self.compute_losses(outputs, batch, prefix)
    predictions = self.compute_predictions(outputs)
    scores = self.compute_metrics(predictions, batch, prefix)
    self.temp_losses_scores[dataloader_idx] = (batch_size, losses, scores)
    
    return batch_size, losses, scores

  def validation_step(self, batch, batch_idx, dataloader_idx=0):
    return self.evaluation_step(batch, batch_idx, prefix='val/', dataloader_idx=dataloader_idx)

  def test_step(self, batch, batch_idx):
    return self.evaluation_step(batch, batch_idx, prefix='test/')

  def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
    # assert batch['spec'].shape[0] == 1, 'Batch size must be 1 for prediction'
    
    outputs: AllInOneOutput = self(batch['spec'])

    db_bool = False
    for key in list(batch.keys()):
      if 'downbeat' in key:
        db_bool = True

    predictions = self.compute_predictions(outputs)
    
    return (batch['track_key'],
            batch['mask'], 
            batch['true_beat'], 
            batch['true_downbeat'] if db_bool else None, 
            batch['true_bpm_int'],
            batch['mag'] if predictions.pred_mags is not None else None,
            outputs.logits_beat,
            outputs.logits_downbeat,
            outputs.logits_nobeat,
            predictions,  
          )

  def compute_losses(self, outputs: AllInOneOutput, batch: Dict, prefix: str = None):
    # print(batch.keys())
    loss = 0.0
    losses = {}

    if self.cfg.focal_loss or self.cfg.dice_loss:
      logits = torch.stack([outputs.logits_downbeat,outputs.logits_beat,outputs.logits_nobeat], dim=1)
      logits = torch.softmax(logits, dim=-1)

      if self.cfg.focal_loss:
        labels = torch.stack([batch['widen_true_downbeat'],batch['widen_true_beat'],batch['widen_true_nobeat']],dim=1)
        loss_focal = FocalLoss(reduce=False)(logits, labels)
      
      if self.cfg.dice_loss:
        prob_downbeat, prob_beat, _ = torch.unbind(logits, dim=-1)
        loss_dice_downbeat = DiceLoss(reduce=False)(prob_downbeat, batch['widen_true_downbeat'])
        loss_dice_beat = DiceLoss(reduce=False)(prob_beat, batch['widen_true_beat'])

    db_bool = True if 'true_downbeat' in batch.keys() else False
    
    loss_beat = F.binary_cross_entropy_with_logits(
      outputs.logits_beat, batch['widen_true_beat'],
      reduction='none',
    )
    if self.cfg.model in ['nobufferallin1']:
      loss_beat = loss_beat.mean(dim=1)
    if db_bool:
      loss_downbeat = F.binary_cross_entropy_with_logits(
        outputs.logits_downbeat, batch['widen_true_downbeat'],
        reduction='none',
      )
      if self.cfg.model in ['nobufferallin1']:
        loss_downbeat = loss_downbeat.mean(dim=1)
    loss_tempo = F.cross_entropy(
      outputs.logits_tempo, batch['widen_true_bpm'],
      reduction='none',
    )
    
    loss_beat *= self.cfg.loss_weight_beat
    if db_bool:
      loss_downbeat *= self.cfg.loss_weight_downbeat
    loss_tempo *= self.cfg.loss_weight_tempo
    
    if self.cfg.learn_rhythm:
      loss += loss_beat + loss_tempo
      if db_bool:
        loss += loss_downbeat
    if self.cfg.focal_loss:
      loss_focal *= self.cfg.loss_weight_focal
      loss += loss_focal
    if self.cfg.dice_loss:
      loss_dice_downbeat *= self.cfg.loss_weight_dice_downbeat
      loss += loss_dice_downbeat
      loss_dice_beat *= self.cfg.loss_weight_dice_beat
      loss += loss_dice_beat

    if db_bool:
      losses.update(
        loss=loss.mean(),
        loss_beat=loss_beat.mean(),
        loss_downbeat=loss_downbeat.mean(),
        loss_tempo=loss_tempo.mean(),
      )
    else:
      losses.update(
        loss=loss.mean(),
        loss_beat=loss_beat.mean(),
        loss_tempo=loss_tempo.mean(),
      )
      
    if self.cfg.focal_loss:
      losses.update(loss_focal=loss_focal.mean())
    if self.cfg.dice_loss:
      losses.update(loss_dice_downbeat=loss_dice_downbeat.mean())
      losses.update(loss_dice_beat=loss_dice_beat.mean())

    if prefix:
      losses = prefix_dict(losses, prefix)
    
    return losses

  def compute_predictions(self, outputs: AllInOneOutput, mask=None):
    raw_prob_tempos = torch.softmax(outputs.logits_tempo.detach(),dim=1)
    prob_tempos = raw_prob_tempos.cpu().numpy()
    pred_tempos = np.argmax(prob_tempos, axis=1)

    if outputs.logits_mag is not None:
      raw_mags = torch.sigmoid(outputs.logits_mag.detach())
      pred_mags = raw_mags.cpu().numpy()

    if self.cfg.focal_loss or self.cfg.dice_loss:
      logits = torch.stack([outputs.logits_downbeat, outputs.logits_beat, outputs.logits_nobeat], dim=1).detach()
      raw_prob_downbeats, raw_prob_beats, _ = torch.unbind(torch.softmax(logits, dim=-1),dim=-1)
    else:
      raw_prob_beats = torch.sigmoid(outputs.logits_beat.detach())
      raw_prob_downbeats = torch.sigmoid(outputs.logits_downbeat.detach())

    prob_beats = raw_prob_beats.cpu().numpy()
    prob_downbeats = raw_prob_downbeats.cpu().numpy()

    if mask is not None:
      prob_beats *= mask
      prob_downbeats *= mask

    pred_beats = prob_beats > self.cfg.threshold_beat
    pred_downbeats = prob_downbeats > self.cfg.threshold_downbeat

    if mask is not None:
      pred_tempos = np.where(mask.cpu().numpy(), pred_tempos, -1)
    
    p = AllInOnePrediction(
      raw_prob_beats=raw_prob_beats,
      raw_prob_downbeats=raw_prob_downbeats,
      raw_prob_tempos=raw_prob_tempos,

      prob_beats=prob_beats,
      prob_downbeats=prob_downbeats,
      prob_tempos=prob_tempos,

      pred_beats=pred_beats,
      pred_downbeats=pred_downbeats,
      pred_tempos=pred_tempos,
      pred_mags=pred_mags if outputs.logits_mag is not None else None,
    )

    return p

  def compute_metrics(self, p: AllInOnePrediction, batch: Dict, prefix: str = None):
    db_bool = False
    for key in list(batch.keys()):
      if "downbeat" in key:
        db_bool = True
        break

    score_beat = self.compute_frame_metrics(p.pred_beats, batch['true_beat'])
    if db_bool:
      score_downbeat = self.compute_frame_metrics(p.pred_downbeats, batch['true_downbeat'])

    true_bpm = batch['true_bpm_int'].cpu().numpy()
    pred_bpm = p.pred_tempos
    
    bpm_f1 = f1_score(true_bpm, pred_bpm, average='macro')
    bpm_accuracy = accuracy_score(true_bpm, pred_bpm)

    if db_bool:
      d = dict(
        beat_f1=score_beat['f1_score'],
        beat_precision=score_beat['precision'],
        beat_recall=score_beat['recall'],
        downbeat_f1=score_downbeat['f1_score'],
        downbeat_precision=score_downbeat['precision'],
        downbeat_recall=score_downbeat['recall'],
        bpm_f1=bpm_f1,
        bpm_acc=bpm_accuracy,
        # mag_rmse=mag_rmse,
      )
    else:
      d = dict(
        beat_f1=score_beat['f1_score'],
        beat_precision=score_beat['precision'],
        beat_recall=score_beat['recall'],
        bpm_f1=bpm_f1,
        bpm_acc=bpm_accuracy,
        # mag_rmse=mag_rmse,
      )
    
    if prefix:
      d = prefix_dict(d, prefix)
    return d
  
  def compute_frame_metrics(self, pred_beats, true_beats):
    true_beats = true_beats.cpu().numpy()
    if self.cfg.model in ['nobufferallin1']:
      true_beats = true_beats.flatten()
      pred_beats = pred_beats.flatten()
    accuracy = accuracy_score(true_beats, pred_beats)
    precision = precision_score(true_beats, pred_beats, zero_division=0)
    recall = recall_score(true_beats, pred_beats, zero_division=0)
    f1 = f1_score(true_beats, pred_beats, zero_division=0)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

  def tensor_to_time(self, tensor: Union[torch.Tensor, NDArray]):
    """
    Args:
      tensor: a binary event tensor with shape (batch, frame)
    """
    if torch.is_tensor(tensor):
      tensor = tensor.cpu().numpy()
    batch_size = tensor.shape[0]
    i_examples, i_frames = np.where(tensor)
    times = librosa.frames_to_time(i_frames, sr=self.cfg.sample_rate, hop_length=self.cfg.hop_size)
    times = [times[i_examples == i] for i in range(batch_size)]
    return times

  def on_fit_end(self):
    print('=> Fit ended.')
    if self.trainer.is_global_zero and self.trainer.checkpoint_callback.best_model_path:
      print('=> Loading best model...')
      # self.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path, cfg=self.cfg)
      self.__class__.load_from_checkpoint(self.trainer.checkpoint_callback.best_model_path, cfg=self.cfg)
      print('=> Loaded best model.')


def prefix_dict(d: Dict, prefix: str):
  return {
    prefix + key: value
    for key, value in d.items()
  }
