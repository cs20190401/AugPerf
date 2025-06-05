import torch
import numpy as np

from madmom.features.downbeats import DBNDownBeatTrackingProcessor
from ..typings import AllInOneOutput
from ..config import Config

np.seterr(divide='ignore')

def postprocess_metrical_structure_minimal(
  logits_beat, logits_downbeat, cfg, buffer,
  logits_tempo=None, logits_nobeat=None,
):
  threshold = cfg.threshold_minimal if hasattr(cfg,"threshold_minimal") else 0.5
  if logits_tempo is not None:
    raw_prob_tempos = torch.softmax(logits_tempo,dim=1)
    tempo = int(torch.argmax(raw_prob_tempos,axis=1).cpu().numpy())
  else:
    tempo = None
  if logits_nobeat is not None:
    logits = torch.stack([logits_downbeat.item(), logits_beat.item(), logits_nobeat.item()])
    prob_downbeat, prob_beat, _ = torch.softmax(logits)
  else:
    prob_beat = torch.sigmoid(logits_beat).item()
    prob_downbeat = torch.sigmoid(logits_downbeat).item()
    activation_no = (2.-prob_beat-prob_downbeat)/2.
    activation_xbeat = max(1e-8,prob_beat-prob_downbeat)
    total = sum([activation_xbeat,prob_downbeat,activation_no])
    prob_beat = activation_xbeat / total
    prob_downbeat = prob_downbeat / total
  activations_combined = np.array([[prob_beat,prob_downbeat]])
  buffer = np.append(buffer[1:,:],activations_combined,axis=0)
  beat_max_index, downbeat_max_index = np.argmax(buffer, axis=0)
  if downbeat_max_index == buffer.shape[0]-1 and buffer[downbeat_max_index,1] > threshold:
    beat_type = 2
  elif beat_max_index == buffer.shape[0]-1 and buffer[beat_max_index,0] > threshold:
    beat_type = 1
  else:
    beat_type = 0

  return beat_type, tempo, buffer


def postprocess_metrical_structure_realtime(
  logits_beat,
  logits_downbeat,
  frame_no:int,
  temp_beat,
  cfg: Config,
  buffer,
  frame_T,
  logits_tempo=None,
  logits_nobeat=None,
  # future=None,
  # latency=None,
):
  postprocessor_downbeat = DBNDownBeatTrackingProcessor(
    beats_per_bar=[3, 4],
    threshold=cfg.best_threshold_downbeat,
    fps=cfg.fps,
  );
  if logits_tempo is not None:
    raw_prob_tempos = torch.softmax(logits_tempo,dim=1)
    tempo = int(torch.argmax(raw_prob_tempos,axis=1).cpu().numpy())
  else:
    tempo = None

  if logits_nobeat is not None:
    logits = torch.cat((logits_downbeat, logits_beat, logits_nobeat))
    prob_downbeat, prob_beat, _ = torch.softmax(logits, dim=0).cpu()
  else:
    prob_beat = torch.sigmoid(logits_beat).item()
    prob_downbeat = torch.sigmoid(logits_downbeat).item()
    activation_no = (2.-prob_beat-prob_downbeat)/2.
    activation_xbeat = max(1e-8,prob_beat-prob_downbeat)
    total = sum([activation_xbeat,prob_downbeat,activation_no])
    prob_beat = activation_xbeat / total
    prob_downbeat = prob_downbeat / total
  activations_combined = np.array([[prob_beat,prob_downbeat]])
  buffer = np.append(buffer[1:,:],activations_combined,axis=0)
  pred_downbeat_times = postprocessor_downbeat(buffer);

  current_time = (frame_no-frame_T+1)*(1./cfg.fps)
  if pred_downbeat_times.size > 0:
    # temptemp = pred_downbeat_times[-1][0]
    
    beats = pred_downbeat_times[:,0]+current_time
    pred_downbeat_times[:, 0] = beats
    if not np.isclose(temp_beat, beats[-1], atol=0.3) and beats[-1]>=temp_beat:
        # print(temptemp)
        beat_type = 2 if pred_downbeat_times[-1][1] == 1. else 1
        # print(f"{frame_no:05} | {tempo:<3} | {beats[-1]:06.02f} | diff:{beats[-1]-temp_beat:05.02f} | {'Downbeat' if beat_type == 2 else 'Beat'}")
        temp_beat = beats[-1]
    else:
      beat_type = 0
  else:
    beat_type = 0

  # if latency is not None and pred_downbeat_times.size > 0:
  #   beat_buffer=pred_downbeat_times[:][1]
  #   if tempo is not None and  future == "tempo":
  #     beat_interval = 60.0 / tempo
  #   elif future == "pattern" and len(beat_buffer)>1:
  #     beat_interval = np.mean(np.diff(beat_buffer))

  #   time_diffs = np.abs(current_time+latency - (temp_beat+np.arange(0,5)*beat_interval))
  #   within_tolerance = np.where(time_diffs<=0.07)[0]
  #   if len(within_tolerance)>0:
  #     closest_beat_index = within_tolerance[0]
  #     last_downbeat_index = len(beat_buffer) - 1 - beat_buffer[::-1].index(1) if 1 in beat_buffer else None
  #     if last_downbeat_index is not None:
  #       expected_downbeat_indices = last_downbeat_index + 4 * np.arange(1, 3)
  #       closest_beat_index += len(beat_buffer)
  #       beat_type = 2 if closest_beat_index in expected_downbeat_indices else 1
  #     else:
  #       beat_type = 1
  #   else:
  #     beat_type = 0

  beat_times = pred_downbeat_times[:,0] - current_time
  beat_position = pred_downbeat_times[:,1]

  return beat_type, tempo, temp_beat, buffer, beat_times, beat_position, activations_combined


def postprocess_metrical_structure(
  result,
  cfg: Config,
):
  beat_result,downbeat_result,nobeat_result = result
  postprocessor_downbeat = DBNDownBeatTrackingProcessor(
    beats_per_bar=[3, 4],
    threshold=cfg.best_threshold_downbeat,
    fps=cfg.fps,
  )

  # raw_prob_tempos = torch.softmax(bpm_result,dim=1)

  if nobeat_result is not None:
    logits = torch.stack((downbeat_result, beat_result, nobeat_result),dim=-1)
    activations_combined = torch.softmax(logits,dim=-1).cpu().numpy()
  else:
    raw_prob_beats = torch.sigmoid(beat_result)
    raw_prob_downbeats = torch.sigmoid(downbeat_result)
    activations_beat = raw_prob_beats
    activations_downbeat = raw_prob_downbeats
    activations_no_beat = 1. - activations_beat
    activations_no_downbeat = 1. - activations_downbeat
    activations_no = (activations_no_beat + activations_no_downbeat) / 2.
    activations_xbeat = torch.maximum(torch.tensor(1e-8), activations_beat - activations_downbeat)
    activations_combined = torch.stack([activations_xbeat, activations_downbeat, activations_no], dim=-1)
    activations_combined /= activations_combined.sum(dim=-1, keepdim=True)
    activations_combined = activations_combined.cpu().numpy()

  pred_downbeat_times = postprocessor_downbeat(activations_combined[:, :2])

  beats = pred_downbeat_times[:, 0]
  beat_positions = pred_downbeat_times[:, 1]
  downbeats = pred_downbeat_times[beat_positions == 1., 0]

  beats = beats.tolist()
  downbeats = downbeats.tolist()
  beat_positions = beat_positions.astype('int').tolist()

  # tempo = torch.argmax(raw_prob_tempos, axis=1).cpu().numpy()

  return {
    'beats': beats,
    'downbeats': downbeats,
    'beat_positions': beat_positions,
    # 'tempo': tempo,
  }