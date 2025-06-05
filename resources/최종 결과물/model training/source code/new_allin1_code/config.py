from typing import List, Optional, Any
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

HARMONIX_LABELS = [
  'start',
  'end',
  'intro',
  'outro',
  'break',
  'bridge',
  'inst',
  'solo',
  'verse',
  'chorus',
]


@dataclass
class DataConfig:
  name: str

  demixed: bool
  num_instruments: int
  num_labels: int

  path_base_dir: str
  path_track_dir: str
  path_demix_dir: str
  path_feature_dir: str
  path_no_demixed_feature_dir: str

  duration_min: float
  duration_max: float

  demucs_model: str = 'htdemucs'


@dataclass
class HarmonixConfig(DataConfig):
  name: str = 'harmonix'

  demixed: bool = False
  num_instruments: int = 4
  num_labels: int = 10

  bpfed: bool = True

  path_base_dir: str = '/home/jongsoo/BeatTrackingDataset/harmonix/'
  path_track_dir: str = '/home/jongsoo/BeatTrackingDataset/harmonix/tracks/'
  path_demix_dir: str = '/home/jongsoo/BeatTrackingDataset/harmonix/demix/'
  path_feature_dir: str = '/home/jongsoo/BeatTrackingDataset/harmonix/features/'
  path_bpf_dir: str = '/home/jongsoo/BeatTrackingDataset/harmonix/bpf/'
  path_bpf_feature_dir: str = '/home/jongsoo/BeatTrackingDataset/harmonix/features_bpf/'
  path_no_demixed_feature_dir: str = '/home/jongsoo/BeatTrackingDataset/harmonix/features_no_demixed/'
  path_metadata: str = '/home/jongsoo/BeatTrackingDataset/harmonix/metadata.csv'

  sources: List[str] = field(default_factory=lambda: ["bass"])
  # sources: List[str] = field(default_factory=lambda: ["bass", "drums"])

  duration_min: int = 76
  duration_max: int = 660


@dataclass
class UnionDatasetConfig(DataConfig):
  name: str = 'union'

  names: List[str] = field(default_factory=lambda: ['Ballroom', 'Carnatic', 'Hainsworth', 'harmonix', 'RWC', 'SMC'])
  test_name: str = 'GTZAN'

  demixed: bool = False
  num_instruments: int = 4
  num_labels: int = 10

  bpfed: bool = False

  path_base_dir: str = '/home/jongsoo/BeatTrackingDataset/'
  path_track_dir: str = '/tracks/'
  path_demix_dir: str = '/demix/'
  path_feature_dir: str = '/features/'
  path_bpf_dir: str = '/bpf/'
  path_bpf_feature_dir: str = '/features_bpf/'
  path_no_demixed_feature_dir: str = '/features_no_demixed/'
  path_metadata: str = '/metadata.csv'

  duration_min: int = 76
  duration_max: int = 660


defaults = [
  '_self_',
  {'data': 'harmonix'},
]

@dataclass
class GTZANConfig(DataConfig):
  name: str = 'gtzan'

  demixed: bool = False
  num_instruments: int = 4
  num_labels: int = 10

  bpfed: bool = True

  # NOTE: Preprocessed with all-in-one-test.ipynb due to nested directories
  path_base_dir: str = '/home/jongsoo/BeatTrackingDataset/GTZAN/'
  path_track_dir: str = '/home/jongsoo/BeatTrackingDataset/GTZAN/tracks/'
  path_demix_dir: str = '/home/jongsoo/BeatTrackingDataset/GTZAN/demix/'
  path_feature_dir: str = '/home/jongsoo/BeatTrackingDataset/GTZAN/features/'
  path_bpf_dir: str = '/home/jongsoo/BeatTrackingDataset/GTZAN/bpf/'
  path_bpf_feature_dir: str = '/home/jongsoo/BeatTrackingDataset/GTZAN/features_bpf/'
  path_no_demixed_feature_dir: str = '/home/jongsoo/BeatTrackingDataset/GTZAN/features_no_demixed/'
  path_metadata: str = '/home/jongsoo/BeatTrackingDataset/GTZAN/metadata.csv'
  # path_beat_annotations: str = '/home/june/projects/other/datasets-bt/GTZAN/annotations/beats/'

  # sources: List[str] = field(default_factory=lambda: ["bass"])
  sources: List[str] = field(default_factory=lambda: ["bass", "drums"])

  duration_min: int = 76
  duration_max: int = 660


@dataclass
class Config:
  # wandb setup
  project_name = "hyundai_final"

  debug: bool = False  # For debugging.
  sanity_check: bool = False  # For overfitting a small subset of data.
  sanity_check_size: int = 1
  offline: bool = False

  case: Optional[str] = None
  model: str = 'allinonetempo'

  data: DataConfig = HarmonixConfig()
  defaults: List[Any] = field(default_factory=lambda: defaults)

  # Data configurations --------------------------------------------------
  sample_rate: int = 22050
  window_size: int = 2048
  num_bands: int = 12
  hop_size: int = 441  # FPS=50
  fps: int = 50
  fmin: int = 30
  fmax: int = 11025
  demucs_model: str = 'htdemucs'
  bpf_band_dir: str = '/home/jongsoo/beat-tracking/band_data'
  buffer_length: int = 5
  latency_t: float = 0.

  # Multi-task learning configurations ------------------------------------
  learn_rhythm: bool = True
  learn_structure: bool = True
  learn_segment: bool = True
  learn_label: bool = True

  # Training configurations -----------------------------------------------
  segment_size: Optional[float] = 5
  batch_size: int = 256

  optimizer: str = 'radam'
  sched: Optional[str] = 'plateau'
  lookahead: bool = False

  lr: float = 0.005
  warmup_lr: float = 1e-5
  warmup_epochs: int = 0
  cooldown_epochs: int = 0
  min_lr: float = 1e-7
  max_epochs: int = 6

  # Plateau scheduler.
  decay_rate: float = 0.3
  patience_epochs: int = 5
  eval_metric: str = 'val/loss'
  epochs: int = 10  # not used. just for creating the scheduler

  validation_interval_epochs: int = 1
  early_stopping_patience: int = 2

  weight_decay: float = 0.00025
  swa_lr: float = 0.15
  gradient_clip: float = 0.5

  # Model configurations --------------------------------------------------
  threshold_minimal: float = 0.5
  threshold_beat: float = 0.19
  threshold_downbeat: float = 0.19
  threshold_section: float = 0.05

  best_threshold_beat: Optional[float] = None
  best_threshold_downbeat: Optional[float] = None

  instrument_attention: bool = True
  double_attention: bool = True

  depth: int = 5
  dilation_factor: int = 2
  dilation_max: int = 3200  # 32 seconds, not in use
  num_heads: int = 2
  kernel_size: int = 5

  dim_input: int = 81
  dim_embed: int = 24
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  causal: bool = False

  drop_conv: float = 0.2
  drop_path: float = 0.1
  drop_hidden: float = 0.2
  drop_attention: float = 0.2
  drop_last: float = 0.0

  act_conv: str = 'elu'
  act_transformer: str = 'gelu'

  layer_norm_eps: float = 1e-5

  # Loss configurations ---------------------------------------------------
  focal_loss: bool = False
  dice_loss: bool = False
  
  loss_weight_beat: float = 1.
  loss_weight_downbeat: float = 3.
  loss_weight_tempo: float = 1.
  
  loss_weight_focal: float = 3.
  loss_weight_dice_downbeat: float = 0.3
  loss_weight_dice_beat: float = 0.1

  loss_weight_mag: float = 1.

  # Misc ------------------------------------------------------------------
  seed: int = 1234
  fold: int = 9
  aafold: Optional[int] = None
  total_folds: int = 8

  bpm_min: int = 55
  bpm_max: int = 240
  min_hops_per_beat: int = 24  # 60 / max_bpm * sample_rate / hop_size


cs = ConfigStore.instance()
cs.store(name='config', node=Config)
cs.store(group='data', name=HarmonixConfig.name, node=HarmonixConfig)
cs.store(name=HarmonixConfig.name, node=HarmonixConfig)