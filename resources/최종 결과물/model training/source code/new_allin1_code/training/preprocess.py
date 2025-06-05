import hydra

from ..config import Config
from ..spectrogram import no_demixed_torch_spectrogram
from ..helpers import expand_paths, mkpath


@hydra.main(version_base=None, config_name='config')
def main(cfg: Config):
  if cfg.data.name == "union":
    names = cfg.data.names + [cfg.data.test_name]
  else:
    names = [cfg.data.name]
  
  for name in names:
    print(f"Preprocess for {name} Dataset...")
    if cfg.data.name == "union":
      path_track_dir = cfg.data.path_base_dir + name + cfg.data.path_track_dir
    else:
      path_track_dir = cfg.data.path_track_dir
      
    if name == 'harmonix':
      track_paths = expand_paths([mkpath(path_track_dir) / '*.mp3'])
    else:
      track_paths = expand_paths([mkpath(path_track_dir) / '*.wav'])
    
    if cfg.data.bpfed:
      feature_dir = mkpath(cfg.data.path_bpf_feature_dir)

      spec_paths = no_demixed_torch_spectrogram(
        paths=track_paths,
        spec_dir=feature_dir,
        cfg=cfg,
      )

    else:
      if cfg.data.name == "union":
        path_no_demixed_feature_dir = cfg.data.path_base_dir + name + cfg.data.path_no_demixed_feature_dir
      else:
        path_no_demixed_feature_dir = cfg.data.path_no_demixed_feature_dir
      no_demixed_feature_dir = mkpath(path_no_demixed_feature_dir)

      spec_paths = no_demixed_torch_spectrogram(
        paths=track_paths,
        spec_dir=no_demixed_feature_dir,
        cfg=cfg,
      )

    print(f'Preprocessing finished. {len(spec_paths)} spectrograms saved.')


if __name__ == '__main__':
  main()
