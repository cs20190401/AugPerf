import os
import shutil

from allin1.training.evaluate import load_wandb_run
from omegaconf import OmegaConf

artifacts_path = "/home/jongsoo/beat-tracking/artifacts"
save_dir = "/home/jongsoo/beat-tracking/save"
run_ids = [
	# type your run_id
	# ex) "cdes17sy"
]

for run_id in run_ids:
	trainer, cfg, _ = load_wandb_run(run_id=run_id, run_dir='eval/', project_name='hyundai_final')
	save_dir = os.path.join(save_dir, run_id)
	os.makedirs(save_dir, exist_ok=True)
	OmegaConf.save(cfg, f'{save_dir}/cfg.yaml')
	for root, dirs, files in os.walk(artifacts_path):
		for dir_name in dirs:
			if run_id in dir_name:
				model_path = os.path.join(root, dir_name, 'model.ckpt')
				if os.path.isfile(model_path):
					target_path = os.path.join(save_dir, 'model.ckpt')
					shutil.copy2(model_path, target_path)

