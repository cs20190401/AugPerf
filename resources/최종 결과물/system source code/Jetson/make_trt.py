import os
import subprocess

import torch
import torch.onnx
from omegaconf import OmegaConf

from allinone_models import AllInOneTempo, AllInOneTempo_with_Head

from com_setup import pretrained_dir
pdir = pretrained_dir

import logging
logging.getLogger("natten.functional").setLevel(logging.ERROR)

def load_model(pretrained_dir='pretrained/', device='cpu'):
	ckpt_path = os.path.join(pretrained_dir, "model.ckpt")
	cfg_path = os.path.join(pretrained_dir, "cfg.yaml")

	cfg = OmegaConf.load(cfg_path)
	model_type = cfg['model']

	checkpoint = torch.load(ckpt_path, map_location=device)
	checkpoint["state_dict"] = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}

	if model_type == "allinonetempo":
		model = AllInOneTempo(cfg).to(device)
	elif model_type == "nobufferallin1":
		model = AllInOneTempo_with_Head(cfg).to(device)
	else:
		raise NotImplementedError(f'Unknown model: {model_type}')

	model.load_state_dict(checkpoint['state_dict'])
	model.eval()
	print("Model Loading Complete!")

	return model, cfg

if os.path.isfile(os.path.join(pdir,"model.onnx")) == False:
	model,cfg = load_model(pretrained_dir=pdir, device='cpu')
	dummy_input = torch.randn(1, 1, cfg.buffer_length*cfg.fps, 83)
	torch.onnx.export(
		model,                           # PyTorch model
		dummy_input,                     # dummpy input
		os.path.join(pdir,"model.onnx"),                    # ONNX file path for save
		export_params=True,              # weights of model
		opset_version=11,                # ONNX opset version
		do_constant_folding=True,        # optimization of constant folding
		input_names=['input'],           # input name
		output_names=['output'],         # output name
		verbose=True,
	)

command = ["trtexec", f"--onnx={os.path.join(pdir,'model.onnx')}", f"--saveEngine={os.path.join(pdir,'model.engine')}"]

try:
	result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

	if result.returncode == 0:
		print("TensorRT engine conversion succeeded!")
		print(result.stdout)
	else:
		print("TensorRT engine conversion failed!")
		print(result.stderr)

except FileNotFoundError:
	print("The 'trtexec' command was not found. Please ensure that TensorRT is installed correctly.")
except Exception as e:
	print(f"An unknown error occurred: {e}")