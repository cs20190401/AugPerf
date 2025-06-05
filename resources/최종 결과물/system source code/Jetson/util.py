import os
import time
from typing import Literal, Optional

import tensorrt as trt
import torch
import torchaudio
from omegaconf import OmegaConf

from madmom.audio.filters import LogarithmicFilterbank

import numpy as np

from com_setup import future_prediction, bpf_band_dir
if future_prediction is not None:
	from com_setup import tolerance

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# Predict the beat type of a future frame
def predict_future_beat_type(
		last_beat_time:float, # Absolute time of the last predicted beat
		frame_no:int, # Index of current frame
		beat_times,
		beat_position,

		tempo = None,

		# Common Parameters
		delay:float = 0., # Time between the current and future frame in seconds (future frame = current frame + delay)
		tolerance:float = 0.07, # Default 0.07
		buffer_length:float = 4.98, # Starting time of the current frame relative to beat_buffer
		fslen = 0.02, # Length of an audio frame (s)
		beat_time_list = None,
	):

	assert len(beat_times) == len(beat_position)
	beat_position = beat_position.tolist()

	if len(beat_times) == 0: # No beats => Cannot predict the beat type of the future frame
		beat_type = 0
	else:
		if tempo is not None:
			beat_interval = 60 / tempo
		elif beat_times.size(0) > 1:
			time_diffs = beat_times[1:] - beat_times[:-1]
			time_diffs = time_diffs[time_diffs != 0]
			beat_interval = torch.mean(time_diffs).item()
		else:
			return 0, last_beat_time, beat_time_list, -1
		time_diffs = np.abs(buffer_length + delay - (float(beat_times[-1]) + np.arange(1,5) * beat_interval))
		within_tolerance = np.where(time_diffs<=tolerance)[0]

		if len(within_tolerance) > 0: # A beat/downbeat is predicted for the future frame
			closest_beat_index = within_tolerance[0]
			temp_beat_time = float(float(beat_times[-1]) + closest_beat_index * beat_interval - buffer_length + frame_no * fslen) # Absolute time of the closest (candidate) beat
			is_not_close = abs(last_beat_time - temp_beat_time) > 0.33
			if is_not_close and temp_beat_time > last_beat_time:
				last_beat_time = temp_beat_time # Update last predicted beat
				last_downbeat_index = len(beat_position) - 1 - beat_position[::-1].index(1) if 1 in beat_position else None
				if last_downbeat_index is not None: # If a downbeat was predicted within the past 5 seconds from the current frame
					expected_downbeat_indices = last_downbeat_index + 4 * np.arange(1,3)
					closest_beat_index += len(beat_position)
					beat_type = 2 if closest_beat_index in expected_downbeat_indices else 1
				else:
					beat_type = 2 if len(beat_position) == 3 else 1
				if beat_time_list is not None:
					beat_time_list = torch.cat((beat_time_list[1:], torch.tensor([last_beat_time], dtype=beat_time_list.dtype, device=beat_time_list.device)), dim=0)
			else:
				beat_type = 0
		else:
			beat_type = 0

	return beat_type, last_beat_time, beat_time_list, int(float(beat_times[-1]/fslen if len(beat_times)>0 else -1))

def get_probs_realtime(logits_beat, logits_downbeat, logits_tempo, buffer, tempo_measure_method: Literal["model","interval"] = "interval"):
	if buffer is None:
		probs_beat = torch.sigmoid(logits_beat)
		probs_downbeat = torch.sigmoid(logits_downbeat)
		activation_no = (2. - probs_beat - probs_downbeat) / 2.
		activation_xbeat = torch.clamp(probs_beat - probs_downbeat, min=1e-8)
		total = activation_xbeat + probs_downbeat + activation_no
		probs_downbeat = probs_downbeat / total
		probs_beat = activation_xbeat / total
		buffer = torch.stack((probs_beat, probs_downbeat), dim=1)
		prob_beat = probs_beat[-1]
		prob_downbeat = probs_downbeat[-1]
	else:
		prob_beat = torch.sigmoid(logits_beat).item()
		prob_downbeat = torch.sigmoid(logits_downbeat).item()
		activation_no = (2. - prob_beat - prob_downbeat) / 2.
		activation_xbeat = max(1e-8,prob_beat-prob_downbeat)
		total = sum([activation_xbeat,prob_downbeat,activation_no])
		prob_beat = activation_xbeat / total
		prob_downbeat = prob_downbeat / total
		activations_combined = torch.tensor([[prob_beat, prob_downbeat]], 
                                    dtype=buffer.dtype, 
                                    device=buffer.device)
		buffer = torch.cat((buffer[1:, :], activations_combined), dim=0)
	if tempo_measure_method == "model":
		raw_prob_tempos = torch.softmax(logits_tempo,dim=0)
		tempo = int(torch.argmax(raw_prob_tempos,axis=0).cpu().numpy())
		if tempo > 180:
			tempo = tempo//2
		elif tempo < 60:
			tempo = tempo*2
	elif tempo_measure_method == "interval":
		tempo = None
	
	return tempo, buffer, prob_beat, prob_downbeat

class Postprocessor:
	def __init__(self, fps: int = 50, kernel_size: int = 7, threshold_beat: float = 0.5, threshold_downbeat: float = 0.1):
		self.fps = fps
		self.kernel_size = kernel_size
		self.padding = kernel_size // 2
		self.width = self.padding // 2
		self.thresholds = torch.tensor([threshold_beat, threshold_downbeat], device='cuda').unsqueeze(1)

	def __call__(self, pred):
		return self.postp_minimal(pred)
	
	# NOTE: using GPU, batch size 1, no masking, output numpy array of shape (num_beats, 2), where [:,0] = beat time and [:,1] = beat position
	def postp_minimal(self, pred_logits):
		# Move inputs to GPU
		if not isinstance(pred_logits, torch.Tensor):
			pred_logits = torch.tensor(pred_logits, device="cuda" if torch.cuda.is_available() else "cpu")
		device = pred_logits.device

		# Reshape for max pooling
		pred_logits = pred_logits.permute(1, 0) # shape (2, T)

		# beat_probs  = beat_probs + downbeat_probs
		pred_logits[0].add_(pred_logits[1])
		pred_logits[0].clamp_(max=1.0)

		# Pick maxima within +/- 70ms
		pred_peaks = pred_logits.masked_fill(
			pred_logits != torch.nn.functional.max_pool1d(pred_logits, kernel_size=self.kernel_size, stride=1, padding=self.padding), 0.
		)

		# Keep maxima with over 0.5 probability (logit > 0)
		pred_peaks = pred_peaks > self.thresholds

		# Rearrange back to two tensors of shape (T,)
		beat_peaks, downbeat_peaks = pred_peaks[0], pred_peaks[1]

		# Piecewise operations
		postp_beat_time_position = self._postp_minimal_item(beat_peaks, downbeat_peaks)

		return postp_beat_time_position

	def _postp_minimal_item(self, beat_peaks, downbeat_peaks):	
		# Find nonzero indices (peaks)
		beat_frame = torch.nonzero(beat_peaks, as_tuple=True)[0]
		downbeat_frame = torch.nonzero(downbeat_peaks, as_tuple=True)[0]
		
		# Remove adjacent peaks
		beat_frame = deduplicate_peaks(beat_frame, width=self.width)
		downbeat_frame = deduplicate_peaks(downbeat_frame, width=self.width)
		
		# Convert frames to time
		beat_time = beat_frame.float() / self.fps
		downbeat_time = downbeat_frame.float() / self.fps
		
		# Move downbeats to the nearest beat
		if len(beat_time) > 0:
			downbeat_time = torch.tensor([
				beat_time[torch.argmin(torch.abs(beat_time - d_time))].item()
				for d_time in downbeat_time
			], device=beat_time.device)

		# Remove duplicate downbeat times and sort in ascending order
		downbeat_time = torch.unique(downbeat_time)

		# Remove beats that coincide with downbeats
		mask = ~torch.isin(beat_time, downbeat_time)
		beat_time = beat_time[mask]

		beat_position = torch.full_like(beat_time, 2, dtype=torch.int)
		
		# Downbeats have position 1
		downbeat_position = torch.ones_like(downbeat_time, dtype=torch.int)
		
		# Combine beat and downbeat times and positions
		all_times = torch.cat((beat_time, downbeat_time))
		all_positions = torch.cat((beat_position, downbeat_position))
		
		# Sort by time (first column)
		sorted_indices = all_times.argsort()
		sorted_times = all_times[sorted_indices]
		sorted_positions = all_positions[sorted_indices]
		
		# Combine the times and positions inside the bar
		beat_time_position = torch.stack((sorted_times, sorted_positions), dim=1)
		
		return beat_time_position
		
def deduplicate_peaks(peaks: torch.Tensor, width=1) -> torch.Tensor:
	if len(peaks) == 0:
		return peaks

	result = []
	peaks = peaks.sort()[0]  # Ensure peaks are sorted
	group = [peaks[0].item()]  # Start the first group

	for p in peaks[1:]:
		if p - group[-1] <= width:
			group.append(p.item())
		else:
			result.append(sum(group) / len(group))  # Add the average of the group
			group = [p.item()]

	result.append(sum(group) / len(group))  # Add the last group
	return torch.tensor(result, device=peaks.device)


def load_engine(engine_path):
	with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
		return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
	inputs = []
	outputs = []
	bindings = []
	count = 0

	for binding in engine:
		size = trt.volume(engine.get_tensor_shape(binding))
		dtype = trt.nptype(engine.get_tensor_dtype(binding))
		
		# PyTorch 텐서로 메모리 할당
		if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
			input_tensor = torch.empty(size, dtype=torch.float32, device='cuda')
			inputs.append(input_tensor)
			bindings.append(input_tensor.data_ptr())
		else:
			output_tensor = torch.empty(size, dtype=torch.float32, device='cuda')
			outputs.append(output_tensor)
			bindings.append(output_tensor.data_ptr())

	return inputs, outputs, bindings

class TORCH_SPECT():
	def __init__(self, cfg, num_channels=1, sample_rate=22050, win_length=2048, hop_size=441, n_bands=12, fmin=30, fmax=11025, device='cuda'):
		self.device = device
		self.stft = torchaudio.transforms.Spectrogram(n_fft=win_length, hop_length=hop_size, normalized=True, power=1).to(device)
		bin_frequencies = np.linspace(0, sample_rate / 2, win_length // 2 + 1)[:-1]
		self.filterbank = LogarithmicFilterbank(bin_frequencies, num_bands=n_bands, fmin=fmin, fmax=fmax, fref=440.0, norm_filters=True, unique_filters=True)
		self.filterbank = torch.from_numpy(self.filterbank).to(device)
		self.cfg = cfg
		if self.cfg.data.bpfed:
			self.bpfed = self.cfg.data.bpfed
			self.sources = self.cfg.data.sources
			bpf_dir_path = bpf_band_dir
			self.subbands = dict()
			for source in self.sources:
			band_freq = np.load(os.path.join(bpf_dir_path, f"total_{source}_freq.npy"))
			band_energ = np.load(os.path.join(bpf_dir_path, f"total_{source}_energy.npy"))
			freqs = np.linspace(0, self.cfg.sample_rate//2, self.cfg.window_size//2+1)
			average_energy = torch.tensor(np.interp(freqs, band_freq, band_energ)[:-1], dtype=torch.float32).to('cuda')
			average_energy = (average_energy / torch.max(average_energy))
			self.subbands[source] = average_energy

	def process_audio(self, audio):
		spectrogram = self.stft(audio)[:-1,:-1].T
		spec_stack = []
		if self.bpfed:
			for source in self.sources:
				source_spec = spectrogram * self.subbands[source]
				spec_stack.append(torch.matmul(source_spec, self.filterbank))
				spectrogram = torch.stack(spec_stack)
		else:
			spectrogram = torch.matmul(spectrogram, self.filterbank)
			spectrogram = torch.log10(spectrogram * 1.0 + 1.0)
		return spectrogram

class TensorRTAllInOne():
	def __init__(self,
			pretrained_dir: str = 'pretrained/',
			kernel_size: int = 7,
			threshold_beat: float = 0.1,
			threshold_downbeat: float = 0.35,
		):
		self.engine = load_engine(os.path.join(pretrained_dir, f"model.engine"))
		self.cfg = OmegaConf.load(os.path.join(pretrained_dir, "cfg.yaml"))
		self.buffer = allocate_buffers(self.engine)
		self.context = self.engine.create_execution_context()
		self.proc = TORCH_SPECT(num_channels=1, sample_rate=self.cfg.sample_rate, win_length=self.cfg.window_size, hop_size=self.cfg.hop_size, n_bands=self.cfg.num_bands, fmin=self.cfg.fmin, fmax=self.cfg.fmax, device='cuda')
		self.window_T = int(self.cfg.buffer_length * self.cfg.sample_rate) // self.cfg.hop_size
		self.fps = self.cfg.fps
		self.postprocessor_downbeat = Postprocessor(fps=self.fps, kernel_size=kernel_size, threshold_beat=threshold_beat, threshold_downbeat=threshold_downbeat)
		
	def process(self, audio_buffer):
		spec = self.proc.process_audio(audio_buffer)
		T, F = spec.shape
		self.buffer[0][0].copy_(spec.reshape(T*F))
		self.context.execute_v2(bindings=self.buffer[-1])
		return self.buffer[1][1:4]

	def get_prob(self, outputs, beat_buffer=None, tempo_measure_method: Literal["model","interval"] = "interval"):
		tempo, beat_buffer, prob_beat, prob_downbeat = get_probs_realtime(
															logits_beat = outputs[0],
															logits_downbeat = outputs[1],
															logits_tempo = outputs[2],
															buffer = beat_buffer,
															tempo_measure_method = tempo_measure_method,
														)
		if tempo_measure_method == "interval":
			return beat_buffer, prob_beat, prob_downbeat
		elif tempo_measure_method == "model":
			return tempo, beat_buffer, prob_beat, prob_downbeat
	
	def postprocessing(self, beat_buffer, frame_no:int, last_beat_time, tempo, beat_time_list=None, future_prediction:Optional[float]=None):
		if beat_time_list is not None:
			time_diffs = beat_time_list[1:] - beat_time_list[:-1]
			time_diffs = time_diffs[time_diffs != 0]
			if time_diffs.numel() > 1:  # numel()은 텐서의 총 원소 수 반환
				mean_diff = torch.mean(time_diffs[1:])
				tempo = int(60/mean_diff)
		pred_downbeat_times = self.postprocessor_downbeat(beat_buffer)

		if future_prediction is None:
			current_time = (frame_no-self.window_T+1)*(1./self.fps)
			if pred_downbeat_times.size(0) > 0:
				beats = pred_downbeat_times[:, 0] + current_time
				pred_downbeat_times[:, 0] = beats
				min_beat_interval = 0.33
				is_not_close = torch.abs(last_beat_time - beats[-1]) > min_beat_interval
				if is_not_close and beats[-1] >= last_beat_time:
					beat_type = 2 if pred_downbeat_times[-1, 1] == 1.0 else 1
					last_beat_time = beats[-1]
					if beat_time_list is not None:
						beat_time_list = torch.cat((beat_time_list[1:], torch.tensor([last_beat_time], dtype=beat_time_list.dtype, device=beat_time_list.device)), dim=0)
				else:
					beat_type = 0
			else:
				beat_type = 0
			index_for_prob = None
		else:
			beat_type, last_beat_time, beat_time_list, index_for_prob = predict_future_beat_type(last_beat_time, frame_no, pred_downbeat_times[:,0], pred_downbeat_times[:,1], tempo=tempo if beat_time_list is None else None, delay=future_prediction, tolerance=tolerance, beat_time_list=beat_time_list)

		
		if beat_time_list is not None:
			return beat_type, last_beat_time, tempo, beat_time_list, index_for_prob
		else:
			return beat_type, last_beat_time, tempo, None, index_for_prob