import torch
import torch.nn.functional as F

class Postprocessor:
	def __init__(self, fps: int = 50, kernel_size: int = 17, threshold_beat: float = 0.5, threshold_downbeat: float = 0.1):
		self.fps = fps
		self.kernel_size = kernel_size
		self.padding = kernel_size // 2
		self.width = self.padding // 2
		self.thresholds = torch.tensor([threshold_beat, threshold_downbeat], device='cuda').unsqueeze(1)

	def __call__(self, pred):
		return self.postp_minimal(pred)
	
	def postprocess(self, result):
		beat_result,downbeat_result,nobeat_result = result
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
		
		pred_downbeat_times = self.postp_minimal(activations_combined[:, :2])

		beats = pred_downbeat_times[:, 0] + 0.035
		beat_positions = pred_downbeat_times[:, 1]
		downbeats = pred_downbeat_times[beat_positions == 1., 0] + 0.035

		beats = beats.tolist()
		downbeats = downbeats.tolist()
		beat_positions = beat_positions.cpu().numpy().astype('int').tolist()

		# tempo = torch.argmax(raw_prob_tempos, axis=1).cpu().numpy()

		return {
			'beats': beats,
			'downbeats': downbeats,
			'beat_positions': beat_positions,
			# 'tempo': tempo,
		}
	
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
			pred_logits != F.max_pool1d(pred_logits, kernel_size=self.kernel_size, stride=1, padding=self.padding), 0.
		)

		# Keep maxima with over 0.5 probability (logit > 0)
		pred_peaks = pred_peaks > self.thresholds

		# Rearrange back to two tensors of shape (T,)
		beat_peaks, downbeat_peaks = pred_peaks[0], pred_peaks[1]
		# TODO: check why almost all beats are also downbeats

		# Piecewise operations
		postp_beat_time_position = self._postp_minimal_item(beat_peaks, downbeat_peaks)
		# print("return: ", postp_beat_time_position)

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

		# NOTE: This part is new
		# TODO: check why it generates too many beats and downbeats (check if it's this or the model's output)

		# Remove beats that coincide with downbeats
		mask = ~torch.isin(beat_time, downbeat_time)
		beat_time = beat_time[mask]

		# ----------------------------------------------------
		# TODO: FIX ERROR AND TEST
		# Assign beat positions based on downbeat intervals
		# Use torch.searchsorted to find the indices where each beat falls into a downbeat interval
		# downbeat_indices = torch.searchsorted(downbeat_time, beat_time, right=False) - 1
		# downbeat_indices = torch.clamp(downbeat_indices, min=0)

		# # Calculate positions relative to the last downbeat
		# beat_position = torch.arange(1, len(beat_time) + 1, dtype=torch.int, device=beat_time.device)
		# beat_position -= torch.cumsum(torch.cat(([0], torch.diff(downbeat_indices).ne(0).int()))[:-1])
		# ----------------------------------------------------
		
		# TODO: Change with increasing value; for now beat position = 2 for testing
		beat_position = torch.full_like(beat_time, 2, dtype=torch.int)
		
		# Downbeats have position 1
		downbeat_position = torch.ones_like(downbeat_time, dtype=torch.int)
		
		# Combine beat and downbeat times and positions
		all_times = torch.cat((beat_time, downbeat_time))
		all_positions = torch.cat((beat_position, downbeat_position))
		
		# Sort by time (first column); TODO: check if necessary
		sorted_indices = all_times.argsort()
		sorted_times = all_times[sorted_indices]
		sorted_positions = all_positions[sorted_indices]
		
		# Combine the times and positions inside the bar
		beat_time_position = torch.stack((sorted_times, sorted_positions), dim=1)
		
		# Convert to numpy array and return
		# return beat_time_position.cpu().numpy()
		return beat_time_position
		

def deduplicate_peaks(peaks: torch.Tensor, width=1) -> torch.Tensor:
	"""
	Replaces groups of adjacent peak frame indices that are each not more
	than `width` frames apart by the average of the frame indices.
	"""
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