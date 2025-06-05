import os
import sys
import time
import torch
import struct
import serial # UART [https://pyserial.readthedocs.io/en/latest/pyserial_api.html]
import socket # TCP-based Ethernet communication
import warnings
import numpy as np

from util import TensorRTAllInOne, predict_future_beat_type
from com_setup import pretrained_dir, threshold_beat, threshold_downbeat, UART, Ethernet, dont_need_beat_buffer, tempo_measure_method, debug_dl, future_prediction

if future_prediction is not None:
	from com_setup import tolerance

warnings.simplefilter("ignore", RuntimeWarning)

def save_data_to_csv(run_info):
	try:
		with open(os.path.join(run_info, "send_history.csv"), 'a') as f:
			for s in send_time_history:
				f.write(" ".join(map(str, s)) + "\n")
			send_time_history.clear()
		with open(os.path.join(run_info, "receive_history.csv"), 'a') as f:
			for r in receive_time_history:
				f.write(" ".join(map(str, r)) + "\n")
			receive_time_history.clear()
	except Exception as e:
		print(f"Error writing to CSV: {e}")

# Model setup
print(f"Loading All-In-One model...")
model = TensorRTAllInOne(pretrained_dir=pretrained_dir, threshold_beat=threshold_beat, threshold_downbeat=threshold_downbeat)

beat_buffer = torch.zeros((250,2), dtype=torch.float32).to('cuda')

depth = 2 # Bit depth (bytes):  2 => 16-bit PCM
sample_rate = model.cfg.sample_rate # Hz

slen = model.cfg.hop_size / sample_rate # Length of an audio frame (s)
mslen = int(slen * 1000) # Length of an audio frame (ms)

BUFFER_SIZE = model.cfg.buffer_length * sample_rate

global_buffer = torch.zeros(BUFFER_SIZE, dtype=torch.float32).to('cuda')

beat_time_list = torch.zeros(50, dtype=torch.float32).to('cuda')
tempo_list = [0 for i in range(50)]

if debug_dl:
	from datetime import datetime
	run_info = str(datetime.now()).replace(" ","_").split(".")[0]
	os.makedirs(f"delay/{run_info}")
	run_info = f"delay/{run_info}"

send_time_history = []
receive_time_history = []

# UART setup
try:
	serial_port = UART.serial_port
	baud_rate = UART.baud_rate
	ser = serial.Serial(serial_port, baud_rate)
except serial.SerialException as e:
	print(f"Serial error: {e}")
	sys.exit(1)

# Ethernet setup
host_ip = Ethernet.host_ip
port = Ethernet.port

try:
	sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1) # Allow port reuse if the socket was not closed
	sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # Disable Nagle’s algorithm
	sock.bind((host_ip, port))
	sock.listen(1)
	print(f"Listening for incoming connections on {host_ip}:{port}...")
	conn, addr = sock.accept()
	print(f"Connection established with {addr}")
	time.sleep(0.1)

	# Receive a pre-warming message
	try:
		pre_warm_msg = conn.recv(2).decode()
	except socket.timeout as e:
		print(f"Socket timeout: {e}")
		sys.exit(1)
	except Exception as e:
		print(f"Error receiving pre-warm message: {e}")

except socket.error as e:
	print(f"Socket error: {e}")
	sys.exit(1)

# Initialization
try:
	data = conn.recv(882)
	audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32)
except Exception as e:
	print("Error occured in an initialization stage:", e)
	exit()

normalized_chunk = audio_chunk / np.iinfo(np.int16).max # audio_chunk / 32767
cuda_chunk = torch.tensor(normalized_chunk).to('cuda')
result = model.process(global_buffer.clone())
if tempo_measure_method == "model":
	tempo, _, beat_prob, downbeat_prob = model.get_prob(result, None if dont_need_beat_buffer else beat_buffer, tempo_measure_method)
	_,_,_,_,_ = model.postprocessing(beat_buffer, 0, -1, 120)
elif tempo_measure_method == "interval":
	beat_buffer, beat_prob, downbeat_prob = model.get_prob(result, None if dont_need_beat_buffer else beat_buffer, tempo_measure_method)
	_,_,_,_,_ = model.postprocessing(beat_buffer, 0, -1, 120, beat_time_list)
beat_data = struct.pack(">HHBffH", 0, 0, 0, 0., 0., 0)
ser.write(beat_data)
print("Start")

# Program Start
try:
	chunk_size = int(slen * sample_rate * depth)
	frame_no = 0
	last_beat_time = -1
	print("Ready!")

	while True:
		try:
			data = conn.recv(chunk_size) # Receive audio chunk
			if debug_dl:
				receive_time_history.append([frame_no, time.time()])
			if not data:
				print("Socket connection closed by client.")
				break
			audio_chunk = np.frombuffer(data, dtype=np.int16).astype(np.float32)
		except:
			ser.write(struct.pack(">HHBffH", 0, 0, 0, 0., 0., 0))
			continue
		
		sound_magnitude = int(np.clip(np.mean(np.abs(audio_chunk)), 0, 65535))

		normalized_chunk = audio_chunk / np.iinfo(np.int16).max # audio_chunk / 32767
		cuda_chunk = torch.tensor(normalized_chunk).to('cuda')
		
		global_buffer = torch.cat((global_buffer[chunk_size//2:], cuda_chunk))
		
		result = model.process(global_buffer.clone())
		if tempo_measure_method == "model":
			tempo, beat_buffer, beat_prob, downbeat_prob = model.get_prob(result, None if dont_need_beat_buffer else beat_buffer, tempo_measure_method)
			beat_type, last_beat_time, tempo, _, index_for_prob = model.postprocessing(beat_buffer, frame_no, last_beat_time, tempo, beat_time_list=None, future_prediction=future_prediction)
		elif tempo_measure_method == "interval":
			beat_buffer, beat_prob, downbeat_prob = model.get_prob(result, None if dont_need_beat_buffer else beat_buffer, tempo_measure_method)
			beat_type, last_beat_time, tempo, beat_time_list, index_for_prob = model.postprocessing(beat_buffer, frame_no, last_beat_time, tempo_list[-1], beat_time_list, future_prediction)
		if future_prediction is not None:
			beat_prob = beat_buffer[index_for_prob][0]
			downbeat_prob = beat_buffer[index_for_prob][1]
		tempo_list = tempo_list[1:] + [tempo]
		
		beat_data = struct.pack(">HHBffH", frame_no, sound_magnitude, beat_type, beat_prob, downbeat_prob, tempo)
		ser.write(beat_data)
		if debug_dl:
			send_time_history.append([frame_no, time.time()])
		frame_no = (frame_no + 1) % 65536

except (KeyboardInterrupt):
	print("Program interrupted. Shutting down...")

except Exception as e:
	print(f"An error occurred: {e}")

finally:
	if ser.is_open:
		ser.close()
		print("Serial connection closed.")
	if sock:
		sock.close()
		print("Socket closed.")
	if debug_dl:
		save_data_to_csv(run_info)