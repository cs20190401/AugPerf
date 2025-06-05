from typing import Literal, Optional

from dataclasses import dataclass


# 바꾸는 값 #############################################################
# pretrained_dir = f"/home/macjet/Documents/jongsoo/17ugvnkd"
# threshold_beat=0.02
# threshold_downbeat=0.08
# dont_need_beat_buffer = False

pretrained_dir = f"/home/macjet/Documents/jongsoo/pretrained"
threshold_beat=0.18
threshold_downbeat=0.12
dont_need_beat_buffer = True
bpf_band_dir = f"home/macjet/Documents/jongsoo/sub_band_data"
#######################################################################


# 바꾸는 값 #############################################################
tempo_measure_method: Literal["model","interval"] = "interval"
#######################################################################


# 바꾸는 값 #############################################################
# future_prediction: Optional[float] = None

future_prediction: Optional[float] = 0.04
tolerance:float = 0.07
#######################################################################


# 바꾸는 값 #############################################################
debug_dl: bool = True
#######################################################################


@dataclass
class UART:
	# 바꾸는 값 #########################################################
	serial_port: str = "/dev/ttyTHS0" # UART serial port
	baud_rate: int = 38400
	###################################################################


@dataclass
class Ethernet:
	# 바꾸는 값 #########################################################
	host_ip: str = "10.0.0.1" # This device's IP
	port: int = 12345 # This device's port
	###################################################################