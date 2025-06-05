# One single UART communication for both energy and beat prediction data

import os
import sys
import time
import struct
import serial # UART [https://pyserial.readthedocs.io/en/latest/pyserial_api.html]
import socket # TCP-based Ethernet communication
import threading
import numpy as np
from pydub import AudioSegment
from pythonosc import udp_client

from com_setup import debug_dl, file_path, UART, Ethernet

if debug_dl:
    from datetime import datetime

stop_event = threading.Event() # Thread stop flag

mslen = 20 # Length of an audio frame (ms)
depth = 2 # Bit depth (bytes) | 1 => 8-bit PCM | 2 => 16-bit PCM | 4 => 32-bit PCM
sample_rate = 22050 # Hz

send_time_history = []
receive_time_history = []
lock = threading.Lock()

# Setup OSC client
osc_ip = "127.0.0.1"
osc_port = 8000
osc_client = udp_client.SimpleUDPClient(osc_ip, osc_port)

# Debug - delay analysis
def save_data_to_csv(run_info):
    try:
        with lock:
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

# Send audio data through TCP
def send_audio_over_ethernet(audio_data, file_path, sock, run_info, stop_event=None, debug=debug_dl):
    chunk_size = int(mslen * 0.001 * sample_rate * depth) # 20ms * 22050Hz * bytes per sample
    frame_no = 0
    start_time = time.monotonic()

    # Send OSC message to start audio playback
    osc_client.send_message("/audio", file_path)

    try:
        for i in range(0, len(audio_data), chunk_size):
            if stop_event and stop_event.is_set():
                break
            chunk = audio_data[i:i+chunk_size]
            if debug:
                send_time_history.append([frame_no, time.time()])
            sock.send(chunk)
            frame_no = (frame_no + 1) % 65536

            elapsed_time = time.monotonic() - start_time
            expected_time = (i + chunk_size) / (sample_rate * depth)
            
            if elapsed_time < expected_time:
                time.sleep(expected_time - elapsed_time)

        print("End of the song.")
        stop_event.set()
    except Exception as e:
        print(f"An error occurred: {e}")
        stop_event.set()
        raise e
    finally:
        if debug:
            save_data_to_csv(run_info)

# Receive a response through UART
def receive_data(ser, stop_event, debug=debug_dl):
    try:
        print(f"{'Frame':^7} | {'Energy':^7} | {'Tempo':^7} | {'Type':^10} | {'BeatProb':^11} | {'DBeatProb':^11}")
        while not stop_event.is_set():
            if ser.in_waiting > 0:
                data = ser.read(15)
                frame_no, energy, beat_type, beat_prob, downbeat_prob, tempo = struct.unpack(">HHBffH", data)
                if debug:
                    receive_time_history.append([frame_no, time.time(), beat_type])
                if beat_type == 0:
                    pass
                elif beat_type == 1:
                    # Send OSC message to play a beat click sound
                    osc_client.send_message("/click", 1)
                    print(f"{frame_no:^7} | {energy:^7} | {tempo:^7} | {'Beat':^10} | {beat_prob:^.3f} | {downbeat_prob:^.3f}")
                else:
                    # Send OSC message to play a downbeat click sound
                    osc_client.send_message("/click", 2)
                    print(f"{frame_no:^7} | {energy:^7} | {tempo:^7} | {'Downbeat':^10} | {beat_prob:^.3f} | {downbeat_prob:^.3f}")
    except Exception as e:
        print(f"An error occurred: {e}")
        stop_event.set()

def main():

    # Debug - delay analysis
    if debug_dl:
        run_info = str(datetime.now()).replace(" ", "_").replace(":", "-").split('.')[0] # Windows
        run_info = os.path.join("delay", run_info)
        os.makedirs(run_info)
    else:
        run_info = ""
        
    try:
        ext = file_path[-3:]
        print(f"Playing {file_path}.\n")
        
        audio = AudioSegment.from_file(file_path, format=ext)
        audio = audio.set_frame_rate(sample_rate).set_channels(1)
        audio = audio.set_sample_width(depth)
        raw_data = audio.get_array_of_samples()
        raw_data = raw_data[:60*sample_rate] if debug_dl else raw_data
        raw_data = np.array(raw_data).tobytes()
        
        # UART setup
        try:
            serial_port = UART.serial_port
            baud_rate = UART.baud_rate
            ser = serial.Serial(serial_port, baud_rate)
        except serial.SerialException as e:
            print(f"Serial error: {e}")
            sys.exit(1)
        
        # Ethernet (TCP) setup
        host_ip = Ethernet.host_ip
        port = Ethernet.port
        client_ip = Ethernet.client_ip
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) # Disable Nagle's algorithm
            sock.bind((client_ip,0))
            sock.connect((host_ip, port))
            # Send a pre-warming message
            pre_warm_msg = "READY"
            sock.send(pre_warm_msg.encode())
        except socket.error as e:
            print(f"Socket error: {e}")
            sock.close()
            ser.close()
            sys.exit(1)
        
		# Initialization
        sock.send(raw_data[:882])
        _ = ser.read(15)
        print("Start")

        # Thread to receive results over UART
        receive_thread = threading.Thread(target=receive_data, args=(ser, stop_event), daemon=True)
        # Thread to send audio data over Ethernet
        send_audio_thread = threading.Thread(target=send_audio_over_ethernet, args=(raw_data, file_path, sock, run_info, stop_event), daemon=True)

        receive_thread.start()
        send_audio_thread.start()

        send_audio_thread.join()
        receive_thread.join()

    except (KeyboardInterrupt):
        print("Program interrupted. Shutting down...")
        stop_event.set()

    except Exception as e:
        print(f"An error occurred: {e}")
        stop_event.set()

    finally:
        time.sleep(1) # Make sure the server has time to process the last chunk
        

if __name__ == "__main__":
    main()