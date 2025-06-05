import socket
from com_setup import Ethernet

JETSON_IP = Ethernet.host_ip
PORT = Ethernet.port
CLIENT_IP = Ethernet.client_ip

def send_data_to_jetson():
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
		client_socket.bind((CLIENT_IP, 0))
		client_socket.connect((JETSON_IP, PORT))
		print("Connected to Jetson server")
		client_socket.sendall(b"Hello from MacBook!")
		data = client_socket.recv(1024)
		print(f"Received: {data.decode()}")

if __name__ == "__main__":
	send_data_to_jetson()