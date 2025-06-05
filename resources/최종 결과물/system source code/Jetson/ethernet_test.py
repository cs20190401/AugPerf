import socket
from com_setup import Ethernet

# 서버 설정
HOST = Ethernet.host_ip
PORT = Ethernet.port

def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(1)
        print(f"Listening for incoming connections on {HOST}:{PORT}...")
        conn, addr = server_socket.accept()
        print(f"Connection established with {addr}")
        with conn:
            print(f"Connected by {addr}")
            data = conn.recv(1024)
            if data:
                print(f"Received: {data.decode()}")
                conn.sendall(b"Hello from Jetson!")

if __name__ == "__main__":
    start_server()