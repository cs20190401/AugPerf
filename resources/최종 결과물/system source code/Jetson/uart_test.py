import serial
import time
from com_setup import UART

port = UART.serial_port
baudrate = 9600

with serial.Serial(port, baudrate, timeout=1) as ser:
    while True:
        ser.write(b'Hello from Jetson\n')
        print("Sent: Hello from Jetson")
        time.sleep(1)