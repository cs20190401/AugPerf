import serial
from com_setup import UART

port = UART.serial_port
baudrate = 9600

with serial.Serial(port, baudrate, timeout=1) as ser:
    while True:
        if ser.in_waiting > 0:
            data = ser.readline().decode('utf-8').strip()
            print("Received:", data)