import socket
import time
from smbus2 import SMBus

# MPU6050 Registers and their Address
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
INT_ENABLE = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H = 0x43
GYRO_YOUT_H = 0x45
GYRO_ZOUT_H = 0x47
ADDRESS_1 = 0x68  # AD0 pin connected to GND
ADDRESS_2 = 0x69  # AD0 pin connected to VCC

# Initialize the I2C bus
bus = SMBus(1)

def MPU_Init(addr):
    bus.write_byte_data(addr, SMPLRT_DIV, 7)
    bus.write_byte_data(addr, PWR_MGMT_1, 1)
    bus.write_byte_data(addr, CONFIG, 0)
    bus.write_byte_data(addr, GYRO_CONFIG, 24)
    bus.write_byte_data(addr, INT_ENABLE, 1)

def read_raw_data(addr, reg):
    high = bus.read_byte_data(addr, reg)
    low = bus.read_byte_data(addr, reg+1)
    value = ((high << 8) | low)
    if(value > 32768):
        value = value - 65536
    return value

# Initialize MPU6050
MPU_Init(ADDRESS_1)
MPU_Init(ADDRESS_2)

# Socket setup
host = '192.168.3.194'  
port = 8080
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.connect((host, port))

try:
    while True:
        # Read sensor data
        accel_x = read_raw_data(ADDRESS_1, ACCEL_XOUT_H)
        accel_y = read_raw_data(ADDRESS_1, ACCEL_YOUT_H)
        accel_z = read_raw_data(ADDRESS_1, ACCEL_ZOUT_H)
        accel_x2 = read_raw_data(ADDRESS_2, ACCEL_XOUT_H)
        accel_y2 = read_raw_data(ADDRESS_2, ACCEL_YOUT_H)
        accel_z2 = read_raw_data(ADDRESS_2, ACCEL_ZOUT_H)
        # Create a string of the data
        data_string = f"{accel_x},{accel_y},{accel_z},{accel_x2},{accel_y2},{accel_z2}"
        # Send the data
        sock.sendall(data_string.encode('utf-8'))
        time.sleep(0.04) 
finally:
    sock.close()
