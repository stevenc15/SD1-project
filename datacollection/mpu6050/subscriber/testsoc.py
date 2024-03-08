import socket
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

# Socket setup
host = '0.0.0.0'
port = 8080
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.bind((host, port))
sock.listen(1)
print("Waiting for connection...")
conn, addr = sock.accept()
print("Connected to", addr)

# Initialize subplots
# Initialize figure and subplots
fig = plt.figure(figsize=(14, 8))
ax1 = fig.add_subplot(2, 2, 1)  # 2D plot for roll and pitch
ax2 = fig.add_subplot(2, 2, 2)  # 2D plot for arm orientation (not used in this example, but kept for consistency)
ax3 = fig.add_subplot(2, 2, (3, 4), projection='3d')  # 3D plot for the arm

# Setting up the 3D plot
ax3.set_xlim3d([-2, 2])
ax3.set_ylim3d([-2, 2])
ax3.set_zlim3d([-2, 2])
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')

# Initial arm representation in 3D
line3d_upper_arm, = ax3.plot([], [], [], 'r-', linewidth=2, label='Upper Arm 3D')
line3d_forearm, = ax3.plot([], [], [], 'g-', linewidth=2, label='Forearm 3D')

# For plotting roll and pitch
x_data, roll_data, pitch_data, roll_data2, pitch_data2 = [], [], [], [], []
ln_roll, = ax1.plot([], [], 'r-', label='Roll')
ln_pitch, = ax1.plot([], [], 'g-', label='Pitch')
ln_roll2, = ax1.plot([], [], 'r--', label='Roll2')
ln_pitch2, = ax1.plot([], [], 'g--', label='Pitch2')

# For plotting arm orientation
line_upper_arm, = ax2.plot([], [], 'r-', linewidth=2, label='Upper Arm')
line_forearm, = ax2.plot([], [], 'g-', linewidth=2, label='Forearm')
ax2.set_xlim(-2, 2)
ax2.set_ylim(-2, 2)
shoulder = np.array([0, 0])  # Shoulder position

# Initialize Kalman Filters for roll and pitch
def init_kalman_filter():
    kf = KalmanFilter(dim_x=2, dim_z=1)
    kf.x = np.array([0., 0.])  # initial state
    kf.F = np.array([[1., 1.], [0., 1.]])  # state transition matrix
    kf.H = np.array([[1., 0.]])  # Measurement function
    kf.P *= 1000.  # covariance matrix
    kf.R = 5  # measurement noise
    kf.Q = Q_discrete_white_noise(dim=2, dt=1., var=0.1)  # process noise
    return kf

kf_roll = init_kalman_filter()
kf_pitch = init_kalman_filter()
kf_roll2 = init_kalman_filter()
kf_pitch2 = init_kalman_filter()

def init():
    ax1.set_xlim(0, 100)
    ax1.set_ylim(-120, 120)  # roll and pitch values range from -180 to 180 degrees
    ax1.legend()
    return ln_roll, ln_pitch, ln_roll2, ln_pitch2

def update(frame):
    global x_data, roll_data, pitch_data, roll_data2, pitch_data2
    data = conn.recv(1024).decode('utf-8').split(',')
    
    # Assuming the data contains 6 values, 3 for each dataset
    if len(data) == 6:
        accel_x, accel_y, accel_z, accel_x2, accel_y2, accel_z2 = map(float, data)
        print(accel_x, accel_y, accel_z, accel_x2, accel_y2, accel_z2)
        roll, pitch = compute_roll_pitch(accel_x, accel_y, accel_z)
        roll2, pitch2 = compute_roll_pitch(accel_x2, accel_y2, accel_z2)
        
        # Update Kalman Filters for the first dataset
        kf_roll.predict()
        kf_roll.update([roll])
        filtered_roll = kf_roll.x[0]
        
        kf_pitch.predict()
        kf_pitch.update([pitch])
        filtered_pitch = kf_pitch.x[0]

        # Update Kalman Filters for the second dataset
        kf_roll2.predict()
        kf_roll2.update([roll2])
        filtered_roll2 = kf_roll2.x[0]
        
        kf_pitch2.predict()
        kf_pitch2.update([pitch2])
        filtered_pitch2 = kf_pitch2.x[0]
        
        x_data.append(frame)
        roll_data.append(filtered_roll)
        pitch_data.append(filtered_pitch)
        roll_data2.append(filtered_roll2)
        pitch_data2.append(filtered_pitch2)
        
        ln_roll.set_data(x_data, roll_data)
        ln_pitch.set_data(x_data, pitch_data)
        ln_roll2.set_data(x_data, roll_data2)
        ln_pitch2.set_data(x_data, pitch_data2)
        
        if len(x_data) > 100:
            ax1.set_xlim(frame - 100, frame)
        
        # Update the orientation of the arm based on the filtered roll and pitch
        update_arm_orientation(filtered_roll, filtered_pitch, filtered_roll2, filtered_pitch2)
        #update_lines_3d(filtered_roll, filtered_pitch, filtered_roll2, filtered_pitch2)
        
    return ln_roll, ln_pitch, ln_roll2, ln_pitch2, line_upper_arm, line_forearm

def compute_roll_pitch(accel_x, accel_y, accel_z):
    roll = np.arctan2(accel_y, accel_z) * 57.2958
    pitch = np.arctan2(-accel_x, np.sqrt(accel_y**2 + accel_z**2)) * 57.2958
    return roll, pitch

def update_arm_orientation(roll_upper, pitch_upper, roll_forearm, pitch_forearm):
    # temp = roll_upper
    # roll_upper = pitch_upper
    # pitch_upper = temp

    # temp = roll_forearm
    # roll_forearm = pitch_forearm
    # pitch_forearm = temp

    # Convert degrees to radians
    roll_upper_rad = np.radians(roll_upper)
    pitch_upper_rad = np.radians(pitch_upper)
    roll_forearm_rad = np.radians(roll_forearm)
    pitch_forearm_rad = np.radians(pitch_forearm)

    # Lengths of the limbs
    upper_arm_length = 1
    forearm_length = 1

    # Calculate upper arm end position (elbow position)
    upper_arm_end = shoulder + np.array([upper_arm_length * np.cos(roll_upper_rad), upper_arm_length * np.sin(roll_upper_rad)])

    # For the forearm, consider its roll and pitch relative to the upper arm's end point
    # This calculation assumes the forearm's movement is somewhat independent but starts from the elbow
    # Adjust this logic if the forearm's pitch should directly follow the upper arm's orientation
    forearm_end = upper_arm_end + np.array([
        forearm_length * np.cos(roll_upper_rad + roll_forearm_rad + pitch_forearm_rad),
        forearm_length * np.sin(roll_upper_rad + roll_forearm_rad + pitch_forearm_rad)
    ])

    # Update line data for the upper arm and forearm
    line_upper_arm.set_data([shoulder[0], upper_arm_end[0]], [shoulder[1], upper_arm_end[1]])
    line_forearm.set_data([upper_arm_end[0], forearm_end[0]], [upper_arm_end[1], forearm_end[1]])

def update_lines_3d(roll_upper, pitch_upper, roll_forearm, pitch_forearm):
    return
    shoulder = np.array([0, 0, 0])  # Origin point for the shoulder in 3D

    # Update 3D arm orientation
    # Convert degrees to radians
    roll_upper_rad = np.radians(roll_upper)
    pitch_upper_rad = np.radians(pitch_upper)
    roll_forearm_rad = np.radians(roll_forearm)
    pitch_forearm_rad = np.radians(pitch_forearm)

    # Define lengths
    upper_arm_length = 1
    forearm_length = 1

    # Calculate upper arm end point in 3D
    upper_arm_end = shoulder + np.array([upper_arm_length * np.sin(roll_upper_rad), 
                                         upper_arm_length * np.cos(roll_upper_rad) * np.cos(pitch_upper_rad), 
                                         upper_arm_length * np.sin(pitch_upper_rad)])
    
    # Calculate forearm end point in 3D relative to the upper arm's end
    forearm_end = upper_arm_end + np.array([forearm_length * np.sin(roll_forearm_rad),
                                            forearm_length * np.cos(roll_forearm_rad) * np.cos(pitch_forearm_rad),
                                            forearm_length * np.sin(pitch_forearm_rad)])

    # Update 3D lines
    line3d_upper_arm.set_data([shoulder[0], upper_arm_end[0]], [shoulder[1], upper_arm_end[1]])
    line3d_upper_arm.set_3d_properties([shoulder[2], upper_arm_end[2]])
    line3d_forearm.set_data([upper_arm_end[0], forearm_end[0]], [upper_arm_end[1], forearm_end[1]])
    line3d_forearm.set_3d_properties([upper_arm_end[2], forearm_end[2]])
    
ani = FuncAnimation(fig, update, init_func=init, blit=True, interval=40, frames=1000)

plt.show()
