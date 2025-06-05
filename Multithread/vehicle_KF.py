import numpy as np
import matplotlib.pyplot as plt

# Time step (e.g., 1 second)
dt = 1.0

# Initial state: [position, velocity]
x = np.array([[0],   # position
              [20]]) # velocity

# Initial uncertainty
P = np.array([[1000, 0],
              [0, 1000]])

# State transition matrix (constant velocity model)
F = np.array([[1, dt],
              [0, 1]])

# Control input matrix (not used in this example)
B = np.array([[0],
              [0]])

# Observation matrix (we only observe position)
H = np.array([[1, 0]])

# Measurement noise covariance (GPS position noise)
R = np.array([[5]])

# Process noise covariance (model uncertainty)
Q = np.array([[1, 0],
              [0, 1]])

# Number of steps
n = 50

# Simulated true position and measurements
true_positions = []
measurements = []
estimated_positions = []

# True position and speed (assumed constant)
true_pos = 0
true_speed = 20

for i in range(n):
    # Simulate true motion
    true_pos += true_speed * dt
    true_positions.append(true_pos)

    # Simulate GPS measurement with noise
    z = true_pos + np.random.normal(0, np.sqrt(R[0, 0]))
    measurements.append(z)

    # --- Kalman Filter Prediction ---
    x = F @ x
    P = F @ P @ F.T + Q

    # --- Kalman Filter Update ---
    y = z - H @ x                            # Measurement residual
    S = H @ P @ H.T + R                      # Residual covariance
    K = P @ H.T @ np.linalg.inv(S)          # Kalman gain
    x = x + K @ y                           # State update
    P = (np.eye(2) - K @ H) @ P             # Covariance update

    estimated_positions.append(x[0, 0])

# --- Plot results ---
plt.figure(figsize=(10, 6))
plt.plot(true_positions, label="True Position")
plt.plot(measurements, label="GPS Measurements", linestyle='dotted')
plt.plot(estimated_positions, label="Kalman Estimated Position")
plt.xlabel("Time Step")
plt.ylabel("Position")
plt.legend()
plt.title("1D Kalman Filter for Car Position Estimation")
plt.grid()
plt.show()
