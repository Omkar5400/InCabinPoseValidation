# src/ekf_refinement.py
import numpy as np

class JointEKF:
    """Constant Velocity 3D Kalman Filter for a single anatomical joint."""
    def __init__(self, dt=0.033):
        self.dt = dt
        # State vector: [x, y, z, vx, vy, vz]
        self.x = np.zeros((6, 1))
        
        # State transition matrix
        self.F = np.eye(6)
        for i in range(3):
            self.F[i, i + 3] = dt

        # Measurement matrix (observing x, y, z)
        self.H = np.zeros((3, 6))
        self.H[0:3, 0:3] = np.eye(3)

        # Covariance matrices
        self.P = np.eye(6) * 1.0
        self.Q = np.eye(6) * 0.01  # Process noise
        self.R = np.eye(3) * 0.05  # Measurement noise

    def predict(self):
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x[0:3].flatten()

    def update(self, measurement):
        z = np.reshape(measurement, (3, 1))
        y = z - np.dot(self.H, self.x)  # Measurement residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x[0:3].flatten()

def enforce_bone_length(parent_joint, child_joint, nominal_length, tolerance=0.15):
    """Enforces constant biological bone length between connected joints."""
    vector = child_joint - parent_joint
    measured_len = np.linalg.norm(vector)
    if measured_len == 0:
        return child_joint
    
    # Restrict bone length within [1 - tol, 1 + tol] * nominal_length
    min_len = nominal_length * (1 - tolerance)
    max_len = nominal_length * (1 + tolerance)
    
    clamped_len = np.clip(measured_len, min_len, max_len)
    corrected_child = parent_joint + (vector / measured_len) * clamped_len
    return corrected_child