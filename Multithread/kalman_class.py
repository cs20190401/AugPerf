import numpy as np
from numpy.linalg import inv, norm, cholesky, LinAlgError

class KalmanFilter:
    # Based on "A Modified Kalman Filtering Approach to On-Line Musical Beat Tracking"
    def __init__(self, F=None, H=None, Q=None, R=None, x_0=None, P_0=None, T_0=100):
        """
        Kalman Filter initialization.
        Args:
            F (np.ndarray): State transition matrix.
            H (np.ndarray): Observation matrix.
            Q (np.ndarray): System noise covariance matrix.
            R (np.ndarray): Measurement noise covariance matrix.
            x_0 (np.ndarray): Initial state estimate.
            P_0 (np.ndarray): Initial error covariance matrix.
            T_0 (int): Period corresponding to 60 BPM at the desired resolution (frames per second).
        """
        # T_0 = 100 # 20ms window with 50% overlap = 10ms resolution = 100 fps
        dsquare = T_0 ** 2
        self.sigma = dsquare / 12
        self.r = self.sigma * 0.001

        self.F = F if F is not None else np.array([
            [1, 1],
            [0, 1] # Constant tempo
        ])
        
        # a = ?
        # self.F = F if F is not None else np.array([
        #     [1, 1],
        #     [0, a] # Variable tempo
        # ])

        self.H = H if H is not None else np.array([ # y = [beat time, beat interval]
            [1, 0], # Beat time
            [0, 1]  # Beat interval
        ])

        self.Q = Q if Q is not None else np.array([
            [self.sigma, 0],
            [0, self.sigma]
        ])
        self.R = R if R is not None else np.array([
            [self.r, 0],
            [0, self.r]
        ])
        
        self.x_0 = x_0 if x_0 is not None else np.zeros((2, 1))
        self.P_0 = P_0 if P_0 is not None else np.diag([0.1, 0.15]) # Measurement unit: seconds^2

        # Current state and covariance matrix
        self.x_esti = self.x_0
        self.P = self.P_0

        # Variables to update Q
        self.M = 3 # Number of prediction errors to use
        self.xi = 2.0 # Threshold; keep between 1.0 and 3.0
        self.prediction_errors = []

    def is_positive_definite(self, matrix):
        """Check if a matrix is positive definite."""
        try:
            cholesky(matrix)
            return True
        except LinAlgError:
            return False

    def update_Q(self):
        """Update Q based on the last M prediction errors (lock detection)."""
        if len(self.prediction_errors) < self.M:
            return

        mean_prediction_error = np.mean(self.prediction_errors)
        sqrt_r = np.sqrt(self.r)

        if mean_prediction_error < self.xi * sqrt_r:
            # Satisfactory => lock
            self.Q = np.array([
                [0, 0],
                [0, 0]
            ])
        else:
            # Unsatisfactory => unlock
            self.Q = np.array([
                [self.sigma, 0],
                [0, self.sigma]
            ])

    # Based on [https://github.com/tbmoon/kalman_filter/tree/master]
    def update(self, y_meas):
        """
        Perform a single Kalman filter update step.
        Args:
            y_meas (np.ndarray): Measurement vector (observation).
        Returns:
            np.ndarray: Updated state estimate.
        """
        # (1) Prediction
        x_pred = self.F @ self.x_esti
        P_pred = self.F @ self.P @ self.F.T + self.Q

        if not self.is_positive_definite(P_pred):
            print("Warning: covariance matrix is not positive definite!")

        # (2) Kalman Gain
        S = self.H @ P_pred @ self.H.T + self.R # Innovation covariance
        K = P_pred @ self.H.T @ inv(S)
        
        # (3) Estimation
        self.x_esti = x_pred + K @ (y_meas - self.H @ x_pred)

        # (4) Error Covariance (standard form)
        # self.P = P_pred - K @ self.H @ P_pred

        # (4) Error Covariance (Joseph form for numerical stability)
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ self.H) @ P_pred @ (I - K @ self.H).T + K @ self.R @ K.T

        # Update Q
        prediction_error = norm(y_meas - self.H @ x_pred)
        self.prediction_errors.append(prediction_error)
        if len(self.prediction_errors) > self.M:
            self.prediction_errors.pop(0)
        self.update_Q()

        return self.x_esti

    def get_state(self):
        """
        Get the current state estimate.
        Returns:
            np.ndarray: Current state estimate.
        """
        return self.x_esti