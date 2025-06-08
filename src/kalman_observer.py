# kalman_observer.py

import numpy as np
import pandas as pd

class DisturbanceObserverKalman:
    def __init__(self, A_d=1.0, B_d=0.0, C_d=1.0, Q=1e-5, R=1e-2, P=1.0, d_est = 5.0):
        # A_d - динаміка збурення (можливо, 1 для постійного збурення)
        # C_d - модель спостереження
        self.A_d = A_d
        self.C_d = C_d
        self.Q = Q  # дисперсія процесу
        self.R = R  # дисперсія вимірювання
        self.d_est = d_est  # початкова оцінка збурення
        self.P = P  # початкова оцінка ковариації

    def update(self, y_meas, y_pred):
        # прогноз збурення
        d_pred = self.A_d * self.d_est
        P_pred = self.A_d * self.P * self.A_d + self.Q

        # інновація
        innov = y_meas - (y_pred + self.C_d * d_pred)
        S = self.C_d * P_pred * self.C_d + self.R
        
        # коефіцієнт Калмана
        K = P_pred * self.C_d / S
        
        # оновлення оцінки
        self.d_est = d_pred + K * innov
        self.P = (1 - K * self.C_d) * P_pred
        return self.d_est
    
class AdaptiveEKFilter:
    def __init__(self,
                 dt: float = 1.0,
                 process_noise_init: float = 0.1,
                 meas_noise_init: float = None,
                 lambda_forget: float = 0.95,
                 min_process_noise: float = 1e-6,
                 max_process_noise: float = 100.0,
                 min_meas_noise: float = 1e-4,
                 max_meas_noise: float = 100.0,
                 adaptive: bool = True):
        """ Ініціалізація адаптивного Калманівського фільтра """
        self.dt = dt
        self.q0 = process_noise_init
        self.r0 = meas_noise_init
        self.lmbd = lambda_forget
        self.q_min = min_process_noise
        self.q_max = max_process_noise
        self.r_min = min_meas_noise
        self.r_max = max_meas_noise
        self.adaptive = adaptive  # Керування адаптацією параметрів

    def filter(self, series: pd.Series) -> pd.Series:
        """ Фільтрує часовий ряд із адаптацією Q та R (якщо увімкнено). """
        z = series.to_numpy(dtype=float)
        n = len(z)
        if n == 0:
            return series

        x = np.array([z[0], 0.0], dtype=float)
        P = np.eye(2) * 1.0
        Q = self.q0 * np.eye(2)
        R = self.r0 if self.r0 is not None else max(np.var(z), self.r_min)

        H = np.array([[1.0, 0.0]])
        I = np.eye(2)

        filtered = np.empty_like(z)
        Rs, Qs = np.zeros(n), np.zeros(n)

        for k in range(n):
            F = np.array([[1.0, self.dt], [0.0, 1.0]])
            x_pred = F @ x
            P_pred = F @ P @ F.T + Q

            y = z[k] - (H @ x_pred)[0]
            S = (H @ P_pred @ H.T)[0, 0] + R
            K = (P_pred @ H.T)[:, 0] / S

            x = x_pred + K * y
            P = (I - np.outer(K, H)) @ P_pred
            filtered[k] = x[0]

            if self.adaptive:
                # Оновлення R на основі інновації
                innov_var = y**2 + (H @ P_pred @ H.T)[0, 0]
                R = self.lmbd * R + (1 - self.lmbd) * innov_var
                R = np.clip(R, self.r_min, self.r_max)

                # Оновлення Q на основі оцінки прискорення
                acc_est = (x[1] - x_pred[1]) / self.dt
                q_var = self.lmbd * (Q[1, 1] / self.dt**2) + (1 - self.lmbd) * acc_est**2
                Q = np.clip(q_var, self.q_min, self.q_max) * np.eye(2)

            Rs[k], Qs[k] = R, Q[1, 1]

        return pd.Series(filtered, index=series.index, name=series.name), Rs, Qs

