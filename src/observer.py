# observer.py

import numpy as np
from noise_constants import (
    ERROR_PERCENTS_NULL,
    ERROR_PERCENTS_LOW,
    ERROR_PERCENTS_MEDIUM,
    ERROR_PERCENTS_HIGH,
    ERROR_RATIOS,
)

class KalmanDisturbanceObserver:
    def __init__(self, A: np.ndarray, C: np.ndarray,
                 error_percents: dict, error_ratios: dict, y_means: dict,
                 lowpass_alpha: float = 0.3, anomaly_thresh: float = 5.0,
                 r_scale: float = 1.0, q_state_scale: float = 1.0, q_dist_scale: float = 1.0):
        self.n = A.shape[0]

        # Масштаби шумів зберігаємо ДО побудови R і Q
        self.r_scale       = r_scale
        self.q_state_scale = q_state_scale
        self.q_dist_scale  = q_dist_scale

        # Розширена модель x̄ = [x; d]
        phi = 0.995
        self.A_bar = np.block([
            [A,                     np.zeros((self.n, self.n))],
            [np.zeros((self.n, self.n)), phi * np.eye(self.n)]
        ])
        self.C_bar = np.hstack([C, np.eye(self.n)])

        # Генерація R та Q із урахуванням масштабів
        self.R = self._build_R(error_percents, error_ratios, y_means)
        self.Q = self._build_Q(error_percents, error_ratios, y_means)

        # Ініціалізація оцінок
        self.x_hat = np.zeros(2 * self.n)
        self.P     = np.eye(2 * self.n) * 1e-2

        # Параметри фільтрації вимірювань
        self.alpha  = lowpass_alpha
        self.thr    = anomaly_thresh
        self.y_prev = None

    def _build_R(self, err_p, err_r, y_means):
        target_keys = [
            'concentrate_fe_percent',
            'concentrate_mass_flow',
            'tailings_fe_percent',
        ]
        if len(target_keys) != self.n:
            raise ValueError(f"n={self.n} != len(target_keys)={len(target_keys)}")

        R = np.zeros((self.n, self.n))
        for i, key in enumerate(target_keys):
            ym   = y_means[key]
            perc = err_p[key] / 100.0 * ym
            abs_c, rel_c, lf_c = (
                perc * err_r[key][0],
                perc * err_r[key][1],
                perc * err_r[key][2],
            )
            sigma = np.sqrt(abs_c**2 + rel_c**2 + lf_c**2)
            R[i, i] = sigma**2

        # масштабування
        return R * self.r_scale

    def _build_Q(self, err_p, err_r, y_means):
        target_keys = [
            'concentrate_fe_percent',
            'concentrate_mass_flow',
            'tailings_fe_percent',
        ]
        if len(target_keys) != self.n:
            raise ValueError(f"n={self.n} != len(target_keys)={len(target_keys)}")

        Q = np.zeros((2 * self.n, 2 * self.n))
        for i, key in enumerate(target_keys):
            ym   = y_means[key]
            perc = err_p[key] / 100.0 * ym
            lf_c = perc * err_r[key][2]
            Q[i, i]                 = perc**2 * 1e-4
            Q[self.n + i, self.n + i] = max(lf_c**2, 1e-6)

        # state-шум
        Q[:self.n, :self.n]     *= self.q_state_scale
        # disturbance-шум
        Q[self.n:, self.n:]     *= self.q_dist_scale
        return Q

    def _filter_measurement(self, y: np.ndarray):
        if self.y_prev is None:
            self.y_prev = y.copy()
        # EWMA-фільтр
        y_f = self.alpha * y + (1 - self.alpha) * self.y_prev
        # Відсікання аномалій
        delta = np.abs(y - y_f)
        stds  = np.sqrt(np.diag(self.R))
        mask  = delta > self.thr * stds
        y_f[mask] = self.y_prev[mask]
        self.y_prev = y_f
        return y_f

    def predict(self, u: np.ndarray = None):
        # Прогнозування без входу B·u
        self.x_hat = self.A_bar @ self.x_hat
        self.P     = self.A_bar @ self.P @ self.A_bar.T + self.Q

    def update(self, y: np.ndarray):
        y = self._filter_measurement(y)
        S = self.C_bar @ self.P @ self.C_bar.T + self.R
        K = self.P @ self.C_bar.T @ np.linalg.inv(S)
        self.x_hat = self.x_hat + K @ (y - self.C_bar @ self.x_hat)
        self.P     = (np.eye(2 * self.n) - K @ self.C_bar) @ self.P
        return self.x_hat.copy()