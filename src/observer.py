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
    def __init__(self, A_full: np.ndarray, C_full: np.ndarray,
                 error_percents: dict, error_ratios: dict, y_means: dict,
                 lowpass_alpha: float = 0.3, anomaly_thresh: float = 5.0,
                 r_scale: float = 1.0, q_state_scale: float = 1.0,
                 q_dist_scales: dict[str, float] = None):

        # 1) Перелік ключів для всіх трьох каналів
        self.target_keys = [
            'concentrate_fe_percent',
            'concentrate_mass_flow',
            'tailings_fe_percent',
        ]
        self.n = len(self.target_keys)  # 3

        # 2) Повні матриці
        A = A_full
        C = C_full

        # 3) Якщо потрібні індивідуальні масштаби disturbance-шуму
        if q_dist_scales is None:
            q_dist_scales = {k: 1e-3 for k in self.target_keys}
        self.q_dist_scales = q_dist_scales

        # 4) Зберігаємо решту масштабів
        self.r_scale       = r_scale
        self.q_state_scale = q_state_scale

        # 5) Побудова розширеної моделі
        phi = 0.999
        self.A_bar = np.block([
            [A,                     np.zeros((self.n, self.n))],
            [np.zeros((self.n, self.n)), phi * np.eye(self.n)]
        ])
        self.C_bar = np.hstack([C, np.eye(self.n)])

        # 6) Генеруємо R і Q
        self.R = self._build_R(error_percents, error_ratios, y_means)
        self.Q = self._build_Q(error_percents, error_ratios, y_means)

        print("R =", self.R)
        print("Q =", self.Q)

        # 7) Ініціалізація фільтра
        self.x_hat = np.zeros(2 * self.n)
        self.P = np.eye(2 * self.n) * 1e-2
        for i in range(self.n, 2*self.n):
            self.P[i, i] = 1e3  # або навіть 1e3, якщо Kalman надто “інертний”
        self.alpha = lowpass_alpha
        self.thr   = anomaly_thresh
        self.y_prev = None

    def _build_R(self, err_p, err_r, y_means):
        # Перевірка: чи всі ключі в наявності
        for key in self.target_keys:
            if key not in err_p or key not in err_r or key not in y_means:
                raise KeyError(f"Канал '{key}' відсутній у одному з вхідних словників.")
        R = np.zeros((self.n, self.n))
        for i, key in enumerate(self.target_keys):
            ym   = y_means[key]
            perc = err_p[key] / 100.0 * ym
            abs_c = perc * err_r[key][0]
            rel_c = perc * err_r[key][1]
            lf_c  = perc * err_r[key][2]
            sigma = np.sqrt(abs_c**2 + rel_c**2 + lf_c**2)
            R[i, i] = sigma**2
        return R * self.r_scale
    
    def _build_Q(self, err_p, err_r, y_means):
        # Перевірка: чи всі ключі в наявності
        for key in self.target_keys:
            if key not in err_p or key not in err_r or key not in y_means:
                raise KeyError(f"Канал '{key}' відсутній у одному з вхідних словників.")
        Q = np.zeros((2*self.n, 2*self.n))
        for i, key in enumerate(self.target_keys):
            ym   = y_means[key]
            perc = err_p[key] / 100.0 * ym
            lf_c = perc * err_r[key][2]
            # шум для стану
            Q[i, i] = perc**2 * 1e-4 * self.q_state_scale
            # шум для disturbance (+ індивідуальний масштаб)
            base = max(lf_c**2, 1e-12)
            Q[self.n+i, self.n+i] = base * self.q_dist_scales[key]
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