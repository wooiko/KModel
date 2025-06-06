# kalman_observer.py

import numpy as np

class DisturbanceObserverKalman:
    def __init__(self, A_d=1.0, B_d=0.0, C_d=1.0, Q=1e-5, R=1e-2):
        # A_d - динаміка збурення (можливо, 1 для постійного збурення)
        # C_d - модель спостереження
        self.A_d = A_d
        self.C_d = C_d
        self.Q = Q  # дисперсія процесу
        self.R = R  # дисперсія вимірювання
        self.d_est = 0.0  # початкова оцінка збурення
        self.P = 1.0  # початкова оцінка ковариації

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