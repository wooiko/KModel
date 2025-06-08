# observers.py

import numpy as np
import pandas as pd
from tqdm import tqdm

class DisturbanceObserverKalman:
    def __init__(self, A_d=1.0, B_d=0.0, C_d=1.0, Q=1e-5, R=1e-2, P=1.0, d_est=5.0,
                tracking=True, tracking_threshold=3.0, tracking_factor=10.0):
        # A_d - динаміка збурення (можливо, 1 для постійного збурення)
        # C_d - модель спостереження
        self.A_d = A_d
        self.C_d = C_d
        self.Q = Q  # дисперсія процесу
        self.R = R  # дисперсія вимірювання
        self.d_est = d_est  # початкова оцінка збурення
        self.P = P  # початкова оцінка ковариації
        
        # Параметри режиму відстеження
        self.tracking = tracking  # увімкнути режим відстеження
        self.tracking_threshold = tracking_threshold  # поріг у кількості стандартних відхилень
        self.tracking_factor = tracking_factor  # множник для Q при великих інноваціях
    
    def update(self, y_meas, y_pred):
        # Обчислення інновації
        innov = y_meas - y_pred - self.C_d * self.d_est
        
        # Стандартний розрахунок підсилення Калмана
        S = self.C_d * self.P * self.C_d + self.R
        K = (self.P * self.C_d) / S
        
        # Оновлення стану та коваріації
        self.d_est = self.d_est + K * innov
        self.P = (1 - K * self.C_d) * self.P
        
        # Параметр збільшення Q для режиму відстеження
        Q_factor = 1.0
        if self.tracking and abs(innov) > self.tracking_threshold * np.sqrt(self.R):
            Q_factor = self.tracking_factor
        
        # Прогноз з можливо збільшеним Q
        self.d_est = self.A_d * self.d_est
        self.P = self.A_d * self.P * self.A_d + Q_factor * self.Q
        
        return self.d_est, innov, S
    
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

class ExponentialFilter:
    def __init__(self, alpha=0.3):
        """
        alpha: параметр згладжування (0 < alpha ≤ 1)
            - більше alpha = більша реакція на зміни
            - менше alpha = більше згладжування
        """
        self.alpha = alpha
        self.state = None
        
        # Додаємо A_d як еквівалент (1-alpha) для сумісності з Калманом
        self.A_d = 1.0 - alpha
        
    def update(self, measurement, prediction=None):
        """Оновлює стан фільтра і повертає фільтроване значення."""
        if self.state is None:
            self.state = measurement
            return measurement, 0, 1.0
        
        # Обчислення інновації
        innov = measurement - self.state
        
        # Оновлення стану
        self.state = self.state + self.alpha * innov
        
        # Повертаємо оцінку, інновацію та 1.0 для сумісності з інтерфейсом Калмана
        return self.state, innov, 1.0
    
def evaluate_params(y_meas: np.ndarray, y_pred: np.ndarray,
                    Q: float, R: float) -> float:
    """
    Проганяємо фільтр з даними Q, R і рахуємо -log likelihood.
    """
    kf = DisturbanceObserverKalman(Q=Q, R=R)
    neg_log_likelihood = 0.0
    for ym, yp in zip(y_meas, y_pred):
        _, innov, S = kf.update(ym, yp)
        # внесок кожного кроку:
        neg_log_likelihood += 0.5 * (np.log(2 * np.pi * S) + innov**2 / S)
    return neg_log_likelihood

def grid_search_qr(y_meas: pd.Series, y_pred: pd.Series,
                   Q_range: np.ndarray, R_range: np.ndarray):
    best_score = np.inf
    best_params = {'Q': None, 'R': None}
    ym = y_meas.to_numpy()
    yp = y_pred.to_numpy()
    for Q in tqdm(Q_range, desc="GridSearch Q"):
        for R in R_range:
            score = evaluate_params(ym, yp, Q, R)
            if score < best_score:
                best_score = score
                best_params = {'Q': Q, 'R': R}
    return best_params, best_score

def select_filter_params():
    """
    Повертає рекомендовані параметри alpha для експоненційних фільтрів
    """
    # Можна підібрати ці значення експериментально
    alpha_feed = 0.2   # Для feed_fe_percent
    alpha_ore = 0.3    # Для ore_mass_flow
    alpha_fe = 0.1     # Для концентрату Fe
    alpha_mass = 0.15  # Для потоку маси
    
    print(f"Параметри фільтрів: alpha_feed={alpha_feed}, alpha_ore={alpha_ore}")
    print(f"Параметри вихідних фільтрів: alpha_fe={alpha_fe}, alpha_mass={alpha_mass}")
    
    return alpha_feed, alpha_ore, alpha_fe, alpha_mass