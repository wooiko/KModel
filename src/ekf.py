# ekf.py
import numpy as np
from model import KernelModel
from sklearn.preprocessing import StandardScaler

class ExtendedKalmanFilter:
    def __init__(self,
                 model: KernelModel,
                 x_scaler: StandardScaler,
                 y_scaler: StandardScaler,
                 x0: np.ndarray,            # Початковий стан [x_phys_unscaled, d_scaled]
                 P0: np.ndarray,            # Початкова коваріація невизначеності
                 Q: np.ndarray,             # Коваріація шуму процесу
                 R: np.ndarray,             # Коваріація шуму вимірювань
                 lag: int):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.L = lag
        self.n_phys = (lag + 1) * 3
        self.n_dist = R.shape[0]
        self.n_aug = self.n_phys + self.n_dist

        # Стан та коваріація (стан зберігаємо в оригінальному масштабі, збурення - в масштабованому)
        self.x_hat = x0.copy()
        self.P = P0.copy()

        # Матриці шумів
        self.Q = Q
        self.R = R
        
        # Матриця переходу стану F (Якобіан від f) - вона буде сталою
        self._build_state_transition_matrix()

    def _build_state_transition_matrix(self):
        """ Будує матрицю F, Якобіан функції переходу стану f. """
        self.F = np.zeros((self.n_aug, self.n_aug))
        # Блок для фізичного стану (зсувний регістр)
        # Викидаємо найстаріший блок (3 змінні) і зсуваємо все вгору
        self.F[:self.n_phys - 3, 3:self.n_phys] = np.eye(self.n_phys - 3)
        # Блок для збурень (випадкове блукання, d_k = d_{k-1})
        self.F[self.n_phys:, self.n_phys:] = np.eye(self.n_dist)

    def predict(self, u_prev: float, d_measured: np.ndarray):
        """ Крок прогнозу EKF """
        x_phys_prev = self.x_hat[:self.n_phys]
        d_prev = self.x_hat[self.n_phys:]

        # 1. Прогноз стану x_hat_k|k-1 = f(x_hat_{k-1|k-1}, u_{k-1})
        x_phys_new = np.roll(x_phys_prev, -3)
        x_phys_new[-3:] = np.hstack([d_measured[0], d_measured[1], u_prev])
        d_new = d_prev # Модель руху для збурень - константа
        self.x_hat = np.hstack([x_phys_new, d_new])
        
        # 2. Прогноз коваріації P_k|k-1 = F * P_{k-1|k-1} * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z_k: np.ndarray):
        """ Крок корекції EKF. z_k - вектор реальних вимірювань (немасштабований). """
        # Розпаковуємо поточний (a priori) стан
        x_phys_unscaled = self.x_hat[:self.n_phys].reshape(1, -1)
        d_scaled = self.x_hat[self.n_phys:]

        # 1. Масштабуємо фізичний стан для моделі
        x_phys_scaled = self.x_scaler.transform(x_phys_unscaled)
        
        # 2. Обчислюємо Якобіан H_k моделі вимірювання h
        W_local_scaled, _ = self.model.linearize(x_phys_scaled)
        H_k = np.zeros((self.n_dist, self.n_aug))
        H_k[:, :self.n_phys] = W_local_scaled.T # W_local має форму (n_phys, n_dist)
        H_k[:, self.n_phys:] = np.eye(self.n_dist)

        # 3. Робимо прогноз вимірювання y_hat = h(x_hat_k|k-1)
        y_pred_scaled = self.model.predict(x_phys_scaled)[0]
        y_hat_scaled = y_pred_scaled + d_scaled
        
        # 4. Обчислюємо інновацію (нев'язку) y_tilde = z_k - y_hat
        z_k_scaled = self.y_scaler.transform(z_k.reshape(1, -1))[0]
        y_tilde = z_k_scaled - y_hat_scaled

        # 5. Обчислюємо коваріацію інновації S_k = H_k * P_k|k-1 * H_k^T + R
        S_k = H_k @ self.P @ H_k.T + self.R

        # 6. Обчислюємо підсилення Калмана K_k = P_k|k-1 * H_k^T * S_k^{-1}
        K_k = self.P @ H_k.T @ np.linalg.inv(S_k)

        # 7. Оновлюємо оцінку стану x_hat_k|k = x_hat_k|k-1 + K_k * y_tilde
        self.x_hat = self.x_hat + K_k @ y_tilde

        # 8. Оновлюємо коваріацію P_k|k = (I - K_k * H_k) * P_k|k-1
        I = np.eye(self.n_aug)
        self.P = (I - K_k @ H_k) @ self.P