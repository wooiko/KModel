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
                 R: np.ndarray,             # Початкова (мінімальна) коваріація шуму вимірювань для адаптивної R
                 lag: int):
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.L = lag
        self.n_phys = (lag + 1) * 3  # Розмірність фізичної частини стану (L+1 блоків по 3 змінні)
        self.n_dist = R.shape[0]     # Розмірність вектора збурень (дисторбансів)
        self.n_aug = self.n_phys + self.n_dist # Загальна розмірність розширеного стану

        # Стан та коваріація (фізичний стан зберігаємо в оригінальному масштабі, збурення - в масштабованому)
        self.x_hat = x0.copy()
        self.P = P0.copy()

        # Матриці шумів
        self.Q = Q
        self._R_initial = R 
        self.R = R 
        
        # Матриця переходу стану F
        self._build_state_transition_matrix()
        
        # Новий атрибут для зберігання останньої інновації
        self.last_innovation = None        

    def _build_state_transition_matrix(self):
        """
        Будує матрицю переходу стану F, яка є якобіаном функції переходу стану f.
        F відображає, як лінеаризований стан x_k залежить від x_{k-1}.
        """
        self.F = np.zeros((self.n_aug, self.n_aug))
        
        # Блок для фізичного стану (зсувний регістр):
        # Перші n_phys - 3 елементів нового фізичного стану є зсунутими елементами попереднього фізичного стану
        # (тобто x_phys_new[i] = x_phys_prev[i+3]). Це відображається через одиничну матрицю,
        # розташовану так, що елемент (i, i+3) дорівнює 1.
        self.F[:self.n_phys - 3, 3:self.n_phys] = np.eye(self.n_phys - 3)
        
        # Останні 3 елементи фізичного стану в predict методі перезаписуються значеннями
        # d_measured та u_prev (які є входами, а не частиною попереднього стану).
        # Тому ці частини якобіана (рядки, що відповідають цим елементам) є нульовими
        # по відношенню до попереднього стану. Це є коректним для даної моделі.

        # Блок для збурень (дисторбансів):
        # Модель руху для збурень: d_k = d_{k-1} (випадкове блукання з нульовим зміщенням).
        # Це відображається одиничною матрицею для блоку збурень.
        self.F[self.n_phys:, self.n_phys:] = np.eye(self.n_dist)

    def predict(self, u_prev: float, d_measured: np.ndarray):
        """
        Крок прогнозу (prediction step) EKF.
        Обчислює a priori оцінку стану та коваріації.

        Args:
            u_prev (float): Попереднє значення керованої змінної.
            d_measured (np.ndarray): Вектор з двох виміряних дисторбансів (d_fe, d_mass).
        """
        x_phys_prev = self.x_hat[:self.n_phys]  # Попередня оцінка фізичного стану
        d_prev = self.x_hat[self.n_phys:]       # Попередня оцінка збурень

        # 1. Прогноз стану x_hat_k|k-1 = f(x_hat_{k-1|k-1}, u_{k-1})
        # Модель переходу для фізичної частини (зсувний регістр):
        # Зсуваємо всі елементи фізичного стану на 3 позиції назад (викидаємо найстаріші 3 елементи)
        x_phys_new = np.roll(x_phys_prev, -3)
        # Останні 3 елементи нового фізичного стану заповнюються новими вимірами (d_fe, d_mass) та u_prev
        x_phys_new[-3:] = np.hstack([d_measured[0], d_measured[1], u_prev])
        
        # Модель переходу для збурень: припускаємо, що вони не змінюються
        # (модель випадкового блукання з нульовим зміщенням)
        d_new = d_prev 
        
        # Оновлюємо повний розширений стан
        self.x_hat = np.hstack([x_phys_new, d_new])
        
        # 2. Прогноз коваріації P_k|k-1 = F * P_{k-1|k-1} * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
    def update(self, z_k: np.ndarray):
        """
        Крок корекції (update step) EKF.
        Обчислює a posteriori оцінку стану та коваріації на основі поточного вимірювання.

        Args:
            z_k (np.ndarray): Вектор реальних (немасштабованих) вимірювань виходу (фактичний Fe і Mass).
        """
        # Розпаковуємо поточний (a priori) стан
        x_phys_unscaled = self.x_hat[:self.n_phys].reshape(1, -1)
        d_scaled = self.x_hat[self.n_phys:]

        # 1. Масштабуємо фізичний стан для використання в моделі (як того вимагає KernelModel)
        x_phys_scaled = self.x_scaler.transform(x_phys_unscaled)
        
        # 2. Обчислюємо Якобіан H_k моделі вимірювання h.
        W_local_scaled, _ = self.model.linearize(x_phys_scaled)
        
        H_k = np.zeros((self.n_dist, self.n_aug))
        H_k[:, :self.n_phys] = W_local_scaled.T @ np.diag(1.0 / self.x_scaler.scale_[:self.n_phys])
        H_k[:, self.n_phys:] = np.eye(self.n_dist)

        # 3. Робимо прогноз вимірювання y_hat = h(x_hat_k|k-1)
        y_pred_scaled = self.model.predict(x_phys_scaled)[0]
        y_hat_scaled = y_pred_scaled + d_scaled 
        
        # 4. Обчислюємо інновацію (нев'язку) y_tilde = z_k - y_hat
        z_k_scaled = self.y_scaler.transform(z_k.reshape(1, -1))[0]
        y_tilde = z_k_scaled - y_hat_scaled

        # 5. Обчислюємо коваріацію інновації S_k = H_k * P_k|k-1 * H_k^T + R
        beta = 0.5 
        self.R = self._R_initial + beta * np.diag(np.abs(y_tilde) + 1e-6) 
        
        S_k = H_k @ self.P @ H_k.T + self.R

        # 6. Обчислюємо підсилення Калмана K_k = P_k|k-1 * H_k^T * S_k^{-1}
        K_k = self.P @ H_k.T @ np.linalg.inv(S_k)

        # 7. Оновлюємо оцінку стану x_hat_k|k = x_hat_k|k-1 + K_k * y_tilde
        self.x_hat = self.x_hat + K_k @ y_tilde
        
        # 8. Оновлюємо коваріацію P_k|k = (I - K_k * H_k) * P_k|k-1
        I = np.eye(self.n_aug)
        self.P = (I - K_k @ H_k) @ self.P
        
        # Зберігаємо інновацію
        self.last_innovation = y_tilde