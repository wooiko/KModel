# ekf.py
import numpy as np
from model import KernelModel
from sklearn.preprocessing import StandardScaler

class ExtendedKalmanFilter:

    def __init__(self,
                 model: KernelModel,
                 x_scaler: StandardScaler,
                 y_scaler: StandardScaler,
                 x0: np.ndarray,            # ✅ ВИПРАВЛЕНО: Консистентний початковий стан
                 P0: np.ndarray,            # Початкова коваріація невизначеності
                 Q: np.ndarray,             # Коваріація шуму процесу
                 R: np.ndarray,             # Початкова коваріація шуму вимірювань для адаптивної R
                 lag: int,
                 beta_R: float = 0.5,
                 q_adaptive_enabled: bool = True,
                 q_alpha: float = 0.98,
                 q_nis_threshold: float = 1.5,
                 use_scaled_state: bool = True):  # ✅ НОВИЙ: Флаг масштабування
    
        self._debug_count = 0  # Для діагностики
    
        self.model = model
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.L = lag
        self.n_phys = (lag + 1) * 3  # Розмірність фізичної частини стану
        self.n_dist = R.shape[0]     # Розмірність вектора збурень
        self.n_aug = self.n_phys + self.n_dist # Загальна розмірність розширеного стану
        
        # ✅ ВИПРАВЛЕННЯ: Флаг для відстеження масштабування
        self.use_scaled_state = use_scaled_state
    
        # ✅ ВИПРАВЛЕННЯ: Стан тепер консистентно масштабований
        # Якщо use_scaled_state=True: ВСЯ x_hat (і фізична, і збурення) масштабовані
        # Якщо use_scaled_state=False: ВСЯ x_hat немасштабована (для зворотної сумісності)
        self.x_hat = x0.copy()
        self.P = P0.copy()
        
        if self.use_scaled_state:
            print(f"   ✅ EKF initialized with SCALED state (all components scaled)")
            print(f"      State range: [{self.x_hat.min():.3f}, {self.x_hat.max():.3f}]")
        else:
            print(f"   ⚠️  EKF initialized with UNSCALED state (legacy mode)")
    
        # Матриці шумів
        self.Q = Q
        self._R_initial = R 
        self.R = R 
        self.beta_R = beta_R 
    
        # Адаптація Q
        self.q_adaptive_enabled = q_adaptive_enabled
        self.q_alpha = q_alpha
        self.q_nis_threshold = q_nis_threshold
        self.q_scale = 1.0
        
        # Матриця переходу стану F
        self._build_state_transition_matrix()
        
        # Атрибут для зберігання останньої інновації
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

        # ---- 1. Прогноз стану x_hat_k|k-1 = f(x_hat_{k-1|k-1}, u_{k-1})
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
        
        # ---- 2. Прогноз коваріації P_k|k-1 = F * P_{k-1|k-1} * F^T + Q
        # Застосовуємо адаптивний коефіцієнт до Q
        self.P = self.F @ self.P @ self.F.T + (self.Q * self.q_scale)
        


    def update(self, z_k: np.ndarray):
        """EKF update step with consistent scaling"""
        
        # ✅ ВИПРАВЛЕННЯ: Обробка стану залежно від масштабування
        if self.use_scaled_state:
            # Весь стан вже масштабований, просто беремо фізичну частину
            x_phys_scaled = self.x_hat[:self.n_phys].reshape(1, -1)
            d_scaled = self.x_hat[self.n_phys:]
            
            # Діагностика для перших кроків
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            
            if self._debug_count < 3:
                print(f"   📊 EKF step {self._debug_count}: Using SCALED state directly")
                print(f"      Physical state range: [{x_phys_scaled.min():.3f}, {x_phys_scaled.max():.3f}]")
                print(f"      Disturbances: [{d_scaled.min():.3f}, {d_scaled.max():.3f}]")
                self._debug_count += 1
            else:
                self._debug_count += 1
                
        else:
            # Legacy режим: фізична частина немасштабована, потрібно масштабувати
            x_phys_unscaled = self.x_hat[:self.n_phys].reshape(1, -1)
            x_phys_scaled = self.x_scaler.transform(x_phys_unscaled)
            d_scaled = self.x_hat[self.n_phys:]
            
            if not hasattr(self, '_debug_count'):
                self._debug_count = 0
            
            if self._debug_count < 3:
                print(f"   ⚠️  EKF step {self._debug_count}: Converting UNSCALED to scaled state")
                self._debug_count += 1
            else:
                self._debug_count += 1
        
        # ✅ Основні обчислення EKF (без змін)
        W_local_scaled, _ = self.model.linearize(x_phys_scaled)
        
        H_k = np.zeros((self.n_dist, self.n_aug))
        start_idx = self.n_phys - 9  # Останні 3 точки (9 значень)
        H_k[:, start_idx:self.n_phys] = (
            np.diag(1.0 / self.y_scaler.scale_) @ W_local_scaled.T
        )
        H_k[:, self.n_phys:] = np.eye(self.n_dist)
        
        # Передбачення виходу
        y_pred_scaled = self.model.predict(x_phys_scaled)[0]
        y_pred_unscaled = self.y_scaler.inverse_transform(y_pred_scaled.reshape(1, -1))[0]
        
        # Масштабування вимірювань для порівняння
        z_k_scaled = self.y_scaler.transform(z_k.reshape(1, -1))[0]
        
        # Обчислення інновації (в масштабованих координатах)
        y_k = z_k_scaled - (y_pred_scaled + d_scaled)
        self.last_innovation = y_k.copy()
        
        # Коваріація інновації та оновлення адаптивної R
        S_k = H_k @ self.P @ H_k.T + self.R
        
        if hasattr(self, 'beta_R') and self.beta_R > 0:
            innovation_cov = np.outer(y_k, y_k)
            self.R = (1 - self.beta_R) * self.R + self.beta_R * innovation_cov
            self.R = np.maximum(self.R, self._R_initial * 0.1)
        
        # Підсилення Калмана
        try:
            K_k = self.P @ H_k.T @ np.linalg.inv(S_k)
        except np.linalg.LinAlgError:
            K_k = self.P @ H_k.T @ np.linalg.pinv(S_k)
        
        # Оновлення стану та коваріації
        self.x_hat = self.x_hat + K_k @ y_k
        I_KH = np.eye(self.n_aug) - K_k @ H_k
        self.P = I_KH @ self.P @ I_KH.T + K_k @ self.R @ K_k.T
        
        # Адаптація Q на основі NIS
        if self.q_adaptive_enabled:
            nis = y_k.T @ np.linalg.inv(S_k) @ y_k
            if nis > self.q_nis_threshold:
                self.q_scale = min(self.q_scale * 1.1, 5.0)
            else:
                self.q_scale = max(self.q_scale * self.q_alpha, 0.1)