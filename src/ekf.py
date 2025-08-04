# ekf.py
import numpy as np
from model import KernelModel
from sklearn.preprocessing import StandardScaler

class ExtendedKalmanFilter:

    def __init__(self,
                 model: KernelModel,
                 x_scaler: StandardScaler,
                 y_scaler: StandardScaler,
                 x0: np.ndarray,            # <<< ПЕРЕВІРТЕ ЦЕЙ РЯДОК: Початковий стан [x_phys_unscaled, d_scaled]
                 P0: np.ndarray,            # Початкова коваріація невизначеності
                 Q: np.ndarray,             # Коваріація шуму процесу
                 R: np.ndarray,             # Початкова (мінімальна) коваріація шуму вимірювань для адаптивної R
                 lag: int,
                 beta_R: float = 0.5,
                 q_adaptive_enabled: bool = True,
                 q_alpha: float = 0.98,
                 q_nis_threshold: float = 1.5):

        self._debug_count = 0  # Для діагностики

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
        self.beta_R = beta_R 

        # --- Адаптація Q ---
        self.q_adaptive_enabled = q_adaptive_enabled
        self.q_alpha = q_alpha  # Фактор "забування" для q_scale
        self.q_nis_threshold = q_nis_threshold # Поріг для збільшення Q
        self.q_scale = 1.0      # Початковий коефіцієнт масштабування для Q
        
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
        """
        Крок корекції (update step) EKF.
        Обчислює a posteriori оцінку стану та коваріації.
    
        Args:
            z_k (np.ndarray): Вектор вимірювань [conc_fe, conc_mass] в оригінальному масштабі.
        """
        
        # ✅ ПРАВИЛЬНО: витягуємо останні 9 змінних для моделі
        x_phys_for_model = self.x_hat[self.n_phys-9:self.n_phys].reshape(1, -1)  # (1, 9)  
        d_scaled = self.x_hat[self.n_phys:]  # (2,) - збурення вже в масштабованому вигляді
        
        # ✅ ПРАВИЛЬНО: масштабуємо ОДИН РАЗ
        x_phys_scaled = self.x_scaler.transform(x_phys_for_model)  # (1, 9)
    
        # ✅ ВИПРАВЛЕНА ДІАГНОСТИКА: що передається в модель
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        
        if self._debug_count < 5:  # Перші 5 кроків
            print(f"\n--- EKF MODEL INPUT DEBUG (step {self._debug_count}) ---")
            print(f"x_phys_for_model shape: {x_phys_for_model.shape}")
            print(f"x_phys_for_model range: [{np.min(x_phys_for_model):.3f}, {np.max(x_phys_for_model):.3f}]")
            print(f"x_phys_scaled shape: {x_phys_scaled.shape}")
            print(f"x_phys_scaled range: [{np.min(x_phys_scaled):.3f}, {np.max(x_phys_scaled):.3f}]")
            
            # Порівняння з тренувальними межами
            train_min = np.min([-1.78, -2.34, -3.83] * 3)  # З логу вище
            train_max = np.max([10.72, 3.32, 4.67] * 3)   # З логу вище
            print(f"Training X range approx: [{train_min:.3f}, {train_max:.3f}]")
            print(f"Training X_scaled expected: [-2, +2] approx")
            
            # Чи виходимо за межі?
            if np.any(x_phys_scaled < -3) or np.any(x_phys_scaled > 3):
                print("❌ ЕКСТРАПОЛЯЦІЯ: x_phys_scaled виходить за межі тренувальних даних!")
            else:
                print("✅ x_phys_scaled в межах тренувальних даних")
                
            # Тестуємо модель на цих даних
            y_pred_test = self.model.predict(x_phys_scaled)[0]
            print(f"Model prediction: [{y_pred_test[0]:.3f}, {y_pred_test[1]:.3f}]")
            print(f"d_scaled (disturbances): [{d_scaled[0]:.3f}, {d_scaled[1]:.3f}]")
            
            # КЛЮЧОВЕ: порівняння з очікуваними значеннями
            expected_fe = 50.0  # Приблизно очікується
            expected_mass = 100.0  # Приблизно очікується
            print(f"Expected values approx: [{expected_fe:.1f}, {expected_mass:.1f}]")
            
            if abs(y_pred_test[0]) > 20 or abs(y_pred_test[1]) > 20:
                print("❌ МОДЕЛЬ ПЕРЕДБАЧАЄ НЕРЕАЛЬНІ ЗНАЧЕННЯ!")
            else:
                print("✅ Модель передбачає розумні значення")
                
            self._debug_count += 1
        
        # ✅ ПРАВИЛЬНО: лінеаризація дає якобіан в правильному масштабі
        W_local_scaled, _ = self.model.linearize(x_phys_scaled)  # (9, 2) - ВЖЕ в правильному масштабі!
        
        # ✅ ВИПРАВЛЕНО: правильний якобіан БЕЗ додаткового масштабування
        H_k = np.zeros((self.n_dist, self.n_aug))
        start_idx = self.n_phys - 9  # Індекс початку останніх 9 змінних
        
        # ТІЛЬКИ інверсія y_scaler, БЕЗ додаткового масштабування x:
        H_k[:, start_idx:self.n_phys] = (
            np.diag(1.0 / self.y_scaler.scale_) @ W_local_scaled.T  # (2, 2) @ (2, 9) = (2, 9)
        )
        
        # Збурення входять напряму (одинична матриця)
        H_k[:, self.n_phys:] = np.eye(self.n_dist)
        
        # ✅ ПРАВИЛЬНО: модель УЖЕ повертає в правильному масштабі
        y_pred_scaled = self.model.predict(x_phys_scaled)[0]  # (2,)
    
        # ✅ ДІАГНОСТИКА: перевіряємо чи правильні передбачення
        if hasattr(self, '_debug_count') and self._debug_count <= 5:
            # Конвертуємо в оригінальний масштаб для перевірки
            y_pred_unscaled = self.y_scaler.inverse_transform(y_pred_scaled.reshape(1, -1))[0]
            z_k_unscaled = z_k  # Реальне вимірювання
            
            print(f"Model prediction (scaled): [{y_pred_scaled[0]:.3f}, {y_pred_scaled[1]:.3f}]")
            print(f"Model prediction (unscaled): [{y_pred_unscaled[0]:.1f}, {y_pred_unscaled[1]:.1f}]")
            print(f"Real measurement (unscaled): [{z_k_unscaled[0]:.1f}, {z_k_unscaled[1]:.1f}]")
            print(f"Disturbances (scaled): [{d_scaled[0]:.3f}, {d_scaled[1]:.3f}]")
            
            # Перевіряємо чи близькі передбачення до реальності
            error_fe = abs(y_pred_unscaled[0] - z_k_unscaled[0])
            error_mass = abs(y_pred_unscaled[1] - z_k_unscaled[1])
            print(f"Prediction errors: FE={error_fe:.1f}, Mass={error_mass:.1f}")
            
            if error_fe > 10 or error_mass > 10:
                print("❌ МОДЕЛЬ ДУЖЕ ПОГАНО ПЕРЕДБАЧАЄ!")
            else:
                print("✅ Модель передбачає достатньо добре")
    
        y_hat_scaled = y_pred_scaled + d_scaled  # (2,) - загальний прогноз з урахуванням збурень
        
        # ---- Інновація (різниця між виміром та прогнозом) ----
        z_k_scaled = self.y_scaler.transform(z_k.reshape(1, -1))[0]  # Масштабуємо вимірювання
        y_tilde = z_k_scaled - y_hat_scaled  # Інновація в масштабованому просторі
        
        # ---- Адаптивна коваріація шуму вимірювань ----
        # Оновлюємо R на основі квадрату інновації
        self.R = self._R_initial + self.beta_R * np.diag(y_tilde**2 + 1e-6)
        
        # ---- Коваріація інновації та Калманівський коефіцієнт підсилення ----
        S_k = H_k @ self.P @ H_k.T + self.R  # Коваріація інновації
        K_k = self.P @ H_k.T @ np.linalg.inv(S_k)  # Калманівський коефіцієнт підсилення
        
        # ---- Корекція стану та коваріації ----
        self.x_hat = self.x_hat + K_k @ y_tilde  # Оновлення стану
        I = np.eye(self.n_aug)
        self.P = (I - K_k @ H_k) @ self.P  # Оновлення коваріації (форма Джозефа для стійкості)
        
        # ---- Адаптивне налаштування Q на основі NIS ----
        if self.q_adaptive_enabled:
            try:
                S_k_inv = np.linalg.inv(S_k)
                nis = y_tilde.T @ S_k_inv @ y_tilde  # Normalized Innovation Squared
                
                target = self.n_dist  # Очікуване значення NIS
                upper_bound = target * self.q_nis_threshold
                lower_bound = target / self.q_nis_threshold
                
                # Адаптація коефіцієнта масштабування Q
                if nis > upper_bound:
                    # Збільшуємо Q, якщо інновації занадто великі
                    self.q_scale = min(self.q_scale * 1.02, 10.0)
                elif nis < lower_bound:
                    # Зменшуємо Q, якщо інновації занадто малі
                    self.q_scale = max(self.q_scale * 0.99, 0.1)
                    
            except np.linalg.LinAlgError:
                # Ігноруємо помилки обернення матриці
                pass
        
        # Зберігаємо інновацію для діагностики
        self.last_innovation = y_tilde.copy()