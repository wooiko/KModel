# mpc.py - Модифіковані методи з покращеною лінеаризацією

import warnings
import numpy as np
import cvxpy as cp
from model import KernelModel
from sklearn.preprocessing import StandardScaler 

class MPCController:
    def __init__(self,
                 model: KernelModel,
                 objective,
                 x_scaler: StandardScaler,
                 y_scaler: StandardScaler,
                 n_targets: int,
                 horizon: int = 6,
                 control_horizon: int = None,
                 lag: int = 2,
                 u_min: float = 25.0,
                 u_max: float = 35.0,
                 delta_u_max: float = 1.0,
                 use_disturbance_estimator: bool = True,
                 y_max: list = None,
                 y_min: list = None,
                 rho_y: float = 1e6,
                 rho_delta_u: float = 1e4,
                 rho_trust: float = 0.1,
                 # === НОВІ ПАРАМЕТРИ ===
                 adaptive_trust_region: bool = True,
                 initial_trust_radius: float = 2.0,
                 min_trust_radius: float = 0.5,
                 max_trust_radius: float = 5.0,
                 trust_decay_factor: float = 0.9,
                 linearization_check_enabled: bool = True,
                 max_linearization_distance: float = 2.0
                ):
        self.Np = horizon
        self.Nc = control_horizon if control_horizon is not None else horizon

        if self.Nc > self.Np:
            raise ValueError("control_horizon (Nc) не може бути більше за horizon (Np)")

        # Збереження основних параметрів
        self.model = model
        self.objective = objective
        self.x_scaler = x_scaler
        self.y_scaler = y_scaler
        self.L = lag
        self.use_disturbance_estimator = use_disturbance_estimator
        self.n_inputs = (lag + 1) * 3
        self.n_targets = n_targets

        # Збереження обмежень та ваг штрафів
        self.u_min = u_min
        self.u_max = u_max
        self.delta_u_max = delta_u_max
        self.y_max = np.array(y_max) if y_max is not None else None
        self.y_min = np.array(y_min) if y_min is not None else None
        self.rho_y = rho_y
        self.rho_delta_u = rho_delta_u
        self.rho_trust = rho_trust

        # === НОВІ АТРИБУТИ ДЛЯ АДАПТИВНОГО TRUST REGION ===
        self.adaptive_trust_region = adaptive_trust_region
        self.trust_region_radius = initial_trust_radius
        self.min_trust_radius = min_trust_radius
        self.max_trust_radius = max_trust_radius
        self.trust_decay_factor = trust_decay_factor
        self.linearization_check_enabled = linearization_check_enabled
        self.max_linearization_distance = max_linearization_distance
        
        # Історія для адаптації
        self.previous_cost = None
        self.predicted_cost_reduction = None
        self.linearization_quality_history = []

        # Ініціалізація історії та оцінки збурень
        self.x_hist = None
        self.d_hat = np.zeros(self.n_targets) if self.use_disturbance_estimator else None

        # Атрибути для збереження задачі CVXPY
        self.problem = None
        self.variables = {}
        self.parameters = {}

        # Налаштовуємо задачу оптимізації один раз при ініціалізації
        self._setup_optimization_problem()

        # Приховуємо специфічне попередження від cvxpy
        warnings.filterwarnings("ignore",
            message="The problem includes expressions that don't support CPP backend.",
            category=UserWarning,
            module="cvxpy.reductions.solvers.solving_chain_utils")


    def _setup_optimization_problem(self):
        """
        Створює задачу оптимізації CVXPY з правильним масштабуванням даних.
        """
        # 1. Змінні оптимізації
        u_var = cp.Variable(self.Nc, name="u_seq")
        eps_delta_u_upper = cp.Variable(self.Nc, nonneg=True, name="eps_delta_u_upper")
        eps_delta_u_lower = cp.Variable(self.Nc, nonneg=True, name="eps_delta_u_lower")
        eps_y_upper = cp.Variable((self.Np, 2), nonneg=True, name="eps_y_upper") if self.y_max is not None else None
        eps_y_lower = cp.Variable((self.Np, 2), nonneg=True, name="eps_y_lower") if self.y_min is not None else None
        
        self.variables = {
            'u': u_var, 
            'eps_delta_u_upper': eps_delta_u_upper,
            'eps_delta_u_lower': eps_delta_u_lower,
            'eps_y_upper': eps_y_upper, 
            'eps_y_lower': eps_y_lower
        }
    
        # 2. Параметри, що будуть оновлюватись на кожному кроці
        W_param = cp.Parameter((9, self.n_targets), name="W_local")  # ✅ ВИПРАВЛЕНО: точна розмірність 9x2
        b_param = cp.Parameter(self.n_targets, name="b_local")
        x_hist_param = cp.Parameter(self.n_inputs, name="x_hist_flat")  # Повна історія (lag+1)*3
        u_prev_param = cp.Parameter(name="u_prev")
        d_hat_param = cp.Parameter(self.n_targets, name="d_hat")
        d_seq_param = cp.Parameter((self.Np, 2), name="d_seq")
        x0_scaled_param = cp.Parameter(9, name="x0_scaled")  # ✅ ВИПРАВЛЕНО: тільки 9 змінних для моделі
        trust_radius_param = cp.Parameter(nonneg=True, name="trust_radius")
        
        self.parameters = {
            'W': W_param, 'b': b_param, 'x_hist': x_hist_param,
            'u_prev': u_prev_param, 'd_hat': d_hat_param, 'd_seq': d_seq_param,
            'x0_scaled': x0_scaled_param, 'trust_radius': trust_radius_param
        }
    
        # ✅ ВИПРАВЛЕНО: Константи для масштабування ТІЛЬКИ останніх 9 змінних
        # Витягуємо скалери для останніх 9 змінних (які подаються в модель)
        model_input_indices = slice(-9, None)  # Останні 9 позицій
        mean_c_model = cp.Constant(self.x_scaler.mean_[model_input_indices])  # (9,)
        scale_c_model = cp.Constant(self.x_scaler.scale_[model_input_indices])  # (9,)
    
        # 3. Побудова прогнозу на горизонті Np з правильним масштабуванням
        pred_fe, pred_mass = [], []
        trust_region_cost = 0
        
        # Початковий стан з параметра (повна історія)
        xk_unscaled_list = [x_hist_param[i*3:(i+1)*3] for i in range(self.L + 1)]
    
        for k in range(self.Np):
            uk = u_var[k] if k < self.Nc else u_var[self.Nc - 1]
            
            # ✅ ПРАВИЛЬНО: Формуємо повний вектор стану для збереження історії
            Xk_unscaled_full = cp.hstack(xk_unscaled_list)  # Повна історія: (lag+1)*3
            
            # ✅ ПРАВИЛЬНО: Витягуємо тільки останні 9 змінних для передачі в модель
            Xk_for_model = Xk_unscaled_full[-9:]  # Останні 9 змінних
            
            # ✅ ПРАВИЛЬНО: Масштабуємо тільки змінні моделі правильними скалерами
            Xk_scaled = (Xk_for_model - mean_c_model) / scale_c_model  # (9,)
            
            # ✅ ПРАВИЛЬНО: Прогноз за локальною лінійною моделлю з правильними розмірностями
            yk = Xk_scaled @ W_param + b_param + d_hat_param  # (9,) @ (9,2) + (2,) + (2,) = (2,)
            pred_fe.append(yk[0])
            pred_mass.append(yk[1])
    
            # === ПРАВИЛЬНИЙ TRUST REGION PENALTY ===
            if self.model.kernel != 'linear':
                # Застосовуємо штраф тільки для змінних моделі
                weight = self.rho_trust * (self.trust_decay_factor ** k)
                
                # ✅ ПРАВИЛЬНО: Нормована відстань для змінних моделі
                model_state_deviation = Xk_scaled - x0_scaled_param  # (9,) - (9,) = (9,)
                trust_region_cost += weight * cp.sum_squares(model_state_deviation) / trust_radius_param
    
            # Оновлюємо стан для наступного кроку
            feed_fe, ore_flow = d_seq_param[k, 0], d_seq_param[k, 1]
            xk_unscaled_list.pop(0)  # Видаляємо найстарший стан
            xk_unscaled_list.append(cp.hstack([feed_fe, ore_flow, uk]))  # Додаємо новий стан
    
        conc_fe_preds = cp.hstack(pred_fe)
        conc_mass_preds = cp.hstack(pred_mass)
    
        # 4. Формування обмежень (без змін)
        cons = [u_var >= self.u_min, u_var <= self.u_max]
        du0 = u_var[0] - u_prev_param
        du_rest = u_var[1:] - u_var[:-1] if self.Nc > 1 else []
        if self.Nc > 1:
            Du_ext = cp.hstack([du0, du_rest])
        else:
            Du_ext = cp.hstack([du0])        
        
        # Обмеження на Du_ext з м'якими змінними
        cons.extend([
            Du_ext <= self.delta_u_max + eps_delta_u_upper,
            Du_ext >= -self.delta_u_max - eps_delta_u_lower
        ])
    
        y_preds_stacked = cp.vstack([conc_fe_preds, conc_mass_preds]).T
        if eps_y_upper is not None:
            cons.append(y_preds_stacked <= self.y_max + eps_y_upper)
        if eps_y_lower is not None:
            cons.append(y_preds_stacked >= self.y_min - eps_y_lower)
    
        # 5. Формування цільової функції (без змін)
        base_cost = self.objective.cost_full(
            conc_fe_preds=conc_fe_preds, conc_mass_preds=conc_mass_preds,
            u_seq=u_var, u_prev=u_prev_param
        )
        
        # Штраф за порушення Du
        penalty_cost = self.rho_delta_u * (cp.sum_squares(eps_delta_u_upper) + cp.sum_squares(eps_delta_u_lower))
        if eps_y_upper is not None:
            penalty_cost += self.rho_y * cp.sum_squares(eps_y_upper)
        if eps_y_lower is not None:
            penalty_cost += self.rho_y * cp.sum_squares(eps_y_lower)
    
        total_cost = base_cost + penalty_cost + trust_region_cost
        
        # 6. Створення об'єкту задачі
        self.problem = cp.Problem(cp.Minimize(total_cost), cons)


    def check_linearization_validity(self, X_current_scaled, X_predicted_scaled):
        """
        ПОКРАЩЕНА перевірка валідності лінеаризації.
        """
        if not self.linearization_check_enabled:
            return True, 0.0
            
        # Обчислюємо евклідову відстань
        distance = np.linalg.norm(X_predicted_scaled - X_current_scaled)
        
        # Також перевіряємо максимальну покомпонентну відстань
        max_component_distance = np.max(np.abs(X_predicted_scaled - X_current_scaled))
        
        # Використовуємо більш строгий критерій
        is_valid = (distance < self.max_linearization_distance and 
                    max_component_distance < self.max_linearization_distance * 0.8)
        
        # Зберігаємо детальнішу історію
        self.linearization_quality_history.append({
            'euclidean_distance': distance,
            'max_component_distance': max_component_distance,
            'is_valid': is_valid
        })
        
        if len(self.linearization_quality_history) > 100:  # Обмежуємо розмір
            self.linearization_quality_history.pop(0)
            
        return is_valid, distance

    def update_trust_region(self, predicted_cost_reduction, actual_cost_reduction):
        """
        ВИПРАВЛЕНИЙ метод адаптивного оновлення регіону довіри.
        """
        if not self.adaptive_trust_region:
            return
            
        # Уникаємо ділення на нуль та дуже малі значення
        if abs(predicted_cost_reduction) < 1e-6:
            return
            
        # Коефіцієнт якості прогнозу
        ratio = actual_cost_reduction / predicted_cost_reduction
        
        print(f"  -> Trust region update: ratio={ratio:.3f}, pred={predicted_cost_reduction:.4f}, actual={actual_cost_reduction:.4f}")
        
        # ВИПРАВЛЕНА логіка адаптації
        if ratio > 0.75:  # Дуже хороший прогноз
            new_radius = min(self.trust_region_radius * 1.25, self.max_trust_radius)
            print(f"  -> Збільшуємо trust region: {self.trust_region_radius:.3f} -> {new_radius:.3f}")
            self.trust_region_radius = new_radius
            
        elif ratio > 0.25:  # Прийнятний прогноз - невелике зменшення
            new_radius = max(self.trust_region_radius * 0.95, self.min_trust_radius)
            if abs(new_radius - self.trust_region_radius) > 0.01:
                print(f"  -> Легко зменшуємо trust region: {self.trust_region_radius:.3f} -> {new_radius:.3f}")
            self.trust_region_radius = new_radius
            
        else:  # Поганий прогноз - суттєве зменшення
            new_radius = max(self.trust_region_radius * 0.7, self.min_trust_radius)
            print(f"  -> Сильно зменшуємо trust region: {self.trust_region_radius:.3f} -> {new_radius:.3f}")
            self.trust_region_radius = new_radius

    def _estimate_cost_reduction(self, u_optimal, u_prev):
        """
        ПОКРАЩЕНИЙ метод оцінки зниження вартості.
        """
        try:
            if self.problem.value is None:
                return 0.0
                
            current_cost = self.problem.value
            
            if self.previous_cost is not None:
                # Фактичне зниження вартості
                actual_reduction = self.previous_cost - current_cost
                
                # Простий прогноз: базується на зміні керування
                if len(u_optimal) > 0:
                    du = abs(u_optimal[0] - u_prev)
                    # Лінійна оцінка: більша зміна керування -> більше потенційне покращення
                    predicted_reduction = du * 0.1  # Масштабуючий коефіцієнт
                else:
                    predicted_reduction = 0.0
                    
                return predicted_reduction
            else:
                return 0.0
                
        except Exception as e:
            print(f"  -> Помилка в _estimate_cost_reduction: {e}")
            return 0.0


    def optimize(self, d_seq: np.ndarray, u_prev: float) -> np.ndarray:
        """СТАБІЛІЗОВАНИЙ MPC з розумним trust region"""
        
        if self.x_hist is None:
            raise RuntimeError("Історія стану не ініціалізована.")
        
        # ВИПРАВЛЕННЯ: витягуємо останні 9 змінних для моделі
        X0_for_model = self.x_hist.flatten()[-9:].reshape(1, -1)  # (1, 9)
        X0_current_scaled = self.x_scaler.transform(X0_for_model)
        
        # Лінеаризація
        W_local, b_local = self.model.linearize(X0_current_scaled)
        
        # Оновлення параметрів
        self.parameters['W'].value = W_local
        self.parameters['b'].value = b_local
        self.parameters['x_hist'].value = self.x_hist.flatten()  # Повна історія для прогнозу
        self.parameters['u_prev'].value = u_prev
        self.parameters['d_seq'].value = d_seq
        self.parameters['x0_scaled'].value = X0_current_scaled.flatten()
        self.parameters['trust_radius'].value = max(self.trust_region_radius, 0.1)  # Мінімум
        
        d_hat_val = self.d_hat if self.use_disturbance_estimator and self.d_hat is not None else np.zeros(self.n_targets)
        self.parameters['d_hat'].value = d_hat_val
        
        # Розв'язування
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.error.SolverError:
            print("ПОПЕРЕДЖЕННЯ: Помилка солвера.")
            return np.array([u_prev] * self.Nc)
        
        if self.problem.status in ["infeasible", "unbounded"]:
            print(f"ПОПЕРЕДЖЕННЯ: Задача не має розв'язку (статус: {self.problem.status}).")
            return np.array([u_prev] * self.Nc)
        
        u_optimal = self.variables['u'].value
        if u_optimal is None:
            print("ПОПЕРЕДЖЕННЯ: Оптимальне керування не знайдено.")
            return np.array([u_prev] * self.Nc)
        
        # СТАБІЛІЗОВАНИЙ trust region update
        if self.adaptive_trust_region:
            current_cost = self.problem.value
            
            if self.previous_cost is not None and current_cost is not None:
                actual_cost_reduction = self.previous_cost - current_cost
                
                # ПРОСТІША оцінка зміни вартості
                if abs(actual_cost_reduction) > 1e-8:
                    # Просто базуємося на знаку зміни вартості
                    if actual_cost_reduction > 0:  # Покращення
                        self.trust_region_radius = min(
                            self.trust_region_radius * 1.1, 
                            self.max_trust_radius
                        )
                    else:  # Погіршення  
                        self.trust_region_radius = max(
                            self.trust_region_radius * 0.8,
                            self.min_trust_radius
                        )
            
            self.previous_cost = current_cost
        
        return u_optimal

    def get_trust_region_stats(self):
        """
        Повертає статистику про роботу адаптивного trust region.
        
        Returns:
            dict: Словник зі статистикою
        """
        stats = {
            'current_radius': self.trust_region_radius,
            'min_radius': self.min_trust_radius,
            'max_radius': self.max_trust_radius,
            'linearization_history_length': len(self.linearization_quality_history)
        }
        
        if self.linearization_quality_history:
            # ВИПРАВЛЕННЯ: правильно обробляємо історію лінеаризації
            if isinstance(self.linearization_quality_history[0], dict):
                # Якщо зберігаються словники з детальною інформацією
                distances = [h['euclidean_distance'] for h in self.linearization_quality_history]
                max_distances = [h['max_component_distance'] for h in self.linearization_quality_history]
                
                stats.update({
                    'avg_linearization_distance': np.mean(distances),
                    'max_linearization_distance': np.max(distances),
                    'recent_avg_distance': np.mean(distances[-10:]) if len(distances) >= 10 else np.mean(distances),
                    'avg_max_component_distance': np.mean(max_distances),
                    'recent_avg_max_component_distance': np.mean(max_distances[-10:]) if len(max_distances) >= 10 else np.mean(max_distances)
                })
            else:
                # Якщо зберігаються лише числа (зворотна сумісність)
                stats.update({
                    'avg_linearization_distance': np.mean(self.linearization_quality_history),
                    'max_linearization_distance': np.max(self.linearization_quality_history),
                    'recent_avg_distance': np.mean(self.linearization_quality_history[-10:]) if len(self.linearization_quality_history) >= 10 else np.mean(self.linearization_quality_history)
                })
        
        return stats

    def reset_trust_region(self):
        """Скидає адаптивний trust region до початкових налаштувань."""
        self.trust_region_radius = (self.min_trust_radius + self.max_trust_radius) / 2
        self.previous_cost = None
        self.predicted_cost_reduction = None
        self.linearization_quality_history.clear()

    # Інші методи класу залишаються без змін...
    def reset_history(self, initial_history: np.ndarray):
        """
        initial_history: numpy array форми (L+1, 3)
        кожний рядок = [feed_fe, ore_flow, u_applied]
        """
        expected = (self.L + 1, 3)
        if initial_history.shape != expected:
            raise ValueError(f"initial_history має форму {expected}, отримано {initial_history.shape}")
        self.x_hist = initial_history.copy()

    def fit(self,
            X_train: np.ndarray,
            Y_train: np.ndarray,
            x0_hist: np.ndarray = None):
        """
        Навчає KernelModel та опціонально ініціалізує історію і оцінювач збурень.
        """
        self.model.fit(X_train, Y_train)
        
        # Перевіряємо, чи була надана історія, перед тим як її встановлювати
        if x0_hist is not None:
            self.reset_history(x0_hist)

        # Ініціалізація оцінювача збурень
        if self.use_disturbance_estimator:
            self.n_targets = Y_train.shape[1]
            self.d_hat = np.zeros(self.n_targets)
            
        # Скидаємо trust region після перенавчання
        if hasattr(self, 'reset_trust_region'):
            self.reset_trust_region()