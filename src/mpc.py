# mpc.py

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
                 # <<< НОВИЙ ПАРАМЕТР для регіону довіри
                 rho_trust: float = 0.1
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
        self.n_inputs = (lag + 1) * 3 # (feed_fe, ore_flow, u) * (L+1)
        self.n_targets = n_targets

        # Збереження обмежень та ваг штрафів
        self.u_min = u_min
        self.u_max = u_max
        self.delta_u_max = delta_u_max
        self.y_max = np.array(y_max) if y_max is not None else None
        self.y_min = np.array(y_min) if y_min is not None else None
        self.rho_y = rho_y
        self.rho_delta_u = rho_delta_u
        self.rho_trust = rho_trust # <<< Зберігаємо вагу регіону довіри

        # Ініціалізація історії та оцінки збурень
        self.x_hist = None
        self.d_hat = np.zeros(self.n_targets) if self.use_disturbance_estimator else None

        # <<< Атрибути для збереження задачі CVXPY
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
        Створює задачу оптимізації CVXPY один раз з використанням параметрів.
        Це значно прискорює послідовні виклики методу optimize.
        """
        # 1. Змінні оптимізації
        u_var = cp.Variable(self.Nc, name="u_seq")
        eps_delta_u = cp.Variable(self.Nc, nonneg=True, name="eps_delta_u")
        eps_y_upper = cp.Variable((self.Np, 2), nonneg=True, name="eps_y_upper") if self.y_max is not None else None
        eps_y_lower = cp.Variable((self.Np, 2), nonneg=True, name="eps_y_lower") if self.y_min is not None else None
        self.variables = {
            'u': u_var, 'eps_delta_u': eps_delta_u,
            'eps_y_upper': eps_y_upper, 'eps_y_lower': eps_y_lower
        }

        # 2. Параметри, що будуть оновлюватись на кожному кроці
        W_param = cp.Parameter((self.n_inputs, self.n_targets), name="W_local")
        b_param = cp.Parameter(self.n_targets, name="b_local")
        x_hist_param = cp.Parameter(self.n_inputs, name="x_hist_flat")
        u_prev_param = cp.Parameter(name="u_prev")
        d_hat_param = cp.Parameter(self.n_targets, name="d_hat")
        d_seq_param = cp.Parameter((self.Np, 2), name="d_seq")
        x0_scaled_param = cp.Parameter(self.n_inputs, name="x0_scaled") # Для регіону довіри
        self.parameters = {
            'W': W_param, 'b': b_param, 'x_hist': x_hist_param,
            'u_prev': u_prev_param, 'd_hat': d_hat_param, 'd_seq': d_seq_param,
            'x0_scaled': x0_scaled_param
        }

        # Константи для масштабування
        mean_c = cp.Constant(self.x_scaler.mean_)
        scale_c = cp.Constant(self.x_scaler.scale_)

        # 3. Побудова прогнозу на горизонті Np
        pred_fe, pred_mass = [], []
        trust_region_cost = 0
        
        # Початковий стан з параметра
        xk_unscaled_list = [x_hist_param[i*3:(i+1)*3] for i in range(self.L + 1)]

        for k in range(self.Np):
            uk = u_var[k] if k < self.Nc else u_var[self.Nc - 1]
            
            # Формуємо вектор стану Xk в ОРИГІНАЛЬНОМУ масштабі
            Xk_unscaled = cp.hstack(xk_unscaled_list)
            
            # >>> КОРЕКТНЕ МАСШТАБУВАННЯ ВЕКТОРА СТАНУ <<<
            Xk_scaled = (Xk_unscaled - mean_c) / scale_c
            
            # Прогноз за локальною лінійною моделлю
            yk = Xk_scaled @ W_param + b_param + d_hat_param
            pred_fe.append(yk[0])
            pred_mass.append(yk[2]) # Індекси відповідають [conc_fe, tail_fe, conc_mass, tail_mass]

            # >>> ШТРАФ РЕГІОНУ ДОВІРИ (TRUST REGION) <<<
            # Штрафуємо за відхилення прогнозованого стану від точки лінеаризації
            if self.model.kernel != 'linear':
                 trust_region_cost += self.rho_trust * cp.sum_squares(Xk_scaled - x0_scaled_param)

            # Оновлюємо стан для наступного кроку
            feed_fe, ore_flow = d_seq_param[k, 0], d_seq_param[k, 1]
            xk_unscaled_list.pop(0)
            xk_unscaled_list.append(cp.hstack([feed_fe, ore_flow, uk]))

        conc_fe_preds = cp.hstack(pred_fe)
        conc_mass_preds = cp.hstack(pred_mass)

        # 4. Формування обмежень
        cons = [u_var >= self.u_min, u_var <= self.u_max]
        du0 = u_var[0] - u_prev_param
        du_rest = u_var[1:] - u_var[:-1] if self.Nc > 1 else []
        Du_ext = cp.hstack([du0] + ([du_rest] if self.Nc > 1 else []))
        bound = self.delta_u_max + eps_delta_u
        cons.extend([Du_ext <= bound, Du_ext >= -bound])

        y_preds_stacked = cp.vstack([conc_fe_preds, conc_mass_preds]).T
        if eps_y_upper is not None:
            cons.append(y_preds_stacked <= self.y_max + eps_y_upper)
        if eps_y_lower is not None:
            cons.append(y_preds_stacked >= self.y_min - eps_y_lower)

        # 5. Формування цільової функції
        base_cost = self.objective.cost_full(
            conc_fe_preds=conc_fe_preds, conc_mass_preds=conc_mass_preds,
            u_seq=u_var, u_prev=u_prev_param
        )
        penalty_cost = self.rho_delta_u * cp.sum_squares(eps_delta_u)
        if eps_y_upper is not None:
            penalty_cost += self.rho_y * cp.sum_squares(eps_y_upper)
        if eps_y_lower is not None:
            penalty_cost += self.rho_y * cp.sum_squares(eps_y_lower)

        total_cost = base_cost + penalty_cost + trust_region_cost
        
        # 6. Створення об'єкту задачі
        self.problem = cp.Problem(cp.Minimize(total_cost), cons)

    def optimize(self, d_seq: np.ndarray, u_prev: float) -> np.ndarray:
        """
        Знаходить оптимальну послідовність керування, оновлюючи параметри
        попередньо скомпільованої задачі оптимізації.
        """
        if self.x_hist is None:
            raise RuntimeError("Історія стану не ініціалізована. Викличте MPCController.reset_history().")
        if self.problem is None:
            raise RuntimeError("Задача оптимізації не була налаштована. Перевірте конструктор.")

        # 1. Лінеаризація моделі в поточній робочій точці
        X0_current_unscaled = self.x_hist.flatten().reshape(1, -1)
        X0_current_scaled = self.x_scaler.transform(X0_current_unscaled)
        W_local, b_local = self.model.linearize(X0_current_scaled)

        # 2. Оновлення значень параметрів у задачі CVXPY
        self.parameters['W'].value = W_local
        self.parameters['b'].value = b_local
        self.parameters['x_hist'].value = X0_current_unscaled.flatten()
        self.parameters['u_prev'].value = u_prev
        self.parameters['d_seq'].value = d_seq
        self.parameters['x0_scaled'].value = X0_current_scaled.flatten()
        
        d_hat_val = self.d_hat if self.use_disturbance_estimator and self.d_hat is not None else np.zeros(self.n_targets)
        self.parameters['d_hat'].value = d_hat_val

        # 3. Розв'язання задачі
        try:
            self.problem.solve(solver=cp.OSQP, warm_start=True)
        except cp.error.SolverError:
            print("ПОПЕРЕДЖЕННЯ: Помилка солвера. Використовується попереднє керування.")
            return np.array([u_prev] * self.Nc)

        # 4. Діагностика та повернення результату
        if self.problem.status not in ["infeasible", "unbounded"]:
            u_optimal = self.variables['u'].value
            # Діагностичний вивід
            if (self.variables['eps_y_upper'] is not None and 
                np.any(self.variables['eps_y_upper'].value > 1e-4)):
                print(f"  -> УВАГА: Порушено верхнє обмеження Y! ε_y_upper = {np.round(self.variables['eps_y_upper'].value.flatten(), 3)}")
            
            if (self.variables['eps_delta_u'] is not None and 
                np.any(self.variables['eps_delta_u'].value > 1e-4)):
                print(f"  -> УВАГА: Порушено обмеження Δu! ε_Δu = {np.round(self.variables['eps_delta_u'].value, 3)}")
        else:
            print(f"ПОПЕРЕДЖЕННЯ: Задача оптимізації не має розв'язку (статус: {self.problem.status}). Використовується попереднє керування.")
            return np.array([u_prev] * self.Nc)

        return u_optimal

    # Інші методи класу (reset_history, fit, update_disturbance) залишаються без змін
    # ... (скопіюйте сюди решту методів з вашого файлу)

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
            x0_hist: np.ndarray = None): # <<< Додаємо значення за замовчуванням
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

    # 4. Новий метод для оновлення оцінки
    def update_disturbance(self, y_meas_k_unscaled: np.ndarray):
        """
        Оновлює оцінку збурення. Всі розрахунки відбуваються в масштабованому просторі.
        """
        if not self.use_disturbance_estimator or self.d_hat is None:
            return
    
        # 1. Беремо історію в оригінальному масштабі і масштабуємо її
        Xk_minus_1_unscaled = self.x_hist.flatten().reshape(1, -1)
        Xk_minus_1_scaled = self.x_scaler.transform(Xk_minus_1_unscaled)
    
        # 2. Робимо прогноз на масштабованих даних, отримуємо масштабований вихід
        y_pred_k_scaled = self.model.predict(Xk_minus_1_scaled)[0]
    
        # 3. Масштабуємо реальне вимірювання, щоб воно було в тому ж просторі
        y_meas_k_scaled = self.y_scaler.transform(y_meas_k_unscaled.reshape(1, -1))[0]
    
        # 4. Розраховуємо збурення в МАСШТАБОВАНОМУ просторі
        raw_disturbance = y_meas_k_scaled - y_pred_k_scaled
    
        # 5. Застосовуємо фільтр (d_hat також зберігається в масштабованому вигляді)
        alpha_filter = 0.1 
        self.d_hat = alpha_filter * raw_disturbance + (1 - alpha_filter) * self.d_hat

