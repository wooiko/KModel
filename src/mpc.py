# mpc.py

import warnings
import numpy as np
import cvxpy as cp
from model import KernelModel


class MPCController:
    def __init__(self,
                 model: KernelModel,
                 objective,
                 horizon: int = 6,
                 control_horizon: int = None,
                 lag: int = 2,
                 u_min: float = 25.0,
                 u_max: float = 35.0,
                 delta_u_max: float = 1.0,
                 use_disturbance_estimator: bool = True,
                 # <<< ПОЧАТОК НОВИХ ПАРАМЕТРІВ ДЛЯ М'ЯКИХ ОБМЕЖЕНЬ
                 y_max: list = None,          # Верхні межі для виходів [max_fe, max_mass]
                 y_min: list = None,          # Нижні межі для виходів [min_fe, min_mass]
                 rho_y: float = 1e6,          # Штраф за порушення обмежень по Y
                 rho_delta_u: float = 1e4     # Штраф за порушення обмежень по Δu
                 # >>> КІНЕЦЬ НОВИХ ПАРАМЕТРІВ
                 ):
        if model.model_type != 'krr' or model.kernel != 'linear':
            raise ValueError("MPCController підтримує тільки model_type='krr' та kernel='linear'")
        self.model = model
        self.objective = objective
        self.Np = horizon
        self.Nc = control_horizon if control_horizon is not None else horizon
        if self.Nc > self.Np:
            raise ValueError("control_horizon (Nc) не може бути більше за horizon (Np)")
        self.L = lag
        # Зберігаємо жорсткі та м'які обмеження
        self.u_min = u_min
        self.u_max = u_max
        self.delta_u_max = delta_u_max
        self.x_hist = None
        self.use_disturbance_estimator = use_disturbance_estimator
        self.d_hat = None
        self.n_targets = None
        
        # <<< Зберігаємо нові параметри
        self.y_max = np.array(y_max) if y_max is not None else None
        self.y_min = np.array(y_min) if y_min is not None else None
        self.rho_y = rho_y
        self.rho_delta_u = rho_delta_u

        # Приховати конкретне попередження UserWarning від cvxpy щодо бекенду
        warnings.filterwarnings(
            "ignore",
            message="The problem includes expressions that don't support CPP backend. Defaulting to the SCIPY backend for canonicalization.",
            category=UserWarning,
            module="cvxpy.reductions.solvers.solving_chain_utils"
        )

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
            x0_hist: np.ndarray):
        """
        Навчає KernelModel та ініціалізує історію і оцінювач збурень.
        """
        self.model.fit(X_train, Y_train)
        self.W_c = cp.Constant(self.model.coef_)
        self.b_c = cp.Constant(self.model.intercept_)
        self.reset_history(x0_hist)

        # 3. Ініціалізація оцінювача
        if self.use_disturbance_estimator:
            self.n_targets = Y_train.shape[1]
            self.d_hat = np.zeros(self.n_targets)

    # 4. Новий метод для оновлення оцінки
    def update_disturbance(self, y_meas_k: np.ndarray):
        """
        Оновлює оцінку постійного збурення на виході: d_hat(k) = y_meas(k) - y_pred(k).
        Цей метод слід викликати на кожному кроці симуляції ПЕРЕД викликом optimize().
        """
        if not self.use_disturbance_estimator or self.d_hat is None:
            return

        # Формуємо вектор X(k-1) з поточної історії self.x_hist
        Xk_minus_1 = self.x_hist.flatten().reshape(1, -1)

        # Робимо прогноз моделі БЕЗ збурення
        y_pred_k = self.model.predict(Xk_minus_1)[0]

        # Оновлюємо оцінку збурення (можна додати фільтрацію для згладжування)
        # self.d_hat = y_meas_k - y_pred_k

        # Покращена реалізація (з фільтрацією)
        raw_disturbance = y_meas_k - y_pred_k 
        
        # Застосовуємо експоненційний фільтр
        alpha_filter = 0.1 # Коефіцієнт згладжування
        self.d_hat = alpha_filter * raw_disturbance + (1 - alpha_filter) * self.d_hat

    def optimize(self,
                 d_seq: np.ndarray,
                 u_prev: float) -> np.ndarray:
        if self.x_hist is None:
            raise RuntimeError("Спочатку викличте MPCController.fit().")

        # 1. Основна керована змінна
        u_var = cp.Variable(self.Nc)

        # 2. Створюємо змінні ослаблення (Slack Variables)
        eps_delta_u = cp.Variable(self.Nc, nonneg=True)
        eps_y_upper = cp.Variable((self.Np, 2), nonneg=True) if self.y_max is not None else None
        eps_y_lower = cp.Variable((self.Np, 2), nonneg=True) if self.y_min is not None else None

        # 3. Жорсткі обмеження (залишаємо тільки на саму змінну u)
        cons = [
            u_var >= self.u_min,
            u_var <= self.u_max,
        ]

        # Побудова прогнозів
        d_hat_c = cp.Constant(self.d_hat) if self.use_disturbance_estimator and self.d_hat is not None else cp.Constant(np.zeros(self.n_targets))
        xk_list = [list(row) for row in self.x_hist]
        pred_fe, pred_mass = [], []
        for k in range(self.Np):
            uk = u_var[k] if k < self.Nc else u_var[self.Nc - 1]
            flat = [v if isinstance(v, cp.Expression) else float(v) for row in xk_list for v in row]
            Xk_cvx = cp.hstack(flat)
            yk = Xk_cvx @ self.W_c + self.b_c + d_hat_c
            pred_fe.append(yk[0])
            pred_mass.append(yk[2])
            feed_fe, ore_flow = d_seq[k]
            xk_list.pop(0)
            xk_list.append([float(feed_fe), float(ore_flow), uk])
        conc_fe_preds = cp.hstack(pred_fe)
        conc_mass_preds = cp.hstack(pred_mass)

        # 4. Додаємо м'які обмеження
        # Обмеження на Δu
        du0 = u_var[0] - u_prev
        # ▼▼▼ ВИПРАВЛЕНИЙ РЯДОК ▼▼▼
        du_rest = u_var[1:] - u_var[:-1] if self.Nc > 1 else []
        # ▲▲▲ ВИПРАВЛЕНИЙ РЯДОК ▲▲▲
        Du_ext = cp.hstack([du0] + ([du_rest] if self.Nc > 1 else []))
        cons.append(cp.abs(Du_ext) <= self.delta_u_max + eps_delta_u)
        
        # Обмеження на виходи Y
        y_preds_stacked = cp.vstack([conc_fe_preds, conc_mass_preds]).T 
        if eps_y_upper is not None:
            cons.append(y_preds_stacked <= self.y_max + eps_y_upper)
        if eps_y_lower is not None:
            cons.append(y_preds_stacked >= self.y_min - eps_y_lower)

        # 5. Розрахунок основної цільової функції
        base_cost = self.objective.cost_full(
            conc_fe_preds=conc_fe_preds,
            conc_mass_preds=conc_mass_preds,
            u_seq=u_var, # тут u_var передається в аргумент u_seq
            u_prev=u_prev
        )

        # 6. Додаємо штрафи за ослаблення до цільової функції
        penalty_cost = self.rho_delta_u * cp.sum_squares(eps_delta_u)
        
        if eps_y_upper is not None:
            penalty_cost += self.rho_y * cp.sum_squares(eps_y_upper)
        if eps_y_lower is not None:
            penalty_cost += self.rho_y * cp.sum_squares(eps_y_lower)
            
        total_cost = base_cost + penalty_cost
        
        # 7. Розв'язуємо задачу
        problem = cp.Problem(cp.Minimize(total_cost), cons)
        problem.solve(solver=cp.OSQP, warm_start=True)

        # ▼▼▼ ДОДАЙТЕ ЦЕЙ БЛОК ДЛЯ ДІАГНОСТИКИ ▼▼▼
        if problem.status not in ["infeasible", "unbounded"]:
            # Перевіряємо, чи були використані змінні ослаблення для Y
            if eps_y_upper is not None and np.any(eps_y_upper.value > 1e-4):
                print(f"  -> УВАГА: Порушено верхнє обмеження Y! ε_y_upper = {np.round(eps_y_upper.value.flatten(), 3)}")
            # Перевіряємо, чи були використані змінні ослаблення для Δu
            if np.any(eps_delta_u.value > 1e-4):
                print(f"  -> УВАГА: Порушено обмеження Δu! ε_Δu = {np.round(eps_delta_u.value, 3)}")
        # ▲▲▲ КІНЕЦЬ ДІАГНОСТИЧНОГО БЛОКУ ▲▲▲

        if problem.status in ["infeasible", "unbounded"]:
             print("ПОПЕРЕДЖЕННЯ: Задача оптимізації не має розв'язку. Використовується попереднє керування.")
             return np.array([u_prev] * self.Nc)

        return u_var.value