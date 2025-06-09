# mpc.py

import numpy as np
import cvxpy as cp
from model import KernelModel


class MPCController:
    def __init__(self,
                 model: KernelModel,
                 objective,
                 horizon: int        = 6,
                 control_horizon: int = None,
                 lag: int            = 2,
                 u_min: float        = 25.0,
                 u_max: float        = 35.0,
                 delta_u_max: float  = 1.0,
                 use_disturbance_estimator: bool = True): # 1. Новий параметр
        """
        MPC-контролер з лінійним KernelRidge.

        ...
        use_disturbance_estimator – чи використовувати оцінювач збурення для ліквідації зсуву
        """
        if model.model_type != 'krr' or model.kernel != 'linear':
            raise ValueError("MPCController підтримує тільки model_type='krr' та kernel='linear'")
        self.model        = model
        self.objective    = objective
        self.Np           = horizon
        self.Nc           = control_horizon if control_horizon is not None else horizon
        if self.Nc > self.Np:
            raise ValueError("control_horizon (Nc) не може бути більше за horizon (Np)")
        self.L            = lag
        self.u_min        = u_min
        self.u_max        = u_max
        self.delta_u_max  = delta_u_max
        self.x_hist       = None

        # 2. Нові атрибути для оцінювача
        self.use_disturbance_estimator = use_disturbance_estimator
        self.d_hat = None      # Оцінка збурення, shape=(n_targets,)
        self.n_targets = None

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
        self.d_hat = y_meas_k - y_pred_k

    def optimize(self,
                 d_seq: np.ndarray,
                 u_prev: float) -> np.ndarray:
        if self.x_hist is None:
            raise RuntimeError("Спочатку викличте MPCController.fit().")

        u_var = cp.Variable(self.Nc)
        
        # 5. Створюємо константу CVXPY для поточної оцінки збурення
        d_hat_c = cp.Constant(np.zeros(self.n_targets))
        if self.use_disturbance_estimator and self.d_hat is not None:
            d_hat_c = cp.Constant(self.d_hat)

        cons = [
            u_var >= self.u_min,
            u_var <= self.u_max,
            cp.abs(u_var[0] - u_prev) <= self.delta_u_max
        ]
        if self.Nc > 1:
            cons.append(cp.abs(u_var[1:] - u_var[:-1]) <= self.delta_u_max)

        xk_list = [list(row) for row in self.x_hist]
        pred_fe, pred_mass = [], []

        for k in range(self.Np):
            uk = u_var[k] if k < self.Nc else u_var[self.Nc - 1]
            flat = [v if isinstance(v, cp.Expression) else float(v) for row in xk_list for v in row]
            Xk_cvx = cp.hstack(flat)

            # 6. Ключова зміна: додаємо оцінку збурення до прогнозу
            yk = Xk_cvx @ self.W_c + self.b_c + d_hat_c

            pred_fe.append(yk[0])
            pred_mass.append(yk[2])

            feed_fe, ore_flow = d_seq[k]
            xk_list.pop(0)
            xk_list.append([float(feed_fe), float(ore_flow), uk])

        conc_fe_preds = cp.hstack(pred_fe)
        conc_mass_preds = cp.hstack(pred_mass)

        total_cost = self.objective.cost_full(
            conc_fe_preds=conc_fe_preds,
            conc_mass_preds=conc_mass_preds,
            u_seq=u_var,
            u_prev=u_prev
        )

        problem = cp.Problem(cp.Minimize(total_cost), cons)
        problem.solve(solver=cp.OSQP, warm_start=True)

        return u_var.value if u_var.value is not None else np.array([u_prev] * self.Nc)