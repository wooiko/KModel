
# mpc.py

import numpy as np
import cvxpy as cp
from model import KernelModel
from observer import KalmanDisturbanceObserver
from noise_constants import (
    ERROR_PERCENTS_NULL,
    ERROR_PERCENTS_LOW,
    ERROR_PERCENTS_MEDIUM,
    ERROR_PERCENTS_HIGH,
    ERROR_RATIOS,
    Y_MEANS,
)


class MPCController:
    def __init__(self,
                 model: KernelModel,
                 objective,
                 horizon: int         = 6,
                 control_horizon: int = None,
                 lag: int             = 2,
                 u_min: float         = 25.0,
                 u_max: float         = 35.0,
                 delta_u_max: float   = 1.0):
        # Перевірка моделі
        if model.model_type != 'krr' or model.kernel != 'linear':
            raise ValueError("MPCController підтримує тільки model_type='krr' та kernel='linear'")
        self.model     = model
        self.objective = objective

        # MPC-параметри
        self.Np = horizon
        self.Nc = control_horizon or horizon
        self.L  = lag
        self.u_min, self.u_max, self.delta_u_max = u_min, u_max, delta_u_max
        self.x_hist = None

        # Спостерігач: матриці A, C для лінійної моделі
        output_size = 3  # concentrate_fe, concentrate_mass_flow, tailings_fe
        A_bar = np.eye(output_size)
        C_bar = np.eye(output_size)

        self.observer = KalmanDisturbanceObserver(
            A_bar, C_bar,
            ERROR_PERCENTS_LOW, ERROR_RATIOS, Y_MEANS,
            lowpass_alpha=0.3, anomaly_thresh=5.0,
            r_scale=0.1,
            q_state_scale=0.01,
            q_dist_scales={
                'concentrate_fe_percent': 1.0e-3,   # сильно пригальмувати перший канал
                'concentrate_mass_flow':    1.0e-3, # пом’якшити середній канал
                'tailings_fe_percent':      1.0e-3, # пригальмувати третій канал
            }
        )


    def reset_history(self, initial_history: np.ndarray):
        expected = (self.L + 1, 3)
        if initial_history.shape != expected:
            raise ValueError(f"initial_history має форму {expected}, "
                             f"отримано {initial_history.shape}")
        self.x_hist = initial_history.copy()

    def fit(self,
            X_train: np.ndarray,
            Y_train: np.ndarray,
            x0_hist: np.ndarray):
        self.model.fit(X_train, Y_train)
        # Зберігаємо коефіцієнти як CVXPY-константи
        self.W_c = cp.Constant(self.model.coef_)    # shape (3, input_size)
        self.b_c = cp.Constant(self.model.intercept_)  # shape (3,)
        self.reset_history(x0_hist)

    def optimize(self,
                 d_seq: np.ndarray,
                 u_prev: float,
                 y_meas: np.ndarray) -> np.ndarray:
        if self.x_hist is None:
            raise RuntimeError("Спочатку викличте fit()")

        # 1) Оновлення спостерігача
        self.observer.predict(np.array([u_prev]))
        xbar = self.observer.update(y_meas)
        d_hat = xbar[self.observer.n:]  # оцінене збурення
        print(f"Оцінене збурення d̂ = {d_hat}") 

        # 2) Формуємо змінні і обмеження
        u_var = cp.Variable(self.Nc)
        cons = [
            u_var >= self.u_min,
            u_var <= self.u_max,
            cp.abs(u_var[0] - u_prev) <= self.delta_u_max
        ]
        for k in range(1, self.Nc):
            cons.append(cp.abs(u_var[k] - u_var[k-1]) <= self.delta_u_max)

        # 3) Прогнозування за горизонтом
        pred_fe, pred_mass = [], []
        # Копіюємо історію у список списків
        xk_list = [list(row) for row in self.x_hist]

        for k in range(self.Np):
            uk = u_var[k] if k < self.Nc else u_var[self.Nc-1]
        
            # збираємо вектор ознак із історії
            feats = [elem for row in xk_list for elem in row]
            Xk_cvx = cp.hstack(feats)   # CVXPY вектор форми (n_features,)
        
            # прогноз (лінійна модель)
            yk = Xk_cvx @ self.W_c + self.b_c   # (n_outputs,)
        
            # КОРЕКТНО: додаємо оцінене збурення!
            alpha_ff = 0.5   # 0…1
            pred_fe.append(   yk[0] + d_hat[0] )
            pred_mass.append( yk[1] + alpha_ff*d_hat[1] )
        
            # оновлюємо історію
            feed_fe, ore_flow = d_seq[k]
            xk_list.pop(0)
            xk_list.append([feed_fe, ore_flow, uk])

        conc_fe_preds   = cp.hstack(pred_fe)
        conc_mass_preds = cp.hstack(pred_mass)

        # 4) Обчислення вартості і рішення
        total_cost = self.objective.cost_full(
            conc_fe_preds, conc_mass_preds, u_var, u_prev
        )
        problem = cp.Problem(cp.Minimize(total_cost), cons)
        problem.solve(solver=cp.OSQP)

        if u_var.value is None:
            raise RuntimeError("MPC optimization failed: no solution")
        return u_var.value.flatten()
