# mpc.py

import numpy as np
import cvxpy as cp
from model import KernelModel


class MPCController:
    def __init__(self,
                 model: KernelModel,
                 objective,
                 horizon: int     = 6,
                 lag: int         = 2,
                 u_min: float     = 25.0,
                 u_max: float     = 35.0,
                 delta_u_max: float = 1.0):
        """
        MPC-контролер з лінійним KernelRidge.

        model       – натренований KernelModel з kernel='linear'
        objective   – об’єкт, що має метод cost_term(y_pred:list, u, u_prev)
        horizon     – глибина прогнозу H
        lag         – довжина історії L (у x_hist має бути L+1 рядків по 3 стовпці)
        u_min,u_max – межі для змінної u
        delta_u_max – максимум зміни між кроками
        """
        if model.model_type != 'krr' or model.kernel != 'linear':
            raise ValueError("MPCController підтримує тільки model_type='krr' та kernel='linear'")
        self.model       = model
        self.objective   = objective
        self.H           = horizon
        self.L           = lag
        self.u_min       = u_min
        self.u_max       = u_max
        self.delta_u_max = delta_u_max
        self.x_hist      = None   # shape = (L+1, 3)

    def reset_history(self, initial_history: np.ndarray):
        """
        initial_history: numpy array форми (L+1, 3)
        кожний рядок = [conc_fe, ore_flow, u_applied]
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
        Навчає KernelModel та ініціалізує історію.
        Після цього можна викликати optimize().
        """
        # навчаємо модель
        self.model.fit(X_train, Y_train)

        # зберігаємо коефіцієнти лінійної регресії як константи CVXPY
        self.W_c = cp.Constant(self.model.coef_)      # shape=(n_features, n_targets)
        self.b_c = cp.Constant(self.model.intercept_) # shape=(n_targets,)

        # ініціалізуємо історію
        self.reset_history(x0_hist)

    def optimize(self,
                 d_seq: np.ndarray,
                 u_prev: float) -> np.ndarray:
        """
        d_seq: масив форми (H, 2) – послідовність зовнішніх впливів [(feed_fe, ore_flow), …]
        u_prev: попереднє прикладене керування (скаляр)
        Повертає оптимальний вектор u довжини H.
        """
        if self.x_hist is None:
            raise RuntimeError("Спочатку викличте MPCController.fit().")

        # змінна управління
        u = cp.Variable(self.H)

        # обмеження
        cons = [
            u >= self.u_min,
            u <= self.u_max
        ]
        for k in range(1, self.H):
            cons.append(cp.abs(u[k] - u[k-1]) <= self.delta_u_max)

        # копія історії як список рядків [conc_fe, ore_flow, u_expr]
        xk_list = [list(row) for row in self.x_hist]

        cost_terms = []

        for k in range(self.H):
            # 1) будуємо вектор ознак Xk: [conc_fe, ore_flow, u_k−L], … , [conc_fe, ore_flow, u_k−1]
            flat = []
            for row in xk_list:
                for v in row:
                    flat.append(v if isinstance(v, cp.Expression) else float(v))
            Xk_cvx = cp.hstack(flat)  # shape = (n_features,)

            # 2) прогноз лінійною моделлю: yk = Xk·W + b
            yk = Xk_cvx @ self.W_c + self.b_c  # shape = (n_targets,)

            # 3) додаємо вклад у вартість
            uk      = u[k]
            prev_uk = u_prev if k == 0 else u[k-1]
            # перетворюємо yk на список скалярів для objective
            n_targets = self.model.intercept_.shape[0]
            y_pred = [yk[i] for i in range(n_targets)]
            cost_terms.append(self.objective.cost_term(y_pred, uk, prev_uk))

            # 4) зсуваємо історію: додаємо новий крок із символічним uk
            feed_fe, ore_flow = d_seq[k]
            xk_list.pop(0)
            xk_list.append([float(feed_fe), float(ore_flow), uk])

        # формуємо та вирішуємо QP
        total_cost = cp.sum(cp.hstack(cost_terms))
        problem    = cp.Problem(cp.Minimize(total_cost), cons)
        problem.solve()

        return u.value