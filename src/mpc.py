
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
                 delta_u_max: float  = 1.0):
        """
        MPC-контролер з лінійним KernelRidge.

        model             – натренований KernelModel з model_type='krr' та kernel='linear'
        objective         – об’єкт, що має метод cost_term(y_pred:list, u, u_prev)
        horizon           – прогнозний горизонт Np
        control_horizon   – горизонт керування Nc (Nc ≤ Np). Якщо None, то Nc = Np
        lag               – довжина історії L (x_hist має L+1 рядків по 3 стовпці)
        u_min, u_max      – межі для змінної u
        delta_u_max       – максимум зміни між кроками
        """
        if model.model_type != 'krr' or model.kernel != 'linear':
            raise ValueError("MPCController підтримує тільки model_type='krr' та kernel='linear'")
        self.model        = model
        self.objective    = objective
        # прогнозний та контрольний горизонти
        self.Np           = horizon
        self.Nc           = control_horizon if control_horizon is not None else horizon
        if self.Nc > self.Np:
            raise ValueError("control_horizon (Nc) не може бути більше за horizon (Np)")
        self.L            = lag
        self.u_min        = u_min
        self.u_max        = u_max
        self.delta_u_max  = delta_u_max
        self.x_hist       = None    # shape = (L+1, 3)

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
        d_seq: масив форми (Np, 2) – послідовність зовнішніх впливів [(feed_fe, ore_flow), …]
        u_prev: попереднє прикладене керування (скаляр)
        Повертає оптимальний вектор u довжини Nc.
        """
        if self.x_hist is None:
            raise RuntimeError("Спочатку викличте MPCController.fit().")

        # 1) Змінна управління довжини Nc
        u_var = cp.Variable(self.Nc)

        # 2) Обмеження на u_var та Δu
        cons = [
            u_var >= self.u_min,
            u_var <= self.u_max,
            cp.abs(u_var[0] - u_prev) <= self.delta_u_max
        ]
        for k in range(1, self.Nc):
            cons.append(cp.abs(u_var[k] - u_var[k-1]) <= self.delta_u_max)

        # 3) Побудова прогнозів на всьому Np кроків
        xk_list = [list(row) for row in self.x_hist]  # копія історії
        pred_fe   = []  # сюди збиратимемо conc_fe_preds
        pred_mass = []  # сюди — conc_mass_preds

        for k in range(self.Np):
            # вибираємо керування: перші Nc – оптимізовані, далі – останній
            uk = u_var[k] if k < self.Nc else u_var[self.Nc - 1]

            # Будуємо вектор ознак Xk
            flat = []
            for row in xk_list:
                for v in row:
                    flat.append(v if isinstance(v, cp.Expression) else float(v))
            Xk_cvx = cp.hstack(flat)  # shape = (n_features,)

            # Прогноз лінійною моделлю: yk = Xk·W + b
            yk = Xk_cvx @ self.W_c + self.b_c  # shape = (n_targets,)

            # Збираємо перший та третій компоненти
            pred_fe.append(   yk[0] )
            pred_mass.append( yk[2] )

            # Оновлюємо історію для наступного кроку
            feed_fe, ore_flow = d_seq[k]
            xk_list.pop(0)
            xk_list.append([
                float(feed_fe),
                float(ore_flow),
                uk
            ])

        # 4) Формуємо вектори прогнозів
        conc_fe_preds   = cp.hstack(pred_fe)    # shape = (Np,)
        conc_mass_preds = cp.hstack(pred_mass)  # shape = (Np,)

        # 5) Закликаємо векторизовану ціль
        total_cost = self.objective.cost_full(
            conc_fe_preds=conc_fe_preds,
            conc_mass_preds=conc_mass_preds,
            u_seq=u_var,
            u_prev=u_prev
        )

        # 6) Розв’язуємо QP
        problem = cp.Problem(cp.Minimize(total_cost), cons)
        problem.solve(solver=cp.OSQP)

        return u_var.value
