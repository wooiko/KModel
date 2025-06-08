# mpc.py

import numpy as np
import cvxpy as cp

from model import KernelModel
from kalman_observer import DisturbanceObserverKalman
from anomaly_detector import AnomalyDetector


class MPCController:
    def __init__(self,
                 model: KernelModel,
                 objective,
                 horizon: int = 6,
                 control_horizon: int = None,
                 lag: int = 2,
                 u_min: float = 25.0,
                 u_max: float = 35.0,
                 delta_u_max: float = 1.0):
        """
        MPC-контролер з лінійним KernelRidge + обробка вхідних/вихідних шумів.
        """
        if model.model_type != 'krr' or model.kernel != 'linear':
            raise ValueError("MPCController підтримує тільки model_type='krr' та kernel='linear'")
        self.model       = model
        self.objective   = objective
        self.Np          = horizon
        self.Nc          = control_horizon if control_horizon is not None else horizon
        if self.Nc > self.Np:
            raise ValueError("control_horizon (Nc) не може бути більше за horizon (Np)")
        self.L           = lag
        self.u_min       = u_min
        self.u_max       = u_max
        self.delta_u_max = delta_u_max
        self.x_hist      = None   # історія розміром (L+1, 3)

        # --- Обробка вхідних сигналів ---
        self.ad_feed     = AnomalyDetector(window=10, z_thresh=4.5)
        self.ad_ore      = AnomalyDetector(window=10, z_thresh=4.5)
        self.d_obs_feed  = DisturbanceObserverKalman(Q=1e-2, R=0.001, P=100)
        self.d_obs_ore   = DisturbanceObserverKalman(Q=1e-2, R=0.001, P=100)

        # --- Спостерігач збурень для виходів ---
        self.d_obs_fe    = DisturbanceObserverKalman()
        self.d_obs_mass  = DisturbanceObserverKalman()

    def reset_history(self, initial_history: np.ndarray):
        """
        initial_history: numpy array форми (L+1, 3)
        кожний рядок = [feed_fe_percent, ore_mass_flow, solid_feed_percent]
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
        """
        # 1) Навчання прогнозної моделі
        self.model.fit(X_train, Y_train)

        # 2) Перетворюємо коефіцієнти на константи CVXPY
        self.W_c = cp.Constant(self.model.coef_)      # (n_features×n_targets)
        self.b_c = cp.Constant(self.model.intercept_) # (n_targets,)

        # 3) Ініціалізуємо історію
        self.reset_history(x0_hist)

    def optimize(self, d_seq: np.ndarray, u_prev: float) -> np.ndarray:
        """
        d_seq: масив форми (Np, 2) – послідовність зовнішніх впливів
               [(feed_fe_percent, ore_mass_flow), …]
        u_prev: попереднє прикладене керування
        Повертає оптимальний вектор u довжини Nc.
        """
        if self.x_hist is None:
            raise RuntimeError("Спочатку викличте MPCController.fit().")
    
        # 1) Змінна керування
        u_var = cp.Variable(self.Nc)
    
        # 2) Обмеження на u та Δu
        cons = [
            u_var >= self.u_min,
            u_var <= self.u_max,
            cp.abs(u_var[0] - u_prev) <= self.delta_u_max
        ]
        for k in range(1, self.Nc):
            cons.append(cp.abs(u_var[k] - u_var[k-1]) <= self.delta_u_max)
    
        # 3) Побудова прогнозу з «чистими» сигналами
        xk_list = [list(row) for row in self.x_hist]
        pred_fe = []
        pred_mass = []
    
        for k in range(self.Np):
            # а) вибір uk
            uk = u_var[k] if k < self.Nc else u_var[self.Nc - 1]
    
            # b) Формуємо Xk з історії
            flat = []
            for row in xk_list:
                for v in row:
                    flat.append(v if isinstance(v, cp.Expression) else float(v))
            Xk_cvx = cp.hstack(flat)
    
            # c) Прогноз моделі
            yk = Xk_cvx @ self.W_c + self.b_c
    
            # d) Offset-free корекція виходів
            d_fe_const = cp.Constant(self.d_obs_fe.d_est)
            d_mass_const = cp.Constant(self.d_obs_mass.d_est)
            yk_augmented = cp.hstack([
                yk[0] + d_fe_const,  # conc_fe + offset
                yk[1],               # tail_fe
                yk[2] + d_mass_const,  # conc_mass + offset
                yk[3]                # tail_mass
            ])
    
            # e) Збираємо прогнозні вектори
            pred_fe.append(yk_augmented[0])
            pred_mass.append(yk_augmented[2])
    
            # f) Оновлення історії:
            raw_feed, raw_ore = d_seq[k]
    
            # — Корекція аномалій
            corr_feed = self.ad_feed.correct(raw_feed)
            corr_ore  = self.ad_ore.correct(raw_ore)
    
            # — Згладжування шуму Калман-спостерігачем
            smooth_feed = self.d_obs_feed.update(corr_feed, corr_feed)
            smooth_ore  = self.d_obs_ore.update(corr_ore, corr_ore)
    
            xk_list.pop(0)
            xk_list.append([
                float(smooth_feed),
                float(smooth_ore),
                uk
            ])
    
        # 4) Фінальні прогнозні вектори
        conc_fe_preds = cp.hstack(pred_fe)
        conc_mass_preds = cp.hstack(pred_mass)
    
        # 5) Цільова функція
        total_cost = self.objective.cost_full(
            conc_fe_preds=conc_fe_preds,
            conc_mass_preds=conc_mass_preds,
            u_seq=u_var,
            u_prev=u_prev
        )
    
        # 6) Розвʼязок QP із try/except для обробки workspace allocation error
        problem = cp.Problem(cp.Minimize(total_cost), cons)
        try:
            problem.solve(
                solver=cp.OSQP,
                warm_start=True,
                eps_abs=1e-4,
                eps_rel=1e-4,
                max_iter=10000,
                verbose=False
            )
        except cp.SolverError:
            try:
                problem.solve(solver=cp.SCS, verbose=False, max_iters=25000)
            except cp.SolverError as e:
                raise cp.SolverError("Не вдалося розв’язати QP задачу за допомогою OSQP або SCS.") from e
    
        if u_var.value is None:
            raise RuntimeError("MPC optimization returned None. Перевірте налаштування QP та масштабування даних.")
    
        return u_var.value